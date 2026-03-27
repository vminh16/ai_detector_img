from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import JpegImagePlugin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OLD_MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
V4_MANIFEST_CSV = PROJECT_ROOT / "data" / "processed_v4_rgb248_r4_exact" / "manifest.csv"
OLD_FEATURE_CSV = PROJECT_ROOT / "features" / "features_dataset.csv"

OUT_ROOT = PROJECT_ROOT / "audit_output" / "studies" / "feature_spec_v1_validation_20260325"
OUT_METRICS = OUT_ROOT / "feature_set_metrics.csv"
OUT_SINGLE = OUT_ROOT / "single_feature_metrics.csv"
OUT_SHIFT = OUT_ROOT / "feature_shift_metrics.csv"
OUT_SUMMARY = OUT_ROOT / "summary.json"

RNG_SEED = 20260325
EPS = 1e-12
NOISE_FLOOR = 1.53e-5
SAMPLING_WORKERS = 8

LABEL_SAMPLE_PER_GROUP = 100
REAL_NUISANCE_PER_CLASS = 700
SHIFT_SAMPLE_PER_GROUP = 80

SUBSAMPLING_MAP = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
DEGRADATIONS = (
    "clean",
    "jpeg95_420",
    "jpeg90_420",
    "resize50_bilinear",
    "resize75_bilinear",
    "resize50_jpeg90_420",
)


def _json_ready_key(key: object) -> str:
    if isinstance(key, tuple):
        return " | ".join(str(part) for part in key)
    return str(key)


def _json_ready_dict(mapping: dict[object, object]) -> dict[str, object]:
    return {_json_ready_key(key): value for key, value in mapping.items()}


def _stable_sample(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    take = min(n, len(group))
    if take == len(group):
        return group.copy()
    return group.sample(n=take, random_state=seed)


def _rel_from_old_raw_path(path_str: str) -> str | None:
    path_str = str(path_str).replace("\\", "/")
    marker = "/data/raw/"
    if marker not in path_str:
        return None
    return path_str.split(marker, 1)[1]


def _processed_rel_from_old_feature_path(path_str: str) -> str | None:
    path_str = str(path_str).replace("\\", "/")
    marker = "/data/processed/"
    if marker not in path_str:
        return None
    return path_str.split(marker, 1)[1]


def load_common_tables() -> pd.DataFrame:
    old = pd.read_csv(
        OLD_MANIFEST_CSV,
        usecols=["file_path", "generator", "label", "action"],
    )
    v4 = pd.read_csv(
        V4_MANIFEST_CSV,
        usecols=["file_path", "generator", "label", "status", "output_path"],
    )

    old["relative_path"] = old["file_path"].map(_rel_from_old_raw_path)
    old["old_output_path"] = old["relative_path"].map(
        lambda rel: str((PROJECT_ROOT / "data" / "processed" / Path(rel).with_suffix(".npy")).resolve())
        if isinstance(rel, str)
        else None
    )
    old = old.loc[old["action"] == "processed"].copy()

    raw_root_resolved = RAW_ROOT.resolve()
    v4["relative_path"] = v4["file_path"].map(
        lambda p: str(Path(p).resolve().relative_to(raw_root_resolved)).replace("\\", "/")
    )
    v4 = v4.loc[v4["status"] == "ACCEPTED"].copy()
    v4 = v4.rename(columns={"output_path": "v4_output_path"})

    common = old.merge(
        v4[["relative_path", "v4_output_path"]],
        on="relative_path",
        how="inner",
    )

    common = common[["relative_path", "generator", "label", "old_output_path", "v4_output_path"]].copy()
    common["generator_norm"] = common["generator"].astype(str)
    common["label_norm"] = common["label"].replace({"nature": "nature", "ai": "ai"})
    return common


def load_old_baseline33(common_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = pd.read_csv(OLD_FEATURE_CSV)
    feature_df["processed_rel"] = feature_df["file_path"].map(_processed_rel_from_old_feature_path)
    common_map = common_df.copy()
    common_map["processed_rel"] = common_map["relative_path"].map(
        lambda rel: str(Path(rel).with_suffix(".npy")).replace("\\", "/")
    )
    merged = common_map.merge(feature_df, on="processed_rel", how="inner", suffixes=("", "_feat"))
    return merged


def detect_sampling(path_str: str) -> str | None:
    try:
        with Image.open(path_str) as im:
            return SUBSAMPLING_MAP.get(JpegImagePlugin.get_sampling(im), "other")
    except Exception:
        return None


def build_label_sample(common_df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    per_group = 24 if smoke else LABEL_SAMPLE_PER_GROUP
    rng = random.Random(RNG_SEED)
    parts: list[pd.DataFrame] = []
    for (_, _), group in common_df.groupby(["generator_norm", "label_norm"], sort=True):
        parts.append(_stable_sample(group, per_group, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def build_shift_sample(common_df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    per_group = 20 if smoke else SHIFT_SAMPLE_PER_GROUP
    rng = random.Random(RNG_SEED + 5)
    parts: list[pd.DataFrame] = []
    for (_, _), group in common_df.groupby(["generator_norm", "label_norm"], sort=True):
        parts.append(_stable_sample(group, per_group, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def build_real_nuisance_sample(common_df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    subset = common_df.loc[common_df["label_norm"] == "nature"].copy()
    subset["raw_path"] = subset["relative_path"].map(lambda rel: str((RAW_ROOT / rel).resolve()))
    subset["raw_ext"] = subset["raw_path"].map(lambda p: Path(p).suffix.lower())
    subset = subset.loc[subset["raw_ext"].isin([".jpg", ".jpeg"])].copy()
    if subset.empty:
        raise RuntimeError("No common accepted real JPEG rows found.")

    sample_cap = 1200 if smoke else min(6000, len(subset))
    sampled = subset.sample(n=sample_cap, random_state=RNG_SEED + 10).copy()
    sampling: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=SAMPLING_WORKERS) as pool:
        futs = {pool.submit(detect_sampling, path): idx for idx, path in sampled["raw_path"].items()}
        for fut in as_completed(futs):
            sampling[futs[fut]] = fut.result()
    sampled["jpeg_subsampling"] = pd.Series(sampling)
    sampled = sampled.loc[sampled["jpeg_subsampling"].isin(["4:4:4", "4:2:0"])].copy()
    per_class = 160 if smoke else REAL_NUISANCE_PER_CLASS
    rng = random.Random(RNG_SEED + 1)
    parts: list[pd.DataFrame] = []
    for subsampling, group in sampled.groupby("jpeg_subsampling", sort=True):
        parts.append(_stable_sample(group, per_class, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def load_rgb_from_preprocess(path: str, preprocess_version: str) -> np.ndarray:
    arr = np.load(path)
    if preprocess_version == "old_v1":
        if arr.shape != (256, 256, 3):
            raise ValueError(f"Unexpected old patch shape: {arr.shape}")
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    if preprocess_version == "v4_exact":
        if arr.shape != (248, 248, 3):
            raise ValueError(f"Unexpected v4 patch shape: {arr.shape}")
        return arr.astype(np.uint8, copy=False)
    raise ValueError(f"Unknown preprocess_version: {preprocess_version}")


def rgb_to_ycrcb(arr_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)


def jpeg_roundtrip_rgb(arr_rgb: np.ndarray, quality: int, subsampling: int) -> np.ndarray:
    image = Image.fromarray(arr_rgb.astype(np.uint8), mode="RGB")
    bio = io.BytesIO()
    image.save(bio, format="JPEG", quality=int(quality), subsampling=int(subsampling))
    bio.seek(0)
    with Image.open(bio) as decoded:
        out = np.array(decoded.convert("RGB"))
    return out


def resize_roundtrip_rgb(arr_rgb: np.ndarray, scale: float) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    new_h = max(16, int(round(h * scale)))
    new_w = max(16, int(round(w * scale)))
    down = cv2.resize(arr_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.rint(up), 0, 255).astype(np.uint8)


def apply_degradation(arr_rgb: np.ndarray, name: str) -> np.ndarray:
    if name == "clean":
        return arr_rgb
    if name == "jpeg95_420":
        return jpeg_roundtrip_rgb(arr_rgb, quality=95, subsampling=2)
    if name == "jpeg90_420":
        return jpeg_roundtrip_rgb(arr_rgb, quality=90, subsampling=2)
    if name == "resize50_bilinear":
        return resize_roundtrip_rgb(arr_rgb, scale=0.50)
    if name == "resize75_bilinear":
        return resize_roundtrip_rgb(arr_rgb, scale=0.75)
    if name == "resize50_jpeg90_420":
        return jpeg_roundtrip_rgb(resize_roundtrip_rgb(arr_rgb, scale=0.50), quality=90, subsampling=2)
    raise ValueError(f"Unknown degradation: {name}")


def _convolve_valid(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(channel, (kh, kw))
    return np.tensordot(windows, kernel[::-1, ::-1], axes=((2, 3), (0, 1)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b))) + EPS
    return float(np.sum(a * b) / denom)


def _safe_excess_kurtosis(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= EPS:
        return 0.0
    return float(np.mean(x ** 4) / (var * var) - 3.0)


def _safe_skew(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= EPS:
        return 0.0
    sigma = math.sqrt(var)
    return float(np.mean(x ** 3) / (sigma ** 3 + EPS))


def _build_lbp_lut_59() -> np.ndarray:
    lut = np.empty(256, dtype=np.int32)
    uniform_idx = 0
    for code in range(256):
        transitions = 0
        for k in range(8):
            b_curr = (code >> k) & 1
            b_next = (code >> ((k + 1) % 8)) & 1
            if b_curr != b_next:
                transitions += 1
        if transitions <= 2:
            lut[code] = uniform_idx
            uniform_idx += 1
        else:
            lut[code] = 58
    return lut


LBP_LUT_59 = _build_lbp_lut_59()
LBP_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1), (0, 1),
    (1, 1), (1, 0), (1, -1), (0, -1),
]
LAP = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
KERNEL_SQUARE3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64)
KERNEL_EDGE3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
KERNEL_SQUARE5 = 0.25 * np.array(
    [[0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [1, -2, 4, -2, 1], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0]],
    dtype=np.float64,
)


def extract_frequency_features(y: np.ndarray) -> dict[str, float]:
    f = np.fft.fft2(y) / y.size
    power = np.abs(np.fft.fftshift(f)) ** 2
    h, w = y.shape
    yy, xx = np.indices((h, w))
    cy = h // 2
    cx = w // 2
    rr = np.rint(np.hypot(yy - cy, xx - cx)).astype(np.int32)
    max_r = int(rr.max())
    ring_sum = np.bincount(rr.ravel(), weights=power.ravel(), minlength=max_r + 1)
    ring_count = np.bincount(rr.ravel(), minlength=max_r + 1).astype(np.float64)
    ring = ring_sum / np.maximum(ring_count, 1.0)

    mid = ring[8:33]
    mid_mean = float(np.mean(mid))
    frs = 0.0 if mid_mean < NOISE_FLOOR else float(np.var(mid) / (mid_mean ** 2 + EPS))

    fit_r = np.arange(20, min(65, len(ring)), dtype=np.float64)
    fit_y = ring[20:min(65, len(ring))]
    if fit_y.size == 0 or np.all(fit_y < NOISE_FLOOR):
        return {"frs_mid_variance": frs, "ps_alpha": 0.0, "ps_deviation_variance": 0.0}

    fit_y = np.log(np.maximum(fit_y, NOISE_FLOOR))
    fit_x = np.log(fit_r)
    x_center = fit_x - fit_x.mean()
    y_center = fit_y - fit_y.mean()
    slope = float(np.sum(x_center * y_center) / (np.sum(x_center * x_center) + EPS))
    intercept = float(fit_y.mean() - slope * fit_x.mean())
    pred = slope * fit_x + intercept
    dev = float(np.var(fit_y - pred))
    return {
        "frs_mid_variance": frs,
        "ps_alpha": float(-slope),
        "ps_deviation_variance": dev,
    }


def extract_color_global(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0]
    cr = arr_ycc[:, :, 1]
    cb = arr_ycc[:, :, 2]
    var_y = float(np.var(y))
    return {
        "pearson_y_cr": _safe_corr(y, cr),
        "pearson_y_cb": _safe_corr(y, cb),
        "pearson_cr_cb": _safe_corr(cr, cb),
        "energy_ratio_chroma": float((np.var(cr) + np.var(cb)) / (var_y + EPS)),
    }


def extract_spatial_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64)
    cr = arr_ycc[:, :, 1].astype(np.float64) - 128.0
    cb = arr_ycc[:, :, 2].astype(np.float64) - 128.0
    ry = _convolve_valid(y, LAP)
    rcr = _convolve_valid(cr, LAP)
    rcb = _convolve_valid(cb, LAP)

    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)[1:-1, 1:-1]
    q_lo = float(np.quantile(grad, 0.25))
    q_hi = float(np.quantile(grad, 0.75))
    flat = grad <= q_lo
    edge = grad >= q_hi
    if int(flat.sum()) < 32:
        flat = grad <= float(np.quantile(grad, 0.4))
    if int(edge.sum()) < 32:
        edge = grad >= float(np.quantile(grad, 0.6))

    abs_ry = np.abs(ry)
    v_edge = float(np.mean(abs_ry[edge])) if int(edge.sum()) > 0 else 0.0
    v_flat = float(np.mean(abs_ry[flat])) if int(flat.sum()) > 0 else 0.0
    snr = 0.0 if v_flat < 1e-6 else float(np.log10((v_edge + NOISE_FLOOR) / (v_flat + NOISE_FLOOR)))

    noise_y = float(np.mean(np.abs(ry)))
    noise_cb = float(np.mean(np.abs(rcb)))
    cross = float((noise_y + NOISE_FLOOR) / (noise_cb + NOISE_FLOOR))

    return {
        "spatial_snr_ratio": snr,
        "cross_noise_ratio": cross,
        "skew_noise_y": _safe_skew(ry),
        "kurt_noise_y": _safe_excess_kurtosis(ry),
        "skew_noise_cr": _safe_skew(rcr),
        "kurt_noise_cr": _safe_excess_kurtosis(rcr),
        "skew_noise_cb": _safe_skew(rcb),
        "kurt_noise_cb": _safe_excess_kurtosis(rcb),
    }


def _periodicity_energies(residual: np.ndarray, prefix: str) -> dict[str, float]:
    h, w = residual.shape
    yy, xx = np.indices((h, w))
    basis = {
        f"{prefix}_pi_x": np.where(xx % 2 == 0, 1.0, -1.0),
        f"{prefix}_pi_y": np.where(yy % 2 == 0, 1.0, -1.0),
        f"{prefix}_pi_xy": np.where((xx + yy) % 2 == 0, 1.0, -1.0),
    }
    energy = float(np.sum(residual * residual))
    denom = energy * residual.size + EPS
    return {name: float((np.sum(residual * mask) ** 2) / denom) for name, mask in basis.items()}


def extract_cfa_chroma(arr_ycc: np.ndarray) -> dict[str, float]:
    cr = arr_ycc[:, :, 1] - 128.0
    cb = arr_ycc[:, :, 2] - 128.0
    cr_res = _convolve_valid(cr, LAP)
    cb_res = _convolve_valid(cb, LAP)
    out: dict[str, float] = {}
    out.update(_periodicity_energies(cr_res, "cfa_cr"))
    out.update(_periodicity_energies(cb_res, "cfa_cb"))
    return out


def extract_cfa_rgbdiff(arr_rgb: np.ndarray) -> dict[str, float]:
    rgb = arr_rgb.astype(np.float64)
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    bg = rgb[:, :, 2] - rgb[:, :, 1]
    rg_res = _convolve_valid(rg, LAP)
    bg_res = _convolve_valid(bg, LAP)
    out: dict[str, float] = {}
    out.update(_periodicity_energies(rg_res, "cfa_rg"))
    out.update(_periodicity_energies(bg_res, "cfa_bg"))
    return out


def _block_view(arr: np.ndarray, block: int) -> np.ndarray:
    h = (arr.shape[0] // block) * block
    w = (arr.shape[1] // block) * block
    trimmed = arr[:h, :w]
    return trimmed.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def extract_noise_level_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64)
    block = 8
    blocks = _block_view(y, block)
    means = blocks.mean(axis=(2, 3)).ravel() / 255.0
    dx = np.diff(blocks, axis=3)
    dy = np.diff(blocks, axis=2)
    roughness = (np.mean(np.abs(dx), axis=(2, 3)) + np.mean(np.abs(dy), axis=(2, 3))).ravel()

    residual = _convolve_valid(y, LAP)
    residual_blocks = _block_view(residual, block - 2)
    variances = residual_blocks.var(axis=(2, 3)).ravel() / (255.0 * 255.0)

    size = min(means.size, variances.size)
    means = means[:size]
    roughness = roughness[:size]
    variances = variances[:size]
    flat_threshold = np.quantile(roughness, 0.3)
    keep = roughness <= flat_threshold
    if int(keep.sum()) < 16:
        keep = np.ones_like(roughness, dtype=bool)

    m = means[keep]
    v = variances[keep]
    order = np.argsort(m)
    m = m[order]
    v = v[order]

    mr = _rankdata(m)
    vr = _rankdata(v)
    spearman = _safe_corr(mr, vr)

    centered_m = m - m.mean()
    centered_v = v - v.mean()
    denom = float(np.sum(centered_m * centered_m))
    slope = float(np.sum(centered_m * centered_v) / (denom + EPS))
    intercept = float(v.mean() - slope * m.mean())
    fit = slope * m + intercept
    sse = float(np.sum((v - fit) ** 2))
    sst = float(np.sum((v - v.mean()) ** 2))
    r2 = float(max(0.0, 1.0 - sse / (sst + EPS)))

    bins = np.quantile(m, np.linspace(0.0, 1.0, 7))
    bin_medians: list[float] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        mask = (m >= left) & (m <= right) if right == bins[-1] else (m >= left) & (m < right)
        if int(mask.sum()) >= 2:
            bin_medians.append(float(np.median(v[mask])))
    if len(bin_medians) >= 2:
        diffs = np.diff(np.asarray(bin_medians))
        violation = float(np.mean(diffs < 0.0))
    else:
        violation = 0.0

    return {
        "nlf_spearman": spearman,
        "nlf_slope": slope,
        "nlf_intercept": intercept,
        "nlf_r2": r2,
        "nlf_monotone_violation": violation,
    }


def _haar_level(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x00 = channel[0::2, 0::2]
    x01 = channel[0::2, 1::2]
    x10 = channel[1::2, 0::2]
    x11 = channel[1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _match_parent_child(child: np.ndarray, parent: np.ndarray) -> float:
    child_abs = np.abs(child)
    grouped = child_abs.reshape(parent.shape[0], 2, parent.shape[1], 2).mean(axis=(1, 3))
    return _safe_corr(grouped.ravel(), np.abs(parent).ravel())


def extract_wavelet_dependency_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64)
    ll1, lh1, hl1, hh1 = _haar_level(y)
    ll2, lh2, hl2, hh2 = _haar_level(ll1)
    return {
        "wav_parent_corr_h": _match_parent_child(lh1, lh2),
        "wav_parent_corr_v": _match_parent_child(hl1, hl2),
        "wav_parent_corr_d": _match_parent_child(hh1, hh2),
        "wav_kurtosis_l1": float(np.mean([
            _safe_excess_kurtosis(lh1),
            _safe_excess_kurtosis(hl1),
            _safe_excess_kurtosis(hh1),
        ])),
        "wav_kurtosis_l2": float(np.mean([
            _safe_excess_kurtosis(lh2),
            _safe_excess_kurtosis(hl2),
            _safe_excess_kurtosis(hh2),
        ])),
    }


def _compute_srm_stats(channel: np.ndarray, prefix: str) -> dict[str, float]:
    kernels = (
        ("square3", KERNEL_SQUARE3),
        ("edge3", KERNEL_EDGE3),
        ("square5", KERNEL_SQUARE5),
    )
    out: dict[str, float] = {}
    for name, kernel in kernels:
        residual = _convolve_valid(channel, kernel)
        out[f"{prefix}_{name}_mar"] = float(np.mean(np.abs(residual)))
        out[f"{prefix}_{name}_energy"] = float(np.mean(residual * residual))
    return out


def _lbp_raw_codes(channel: np.ndarray) -> np.ndarray:
    center = channel[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(LBP_OFFSETS):
        neigh = channel[1 + dy: channel.shape[0] - 1 + dy, 1 + dx: channel.shape[1] - 1 + dx]
        codes |= ((neigh >= center).astype(np.uint8) << bit)
    return codes


def _extract_lbp_stats(channel: np.ndarray, prefix: str) -> dict[str, float]:
    codes = _lbp_raw_codes(channel.astype(np.float64))
    mapped = LBP_LUT_59[codes]
    hist = np.bincount(mapped.ravel(), minlength=59).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    nz = probs[probs > 0]
    return {
        f"{prefix}_nonuniform_ratio": float(probs[58]),
        f"{prefix}_entropy": float(-np.sum(nz * np.log2(nz))),
    }


def extract_feature_dict(arr_rgb: np.ndarray) -> dict[str, float]:
    arr_ycc = rgb_to_ycrcb(arr_rgb)
    y = arr_ycc[:, :, 0]
    cr = arr_ycc[:, :, 1] - 128.0
    out: dict[str, float] = {}
    out.update(extract_frequency_features(y))
    out.update(extract_color_global(arr_ycc))
    out.update(extract_spatial_features(arr_ycc))
    out.update(extract_cfa_chroma(arr_ycc))
    out.update(extract_cfa_rgbdiff(arr_rgb))
    out.update(extract_noise_level_features(arr_ycc))
    out.update(extract_wavelet_dependency_features(arr_ycc))
    out.update(_compute_srm_stats(y, "ysrm"))
    out.update(_compute_srm_stats(cr, "crsrm"))
    out.update(_extract_lbp_stats(y, "ylbp"))
    out.update(_extract_lbp_stats(arr_ycc[:, :, 1], "crlbp"))
    out.update(_extract_lbp_stats(arr_ycc[:, :, 2], "cblbp"))
    return out


KEEP_CORE_KEYS = (
    "frs_mid_variance",
    "wav_parent_corr_h",
    "wav_parent_corr_v",
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
    "spatial_snr_ratio",
    "skew_noise_y",
    "kurt_noise_y",
)

WAVELET_FULL_KEYS = (
    "wav_parent_corr_h",
    "wav_parent_corr_v",
    "wav_parent_corr_d",
    "wav_kurtosis_l1",
    "wav_kurtosis_l2",
)

WAVELET_PARENT_KEYS = (
    "wav_parent_corr_h",
    "wav_parent_corr_v",
)

NLF_KEYS = (
    "nlf_spearman",
    "nlf_slope",
    "nlf_intercept",
    "nlf_r2",
    "nlf_monotone_violation",
)

CFA_CHROMA_KEYS = (
    "cfa_cr_pi_x",
    "cfa_cr_pi_y",
    "cfa_cr_pi_xy",
    "cfa_cb_pi_x",
    "cfa_cb_pi_y",
    "cfa_cb_pi_xy",
)

CFA_RGB_KEYS = (
    "cfa_rg_pi_x",
    "cfa_rg_pi_y",
    "cfa_rg_pi_xy",
    "cfa_bg_pi_x",
    "cfa_bg_pi_y",
    "cfa_bg_pi_xy",
)

YSRM_KEYS = (
    "ysrm_square3_mar",
    "ysrm_square3_energy",
    "ysrm_edge3_mar",
    "ysrm_edge3_energy",
    "ysrm_square5_mar",
    "ysrm_square5_energy",
)

CRSRM_KEYS = (
    "crsrm_square3_mar",
    "crsrm_square3_energy",
    "crsrm_edge3_mar",
    "crsrm_edge3_energy",
    "crsrm_square5_mar",
    "crsrm_square5_energy",
)

YLBP_KEYS = (
    "ylbp_nonuniform_ratio",
    "ylbp_entropy",
)

CHROMA_LBP_KEYS = (
    "crlbp_nonuniform_ratio",
    "crlbp_entropy",
    "cblbp_nonuniform_ratio",
    "cblbp_entropy",
)

TOXIC_CONTROL_KEYS = (
    "cross_noise_ratio",
    "ps_alpha",
    "ps_deviation_variance",
    "skew_noise_cr",
    "kurt_noise_cr",
    "skew_noise_cb",
    "kurt_noise_cb",
)

ALL_FEATURE_KEYS = tuple(dict.fromkeys(
    list(KEEP_CORE_KEYS)
    + list(WAVELET_FULL_KEYS)
    + list(NLF_KEYS)
    + list(CFA_CHROMA_KEYS)
    + list(CFA_RGB_KEYS)
    + list(YSRM_KEYS)
    + list(CRSRM_KEYS)
    + list(YLBP_KEYS)
    + list(CHROMA_LBP_KEYS)
    + list(TOXIC_CONTROL_KEYS)
))

BASELINE33_KEYS = (
    "frs_mid_variance",
    "dct_mid_mean",
    "dct_mid_variance",
    "dct_mid_skewness",
    "ps_alpha",
    "ps_deviation_variance",
    "local_color_inconsistency",
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
    "glcm_contrast_cr",
    "glcm_correlation_cr",
    "glcm_energy_cr",
    "glcm_homogeneity_cr",
    "srm_square3_mar_cr",
    "srm_square3_energy_cr",
    "srm_edge3_mar_cr",
    "srm_edge3_energy_cr",
    "srm_square5_mar_cr",
    "srm_square5_energy_cr",
    "lbp_nonuniform_ratio_cr",
    "lbp_entropy_cr",
    "lbp_nonuniform_ratio_cb",
    "lbp_entropy_cb",
    "spatial_snr_ratio",
    "cross_noise_ratio",
    "skew_noise_y",
    "kurt_noise_y",
    "skew_noise_cr",
    "kurt_noise_cr",
    "skew_noise_cb",
    "kurt_noise_cb",
)

FEATURE_SETS = {
    "safe_core": list(KEEP_CORE_KEYS),
    "safe_core_plus_ysrm": list(KEEP_CORE_KEYS + YSRM_KEYS),
    "safe_core_plus_cfa_rgb": list(KEEP_CORE_KEYS + CFA_RGB_KEYS),
    "safe_core_plus_cfa_chroma": list(KEEP_CORE_KEYS + CFA_CHROMA_KEYS),
    "safe_core_plus_wavelet_full": list(KEEP_CORE_KEYS + WAVELET_FULL_KEYS),
    "safe_core_plus_nlf": list(KEEP_CORE_KEYS + NLF_KEYS),
    "safe_core_plus_ytexture": list(KEEP_CORE_KEYS + YSRM_KEYS + YLBP_KEYS),
    "priority_bundle": list(KEEP_CORE_KEYS + WAVELET_PARENT_KEYS + YSRM_KEYS + CFA_RGB_KEYS),
    "ysrm_only": list(YSRM_KEYS),
    "ylbp_only": list(YLBP_KEYS),
    "cfa_rgb_only": list(CFA_RGB_KEYS),
    "cfa_chroma_only": list(CFA_CHROMA_KEYS),
    "wavelet_parent_only": list(WAVELET_PARENT_KEYS),
    "wavelet_full_only": list(WAVELET_FULL_KEYS),
    "nlf_only": list(NLF_KEYS),
    "crsrm_only": list(CRSRM_KEYS),
    "chroma_lbp_only": list(CHROMA_LBP_KEYS),
    "toxic_controls": list(TOXIC_CONTROL_KEYS),
}


def extract_record(row: pd.Series, preprocess_version: str, degradation: str) -> dict[str, object]:
    path_col = "old_output_path" if preprocess_version == "old_v1" else "v4_output_path"
    rgb = load_rgb_from_preprocess(str(row[path_col]), preprocess_version=preprocess_version)
    rgb = apply_degradation(rgb, degradation)
    feats = extract_feature_dict(rgb)
    record = {
        "relative_path": row["relative_path"],
        "generator_norm": row["generator_norm"],
        "label_norm": row["label_norm"],
        "preprocess_version": preprocess_version,
        "degradation": degradation,
    }
    if "jpeg_subsampling" in row:
        record["jpeg_subsampling"] = row["jpeg_subsampling"]
    record.update(feats)
    return record


def build_feature_table(df: pd.DataFrame, preprocess_version: str, degradation: str, workers: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(extract_record, row, preprocess_version, degradation) for _, row in df.iterrows()]
        for fut in as_completed(futs):
            records.append(fut.result())
    out = pd.DataFrame(records)
    return out.sort_values("relative_path").reset_index(drop=True)


def evaluate_logo_auc(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str = "generator_norm") -> float:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    preds = np.full(len(df), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    for train_idx, test_idx in splitter.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    return float(roc_auc_score(y, preds))


def evaluate_crossdeg_logo_auc(clean_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> float:
    clean_df = clean_df.sort_values("relative_path").reset_index(drop=True)
    test_df = test_df.sort_values("relative_path").reset_index(drop=True)
    y_test = (test_df["label_norm"] == "ai").astype(int).to_numpy()
    groups = clean_df["generator_norm"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    preds = np.full(len(test_df), np.nan, dtype=np.float64)
    for group in unique_groups:
        train_mask = clean_df["generator_norm"] != group
        test_mask = test_df["generator_norm"] == group
        pipe.fit(clean_df.loc[train_mask, feature_cols], (clean_df.loc[train_mask, "label_norm"] == "ai").astype(int))
        preds[np.asarray(test_mask)] = pipe.predict_proba(test_df.loc[test_mask, feature_cols])[:, 1]
    return float(roc_auc_score(y_test, preds))


def single_feature_metrics(
    clean_df: pd.DataFrame,
    nuisance_df: pd.DataFrame,
    degraded_tables: dict[str, pd.DataFrame],
    preprocess_version: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clean_df = clean_df.copy()
    clean_df["target"] = (clean_df["label_norm"] == "ai").astype(int)
    nuisance_df = nuisance_df.copy()
    nuisance_df["target"] = (nuisance_df["jpeg_subsampling"] == "4:2:0").astype(int)
    for feature in ALL_FEATURE_KEYS:
        rows.append({
            "preprocess_version": preprocess_version,
            "task": "label_logo_clean",
            "feature": feature,
            "auc": evaluate_logo_auc(clean_df[[feature, "target", "generator_norm"]], [feature], "target"),
        })
        rows.append({
            "preprocess_version": preprocess_version,
            "task": "real_jpeg_444_vs_420",
            "feature": feature,
            "auc": evaluate_logo_auc(nuisance_df[[feature, "target", "generator_norm"]], [feature], "target"),
        })
        for degradation in ("jpeg90_420", "resize50_bilinear", "resize50_jpeg90_420"):
            rows.append({
                "preprocess_version": preprocess_version,
                "task": f"xdeg_{degradation}",
                "feature": feature,
                "auc": evaluate_crossdeg_logo_auc(clean_df, degraded_tables[degradation], [feature]),
            })
    return rows


def compute_shift_table(clean_df: pd.DataFrame, degraded_df: pd.DataFrame, preprocess_version: str, label_name: str, degradation: str) -> pd.DataFrame:
    clean_df = clean_df.sort_values("relative_path").reset_index(drop=True)
    degraded_df = degraded_df.sort_values("relative_path").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for feature in ALL_FEATURE_KEYS:
        base = clean_df[feature].to_numpy(dtype=np.float64)
        shifted = degraded_df[feature].to_numpy(dtype=np.float64)
        std = float(np.std(base)) + EPS
        rows.append({
            "preprocess_version": preprocess_version,
            "label": label_name,
            "degradation": degradation,
            "feature": feature,
            "mean_delta": float(np.mean(shifted - base)),
            "mean_abs_z_shift": float(np.mean(np.abs((shifted - base) / std))),
            "median_abs_z_shift": float(np.median(np.abs((shifted - base) / std))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=SAMPLING_WORKERS)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    common = load_common_tables()
    legacy33 = load_old_baseline33(common)
    label_sample = build_label_sample(common, smoke=args.smoke)
    shift_sample = build_shift_sample(common, smoke=args.smoke)
    nuisance_sample = build_real_nuisance_sample(common, smoke=args.smoke)

    label_tables: dict[tuple[str, str], pd.DataFrame] = {}
    shift_tables: dict[tuple[str, str], pd.DataFrame] = {}
    nuisance_tables: dict[str, pd.DataFrame] = {}

    for preprocess_version in ("old_v1", "v4_exact"):
        for degradation in DEGRADATIONS:
            label_tables[(preprocess_version, degradation)] = build_feature_table(
                label_sample,
                preprocess_version=preprocess_version,
                degradation=degradation,
                workers=args.workers,
            )
        for degradation in DEGRADATIONS:
            shift_tables[(preprocess_version, degradation)] = build_feature_table(
                shift_sample,
                preprocess_version=preprocess_version,
                degradation=degradation,
                workers=args.workers,
            )
        nuisance_tables[preprocess_version] = build_feature_table(
            nuisance_sample,
            preprocess_version=preprocess_version,
            degradation="clean",
            workers=args.workers,
        )

    metric_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []
    shift_frames: list[pd.DataFrame] = []

    for preprocess_version in ("old_v1", "v4_exact"):
        clean_df = label_tables[(preprocess_version, "clean")].copy()
        clean_df["target"] = (clean_df["label_norm"] == "ai").astype(int)

        nuisance_df = nuisance_tables[preprocess_version].copy()
        nuisance_df["target"] = (nuisance_df["jpeg_subsampling"] == "4:2:0").astype(int)

        for set_name, cols in FEATURE_SETS.items():
            clean_auc = evaluate_logo_auc(clean_df[cols + ["target", "generator_norm"]], cols, "target")
            metric_rows.append({
                "preprocess_version": preprocess_version,
                "task": "label_logo_clean",
                "feature_set": set_name,
                "n_features": len(cols),
                "auc": clean_auc,
                "gap_vs_clean": 0.0,
            })

            nuisance_auc = evaluate_logo_auc(nuisance_df[cols + ["target", "generator_norm"]], cols, "target")
            metric_rows.append({
                "preprocess_version": preprocess_version,
                "task": "real_jpeg_444_vs_420",
                "feature_set": set_name,
                "n_features": len(cols),
                "auc": nuisance_auc,
                "gap_vs_clean": float(clean_auc - nuisance_auc),
            })

            for degradation in DEGRADATIONS[1:]:
                degraded_df = label_tables[(preprocess_version, degradation)]
                auc = evaluate_crossdeg_logo_auc(clean_df, degraded_df, cols)
                metric_rows.append({
                    "preprocess_version": preprocess_version,
                    "task": f"xdeg_{degradation}",
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "auc": auc,
                    "gap_vs_clean": float(clean_auc - auc),
                })

        single_rows.extend(
            single_feature_metrics(
                clean_df=label_tables[(preprocess_version, "clean")],
                nuisance_df=nuisance_tables[preprocess_version],
                degraded_tables={name: label_tables[(preprocess_version, name)] for name in DEGRADATIONS[1:]},
                preprocess_version=preprocess_version,
            )
        )

        for label_name in ("nature", "ai"):
            base = shift_tables[(preprocess_version, "clean")].loc[
                shift_tables[(preprocess_version, "clean")]["label_norm"] == label_name
            ].copy()
            for degradation in DEGRADATIONS[1:]:
                shifted = shift_tables[(preprocess_version, degradation)].loc[
                    shift_tables[(preprocess_version, degradation)]["label_norm"] == label_name
                ].copy()
                shift_frames.append(
                    compute_shift_table(
                        clean_df=base,
                        degraded_df=shifted,
                        preprocess_version=preprocess_version,
                        label_name=label_name,
                        degradation=degradation,
                    )
                )

    legacy_label = legacy33.loc[legacy33["relative_path"].isin(set(label_sample["relative_path"]))].copy()
    legacy_label["target"] = (legacy_label["label_norm"] == "ai").astype(int)
    legacy_nuisance = legacy33.merge(
        nuisance_sample[["relative_path", "jpeg_subsampling"]],
        on="relative_path",
        how="inner",
    )
    legacy_nuisance["target"] = (legacy_nuisance["jpeg_subsampling"] == "4:2:0").astype(int)
    metric_rows.append({
        "preprocess_version": "old_v1_baseline33",
        "task": "label_logo_clean",
        "feature_set": "baseline33",
        "n_features": len(BASELINE33_KEYS),
        "auc": evaluate_logo_auc(legacy_label[list(BASELINE33_KEYS) + ["target", "generator_norm"]], list(BASELINE33_KEYS), "target"),
        "gap_vs_clean": 0.0,
    })
    legacy_clean_auc = metric_rows[-1]["auc"]
    metric_rows.append({
        "preprocess_version": "old_v1_baseline33",
        "task": "real_jpeg_444_vs_420",
        "feature_set": "baseline33",
        "n_features": len(BASELINE33_KEYS),
        "auc": evaluate_logo_auc(legacy_nuisance[list(BASELINE33_KEYS) + ["target", "generator_norm"]], list(BASELINE33_KEYS), "target"),
        "gap_vs_clean": float(legacy_clean_auc - evaluate_logo_auc(legacy_nuisance[list(BASELINE33_KEYS) + ["target", "generator_norm"]], list(BASELINE33_KEYS), "target")),
    })

    metrics_df = pd.DataFrame(metric_rows).sort_values(["preprocess_version", "task", "auc"], ascending=[True, True, False])
    single_df = pd.DataFrame(single_rows).sort_values(["preprocess_version", "task", "auc"], ascending=[True, True, False])
    shift_df = pd.concat(shift_frames, ignore_index=True).sort_values(
        ["preprocess_version", "label", "degradation", "mean_abs_z_shift"],
        ascending=[True, True, True, False],
    )

    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    single_df.to_csv(OUT_SINGLE, index=False, encoding="utf-8-sig")
    shift_df.to_csv(OUT_SHIFT, index=False, encoding="utf-8-sig")

    summary = {
        "smoke": bool(args.smoke),
        "samples": {
            "common_rows": int(len(common)),
            "label_rows": int(len(label_sample)),
            "shift_rows": int(len(shift_sample)),
            "real_nuisance_rows": int(len(nuisance_sample)),
            "label_by_generator_label": _json_ready_dict(label_sample.groupby(["generator_norm", "label_norm"]).size().to_dict()),
            "real_nuisance_by_subsampling": _json_ready_dict(nuisance_sample.groupby("jpeg_subsampling").size().to_dict()),
        },
        "best_label_auc_by_preprocess": _json_ready_dict(
            metrics_df.loc[metrics_df["task"] == "label_logo_clean"].groupby("preprocess_version")["auc"].max().round(6).to_dict()
        ),
        "best_crossdeg_auc_by_preprocess_task": _json_ready_dict(
            metrics_df.loc[metrics_df["task"].str.startswith("xdeg_")].groupby(["preprocess_version", "task"])["auc"].max().round(6).to_dict()
        ),
        "files": {
            "metrics_csv": str(OUT_METRICS),
            "single_csv": str(OUT_SINGLE),
            "shift_csv": str(OUT_SHIFT),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_SINGLE}")
    print(f"Saved: {OUT_SHIFT}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
