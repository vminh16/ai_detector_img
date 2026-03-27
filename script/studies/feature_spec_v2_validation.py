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

OUT_ROOT = PROJECT_ROOT / "audit_output" / "studies" / "feature_spec_v2_validation_20260325"
OUT_METRICS = OUT_ROOT / "feature_set_metrics.csv"
OUT_SINGLE = OUT_ROOT / "single_feature_metrics.csv"
OUT_SHIFT = OUT_ROOT / "feature_shift_metrics.csv"
OUT_SET_SHIFT = OUT_ROOT / "feature_set_shift_metrics.csv"
OUT_FEATURE_GATE = OUT_ROOT / "feature_gate_summary.csv"
OUT_SUMMARY = OUT_ROOT / "summary.json"

RNG_SEED = 20260325
EPS = 1e-12
NOISE_FLOOR = 1.53e-5
SAMPLING_WORKERS = 8

LABEL_SAMPLE_PER_GROUP = 120
SHIFT_SAMPLE_PER_GROUP = 96
SHIFT_GATE_DEGRADATION = "resize50_bilinear"
STRICT_CLEAN_AUC_MIN = 0.75
STRICT_NUISANCE_AUC_MAX = 0.65
STRICT_SHIFT_MAX = 1.0

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
    per_class = 160 if smoke else None
    rng = random.Random(RNG_SEED + 1)
    groups = {name: group for name, group in sampled.groupby("jpeg_subsampling", sort=True)}
    if "4:4:4" not in groups or "4:2:0" not in groups:
        raise RuntimeError("Expected both 4:4:4 and 4:2:0 in real nuisance sample.")
    if per_class is None:
        per_class = min(len(groups["4:4:4"]), len(groups["4:2:0"]))
    parts: list[pd.DataFrame] = [
        _stable_sample(groups["4:4:4"], per_class, rng.randint(0, 1_000_000)),
        _stable_sample(groups["4:2:0"], per_class, rng.randint(0, 1_000_000)),
    ]
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


def _safe_log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(np.log((a + NOISE_FLOOR) / (b + NOISE_FLOOR)))


def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 8:
        return 0.0, 0.0
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = float(np.sum(x_center * x_center))
    if denom <= EPS:
        return 0.0, 0.0
    slope = float(np.sum(x_center * y_center) / denom)
    intercept = float(y.mean() - slope * x.mean())
    pred = slope * x + intercept
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(max(0.0, 1.0 - sse / (sst + EPS)))
    return slope, r2


def _box_mean(arr: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.boxFilter(
        arr.astype(np.float64),
        ddepth=-1,
        ksize=(ksize, ksize),
        normalize=True,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _quantile_mask(values: np.ndarray, q: float, kind: str) -> np.ndarray:
    threshold = float(np.quantile(values, q))
    if kind == "low":
        mask = values <= threshold
    else:
        mask = values >= threshold
    return mask


def _boundary_keep_mask(
    shape: tuple[int, int],
    phases: tuple[int, ...],
    *,
    width: int,
) -> np.ndarray:
    yy, xx = np.indices(shape)
    near = np.zeros(shape, dtype=bool)
    for phase in phases:
        xmod = (xx - phase) % 8
        ymod = (yy - phase) % 8
        near |= (xmod <= width) | (xmod >= 8 - width) | (ymod <= width) | (ymod >= 8 - width)
    return ~near


def _crop_center_mask(mask: np.ndarray, residual_shape: tuple[int, int], kernel_size: int) -> np.ndarray:
    offset = kernel_size // 2
    h, w = residual_shape
    return mask[offset : offset + h, offset : offset + w]


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


def _compute_masked_srm_stats(
    channel: np.ndarray,
    prefix: str,
    phases: tuple[int, ...],
    *,
    width: int = 1,
) -> dict[str, float]:
    kernels = (
        ("square3", KERNEL_SQUARE3),
        ("edge3", KERNEL_EDGE3),
        ("square5", KERNEL_SQUARE5),
    )
    base_mask = _boundary_keep_mask(channel.shape, phases=phases, width=width)
    out: dict[str, float] = {}
    for name, kernel in kernels:
        residual = _convolve_valid(channel, kernel)
        mask = _crop_center_mask(base_mask, residual.shape, kernel.shape[0])
        values = residual[mask]
        if values.size < 128:
            out[f"{prefix}_{name}_mar"] = 0.0
            out[f"{prefix}_{name}_energy"] = 0.0
            continue
        out[f"{prefix}_{name}_mar"] = float(np.mean(np.abs(values)))
        out[f"{prefix}_{name}_energy"] = float(np.mean(values * values))
    return out


def extract_local_hetero_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64) / 255.0
    residual = _convolve_valid(y, LAP)
    grad_x = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)[1:-1, 1:-1]

    local_mean = _box_mean(y, 7)[1:-1, 1:-1]
    local_var = np.maximum(_box_mean(residual * residual, 7), 0.0)

    flat_mask = _quantile_mask(grad_mag, 0.35, "low")
    edge_mask = _quantile_mask(grad_mag, 0.80, "high")
    if int(flat_mask.sum()) < 128:
        flat_mask = _quantile_mask(grad_mag, 0.50, "low")
    if int(edge_mask.sum()) < 128:
        edge_mask = _quantile_mask(grad_mag, 0.65, "high")

    flat_mean = local_mean[flat_mask]
    flat_var = local_var[flat_mask]
    edge_var = local_var[edge_mask]

    slope, r2 = _linear_fit_stats(flat_mean, flat_var)
    median_flat = float(np.median(flat_var)) if flat_var.size else 0.0
    median_edge = float(np.median(edge_var)) if edge_var.size else 0.0
    flat_cv = float(np.std(flat_var) / (np.mean(flat_var) + EPS)) if flat_var.size else 0.0

    bin_edges = np.quantile(flat_mean, np.linspace(0.0, 1.0, 6)) if flat_mean.size >= 16 else np.array([])
    bin_medians: list[float] = []
    if bin_edges.size:
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            if right <= left:
                continue
            mask = (flat_mean >= left) & (flat_mean <= right) if right == bin_edges[-1] else (flat_mean >= left) & (flat_mean < right)
            if int(mask.sum()) >= 8:
                bin_medians.append(float(np.median(flat_var[mask])))
    if len(bin_medians) >= 2:
        diffs = np.diff(np.asarray(bin_medians, dtype=np.float64))
        violation = float(np.mean(diffs < 0.0))
    else:
        violation = 0.0

    return {
        "lochet_flat_slope": slope,
        "lochet_flat_r2": r2,
        "lochet_flat_cv": flat_cv,
        "lochet_edge_flat_logratio": float(np.log((median_edge + NOISE_FLOOR) / (median_flat + NOISE_FLOOR))),
        "lochet_flat_monotone_violation": violation,
    }


def extract_edge_consistency_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64) / 255.0
    residual = np.abs(_convolve_valid(y, LAP))
    grad_x = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)[1:-1, 1:-1]
    grad_y = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)[1:-1, 1:-1]
    grad_mag = np.hypot(grad_x, grad_y)

    strong = _quantile_mask(grad_mag, 0.90, "high")
    if int(strong.sum()) < 128:
        strong = _quantile_mask(grad_mag, 0.75, "high")

    res_c = residual[1:-1, 1:-1]
    left = residual[1:-1, :-2]
    right = residual[1:-1, 2:]
    up = residual[:-2, 1:-1]
    down = residual[2:, 1:-1]

    gx_i = grad_x[1:-1, 1:-1]
    gy_i = grad_y[1:-1, 1:-1]
    strong_i = strong[1:-1, 1:-1]
    vertical = strong_i & (np.abs(gx_i) >= np.abs(gy_i))
    horizontal = strong_i & (np.abs(gy_i) > np.abs(gx_i))

    cross_v = _safe_log_ratio(left, right)
    along_v = _safe_log_ratio(up, down)
    cross_h = _safe_log_ratio(up, down)
    along_h = _safe_log_ratio(left, right)

    cross_vals = np.concatenate([cross_v[vertical], cross_h[horizontal]])
    ratio_vals = np.concatenate([
        ((cross_v + EPS) / (along_v + EPS))[vertical],
        ((cross_h + EPS) / (along_h + EPS))[horizontal],
    ])
    resid_vals = res_c[strong_i]
    grad_vals = grad_mag[1:-1, 1:-1][strong_i]

    if cross_vals.size < 32:
        return {
            "edge_cross_jump_median": 0.0,
            "edge_cross_jump_p90": 0.0,
            "edge_cross_along_ratio_median": 0.0,
            "edge_grad_resid_corr": 0.0,
        }

    return {
        "edge_cross_jump_median": float(np.median(cross_vals)),
        "edge_cross_jump_p90": float(np.quantile(cross_vals, 0.90)),
        "edge_cross_along_ratio_median": float(np.median(ratio_vals)),
        "edge_grad_resid_corr": _safe_corr(grad_vals, resid_vals),
    }


def _autocorr_peak(series: np.ndarray, lag_min: int = 2, lag_max: int = 32) -> float:
    x = np.asarray(series, dtype=np.float64).ravel()
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if x.size < lag_max + 4 or denom <= EPS:
        return 0.0
    vals: list[float] = []
    for lag in range(lag_min, min(lag_max + 1, x.size - 1)):
        vals.append(float(np.sum(x[:-lag] * x[lag:]) / denom))
    return float(max(vals)) if vals else 0.0


def extract_resampling_periodicity_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64) / 255.0
    dxx = y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]
    dyy = y[2:, :] - 2.0 * y[1:-1, :] + y[:-2, :]
    proj_x = np.mean(np.abs(dxx), axis=0)
    proj_y = np.mean(np.abs(dyy), axis=1)
    peak_x = _autocorr_peak(proj_x)
    peak_y = _autocorr_peak(proj_y)
    return {
        "resamp_peak_x": peak_x,
        "resamp_peak_y": peak_y,
        "resamp_peak_mean": float(0.5 * (peak_x + peak_y)),
    }


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


def extract_feature_dict(arr_rgb: np.ndarray, preprocess_version: str) -> dict[str, float]:
    arr_ycc = rgb_to_ycrcb(arr_rgb)
    y = arr_ycc[:, :, 0]
    cr = arr_ycc[:, :, 1] - 128.0
    native_phases = (0,) if preprocess_version == "old_v1" else (4,)
    union_phases = (0, 4)
    out: dict[str, float] = {}
    out.update(extract_frequency_features(y))
    out.update(extract_color_global(arr_ycc))
    out.update(extract_spatial_features(arr_ycc))
    out.update(extract_cfa_chroma(arr_ycc))
    out.update(extract_cfa_rgbdiff(arr_rgb))
    out.update(extract_noise_level_features(arr_ycc))
    out.update(extract_wavelet_dependency_features(arr_ycc))
    out.update(_compute_srm_stats(y, "ysrm"))
    out.update(_compute_masked_srm_stats(y, "ysrm_native_mask", native_phases))
    out.update(_compute_masked_srm_stats(y, "ysrm_union_mask", union_phases))
    out.update(_compute_srm_stats(cr, "crsrm"))
    out.update(_extract_lbp_stats(y, "ylbp"))
    out.update(_extract_lbp_stats(arr_ycc[:, :, 1], "crlbp"))
    out.update(_extract_lbp_stats(arr_ycc[:, :, 2], "cblbp"))
    out.update(extract_local_hetero_features(arr_ycc))
    out.update(extract_edge_consistency_features(arr_ycc))
    out.update(extract_resampling_periodicity_features(arr_ycc))
    return out


CONTROL_MINIMAL_KEYS = (
    "frs_mid_variance",
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

CONTROL_WITH_WAVELET_KEYS = tuple(dict.fromkeys(CONTROL_MINIMAL_KEYS + WAVELET_PARENT_KEYS))

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

CFA_XY_KEYS = (
    "cfa_cr_pi_xy",
    "cfa_cb_pi_xy",
    "cfa_rg_pi_xy",
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

YSRM_NATIVE_MASK_KEYS = (
    "ysrm_native_mask_square3_mar",
    "ysrm_native_mask_square3_energy",
    "ysrm_native_mask_edge3_mar",
    "ysrm_native_mask_edge3_energy",
    "ysrm_native_mask_square5_mar",
    "ysrm_native_mask_square5_energy",
)

YSRM_UNION_MASK_KEYS = (
    "ysrm_union_mask_square3_mar",
    "ysrm_union_mask_square3_energy",
    "ysrm_union_mask_edge3_mar",
    "ysrm_union_mask_edge3_energy",
    "ysrm_union_mask_square5_mar",
    "ysrm_union_mask_square5_energy",
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

LOCAL_HETERO_KEYS = (
    "lochet_flat_slope",
    "lochet_flat_r2",
    "lochet_flat_cv",
    "lochet_edge_flat_logratio",
    "lochet_flat_monotone_violation",
)

EDGE_CONSIST_KEYS = (
    "edge_cross_jump_median",
    "edge_cross_jump_p90",
    "edge_cross_along_ratio_median",
    "edge_grad_resid_corr",
)

RESAMP_KEYS = (
    "resamp_peak_x",
    "resamp_peak_y",
    "resamp_peak_mean",
)

LOCAL_PHYSICAL_KEYS = tuple(dict.fromkeys(LOCAL_HETERO_KEYS + EDGE_CONSIST_KEYS + RESAMP_KEYS))

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
    list(CONTROL_MINIMAL_KEYS)
    + list(CONTROL_WITH_WAVELET_KEYS)
    + list(WAVELET_FULL_KEYS)
    + list(NLF_KEYS)
    + list(CFA_CHROMA_KEYS)
    + list(CFA_RGB_KEYS)
    + list(CFA_XY_KEYS)
    + list(YSRM_KEYS)
    + list(YSRM_NATIVE_MASK_KEYS)
    + list(YSRM_UNION_MASK_KEYS)
    + list(CRSRM_KEYS)
    + list(YLBP_KEYS)
    + list(CHROMA_LBP_KEYS)
    + list(LOCAL_HETERO_KEYS)
    + list(EDGE_CONSIST_KEYS)
    + list(RESAMP_KEYS)
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
    "control_minimal": list(CONTROL_MINIMAL_KEYS),
    "control_with_wavelet": list(CONTROL_WITH_WAVELET_KEYS),
    "control_plus_cfa_xy": list(CONTROL_WITH_WAVELET_KEYS + CFA_XY_KEYS),
    "control_plus_local_physical": list(CONTROL_MINIMAL_KEYS + LOCAL_PHYSICAL_KEYS),
    "control_plus_masked_ysrm_union": list(CONTROL_MINIMAL_KEYS + YSRM_UNION_MASK_KEYS),
    "control_plus_masked_ysrm_native": list(CONTROL_MINIMAL_KEYS + YSRM_NATIVE_MASK_KEYS),
    "control_plus_local_physical_plus_masked_ysrm": list(CONTROL_MINIMAL_KEYS + LOCAL_PHYSICAL_KEYS + YSRM_UNION_MASK_KEYS),
    "control_plus_resamp": list(CONTROL_MINIMAL_KEYS + RESAMP_KEYS),
    "control_plus_local_hetero": list(CONTROL_MINIMAL_KEYS + LOCAL_HETERO_KEYS),
    "control_plus_edge_consistency": list(CONTROL_MINIMAL_KEYS + EDGE_CONSIST_KEYS),
    "control_plus_wavelet_full": list(CONTROL_MINIMAL_KEYS + WAVELET_FULL_KEYS),
    "control_plus_nlf": list(CONTROL_MINIMAL_KEYS + NLF_KEYS),
    "control_plus_ytexture": list(CONTROL_MINIMAL_KEYS + YSRM_KEYS + YLBP_KEYS),
    "ysrm_only": list(YSRM_KEYS),
    "ysrm_native_mask_only": list(YSRM_NATIVE_MASK_KEYS),
    "ysrm_union_mask_only": list(YSRM_UNION_MASK_KEYS),
    "ylbp_only": list(YLBP_KEYS),
    "cfa_xy_only": list(CFA_XY_KEYS),
    "cfa_rgb_only": list(CFA_RGB_KEYS),
    "cfa_chroma_only": list(CFA_CHROMA_KEYS),
    "wavelet_parent_only": list(WAVELET_PARENT_KEYS),
    "wavelet_full_only": list(WAVELET_FULL_KEYS),
    "nlf_only": list(NLF_KEYS),
    "local_hetero_only": list(LOCAL_HETERO_KEYS),
    "edge_consistency_only": list(EDGE_CONSIST_KEYS),
    "resamp_only": list(RESAMP_KEYS),
    "local_physical_only": list(LOCAL_PHYSICAL_KEYS),
    "crsrm_only": list(CRSRM_KEYS),
    "chroma_lbp_only": list(CHROMA_LBP_KEYS),
    "toxic_controls": list(TOXIC_CONTROL_KEYS),
}


def extract_record(row: pd.Series, preprocess_version: str, degradation: str) -> dict[str, object]:
    path_col = "old_output_path" if preprocess_version == "old_v1" else "v4_output_path"
    rgb = load_rgb_from_preprocess(str(row[path_col]), preprocess_version=preprocess_version)
    rgb = apply_degradation(rgb, degradation)
    feats = extract_feature_dict(rgb, preprocess_version=preprocess_version)
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


def build_feature_gate_summary(single_df: pd.DataFrame, shift_df: pd.DataFrame) -> pd.DataFrame:
    clean = single_df.loc[single_df["task"] == "label_logo_clean", ["preprocess_version", "feature", "auc"]].rename(columns={"auc": "clean_auc"})
    nat = single_df.loc[single_df["task"] == "real_jpeg_444_vs_420", ["preprocess_version", "feature", "auc"]].rename(columns={"auc": "natural_nuisance_auc"})
    shift = shift_df.loc[shift_df["degradation"] == SHIFT_GATE_DEGRADATION, ["preprocess_version", "label", "feature", "mean_abs_z_shift"]].copy()
    shift = shift.groupby(["preprocess_version", "feature"], as_index=False)["mean_abs_z_shift"].max().rename(columns={"mean_abs_z_shift": "resize50_max_shift"})

    out = clean.merge(nat, on=["preprocess_version", "feature"], how="inner").merge(
        shift,
        on=["preprocess_version", "feature"],
        how="inner",
    )
    out["pass_strict_gate"] = (
        (out["clean_auc"] > STRICT_CLEAN_AUC_MIN)
        & (out["natural_nuisance_auc"] < STRICT_NUISANCE_AUC_MAX)
        & (out["resize50_max_shift"] < STRICT_SHIFT_MAX)
    )
    return out.sort_values(
        ["preprocess_version", "pass_strict_gate", "clean_auc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_feature_set_shift_summary(shift_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = shift_df.groupby(
        ["preprocess_version", "degradation", "feature"],
        as_index=False,
    )["mean_abs_z_shift"].max()
    for (preprocess_version, degradation), group in base.groupby(["preprocess_version", "degradation"], sort=True):
        feature_shift = dict(zip(group["feature"], group["mean_abs_z_shift"]))
        for set_name, cols in FEATURE_SETS.items():
            vals = [float(feature_shift[col]) for col in cols if col in feature_shift]
            if not vals:
                continue
            rows.append({
                "preprocess_version": preprocess_version,
                "degradation": degradation,
                "feature_set": set_name,
                "n_features": len(cols),
                "mean_feature_shift": float(np.mean(vals)),
                "median_feature_shift": float(np.median(vals)),
                "max_feature_shift": float(np.max(vals)),
            })
    return pd.DataFrame(rows).sort_values(
        ["preprocess_version", "degradation", "mean_feature_shift"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


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
    set_shift_df = build_feature_set_shift_summary(shift_df)
    gate_df = build_feature_gate_summary(single_df, shift_df)

    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    single_df.to_csv(OUT_SINGLE, index=False, encoding="utf-8-sig")
    shift_df.to_csv(OUT_SHIFT, index=False, encoding="utf-8-sig")
    set_shift_df.to_csv(OUT_SET_SHIFT, index=False, encoding="utf-8-sig")
    gate_df.to_csv(OUT_FEATURE_GATE, index=False, encoding="utf-8-sig")

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
        "strict_gate_pass_count_by_preprocess": _json_ready_dict(
            gate_df.groupby("preprocess_version")["pass_strict_gate"].sum().to_dict()
        ),
        "files": {
            "metrics_csv": str(OUT_METRICS),
            "single_csv": str(OUT_SINGLE),
            "shift_csv": str(OUT_SHIFT),
            "set_shift_csv": str(OUT_SET_SHIFT),
            "feature_gate_csv": str(OUT_FEATURE_GATE),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_SINGLE}")
    print(f"Saved: {OUT_SHIFT}")
    print(f"Saved: {OUT_SET_SHIFT}")
    print(f"Saved: {OUT_FEATURE_GATE}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
