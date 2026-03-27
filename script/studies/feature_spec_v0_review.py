from __future__ import annotations

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
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed_v4_rgb248_r4_exact" / "manifest.csv"
METADATA_CSV = PROJECT_ROOT / "audit_output" / "data_audit" / "metadata" / "per_file_metadata.csv"

OUT_ROOT = PROJECT_ROOT / "audit_output" / "studies" / "feature_spec_v0_review_20260325"
OUT_METRICS = OUT_ROOT / "feature_set_task_metrics.csv"
OUT_SINGLE = OUT_ROOT / "single_feature_task_auc.csv"
OUT_SHIFT = OUT_ROOT / "jpeg_sensitivity_shift.csv"
OUT_SUMMARY = OUT_ROOT / "summary.json"

RNG_SEED = 20260325
EPS = 1e-12
NOISE_FLOOR = 1.53e-5
LABEL_SAMPLE_PER_GROUP = 120
REAL_NUISANCE_PER_CLASS = 600
AI_SENSITIVITY_PER_GROUP = 140
SAMPLING_WORKERS = 8

CLEAN_GENERATORS = ("midjourney", "sdv14", "sdv15", "wukong")
SUBSAMPLING_MAP = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}

CFA_KEYS = (
    "cfa_cr_pi_x",
    "cfa_cr_pi_y",
    "cfa_cr_pi_xy",
    "cfa_cb_pi_x",
    "cfa_cb_pi_y",
    "cfa_cb_pi_xy",
)

NLF_KEYS = (
    "nlf_spearman",
    "nlf_slope",
    "nlf_intercept",
    "nlf_r2",
    "nlf_monotone_violation",
)

WAVELET_KEYS = (
    "wav_parent_corr_h",
    "wav_parent_corr_v",
    "wav_parent_corr_d",
    "wav_kurtosis_l1",
    "wav_kurtosis_l2",
)

KEEP_GLOBAL_KEYS = (
    "frs_mid_variance",
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
    "spatial_snr_ratio",
    "skew_noise_y",
    "kurt_noise_y",
    "skew_noise_cr",
    "kurt_noise_cr",
    "skew_noise_cb",
    "kurt_noise_cb",
)

CONTROL_KEYS = (
    "ps_alpha",
    "ps_deviation_variance",
    "cross_noise_ratio",
)

ALL_FEATURE_KEYS = tuple(CFA_KEYS + NLF_KEYS + WAVELET_KEYS + KEEP_GLOBAL_KEYS + CONTROL_KEYS)

FEATURE_SETS = {
    "spec0_keep": list(CFA_KEYS + KEEP_GLOBAL_KEYS),
    "spec0_keep_wavelet": list(CFA_KEYS + KEEP_GLOBAL_KEYS + WAVELET_KEYS),
    "spec0_keep_nlf": list(CFA_KEYS + KEEP_GLOBAL_KEYS + NLF_KEYS),
    "spec0_keep_all_new": list(CFA_KEYS + KEEP_GLOBAL_KEYS + NLF_KEYS + WAVELET_KEYS),
    "spec0_keep_plus_controls": list(CFA_KEYS + KEEP_GLOBAL_KEYS + CONTROL_KEYS),
    "cfa_only": list(CFA_KEYS),
    "nlf_only": list(NLF_KEYS),
    "wavelet_only": list(WAVELET_KEYS),
}


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


def load_merged_tables() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest.loc[manifest["status"] == "ACCEPTED"].copy()
    manifest["relative_path"] = [
        str(Path(path).resolve().relative_to(RAW_ROOT.resolve())).replace("\\", "/")
        for path in manifest["file_path"]
    ]
    manifest["label"] = manifest["label"].replace({"nature": "nature", "ai": "ai"})
    manifest["generator_norm"] = manifest["generator"].str.lower()

    meta = pd.read_csv(METADATA_CSV)
    meta["relative_path"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    meta["generator_norm"] = meta["inferred_generator"].fillna("").str.lower()
    meta["label_norm"] = meta["inferred_label"].replace({"fake": "ai", "real": "nature"})

    cols = [
        "relative_path",
        "generator_norm",
        "label_norm",
        "format_detected",
        "jpeg_subsampling",
        "quality_estimate",
        "eval_group",
        "image_mode",
        "has_alpha",
        "is_grayscale",
    ]
    merged = manifest.merge(meta[cols], on="relative_path", how="left")
    merged["generator_norm"] = merged["generator_norm_x"].fillna(merged["generator_norm_y"])
    merged["label_norm"] = merged["label_norm"].fillna(merged["label"])
    merged["generator_norm"] = merged["generator_norm"].astype(str).str.lower()
    return merged


def rgb_to_ycrcb(arr_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)


def jpeg_roundtrip_rgb(arr_rgb: np.ndarray, quality: int, subsampling: int) -> np.ndarray:
    image = Image.fromarray(arr_rgb, mode="RGB")
    bio = io.BytesIO()
    image.save(bio, format="JPEG", quality=int(quality), subsampling=int(subsampling))
    bio.seek(0)
    with Image.open(bio) as decoded:
        out = np.array(decoded.convert("RGB"))
    return out


def load_patch(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.uint8 or arr.shape != (248, 248, 3):
        raise ValueError(f"Unexpected patch shape/dtype: {path} -> {arr.shape} {arr.dtype}")
    return arr


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


def extract_frequency_features(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
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
    if mid_mean < NOISE_FLOOR:
        frs = 0.0
    else:
        frs = float(np.var(mid) / (mid_mean ** 2 + EPS))

    fit_r = np.arange(20, min(65, len(ring)), dtype=np.float64)
    fit_y = ring[20:min(65, len(ring))]
    if fit_y.size == 0 or np.all(fit_y < NOISE_FLOOR):
        return {
            "frs_mid_variance": frs,
            "ps_alpha": 0.0,
            "ps_deviation_variance": 0.0,
        }

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
    feats = {
        "pearson_y_cr": _safe_corr(y, cr),
        "pearson_y_cb": _safe_corr(y, cb),
        "pearson_cr_cb": _safe_corr(cr, cb),
        "energy_ratio_chroma": float((np.var(cr) + np.var(cb)) / (var_y + EPS)),
    }
    return feats


def extract_spatial_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64)
    cr = arr_ycc[:, :, 1].astype(np.float64) - 128.0
    cb = arr_ycc[:, :, 2].astype(np.float64) - 128.0
    lap = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    ry = _convolve_valid(y, lap)
    rcr = _convolve_valid(cr, lap)
    rcb = _convolve_valid(cb, lap)

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
    if v_flat < 1e-6:
        snr = 0.0
    else:
        snr = float(np.log10((v_edge + NOISE_FLOOR) / (v_flat + NOISE_FLOOR)))

    noise_y = float(np.mean(np.abs(ry)))
    noise_cb = float(np.mean(np.abs(rcb)))
    cross = float((noise_y + NOISE_FLOOR) / (noise_cb + NOISE_FLOOR))

    feats = {
        "spatial_snr_ratio": snr,
        "cross_noise_ratio": cross,
        "skew_noise_y": _safe_skew(ry),
        "kurt_noise_y": _safe_excess_kurtosis(ry),
        "skew_noise_cr": _safe_skew(rcr),
        "kurt_noise_cr": _safe_excess_kurtosis(rcr),
        "skew_noise_cb": _safe_skew(rcb),
        "kurt_noise_cb": _safe_excess_kurtosis(rcb),
    }
    return feats


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
    return {
        name: float((np.sum(residual * mask) ** 2) / denom)
        for name, mask in basis.items()
    }


def extract_cfa_periodicity(arr_ycc: np.ndarray) -> dict[str, float]:
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    cr = arr_ycc[:, :, 1] - 128.0
    cb = arr_ycc[:, :, 2] - 128.0
    cr_res = _convolve_valid(cr, kernel)
    cb_res = _convolve_valid(cb, kernel)
    feats: dict[str, float] = {}
    feats.update(_periodicity_energies(cr_res, "cfa_cr"))
    feats.update(_periodicity_energies(cb_res, "cfa_cb"))
    return feats


def _block_view(arr: np.ndarray, block: int) -> np.ndarray:
    h = (arr.shape[0] // block) * block
    w = (arr.shape[1] // block) * block
    trimmed = arr[:h, :w]
    return trimmed.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)


def extract_noise_level_features(arr_ycc: np.ndarray) -> dict[str, float]:
    y = arr_ycc[:, :, 0].astype(np.float64)
    block = 8
    blocks = _block_view(y, block)
    means = blocks.mean(axis=(2, 3)).ravel() / 255.0
    dx = np.diff(blocks, axis=3)
    dy = np.diff(blocks, axis=2)
    roughness = (np.mean(np.abs(dx), axis=(2, 3)) + np.mean(np.abs(dy), axis=(2, 3))).ravel()

    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    residual = _convolve_valid(y, kernel)
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


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


def extract_feature_dict(arr_rgb: np.ndarray) -> dict[str, float]:
    arr_ycc = rgb_to_ycrcb(arr_rgb)
    feats: dict[str, float] = {}
    feats.update(extract_frequency_features(arr_ycc[:, :, 0]))
    feats.update(extract_color_global(arr_ycc))
    feats.update(extract_spatial_features(arr_ycc))
    feats.update(extract_cfa_periodicity(arr_ycc))
    feats.update(extract_noise_level_features(arr_ycc))
    feats.update(extract_wavelet_dependency_features(arr_ycc))
    return feats


def split_quadrants(arr_rgb: np.ndarray) -> list[np.ndarray]:
    mid = arr_rgb.shape[0] // 2
    return [
        arr_rgb[:mid, :mid, :],
        arr_rgb[:mid, mid:, :],
        arr_rgb[mid:, :mid, :],
        arr_rgb[mid:, mid:, :],
    ]


def aggregate_patch_features(arr_rgb: np.ndarray, mode: str) -> dict[str, float]:
    if mode == "whole":
        return extract_feature_dict(arr_rgb)

    sub = [extract_feature_dict(patch) for patch in split_quadrants(arr_rgb)]
    keys = sub[0].keys()
    if mode == "quad_mean":
        return {key: float(np.mean([row[key] for row in sub])) for key in keys}
    if mode == "quad_meanstd":
        out: dict[str, float] = {}
        for key in keys:
            values = np.asarray([row[key] for row in sub], dtype=np.float64)
            out[f"{key}__mean"] = float(np.mean(values))
            out[f"{key}__std"] = float(np.std(values))
        return out
    raise ValueError(f"Unknown representation mode: {mode}")


def build_label_sample(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        df["generator_norm"].isin(CLEAN_GENERATORS)
        & (df["label"].isin(["nature", "ai"]))
        & (df["eval_group"].fillna("ID") == "ID")
    ].copy()
    rng = random.Random(RNG_SEED)
    parts: list[pd.DataFrame] = []
    for (_, _), group in subset.groupby(["generator_norm", "label"], sort=True):
        parts.append(_stable_sample(group, LABEL_SAMPLE_PER_GROUP, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def build_real_nuisance_sample(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        (df["label"] == "nature")
        & (df["eval_group"].fillna("ID") == "ID")
        & (df["input_format"] == "JPEG")
    ].copy()
    if subset.empty:
        raise RuntimeError("No accepted real JPEG rows found for nuisance audit.")

    def detect_sampling(path_str: str) -> str | None:
        try:
            with Image.open(path_str) as im:
                return SUBSAMPLING_MAP.get(JpegImagePlugin.get_sampling(im), "other")
        except Exception:
            return None

    sampled = subset.sample(n=min(6000, len(subset)), random_state=RNG_SEED + 10).copy()
    sampling: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=SAMPLING_WORKERS) as pool:
        futs = {pool.submit(detect_sampling, path): idx for idx, path in sampled["file_path"].items()}
        for fut in as_completed(futs):
            sampling[futs[fut]] = fut.result()
    sampled["jpeg_subsampling"] = pd.Series(sampling)
    subset = sampled.loc[sampled["jpeg_subsampling"].isin(["4:4:4", "4:2:0"])].copy()
    rng = random.Random(RNG_SEED + 1)
    parts: list[pd.DataFrame] = []
    for subsampling, group in subset.groupby("jpeg_subsampling", sort=True):
        take = REAL_NUISANCE_PER_CLASS if subsampling in {"4:4:4", "4:2:0"} else len(group)
        parts.append(_stable_sample(group, take, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def build_ai_sensitivity_sample(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        (df["label"] == "ai")
        & (df["generator_norm"].isin(CLEAN_GENERATORS))
        & (df["eval_group"].fillna("ID") == "ID")
    ].copy()
    rng = random.Random(RNG_SEED + 2)
    parts: list[pd.DataFrame] = []
    for generator, group in subset.groupby("generator_norm", sort=True):
        parts.append(_stable_sample(group, AI_SENSITIVITY_PER_GROUP, rng.randint(0, 1_000_000)))
    return pd.concat(parts, ignore_index=True)


def evaluate_auc(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str | None = None) -> float:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    if group_col is not None and df[group_col].nunique() >= 3:
        groups = df[group_col].astype(str).to_numpy()
        preds = np.full(len(df), np.nan, dtype=np.float64)
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        for train_idx, test_idx in splitter.split(X, y, groups):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        return float(roc_auc_score(y, preds))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    preds = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return float(roc_auc_score(y, preds))


def build_representation_table(rows: list[pd.Series], mode: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        arr = load_patch(row["output_path"])
        feats = aggregate_patch_features(arr, mode=mode)
        record = {
            "relative_path": row["relative_path"],
            "generator_norm": row["generator_norm"],
            "label": row["label"],
            "jpeg_subsampling": row.get("jpeg_subsampling", ""),
            "format_detected": row.get("format_detected", ""),
            "representation": mode,
        }
        record.update(feats)
        records.append(record)
    return pd.DataFrame(records)


def build_ai_transform_table(rows: list[pd.Series], quality: int, subsampling: int, mode: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    transform_name = f"jpeg_q{quality}_sub{subsampling}"
    for row in rows:
        arr = load_patch(row["output_path"])
        for tag, rgb in {
            "identity": arr,
            transform_name: jpeg_roundtrip_rgb(arr, quality=quality, subsampling=subsampling),
        }.items():
            record = {
                "relative_path": row["relative_path"],
                "generator_norm": row["generator_norm"],
                "variant": tag,
                "representation": mode,
            }
            record.update(aggregate_patch_features(rgb, mode=mode))
            records.append(record)
    return pd.DataFrame(records)


def feature_cols_for_representation(df: pd.DataFrame, base_cols: list[str]) -> list[str]:
    if any(col.endswith("__mean") for col in df.columns):
        cols: list[str] = []
        for key in base_cols:
            cols.append(f"{key}__mean")
            cols.append(f"{key}__std")
        return cols
    return base_cols


def single_feature_auc(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_name: str,
    representation: str,
    group_col: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in feature_cols:
        auc = evaluate_auc(df[[col, target_col] + ([group_col] if group_col else [])], [col], target_col, group_col)
        rows.append({
            "task": task_name,
            "representation": representation,
            "feature": col,
            "auc": auc,
        })
    return rows


def compute_shift_table(df: pd.DataFrame, transform_name: str) -> pd.DataFrame:
    id_df = df.loc[df["variant"] == "identity"].set_index("relative_path")
    tr_df = df.loc[df["variant"] == transform_name].set_index("relative_path")
    common = id_df.index.intersection(tr_df.index)
    id_df = id_df.loc[common]
    tr_df = tr_df.loc[common]
    rows: list[dict[str, object]] = []
    for feature in ALL_FEATURE_KEYS:
        base = id_df[feature].to_numpy(dtype=np.float64)
        shifted = tr_df[feature].to_numpy(dtype=np.float64)
        std = float(np.std(base)) + EPS
        rows.append({
            "transform": transform_name,
            "feature": feature,
            "mean_delta": float(np.mean(shifted - base)),
            "mean_abs_z_shift": float(np.mean(np.abs((shifted - base) / std))),
            "median_abs_z_shift": float(np.median(np.abs((shifted - base) / std))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = load_merged_tables()

    label_sample = build_label_sample(merged)
    nuisance_sample = build_real_nuisance_sample(merged)
    ai_sample = build_ai_sensitivity_sample(merged)

    label_tables = {
        mode: build_representation_table([row for _, row in label_sample.iterrows()], mode)
        for mode in ("whole", "quad_mean", "quad_meanstd")
    }
    nuisance_tables = {
        mode: build_representation_table([row for _, row in nuisance_sample.iterrows()], mode)
        for mode in ("whole", "quad_mean", "quad_meanstd")
    }

    ai_transform_tables = {
        "ai_identity_vs_jpeg95_420": {
            mode: build_ai_transform_table([row for _, row in ai_sample.iterrows()], quality=95, subsampling=2, mode=mode)
            for mode in ("whole", "quad_mean", "quad_meanstd")
        },
        "ai_identity_vs_jpeg90_420": {
            mode: build_ai_transform_table([row for _, row in ai_sample.iterrows()], quality=90, subsampling=2, mode=mode)
            for mode in ("whole", "quad_mean", "quad_meanstd")
        },
    }

    metric_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []

    for mode, table in label_tables.items():
        table = table.copy()
        table["target"] = (table["label"] == "ai").astype(int)
        for set_name, base_cols in FEATURE_SETS.items():
            cols = feature_cols_for_representation(table, base_cols)
            auc = evaluate_auc(table[cols + ["target", "generator_norm"]], cols, "target", "generator_norm")
            metric_rows.append({
                "task": "label_logo_clean",
                "representation": mode,
                "feature_set": set_name,
                "n_features": len(cols),
                "auc": auc,
            })
        if mode == "whole":
            single_rows.extend(
                single_feature_auc(
                    table,
                    list(ALL_FEATURE_KEYS),
                    "target",
                    "label_logo_clean",
                    mode,
                    "generator_norm",
                )
            )

    for mode, table in nuisance_tables.items():
        table = table.copy()
        table["target"] = (table["jpeg_subsampling"] == "4:2:0").astype(int)
        for set_name, base_cols in FEATURE_SETS.items():
            cols = feature_cols_for_representation(table, base_cols)
            auc = evaluate_auc(table[cols + ["target", "generator_norm"]], cols, "target", "generator_norm")
            metric_rows.append({
                "task": "real_jpeg_444_vs_420",
                "representation": mode,
                "feature_set": set_name,
                "n_features": len(cols),
                "auc": auc,
            })

    for task_name, tables_by_mode in ai_transform_tables.items():
        for mode, transform_df in tables_by_mode.items():
            transform_df = transform_df.copy()
            transform_df["target"] = (transform_df["variant"] != "identity").astype(int)
            for set_name, base_cols in FEATURE_SETS.items():
                cols = feature_cols_for_representation(transform_df, base_cols)
                auc = evaluate_auc(transform_df[cols + ["target", "generator_norm"]], cols, "target", "generator_norm")
                metric_rows.append({
                    "task": task_name,
                    "representation": mode,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "auc": auc,
                })
            if mode == "whole":
                single_rows.extend(
                    single_feature_auc(
                        transform_df,
                        list(ALL_FEATURE_KEYS),
                        "target",
                        task_name,
                        mode,
                        "generator_norm",
                    )
                )

    nuisance_whole = nuisance_tables["whole"].copy()
    nuisance_whole["target"] = (nuisance_whole["jpeg_subsampling"] == "4:2:0").astype(int)
    single_rows.extend(
        single_feature_auc(
            nuisance_whole,
            list(ALL_FEATURE_KEYS),
            "target",
            "real_jpeg_444_vs_420",
            "whole",
            "generator_norm",
        )
    )

    shift_df = pd.concat(
        [
            compute_shift_table(ai_transform_tables["ai_identity_vs_jpeg95_420"]["whole"], "jpeg_q95_sub2"),
            compute_shift_table(ai_transform_tables["ai_identity_vs_jpeg90_420"]["whole"], "jpeg_q90_sub2"),
        ],
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["task", "representation", "auc"], ascending=[True, True, False])
    single_df = pd.DataFrame(single_rows).sort_values(["task", "auc"], ascending=[True, False])

    OUT_METRICS.write_text(metrics_df.to_csv(index=False), encoding="utf-8-sig")
    OUT_SINGLE.write_text(single_df.to_csv(index=False), encoding="utf-8-sig")
    OUT_SHIFT.write_text(shift_df.to_csv(index=False), encoding="utf-8-sig")

    summary = {
        "label_sample": {
            "n_rows": int(len(label_sample)),
            "by_generator_label": _json_ready_dict(label_sample.groupby(["generator_norm", "label"]).size().to_dict()),
        },
        "natural_nuisance_sample": {
            "n_rows": int(len(nuisance_sample)),
            "by_subsampling": _json_ready_dict(nuisance_sample.groupby("jpeg_subsampling").size().to_dict()),
        },
        "ai_sensitivity_sample": {
            "n_rows": int(len(ai_sample)),
            "by_generator": _json_ready_dict(ai_sample.groupby("generator_norm").size().to_dict()),
        },
        "best_auc_by_task_representation": _json_ready_dict(
            metrics_df.groupby(["task", "representation"])["auc"].max().round(6).to_dict()
        ),
        "files": {
            "metrics_csv": str(OUT_METRICS),
            "single_feature_csv": str(OUT_SINGLE),
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
