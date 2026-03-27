from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.color import FEATURE_KEYS as COLOR_KEYS
from src.feature_extraction.color import extract_color_features
from src.feature_extraction.frequency import FEATURE_KEYS as FREQ_KEYS
from src.feature_extraction.frequency import extract_frequency_features
from src.feature_extraction.microtexture import FEATURE_KEYS as MICRO_KEYS
from src.feature_extraction.microtexture import extract_microtexture_features
from src.preprocessing.pipeline import apply_orientation, read_orientation
from src.feature_extraction.spatial import FEATURE_KEYS as SPATIAL_KEYS
from src.feature_extraction.spatial import extract_spatial_features

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_PARQUET = PROJECT_ROOT / "audit_output" / "data_audit" / "metadata" / "per_file_metadata.parquet"
OUT_ROOT = PROJECT_ROOT / "audit_output" / "studies" / "update_feature"
OUT_JSON = OUT_ROOT / "update_feature_benchmark.json"
OUT_CSV = OUT_ROOT / "update_feature_table.csv"
OUT_FEATURE_CSV = OUT_ROOT / "update_feature_individual_metrics.csv"

CROP_SIZE = 256
PAD_MIN_SIZE = 259
GRID_OFFSET = 3
JPEG_Q_MIN = 90
JPEG_Q_MAX = 98
SAMPLE_SEED = 42
SAMPLE_PER_GROUP = 100
WORKERS = 8
EPS = 1e-12

BASELINE_KEYS = list(FREQ_KEYS) + list(COLOR_KEYS) + list(MICRO_KEYS) + list(SPATIAL_KEYS)

CFA_KEYS = [
    "cfa_cr_pi_x",
    "cfa_cr_pi_y",
    "cfa_cr_pi_xy",
    "cfa_cb_pi_x",
    "cfa_cb_pi_y",
    "cfa_cb_pi_xy",
]
NLF_KEYS = [
    "nlf_spearman",
    "nlf_slope",
    "nlf_intercept",
    "nlf_r2",
    "nlf_monotone_violation",
]
WAVELET_KEYS = [
    "wav_parent_corr_h",
    "wav_parent_corr_v",
    "wav_parent_corr_d",
    "wav_kurtosis_l1",
    "wav_kurtosis_l2",
]
NEW_KEYS = CFA_KEYS + NLF_KEYS + WAVELET_KEYS

FEATURE_SETS = {
    "baseline33": BASELINE_KEYS,
    "cfa_periodicity": CFA_KEYS,
    "noise_level_function": NLF_KEYS,
    "wavelet_dependencies": WAVELET_KEYS,
    "all_new": NEW_KEYS,
    "baseline_plus_cfa": BASELINE_KEYS + CFA_KEYS,
    "baseline_plus_nlf": BASELINE_KEYS + NLF_KEYS,
    "baseline_plus_wavelet": BASELINE_KEYS + WAVELET_KEYS,
    "baseline_plus_all_new": BASELINE_KEYS + NEW_KEYS,
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    shift_mode: str
    bottleneck: str | None
    fake_precompress_444: bool = False


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("pil_only_allshift", shift_mode="all", bottleneck="pil420"),
    VariantSpec("method_a_allshift", shift_mode="all", bottleneck=None, fake_precompress_444=True),
)


def read_bgr(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def pad_to_min_size(img: np.ndarray, min_size: int = PAD_MIN_SIZE) -> tuple[np.ndarray, bool, int, int]:
    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img, False, 0, 0

    pad_top = max((min_size - h) // 2, 0)
    pad_bottom = max(min_size - h - pad_top, 0)
    pad_left = max((min_size - w) // 2, 0)
    pad_right = max(min_size - w - pad_left, 0)
    out = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return out, True, pad_left, pad_top


def center_crop_with_mode(
    img: np.ndarray,
    shift_mode: str,
    size: int = CROP_SIZE,
    misalign_offset: int = GRID_OFFSET,
    pad_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = w // 2 - size // 2
    y0 = h // 2 - size // 2
    orig_x0 = x0 - pad_origin[0]
    orig_y0 = y0 - pad_origin[1]

    trigger = (orig_x0 % 8 == 0) or (orig_y0 % 8 == 0)
    if shift_mode == "conditional":
        apply_shift = trigger
    elif shift_mode == "all":
        apply_shift = True
    elif shift_mode == "none":
        apply_shift = False
    else:
        raise ValueError(f"Unknown shift mode: {shift_mode}")

    if apply_shift:
        x0 += misalign_offset
        y0 += misalign_offset

    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    return img[y0:y0 + size, x0:x0 + size]


def deterministic_q(key: str, q_min: int = JPEG_Q_MIN, q_max: int = JPEG_Q_MAX) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little")
    return q_min + (value % (q_max - q_min + 1))


def deterministic_pick(key: str, values: np.ndarray) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little")
    return int(values[value % len(values)])


def jpeg_roundtrip_pil(img: np.ndarray, quality: int, subsampling: int) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    bio = io.BytesIO()
    pil_img.save(bio, format="JPEG", quality=quality, subsampling=subsampling)
    bio.seek(0)
    with Image.open(bio) as decoded:
        rgb_decoded = np.array(decoded.convert("RGB"))
    return cv2.cvtColor(rgb_decoded, cv2.COLOR_RGB2BGR)


def bgr_to_ycrcb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def compute_misalignment_from_dims(width: np.ndarray, height: np.ndarray) -> np.ndarray:
    pad_left = np.where(width >= PAD_MIN_SIZE, 0, ((PAD_MIN_SIZE - width) // 2).astype(int))
    pad_top = np.where(height >= PAD_MIN_SIZE, 0, ((PAD_MIN_SIZE - height) // 2).astype(int))
    padded_width = np.maximum(width, PAD_MIN_SIZE)
    padded_height = np.maximum(height, PAD_MIN_SIZE)
    x0 = (padded_width // 2 - CROP_SIZE // 2).astype(int)
    y0 = (padded_height // 2 - CROP_SIZE // 2).astype(int)
    return ((x0 - pad_left) % 8 == 0) | ((y0 - pad_top) % 8 == 0)


def build_clean_sample() -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_parquet(METADATA_PARQUET)
    meta["needs_pad"] = (meta["width"] < PAD_MIN_SIZE) | (meta["height"] < PAD_MIN_SIZE)
    meta["misalign_trigger"] = compute_misalignment_from_dims(meta["width"].to_numpy(), meta["height"].to_numpy())
    meta["relative_path"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    meta["raw_path"] = [str((RAW_ROOT / Path(rel)).resolve()) for rel in meta["relative_path"]]
    meta["exists_on_disk"] = [Path(path).exists() for path in meta["raw_path"]]

    clean_generators = ["midjourney", "sdv14", "sdv15", "wukong"]
    usable = meta[
        meta["inferred_generator"].isin(clean_generators)
        & (~meta["needs_pad"])
        & (meta["format_detected"].isin(["JPEG", "PNG"]))
        & (meta["exists_on_disk"])
    ].copy()
    usable["generator"] = usable["inferred_generator"].replace({
        "midjourney": "Midjourney",
        "sdv14": "SDv14",
        "sdv15": "SDv15",
        "wukong": "Wukong",
    })
    usable["label"] = usable["inferred_label"].replace({"fake": "ai", "real": "nature"})

    sampled: list[pd.DataFrame] = []
    for (_, _), group in usable.groupby(["generator", "label"]):
        take = min(SAMPLE_PER_GROUP, len(group))
        sampled.append(group.sample(n=take, random_state=SAMPLE_SEED))
    sample_df = pd.concat(sampled, ignore_index=True)

    real_quality_pool = (
        usable.loc[usable["label"] == "nature", "quality_estimate"]
        .dropna()
        .astype(int)
        .to_numpy()
    )
    return sample_df, np.sort(real_quality_pool)


def preprocess_to_arr(row: pd.Series, spec: VariantSpec, real_quality_pool: np.ndarray) -> np.ndarray:
    raw_path = Path(row["raw_path"])
    label = str(row["label"])
    rel_path = str(row["relative_path"]).replace("\\", "/")

    bgr = read_bgr(raw_path)
    orientation = read_orientation(raw_path)
    bgr = apply_orientation(bgr, orientation)

    if spec.fake_precompress_444 and label == "ai":
        q_pre = deterministic_pick(f"pre:{rel_path}", real_quality_pool)
        bgr = jpeg_roundtrip_pil(bgr, quality=q_pre, subsampling=0)

    bgr, _, pad_left, pad_top = pad_to_min_size(bgr, min_size=PAD_MIN_SIZE)
    bgr = center_crop_with_mode(
        bgr,
        shift_mode=spec.shift_mode,
        size=CROP_SIZE,
        misalign_offset=GRID_OFFSET,
        pad_origin=(pad_left, pad_top),
    )

    if spec.bottleneck == "pil420":
        q_bottle = deterministic_q(rel_path, JPEG_Q_MIN, JPEG_Q_MAX)
        bgr = jpeg_roundtrip_pil(bgr, quality=q_bottle, subsampling=2)

    return bgr_to_ycrcb(bgr).astype(np.float64)


def _convolve_valid(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    out_h = channel.shape[0] - kh + 1
    out_w = channel.shape[1] - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel larger than input.")
    windows = np.lib.stride_tricks.sliding_window_view(channel, (kh, kw))
    return np.tensordot(windows, kernel[::-1, ::-1], axes=((2, 3), (0, 1)))


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


def extract_cfa_periodicity(arr: np.ndarray) -> dict[str, float]:
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    cr = arr[:, :, 1] - 128.0
    cb = arr[:, :, 2] - 128.0
    cr_res = _convolve_valid(cr, kernel)
    cb_res = _convolve_valid(cb, kernel)
    feats = {}
    feats.update(_periodicity_energies(cr_res, "cfa_cr"))
    feats.update(_periodicity_energies(cb_res, "cfa_cb"))
    return feats


def _block_view(arr: np.ndarray, block: int) -> np.ndarray:
    h = (arr.shape[0] // block) * block
    w = (arr.shape[1] // block) * block
    trimmed = arr[:h, :w]
    return trimmed.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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


def extract_noise_level_features(arr: np.ndarray) -> dict[str, float]:
    y = arr[:, :, 0].astype(np.float64)
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

    if means.size != variances.size:
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

    mr = rankdata(m)
    vr = rankdata(v)
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
        if right == bins[-1]:
            mask = (m >= left) & (m <= right)
        else:
            mask = (m >= left) & (m < right)
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


def extract_wavelet_dependency_features(arr: np.ndarray) -> dict[str, float]:
    y = arr[:, :, 0].astype(np.float64)
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


def extract_baseline_features(arr: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(extract_frequency_features(arr))
    feats.update(extract_color_features(arr))
    feats.update(extract_microtexture_features(arr))
    feats.update(extract_spatial_features(arr))
    return feats


def extract_candidate_features(arr: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(extract_cfa_periodicity(arr))
    feats.update(extract_noise_level_features(arr))
    feats.update(extract_wavelet_dependency_features(arr))
    return feats


def process_sample(row: pd.Series, spec: VariantSpec, real_quality_pool: np.ndarray) -> tuple[dict[str, float | str], np.ndarray]:
    arr = preprocess_to_arr(row, spec, real_quality_pool)
    record: dict[str, float | str] = {
        "variant": spec.name,
        "generator": str(row["generator"]),
        "label": str(row["label"]),
        "relative_path": str(row["relative_path"]).replace("\\", "/"),
    }
    record.update(extract_baseline_features(arr))
    record.update(extract_candidate_features(arr))
    return record, arr


def run_extraction(sample_df: pd.DataFrame, real_quality_pool: np.ndarray) -> tuple[pd.DataFrame, list[np.ndarray]]:
    records: list[dict[str, float | str]] = []
    latency_arrays: list[np.ndarray] = []
    tasks = [
        (row, spec)
        for _, row in sample_df.iterrows()
        for spec in VARIANTS
    ]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(process_sample, row=row, spec=spec, real_quality_pool=real_quality_pool): (row["relative_path"], spec.name)
            for row, spec in tasks
        }
        for fut in as_completed(futs):
            rel_path, variant = futs[fut]
            try:
                record, arr = fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed on {variant} :: {rel_path}: {exc}") from exc
            records.append(record)
            if variant == VARIANTS[0].name and len(latency_arrays) < 64:
                latency_arrays.append(arr)
    return pd.DataFrame(records), latency_arrays


def _probabilities_by_group(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    preds = np.full(len(X), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    for train_idx, test_idx in splitter.split(X, y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    return preds


def stratified_auc(df: pd.DataFrame, feature_cols: list[str]) -> float:
    X = df[feature_cols]
    y = (df["label"] == "ai").astype(int).to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SAMPLE_SEED)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return float(roc_auc_score(y, prob))


def group_auc(df: pd.DataFrame, feature_cols: list[str]) -> tuple[float, dict[str, float]]:
    X = df[feature_cols]
    y = (df["label"] == "ai").astype(int).to_numpy()
    groups = df["generator"].to_numpy()
    prob = _probabilities_by_group(X, y, groups)
    overall = float(roc_auc_score(y, prob))
    per_generator = {}
    for generator in sorted(df["generator"].unique()):
        mask = groups == generator
        per_generator[str(generator)] = float(roc_auc_score(y[mask], prob[mask]))
    return overall, per_generator


def cohen_d(real: np.ndarray, fake: np.ndarray) -> float:
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    real = real[np.isfinite(real)]
    fake = fake[np.isfinite(fake)]
    if real.size < 2 or fake.size < 2:
        return 0.0
    vr = float(np.var(real, ddof=1))
    vf = float(np.var(fake, ddof=1))
    pooled = ((real.size - 1) * vr + (fake.size - 1) * vf) / max(real.size + fake.size - 2, 1)
    if pooled <= EPS:
        return 0.0
    return float((fake.mean() - real.mean()) / math.sqrt(pooled))


def feature_family_correlation(df: pd.DataFrame, family_cols: list[str]) -> dict[str, float]:
    baseline = df[BASELINE_KEYS].copy()
    family = df[family_cols].copy()
    imp = SimpleImputer(strategy="median")
    baseline_imp = pd.DataFrame(imp.fit_transform(baseline), columns=BASELINE_KEYS)
    family_imp = pd.DataFrame(imp.fit_transform(family), columns=family_cols)
    full = np.corrcoef(
        np.concatenate([family_imp.to_numpy().T, baseline_imp.to_numpy().T], axis=0)
    )
    f = len(family_cols)
    sub = np.abs(full[:f, f:])
    per_feature = np.nanmax(sub, axis=1)
    return {
        "mean_max_abs_corr_to_baseline": float(np.nanmean(per_feature)),
        "max_abs_corr_to_baseline": float(np.nanmax(per_feature)),
    }


def individual_feature_metrics(df: pd.DataFrame, variant: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for family_name, cols in {
        "cfa_periodicity": CFA_KEYS,
        "noise_level_function": NLF_KEYS,
        "wavelet_dependencies": WAVELET_KEYS,
    }.items():
        for col in cols:
            real = df.loc[df["label"] == "nature", col].to_numpy()
            fake = df.loc[df["label"] == "ai", col].to_numpy()
            values = df[[col, "label", "generator"]].copy()
            rows.append({
                "variant": variant,
                "family": family_name,
                "feature": col,
                "cohen_d": cohen_d(real, fake),
                "stratified_auc": stratified_auc(values, [col]),
                "logo_auc": group_auc(values, [col])[0],
            })
    return rows


def benchmark_sets(df: pd.DataFrame, variant: str) -> tuple[list[dict[str, object]], list[dict[str, float | str]]]:
    rows: list[dict[str, object]] = []
    individual = individual_feature_metrics(df, variant)
    baseline_logo = None
    for set_name, cols in FEATURE_SETS.items():
        strat_auc = stratified_auc(df, cols)
        logo_auc, per_generator = group_auc(df, cols)
        if set_name == "baseline33":
            baseline_logo = logo_auc
        row: dict[str, object] = {
            "variant": variant,
            "feature_set": set_name,
            "n_features": len(cols),
            "stratified_auc": strat_auc,
            "logo_auc": logo_auc,
            "per_generator_auc": per_generator,
        }
        if set_name in {"cfa_periodicity", "noise_level_function", "wavelet_dependencies", "all_new"}:
            row.update(feature_family_correlation(df, cols))
        rows.append(row)

    if baseline_logo is not None:
        for row in rows:
            row["delta_logo_vs_baseline33"] = float(row["logo_auc"] - baseline_logo)
    return rows, individual


def latency_benchmark(arrays: list[np.ndarray]) -> dict[str, float]:
    results: dict[str, float] = {}
    benchmarks = {
        "baseline33_ms_per_image": extract_baseline_features,
        "cfa_periodicity_ms_per_image": extract_cfa_periodicity,
        "noise_level_function_ms_per_image": extract_noise_level_features,
        "wavelet_dependencies_ms_per_image": extract_wavelet_dependency_features,
        "all_new_ms_per_image": extract_candidate_features,
    }
    for key, fn in benchmarks.items():
        start = time.perf_counter()
        for _ in range(3):
            for arr in arrays:
                fn(arr)
        elapsed = time.perf_counter() - start
        results[key] = float(elapsed * 1000.0 / (3 * len(arrays)))
    results["baseline_plus_all_new_ms_per_image"] = (
        results["baseline33_ms_per_image"] + results["all_new_ms_per_image"]
    )
    return results


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(key): stringify_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(value) for value in obj]
    return obj


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_df, real_quality_pool = build_clean_sample()
    feature_df, latency_arrays = run_extraction(sample_df, real_quality_pool)

    summary_rows: list[dict[str, object]] = []
    individual_rows: list[dict[str, float | str]] = []
    per_variant_summary: dict[str, object] = {}
    for variant, sub in feature_df.groupby("variant"):
        rows, individual = benchmark_sets(sub.reset_index(drop=True), variant)
        summary_rows.extend(rows)
        individual_rows.extend(individual)
        per_variant_summary[variant] = {row["feature_set"]: row for row in rows}

    latency = latency_benchmark(latency_arrays)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["variant", "logo_auc"], ascending=[True, False])
    individual_df = pd.DataFrame(individual_rows)
    individual_df = individual_df.sort_values(["variant", "logo_auc"], ascending=[True, False])

    OUT_CSV.write_text(summary_df.to_csv(index=False), encoding="utf-8-sig")
    OUT_FEATURE_CSV.write_text(individual_df.to_csv(index=False), encoding="utf-8-sig")

    output = {
        "sample": {
            "n_total": int(len(sample_df)),
            "by_generator_label": sample_df.groupby(["generator", "label"]).size().to_dict(),
        },
        "variants": per_variant_summary,
        "latency_ms_per_image": latency,
    }
    OUT_JSON.write_text(json.dumps(stringify_keys(output), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved summary CSV: {OUT_CSV}")
    print(f"Saved individual CSV: {OUT_FEATURE_CSV}")
    print(f"Saved JSON: {OUT_JSON}")
    print(summary_df[[
        "variant",
        "feature_set",
        "n_features",
        "stratified_auc",
        "logo_auc",
        "delta_logo_vs_baseline33",
    ]].to_string(index=False))
    print("Latency (ms/image):")
    for key, value in latency.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
