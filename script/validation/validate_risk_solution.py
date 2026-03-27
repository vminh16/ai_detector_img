from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
from src.feature_extraction.spatial import FEATURE_KEYS as SPATIAL_KEYS
from src.feature_extraction.spatial import extract_spatial_features
from src.preprocessing.pipeline import apply_orientation, read_orientation

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FEATURES_CSV = PROJECT_ROOT / "features" / "features_dataset.csv"
METADATA_PARQUET = PROJECT_ROOT / "audit_output" / "data_audit" / "metadata" / "per_file_metadata.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "param" / "champion_lgbm.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "param" / "platt_calibrator.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "param" / "threshold_lock.json"
SCHEMA_PATH = PROJECT_ROOT / "models" / "param" / "feature_schema.json"
OUT_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "risk_solution"
OUT_JSON = OUT_ROOT / "risk_solution_validation.json"
OUT_CSV = OUT_ROOT / "risk_solution_variant_metrics.csv"
OUT_VARIANT_PARQUET = OUT_ROOT / "risk_solution_variant_features.parquet"

CROP_SIZE = 256
PAD_MIN_SIZE = 259
GRID_OFFSET = 3
JPEG_Q_MIN = 90
JPEG_Q_MAX = 98
SAMPLE_SEED = 42
SAMPLE_PER_GROUP = 100
BOOTSTRAP_ROUNDS = 200
WORKERS = 8

ALL_FEATURES = list(FREQ_KEYS) + list(COLOR_KEYS) + list(MICRO_KEYS) + list(SPATIAL_KEYS)
GROUPS = {
    "frequency": list(FREQ_KEYS),
    "color": list(COLOR_KEYS),
    "microtexture": list(MICRO_KEYS),
    "spatial": list(SPATIAL_KEYS),
}

SUBSAMPLING_MAP = {
    0: "4:4:4",
    1: "4:2:2",
    2: "4:2:0",
    -1: "other",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    shift_mode: str
    bottleneck: str | None
    fake_precompress_444: bool = False


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("baseline_conditional_cv2", shift_mode="conditional", bottleneck="cv2"),
    VariantSpec("baseline_allshift_cv2", shift_mode="all", bottleneck="cv2"),
    VariantSpec("pil_only_conditional", shift_mode="conditional", bottleneck="pil420"),
    VariantSpec("pil_only_allshift", shift_mode="all", bottleneck="pil420"),
    VariantSpec(
        "method_b_conditional",
        shift_mode="conditional",
        bottleneck="cv2",
        fake_precompress_444=True,
    ),
    VariantSpec(
        "method_b_allshift",
        shift_mode="all",
        bottleneck="cv2",
        fake_precompress_444=True,
    ),
    VariantSpec(
        "method_a_allshift",
        shift_mode="all",
        bottleneck=None,
        fake_precompress_444=True,
    ),
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
) -> tuple[np.ndarray, bool]:
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
    return img[y0:y0 + size, x0:x0 + size], apply_shift


def deterministic_q(key: str, q_min: int = JPEG_Q_MIN, q_max: int = JPEG_Q_MAX) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little")
    return q_min + (value % (q_max - q_min + 1))


def deterministic_pick(key: str, values: np.ndarray) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little")
    return int(values[value % len(values)])


def jpeg_bottleneck_cv2(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("cv2.imdecode failed")
    return out


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


def extract_features(arr: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(extract_frequency_features(arr))
    features.update(extract_color_features(arr))
    features.update(extract_microtexture_features(arr))
    features.update(extract_spatial_features(arr))
    return features


def compute_misalignment_from_dims(width: np.ndarray, height: np.ndarray) -> np.ndarray:
    pad_left = np.where(width >= PAD_MIN_SIZE, 0, ((PAD_MIN_SIZE - width) // 2).astype(int))
    pad_top = np.where(height >= PAD_MIN_SIZE, 0, ((PAD_MIN_SIZE - height) // 2).astype(int))
    padded_width = np.maximum(width, PAD_MIN_SIZE)
    padded_height = np.maximum(height, PAD_MIN_SIZE)
    x0 = (padded_width // 2 - CROP_SIZE // 2).astype(int)
    y0 = (padded_height // 2 - CROP_SIZE // 2).astype(int)
    return ((x0 - pad_left) % 8 == 0) | ((y0 - pad_top) % 8 == 0)


def get_sampling(path: Path) -> str:
    with Image.open(path) as im:
        return SUBSAMPLING_MAP.get(JpegImagePlugin.get_sampling(im), "other")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / math.sqrt(pooled))


def fisher_dprime(df: pd.DataFrame) -> float:
    X = df[ALL_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    y = (df["label"] == "ai").astype(int).to_numpy()
    mu_real = X_imp[y == 0].mean(axis=0)
    mu_fake = X_imp[y == 1].mean(axis=0)
    centered_real = X_imp[y == 0] - mu_real
    centered_fake = X_imp[y == 1] - mu_fake
    pooled = np.vstack([centered_real, centered_fake])
    cov = LedoitWolf().fit(pooled).covariance_
    delta = mu_fake - mu_real
    return float(math.sqrt(max(delta @ np.linalg.pinv(cov) @ delta, 0.0)))


def logistic_cv_auc(df: pd.DataFrame) -> float:
    X = df[ALL_FEATURES]
    y = (df["label"] == "ai").astype(int).to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SAMPLE_SEED)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return float(roc_auc_score(y, prob))


def variant_metrics(df: pd.DataFrame) -> dict[str, float | dict[str, float]]:
    real = df[df["label"] == "nature"]
    fake = df[df["label"] == "ai"]

    feature_ds = {name: cohen_d(real[name].to_numpy(dtype=float), fake[name].to_numpy(dtype=float)) for name in ALL_FEATURES}
    group_sums = {
        group: float(np.nansum([abs(feature_ds[name]) for name in names]))
        for group, names in GROUPS.items()
    }
    sum_abs_d = float(np.nansum([abs(v) for v in feature_ds.values()]))
    dprime = fisher_dprime(df)
    return {
        "n_total": int(len(df)),
        "n_real": int(len(real)),
        "n_fake": int(len(fake)),
        "sum_abs_d": sum_abs_d,
        "group_sum_abs_d": group_sums,
        "fisher_dprime": dprime,
        "fisher_auc_est": float(norm.cdf(dprime / math.sqrt(2.0))),
        "logreg_cv_auc": logistic_cv_auc(df),
    }


def bootstrap_metric_ci(df: pd.DataFrame, metric_key: str) -> tuple[float, float]:
    rng = np.random.default_rng(SAMPLE_SEED)
    real = df[df["label"] == "nature"]
    fake = df[df["label"] == "ai"]
    values: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        draw_real = real.iloc[rng.integers(0, len(real), len(real))]
        draw_fake = fake.iloc[rng.integers(0, len(fake), len(fake))]
        draw = pd.concat([draw_real, draw_fake], ignore_index=True)
        value = variant_metrics(draw)[metric_key]
        values.append(float(value))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def preprocess_variant(
    row: pd.Series,
    spec: VariantSpec,
    real_quality_pool: np.ndarray,
) -> dict[str, float | str | bool]:
    raw_path = Path(row["raw_path"])
    label = str(row["label"])
    rel_path = str(row["relative_path"]).replace("\\", "/")
    bgr = read_bgr(raw_path)
    orientation = read_orientation(raw_path)
    bgr = apply_orientation(bgr, orientation)

    q_pre = None
    if spec.fake_precompress_444 and label == "ai":
        q_pre = deterministic_pick(f"pre:{rel_path}", real_quality_pool)
        bgr = jpeg_roundtrip_pil(bgr, quality=q_pre, subsampling=0)

    bgr, padded, pad_left, pad_top = pad_to_min_size(bgr, min_size=PAD_MIN_SIZE)
    bgr, shift_applied = center_crop_with_mode(
        bgr,
        shift_mode=spec.shift_mode,
        size=CROP_SIZE,
        misalign_offset=GRID_OFFSET,
        pad_origin=(pad_left, pad_top),
    )

    q_bottle = None
    if spec.bottleneck == "cv2":
        q_bottle = deterministic_q(rel_path, JPEG_Q_MIN, JPEG_Q_MAX)
        bgr = jpeg_bottleneck_cv2(bgr, quality=q_bottle)
    elif spec.bottleneck == "pil420":
        q_bottle = deterministic_q(rel_path, JPEG_Q_MIN, JPEG_Q_MAX)
        bgr = jpeg_roundtrip_pil(bgr, quality=q_bottle, subsampling=2)

    arr = bgr_to_ycrcb(bgr)
    feats = extract_features(arr)
    return {
        "variant": spec.name,
        "generator": str(row["generator"]),
        "label": label,
        "relative_path": rel_path,
        "raw_width": int(row["width"]),
        "raw_height": int(row["height"]),
        "orig_needs_pad": bool(row["needs_pad"]),
        "orig_misalign_trigger": bool(row["misalign_trigger"]),
        "shift_applied": bool(shift_applied),
        "pad_applied": bool(padded),
        "q_pre": q_pre,
        "q_bottle": q_bottle,
        **feats,
    }


def run_variant_experiments(sample_df: pd.DataFrame, real_quality_pool: np.ndarray) -> pd.DataFrame:
    tasks = [
        (row, spec)
        for _, row in sample_df.iterrows()
        for spec in VARIANTS
    ]
    records: list[dict[str, float | str | bool]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(preprocess_variant, row=row, spec=spec, real_quality_pool=real_quality_pool): (row["relative_path"], spec.name)
            for row, spec in tasks
        }
        for fut in as_completed(futs):
            rel_path, variant = futs[fut]
            try:
                records.append(fut.result())
            except Exception as exc:
                raise RuntimeError(f"Failed on {variant} :: {rel_path}: {exc}") from exc
    return pd.DataFrame(records)


def build_full_audit() -> dict[str, object]:
    manifest = pd.read_csv(PROCESSED_MANIFEST)
    meta = pd.read_parquet(METADATA_PARQUET)
    features = pd.read_csv(FEATURES_CSV)

    meta["needs_pad"] = (meta["width"] < PAD_MIN_SIZE) | (meta["height"] < PAD_MIN_SIZE)
    meta["misalign_trigger"] = compute_misalignment_from_dims(meta["width"].to_numpy(), meta["height"].to_numpy())

    fake_sizes = (
        meta.loc[meta["inferred_label"] == "fake", ["width", "height"]]
        .astype(int)
        .astype(str)
        .agg("x".join, axis=1)
        .value_counts()
        .head(10)
        .to_dict()
    )
    real_sizes = (
        meta.loc[meta["inferred_label"] == "real", ["width", "height"]]
        .astype(int)
        .astype(str)
        .agg("x".join, axis=1)
        .value_counts()
        .head(10)
        .to_dict()
    )

    current_real_paths = [
        RAW_ROOT / Path(rel)
        for rel in manifest.loc[manifest["label"] == "nature", "file_path"]
        .str.split("data\\raw\\", n=1, regex=False)
        .str[-1]
        .str.replace("\\", "/", regex=False)
        .tolist()
    ]
    subsampling_counter: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(get_sampling, path) for path in current_real_paths]
        for fut in as_completed(futs):
            subsampling_counter[fut.result()] += 1

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["tau_op"]
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    feature_order = schema["active_features"]

    meta["raw_rel"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    manifest["raw_rel"] = manifest["file_path"].str.split("data\\raw\\", n=1, regex=False).str[-1].str.replace("\\", "/", regex=False)
    features["proc_rel"] = features["file_path"].str.split("data\\processed\\", n=1, regex=False).str[-1].str.replace("\\", "/", regex=False)
    manifest["proc_rel"] = manifest["output_path"].str.split("data\\processed\\", n=1, regex=False).str[-1].str.replace("\\", "/", regex=False)

    merged = (
        features
        .merge(manifest[["proc_rel", "padded", "raw_rel"]], on="proc_rel", how="left")
        .merge(meta[["raw_rel", "misalign_trigger"]], on="raw_rel", how="left")
    )
    raw_scores = model.predict(merged[feature_order])
    merged["score"] = calibrator.predict_proba(np.asarray(raw_scores).reshape(-1, 1))[:, 1]
    merged["pred_high"] = merged["score"] >= threshold

    def subgroup_stats(label: str, flag: str) -> dict[str, dict[str, float]]:
        sub = merged[merged["label"] == label].groupby(flag).agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            high_rate=("pred_high", "mean"),
        )
        return {
            str(idx): {
                "n": int(row["n"]),
                "mean_score": float(row["mean_score"]),
                "high_rate": float(row["high_rate"]),
            }
            for idx, row in sub.iterrows()
        }

    return {
        "dataset_counts": {
            "manifest_rows": int(len(manifest)),
            "fake_rows": int((meta["inferred_label"] == "fake").sum()),
            "real_rows": int((meta["inferred_label"] == "real").sum()),
        },
        "format_by_label": pd.crosstab(meta["inferred_label"], meta["format_detected"], normalize="index").round(4).to_dict(),
        "pad_rate_by_label": (meta.groupby("inferred_label")["needs_pad"].mean().round(4)).to_dict(),
        "misalign_rate_by_label": (meta.groupby("inferred_label")["misalign_trigger"].mean().round(4)).to_dict(),
        "pad_rate_by_generator_label": (
            meta.groupby(["inferred_generator", "inferred_label"])["needs_pad"].mean().round(4).to_dict()
        ),
        "misalign_rate_by_generator_label": (
            meta.groupby(["inferred_generator", "inferred_label"])["misalign_trigger"].mean().round(4).to_dict()
        ),
        "top_sizes_fake": fake_sizes,
        "top_sizes_real": real_sizes,
        "real_jpeg_subsampling": {
            key: {
                "count": int(value),
                "rate": float(value / len(current_real_paths)),
            }
            for key, value in sorted(subsampling_counter.items())
        },
        "model_score_by_real_flags": {
            "padded": subgroup_stats("nature", "padded"),
            "misalign_trigger": subgroup_stats("nature", "misalign_trigger"),
        },
        "model_score_by_fake_flags": {
            "padded": subgroup_stats("ai", "padded"),
        },
    }


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
    for (generator, label), group in usable.groupby(["generator", "label"]):
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


def summarise_variants(df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    summary: dict[str, object] = {}
    rows: list[dict[str, float | str]] = []
    for variant, sub in df.groupby("variant"):
        metrics = variant_metrics(sub)
        ci_d = bootstrap_metric_ci(sub, "sum_abs_d")
        ci_dp = bootstrap_metric_ci(sub, "fisher_dprime")
        metrics["sum_abs_d_ci95"] = [ci_d[0], ci_d[1]]
        metrics["fisher_dprime_ci95"] = [ci_dp[0], ci_dp[1]]
        summary[variant] = metrics
        rows.append({
            "variant": variant,
            "n_total": metrics["n_total"],
            "sum_abs_d": metrics["sum_abs_d"],
            "sum_abs_d_ci95_low": ci_d[0],
            "sum_abs_d_ci95_high": ci_d[1],
            "fisher_dprime": metrics["fisher_dprime"],
            "fisher_dprime_ci95_low": ci_dp[0],
            "fisher_dprime_ci95_high": ci_dp[1],
            "fisher_auc_est": metrics["fisher_auc_est"],
            "logreg_cv_auc": metrics["logreg_cv_auc"],
            "frequency_sum_abs_d": metrics["group_sum_abs_d"]["frequency"],
            "color_sum_abs_d": metrics["group_sum_abs_d"]["color"],
            "microtexture_sum_abs_d": metrics["group_sum_abs_d"]["microtexture"],
            "spatial_sum_abs_d": metrics["group_sum_abs_d"]["spatial"],
        })
    return summary, pd.DataFrame(rows).sort_values("sum_abs_d", ascending=False)


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(key): stringify_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(value) for value in obj]
    return obj


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    full_audit = build_full_audit()
    sample_df, real_quality_pool = build_clean_sample()
    variant_df = run_variant_experiments(sample_df, real_quality_pool)
    variant_summary, variant_table = summarise_variants(variant_df)

    out = stringify_keys({
        "full_audit": full_audit,
        "clean_sample": {
            "n_total": int(len(sample_df)),
            "by_generator_label": sample_df.groupby(["generator", "label"]).size().to_dict(),
            "misalign_rate_by_label": sample_df.groupby("label")["misalign_trigger"].mean().round(4).to_dict(),
        },
        "variant_summary": variant_summary,
    })
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    variant_table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    variant_df.to_parquet(OUT_VARIANT_PARQUET, index=False)
    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved CSV : {OUT_CSV}")
    print(f"Saved variant features: {OUT_VARIANT_PARQUET}")
    print(variant_table.to_string(index=False))


if __name__ == "__main__":
    main()
