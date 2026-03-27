from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
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
OUT_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "pipeline_revision"
OUT_JSON = OUT_ROOT / "pipeline_revision_validation.json"
OUT_CSV = OUT_ROOT / "pipeline_revision_variant_metrics.csv"
OUT_SUBS_CSV = OUT_ROOT / "pipeline_revision_subsampling_metrics.csv"

CROP_SIZE = 256
GRID_OFFSET = 3
JPEG_Q_MIN = 90
JPEG_Q_MAX = 98
SAMPLE_SEED = 42
SAMPLE_PER_GROUP = 100
SUBSAMPLE_PER_CLASS = 400
WORKERS = 8
SUBSAMPLING_MAP = {
    0: "4:4:4",
    1: "4:2:2",
    2: "4:2:0",
    -1: "other",
}

BASELINE_KEYS = list(FREQ_KEYS) + list(COLOR_KEYS) + list(MICRO_KEYS) + list(SPATIAL_KEYS)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    resize_short_side: int | None
    apply_chroma_canonicalization: bool
    fixed_residue: tuple[int, int]


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("native_nojpeg", resize_short_side=None, apply_chroma_canonicalization=False, fixed_residue=(3, 3)),
    VariantSpec("native_chroma420", resize_short_side=None, apply_chroma_canonicalization=True, fixed_residue=(3, 3)),
    VariantSpec("resize263_nojpeg", resize_short_side=263, apply_chroma_canonicalization=False, fixed_residue=(3, 3)),
    VariantSpec("resize263_chroma420", resize_short_side=263, apply_chroma_canonicalization=True, fixed_residue=(3, 3)),
)


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(key): stringify_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(value) for value in obj]
    return obj


def mutual_information_binary(
    p_positive_given_y0: float,
    p_positive_given_y1: float,
    p_y0: float,
    p_y1: float,
) -> float:
    p = {
        (0, 1): p_y0 * p_positive_given_y0,
        (0, 0): p_y0 * (1.0 - p_positive_given_y0),
        (1, 1): p_y1 * p_positive_given_y1,
        (1, 0): p_y1 * (1.0 - p_positive_given_y1),
    }
    p_n = {
        0: p[(0, 0)] + p[(1, 0)],
        1: p[(0, 1)] + p[(1, 1)],
    }
    p_y = {0: p_y0, 1: p_y1}
    out = 0.0
    for y in (0, 1):
        for n in (0, 1):
            joint = p[(y, n)]
            if joint > 0.0:
                out += joint * math.log2(joint / (p_y[y] * p_n[n]))
    return float(out)


def read_bgr(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def resize_short_side(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side == target:
        return img
    scale = target / short_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def nonzero_phase_center_crop(img: np.ndarray, size: int, residue: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size + 7 or w < size + 7:
        raise ValueError(f"Image too small for residue-safe crop: {img.shape[:2]}")
    rx, ry = residue
    center_x = w // 2 - size // 2
    center_y = h // 2 - size // 2

    x_candidates = np.arange(rx, w - size + 1, 8)
    y_candidates = np.arange(ry, h - size + 1, 8)
    x0 = int(x_candidates[np.argmin(np.abs(x_candidates - center_x))])
    y0 = int(y_candidates[np.argmin(np.abs(y_candidates - center_y))])
    return img[y0:y0 + size, x0:x0 + size]


def bgr_to_ycrcb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def chroma_canonicalize_420(ycrcb: np.ndarray) -> np.ndarray:
    arr = ycrcb.astype(np.float32)
    out = arr.copy()
    for idx in (1, 2):
        ch = arr[:, :, idx]
        blurred = cv2.sepFilter2D(ch, -1, np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0, np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0, borderType=cv2.BORDER_REFLECT_101)
        down = cv2.resize(blurred, (ch.shape[1] // 2, ch.shape[0] // 2), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_CUBIC)
        out[:, :, idx] = up
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def extract_baseline_features(arr: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(extract_frequency_features(arr))
    feats.update(extract_color_features(arr))
    feats.update(extract_microtexture_features(arr))
    feats.update(extract_spatial_features(arr))
    return feats


def chroma_highband_ratio(arr: np.ndarray) -> float:
    total_energy = 0.0
    high_energy = 0.0
    for idx in (1, 2):
        ch = arr[:, :, idx].astype(np.float64) - 128.0
        spec = np.fft.fftshift(np.fft.fft2(ch))
        power = np.abs(spec) ** 2
        h, w = ch.shape
        yy, xx = np.ogrid[-h // 2:h // 2, -w // 2:w // 2]
        mask = (np.abs(xx) > (w // 4)) | (np.abs(yy) > (h // 4))
        total_energy += float(power.sum())
        high_energy += float(power[mask].sum())
    return float(high_energy / total_energy) if total_energy > 0 else 0.0


def subsampling_binary_label(value: str | None) -> int | None:
    if value == "4:4:4":
        return 1
    if value == "4:2:0":
        return 0
    return None


def preprocess_variant(row: pd.Series, spec: VariantSpec) -> tuple[dict[str, object], np.ndarray]:
    raw_path = Path(row["raw_path"])
    bgr = read_bgr(raw_path)
    orientation = read_orientation(raw_path)
    bgr = apply_orientation(bgr, orientation)
    if spec.resize_short_side is not None:
        bgr = resize_short_side(bgr, spec.resize_short_side)
    bgr = nonzero_phase_center_crop(bgr, CROP_SIZE, spec.fixed_residue)
    arr = bgr_to_ycrcb(bgr)
    if spec.apply_chroma_canonicalization:
        arr = chroma_canonicalize_420(arr)
    feats = extract_baseline_features(arr)
    return {
        "variant": spec.name,
        "generator": str(row["generator"]),
        "label": str(row["label"]),
        "relative_path": str(row["relative_path"]).replace("\\", "/"),
        "jpeg_subsampling": row.get("jpeg_subsampling"),
        "format_detected": row.get("format_detected"),
        "chroma_highband_ratio": chroma_highband_ratio(arr),
        **feats,
    }, arr


def stratified_auc(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> float:
    X = df[feature_cols]
    y = df[target_col].to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SAMPLE_SEED)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return float(roc_auc_score(y, prob))


def group_auc(df: pd.DataFrame, target_col: str, feature_cols: list[str], group_col: str) -> float:
    X = df[feature_cols]
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()
    pred = np.full(len(df), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    for train_idx, test_idx in splitter.split(X, y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    return float(roc_auc_score(y, pred))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
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


def build_meta() -> pd.DataFrame:
    meta = pd.read_parquet(METADATA_PARQUET)
    meta["min_side"] = meta[["width", "height"]].min(axis=1)
    meta["label"] = meta["inferred_label"].replace({"real": "nature", "fake": "ai"})
    meta["generator"] = meta["inferred_generator"].str.lower()
    meta["relative_path"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    meta["raw_path"] = [str((RAW_ROOT / Path(rel)).resolve()) for rel in meta["relative_path"]]
    meta["exists_on_disk"] = [Path(path).exists() for path in meta["raw_path"]]
    return meta[meta["exists_on_disk"]].copy()


def dataset_branch_stats(meta: pd.DataFrame) -> dict[str, object]:
    p_y0 = float((meta["label"] == "nature").mean())
    p_y1 = 1.0 - p_y0
    main_mask = meta["min_side"] >= (CROP_SIZE + 7)
    rate_real = float(main_mask[meta["label"] == "nature"].mean())
    rate_fake = float(main_mask[meta["label"] == "ai"].mean())
    mi = mutual_information_binary(rate_real, rate_fake, p_y0, p_y1)
    per_gen = (
        meta.groupby(["generator", "label"])["min_side"]
        .apply(lambda s: float((s >= CROP_SIZE + 7).mean()))
        .to_dict()
    )
    top_sizes = {}
    for label in ("nature", "ai"):
        top_sizes[label] = (
            meta.loc[meta["label"] == label, "min_side"]
            .value_counts()
            .head(15)
            .to_dict()
        )
    return {
        "main_branch_threshold": CROP_SIZE + 7,
        "main_branch_rate_by_label": {"nature": rate_real, "ai": rate_fake},
        "mi_bits_main_branch_vs_label": mi,
        "per_generator_label_main_rate": per_gen,
        "top_min_side_counts": top_sizes,
    }


def build_clean_classification_sample(meta: pd.DataFrame) -> pd.DataFrame:
    clean_generators = ["midjourney", "sdv14", "sdv15", "wukong"]
    usable = meta[
        meta["generator"].isin(clean_generators)
        & (meta["min_side"] >= (CROP_SIZE + 7))
        & (meta["format_detected"].isin(["JPEG", "PNG"]))
    ].copy()
    usable["generator"] = usable["generator"].replace({
        "midjourney": "Midjourney",
        "sdv14": "SDv14",
        "sdv15": "SDv15",
        "wukong": "Wukong",
    })
    sampled = []
    for (_, _), group in usable.groupby(["generator", "label"]):
        take = min(SAMPLE_PER_GROUP, len(group))
        sampled.append(group.sample(n=take, random_state=SAMPLE_SEED))
    return pd.concat(sampled, ignore_index=True)


def build_real_subsampling_sample(meta: pd.DataFrame) -> pd.DataFrame:
    usable = meta[
        (meta["label"] == "nature")
        & (meta["min_side"] >= (CROP_SIZE + 7))
        & (meta["extension"].str.lower().isin([".jpg", ".jpeg"]))
    ].copy()
    if usable.empty:
        raise RuntimeError("No real JPEG candidates found for subsampling audit.")

    def detect_sampling(path_str: str) -> str | None:
        try:
            with Image.open(path_str) as im:
                return SUBSAMPLING_MAP.get(JpegImagePlugin.get_sampling(im), "other")
        except Exception:
            return None

    candidate = usable.sample(n=min(6000, len(usable)), random_state=SAMPLE_SEED).copy()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(detect_sampling, path): idx for idx, path in candidate["raw_path"].items()}
        sampling = {}
        for fut in as_completed(futs):
            sampling[futs[fut]] = fut.result()
    candidate["jpeg_subsampling"] = pd.Series(sampling)
    candidate = candidate[candidate["jpeg_subsampling"].isin(["4:4:4", "4:2:0"])].copy()

    sampled = []
    for subsampling_value in ("4:4:4", "4:2:0"):
        group = candidate[candidate["jpeg_subsampling"] == subsampling_value]
        if len(group) < SUBSAMPLE_PER_CLASS:
            raise RuntimeError(
                f"Insufficient {subsampling_value} samples after detection: {len(group)}"
            )
        sampled.append(group.sample(n=SUBSAMPLE_PER_CLASS, random_state=SAMPLE_SEED))
    return pd.concat(sampled, ignore_index=True)


def run_variant_processing(df: pd.DataFrame) -> pd.DataFrame:
    tasks = [(row, spec) for _, row in df.iterrows() for spec in VARIANTS]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(preprocess_variant, row=row, spec=spec): (row["relative_path"], spec.name)
            for row, spec in tasks
        }
        for fut in as_completed(futs):
            rel, name = futs[fut]
            try:
                record, _ = fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed on {name} :: {rel}: {exc}") from exc
            records.append(record)
    return pd.DataFrame(records)


def classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in df.groupby("variant"):
        work = sub.copy()
        work["target"] = (work["label"] == "ai").astype(int)
        rows.append({
            "variant": variant,
            "task": "real_vs_fake",
            "stratified_auc": stratified_auc(work, "target", BASELINE_KEYS),
            "logo_auc": group_auc(work, "target", BASELINE_KEYS, "generator"),
            "mean_chroma_highband_real": float(work.loc[work["label"] == "nature", "chroma_highband_ratio"].mean()),
            "mean_chroma_highband_fake": float(work.loc[work["label"] == "ai", "chroma_highband_ratio"].mean()),
            "cohen_d_chroma_highband_fake_minus_real": cohen_d(
                work.loc[work["label"] == "nature", "chroma_highband_ratio"].to_numpy(),
                work.loc[work["label"] == "ai", "chroma_highband_ratio"].to_numpy(),
            ),
        })
    return pd.DataFrame(rows)


def subsampling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in df.groupby("variant"):
        work = sub.copy()
        work["target"] = work["jpeg_subsampling"].map(subsampling_binary_label)
        work = work.dropna(subset=["target"]).copy()
        work["target"] = work["target"].astype(int)
        rows.append({
            "variant": variant,
            "task": "real_jpeg_444_vs_420",
            "stratified_auc": stratified_auc(work, "target", BASELINE_KEYS),
            "mean_chroma_highband_420": float(work.loc[work["target"] == 0, "chroma_highband_ratio"].mean()),
            "mean_chroma_highband_444": float(work.loc[work["target"] == 1, "chroma_highband_ratio"].mean()),
            "cohen_d_chroma_highband_444_minus_420": cohen_d(
                work.loc[work["target"] == 0, "chroma_highband_ratio"].to_numpy(),
                work.loc[work["target"] == 1, "chroma_highband_ratio"].to_numpy(),
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = build_meta()
    branch_stats = dataset_branch_stats(meta)

    class_sample = build_clean_classification_sample(meta)
    class_variant_df = run_variant_processing(class_sample)
    class_metrics = classification_metrics(class_variant_df)

    subs_sample = build_real_subsampling_sample(meta)
    subs_variant_df = run_variant_processing(subs_sample)
    subs_metrics = subsampling_metrics(subs_variant_df)

    combined = pd.concat([class_metrics, subs_metrics], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    subs_metrics.to_csv(OUT_SUBS_CSV, index=False, encoding="utf-8-sig")

    output = {
        "branch_stats": branch_stats,
        "classification_sample": {
            "n_total": int(len(class_sample)),
            "by_generator_label": class_sample.groupby(["generator", "label"]).size().to_dict(),
        },
        "subsampling_sample": {
            "n_total": int(len(subs_sample)),
            "by_subsampling": subs_sample["jpeg_subsampling"].value_counts().to_dict(),
        },
        "classification_metrics": class_metrics.to_dict(orient="records"),
        "subsampling_metrics": subs_metrics.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(stringify_keys(output), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved CSV : {OUT_CSV}")
    print(class_metrics.to_string(index=False))
    print(subs_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
