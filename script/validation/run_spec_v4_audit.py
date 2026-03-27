from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from scipy.signal import convolve2d
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import apply_orientation, read_orientation

AUDIT_ROOT = PROJECT_ROOT / "audit" / "spec_v4_20260319"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_PARQUET = PROJECT_ROOT / "audit_output" / "data_audit" / "metadata" / "per_file_metadata.parquet"
FEATURES_CSV = PROJECT_ROOT / "features" / "features_dataset.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

SEED = 42
BOOTSTRAP_ROUNDS = 2000
SAMPLING_WORKERS = 12
RAW_WORKERS = 8
PROXY_LABEL_PER_GROUP = 80
PROXY_NUISANCE_PER_CLASS = 300

FEATURE_FAMILIES: dict[str, list[str]] = {
    "frequency": [
        "frs_mid_variance",
        "dct_mid_mean",
        "dct_mid_variance",
        "dct_mid_skewness",
        "ps_alpha",
        "ps_deviation_variance",
    ],
    "color": [
        "local_color_inconsistency",
        "pearson_y_cr",
        "pearson_y_cb",
        "pearson_cr_cb",
        "energy_ratio_chroma",
        "glcm_contrast_cr",
        "glcm_correlation_cr",
        "glcm_energy_cr",
        "glcm_homogeneity_cr",
    ],
    "microtexture": [
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
    ],
    "spatial": [
        "spatial_snr_ratio",
        "cross_noise_ratio",
        "skew_noise_y",
        "kurt_noise_y",
        "skew_noise_cr",
        "kurt_noise_cr",
        "skew_noise_cb",
        "kurt_noise_cb",
    ],
}

CURRENT_KEEP_CANDIDATE_FEATURES: list[str] = [
    "frs_mid_variance",
    "local_color_inconsistency",
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
    "glcm_contrast_cr",
    "glcm_correlation_cr",
    "glcm_energy_cr",
    "glcm_homogeneity_cr",
    "spatial_snr_ratio",
    "skew_noise_y",
    "kurt_noise_y",
    "skew_noise_cr",
    "kurt_noise_cr",
    "skew_noise_cb",
    "kurt_noise_cb",
]

PROXY_KERNEL = np.array(
    [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
    dtype=np.float64,
)
LBP_OFFSETS: list[tuple[int, int]] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]
LBP_NONUNIFORM_LUT = np.empty(256, dtype=np.uint8)
for code in range(256):
    bits = [(code >> i) & 1 for i in range(8)]
    transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
    LBP_NONUNIFORM_LUT[code] = 1 if transitions > 2 else 0


@dataclass(frozen=True)
class ProxyVariant:
    name: str
    apply_chroma420: bool


PROXY_VARIANTS: tuple[ProxyVariant, ...] = (
    ProxyVariant("identity", apply_chroma420=False),
    ProxyVariant("chroma420", apply_chroma420=True),
)

SUBSAMPLING_MAP = {
    0: "4:4:4",
    1: "4:2:2",
    2: "4:2:0",
}


def ensure_audit_root() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(v) for v in obj]
    return obj


def logistic_pipe() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def norm_processed(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    idx = text.lower().find("/data/processed/")
    if idx < 0:
        raise ValueError(f"Cannot normalize processed path: {path}")
    return text[idx + len("/data/processed/") :]


def norm_raw(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    idx = text.lower().find("/data/raw/")
    if idx < 0:
        raise ValueError(f"Cannot normalize raw path: {path}")
    return text[idx + len("/data/raw/") :]


def current_raw_snapshot() -> tuple[list[str], set[str]]:
    rels = sorted(
        str(path.relative_to(RAW_ROOT)).replace("\\", "/")
        for path in RAW_ROOT.rglob("*")
        if path.is_file()
    )
    return rels, set(rels)


def snapshot_hash(rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_current_metadata(raw_rel_set: set[str]) -> pd.DataFrame:
    meta = pd.read_parquet(METADATA_PARQUET).copy()
    meta["relative_path"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    meta["exists_on_disk"] = meta["relative_path"].isin(raw_rel_set)
    meta = meta[meta["exists_on_disk"]].copy()
    meta["label"] = meta["inferred_label"].map({"real": "nature", "fake": "ai"})
    meta["generator"] = meta["inferred_generator"].str.lower()
    meta["raw_rel"] = meta["relative_path"]
    meta["raw_path"] = meta["raw_rel"].map(lambda rel: str((RAW_ROOT / rel).resolve()))
    meta["S"] = meta[["width", "height"]].min(axis=1)
    return meta


def input_mode_audit(meta: pd.DataFrame) -> None:
    mode_counts = meta["image_mode"].value_counts(dropna=False).rename_axis("image_mode").reset_index(name="count")
    mode_counts.to_csv(AUDIT_ROOT / "input_mode_counts.csv", index=False, encoding="utf-8-sig")

    alpha_by_label = pd.crosstab(meta["label"], meta["has_alpha"])
    gray_by_label = pd.crosstab(meta["label"], meta["is_grayscale"])

    summary = {
        "has_alpha_counts": meta["has_alpha"].value_counts(dropna=False).to_dict(),
        "is_grayscale_counts": meta["is_grayscale"].value_counts(dropna=False).to_dict(),
        "image_mode_counts": meta["image_mode"].value_counts(dropna=False).to_dict(),
        "alpha_by_label": alpha_by_label.to_dict(),
        "grayscale_by_label": gray_by_label.to_dict(),
        "cmyk_count": int((meta["image_mode"] == "CMYK").sum()),
    }
    with open(AUDIT_ROOT / "input_mode_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stringify_keys(summary), fh, indent=2, ensure_ascii=False)


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


def nearest_residue_start(lengths: np.ndarray, crop_size: int, residue: int) -> tuple[np.ndarray, np.ndarray]:
    max_start = lengths - crop_size
    starts = np.full_like(lengths, fill_value=-1, dtype=np.int32)
    ok = max_start >= residue
    if not np.any(ok):
        return starts, ok
    centers = max_start[ok] / 2.0
    k = np.round((centers - residue) / 8.0).astype(np.int32)
    cand = residue + 8 * k
    lower = residue + 8 * np.floor((max_start[ok] - residue) / 8.0).astype(np.int32)
    cand = np.clip(cand, residue, max_start[ok]).astype(np.int32)
    cand = np.where((cand - residue) % 8 == 0, cand, lower)
    starts[ok] = cand
    return starts, ok


def geometry_audit(meta: pd.DataFrame) -> dict[str, object]:
    p_y0 = float((meta["label"] == "nature").mean())
    p_y1 = 1.0 - p_y0
    widths = meta["width"].to_numpy(dtype=np.int32)
    heights = meta["height"].to_numpy(dtype=np.int32)
    short_side = meta["S"].to_numpy(dtype=np.int32)

    rows: list[dict[str, float | int]] = []
    for crop_size in range(216, 257):
        for residue in range(1, 8):
            threshold = crop_size + residue
            accepted = short_side >= threshold
            p_ai = float(np.mean(accepted[meta["label"] == "ai"]))
            p_real = float(np.mean(accepted[meta["label"] == "nature"]))
            accepted_df = meta[accepted]
            retained = (
                float(np.mean((crop_size * crop_size) / (accepted_df["S"].to_numpy(dtype=float) ** 2)))
                if len(accepted_df)
                else float("nan")
            )

            x0, ok_x = nearest_residue_start(widths, crop_size, residue)
            y0, ok_y = nearest_residue_start(heights, crop_size, residue)
            ok = ok_x & ok_y
            cx = (widths[ok] - crop_size) / 2.0
            cy = (heights[ok] - crop_size) / 2.0
            dx = np.abs(x0[ok] - cx)
            dy = np.abs(y0[ok] - cy)
            mean_center_linf = float(np.mean(np.maximum(dx, dy))) if len(dx) else float("nan")

            rows.append(
                {
                    "crop_size": crop_size,
                    "residue": residue,
                    "threshold": threshold,
                    "accepted_ai": p_ai,
                    "accepted_real": p_real,
                    "coverage_gap_real_minus_ai": p_real - p_ai,
                    "coverage_total": float(np.mean(accepted)),
                    "accepted_total": int(np.sum(accepted)),
                    "mi_bits": mutual_information_binary(p_real, p_ai, p_y0, p_y1),
                    "mean_retained_area_vs_inscribed_square": retained,
                    "mean_center_linf": mean_center_linf,
                    "phase_distance_to_nearest_8_boundary": min(residue, 8 - residue),
                }
            )

    sweep = pd.DataFrame(rows).sort_values(["crop_size", "residue"]).reset_index(drop=True)
    sweep.to_csv(AUDIT_ROOT / "geometry_frontier.csv", index=False, encoding="utf-8-sig")

    mult8 = sweep[sweep["crop_size"] % 8 == 0].copy()
    mult8.to_csv(AUDIT_ROOT / "geometry_frontier_mult8.csv", index=False, encoding="utf-8-sig")

    recommended_crop = int(
        mult8.loc[mult8["accepted_ai"] >= 0.999999, "crop_size"].max()
    )
    residue_candidates = mult8[mult8["crop_size"] == recommended_crop].copy()
    residue_candidates = residue_candidates.sort_values(
        [
            "phase_distance_to_nearest_8_boundary",
            "mean_center_linf",
            "coverage_total",
        ],
        ascending=[False, True, False],
    )
    recommended_residue = int(residue_candidates.iloc[0]["residue"])

    residue_df = sweep[sweep["crop_size"] == recommended_crop].copy()
    residue_df.to_csv(
        AUDIT_ROOT / f"geometry_residue_scan_crop_{recommended_crop}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rec_row = sweep[
        (sweep["crop_size"] == recommended_crop) & (sweep["residue"] == recommended_residue)
    ].iloc[0]
    by_generator = (
        meta.assign(accepted=meta["S"] >= int(rec_row["threshold"]))
        .groupby(["generator", "label"], as_index=False)["accepted"]
        .mean()
        .sort_values(["generator", "label"])
    )
    by_generator.to_csv(
        AUDIT_ROOT / "geometry_recommendation_by_generator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recommendation = {
        "crop_size": recommended_crop,
        "residue": recommended_residue,
        "threshold": int(rec_row["threshold"]),
        "accepted_ai": float(rec_row["accepted_ai"]),
        "accepted_real": float(rec_row["accepted_real"]),
        "coverage_total": float(rec_row["coverage_total"]),
        "mi_bits": float(rec_row["mi_bits"]),
        "mean_retained_area_vs_inscribed_square": float(rec_row["mean_retained_area_vs_inscribed_square"]),
        "mean_center_linf": float(rec_row["mean_center_linf"]),
        "selection_rule": {
            "crop_size": "max crop_size divisible by 8 with accepted_ai == 1 on current raw snapshot",
            "residue": "maximize distance from residue 0 mod 8, then minimize center drift, then maximize coverage",
        },
    }
    with open(AUDIT_ROOT / "geometry_recommendation.json", "w", encoding="utf-8") as fh:
        json.dump(stringify_keys(recommendation), fh, indent=2, ensure_ascii=False)

    return {
        "sweep": sweep,
        "recommended_crop": recommended_crop,
        "recommended_residue": recommended_residue,
        "recommendation": recommendation,
    }


def load_features_with_raw_paths() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV).copy()
    manifest = pd.read_csv(MANIFEST_CSV, usecols=["file_path", "output_path"]).copy()
    df["proc_rel"] = df["file_path"].map(norm_processed)
    manifest["proc_rel"] = manifest["output_path"].map(norm_processed)
    manifest["raw_rel"] = manifest["file_path"].map(norm_raw)
    df = df.merge(manifest[["proc_rel", "raw_rel"]], on="proc_rel", how="left", validate="one_to_one")
    df["raw_path"] = df["raw_rel"].map(lambda rel: str((RAW_ROOT / rel).resolve()))
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "file_path",
            "generator",
            "label",
            "split_role",
            "dataset_name",
            "preprocess_version",
            "feature_version",
            "status",
            "error",
            "proc_rel",
            "raw_rel",
            "raw_path",
        }
    ]
    return df, feature_cols


def get_sampling(path: str | Path) -> str:
    try:
        with Image.open(path) as im:
            return SUBSAMPLING_MAP.get(JpegImagePlugin.get_sampling(im), "other")
    except Exception:
        return "error"


def sampling_map(paths: pd.Series) -> dict[str, str]:
    unique_paths = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=SAMPLING_WORKERS) as ex:
        labels = list(ex.map(get_sampling, unique_paths))
    return dict(zip(unique_paths, labels))


def fit_predict_prob(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    model = logistic_pipe()
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy()
    X_test = test_df[feature_cols]
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def bootstrap_delta_ci(
    y_true: np.ndarray,
    pred_full: np.ndarray,
    pred_drop: np.ndarray,
    seed: int,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        values.append(float(roc_auc_score(y_b, pred_full[idx]) - roc_auc_score(y_b, pred_drop[idx])))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def feature_governance_audit() -> dict[str, object]:
    df, feature_cols = load_features_with_raw_paths()
    feature_to_family = {
        feature: family for family, features in FEATURE_FAMILIES.items() for feature in features
    }

    train_label = df[df["split_role"] == "train_core"].copy()
    test_label = df[df["split_role"] == "id_test"].copy()
    train_label["y_lab"] = (train_label["label"] == "ai").astype(int)
    test_label["y_lab"] = (test_label["label"] == "ai").astype(int)

    real_df = df[df["label"] == "nature"].copy()
    real_sampling = sampling_map(real_df["raw_path"])
    real_df["jpeg_sampling"] = real_df["raw_path"].map(real_sampling)
    real_df = real_df[real_df["jpeg_sampling"].isin(["4:4:4", "4:2:0"])].copy()
    real_df["y_nuis"] = (real_df["jpeg_sampling"] == "4:4:4").astype(int)
    train_nuis = real_df[real_df["split_role"] == "train_core"].copy()
    test_nuis = real_df[real_df["split_role"] == "id_test"].copy()

    full_pred_lab = fit_predict_prob(train_label, test_label, feature_cols, "y_lab")
    full_pred_nuis = fit_predict_prob(train_nuis, test_nuis, feature_cols, "y_nuis")
    full_lab_auc = float(roc_auc_score(test_label["y_lab"], full_pred_lab))
    full_nuis_auc = float(roc_auc_score(test_nuis["y_nuis"], full_pred_nuis))

    family_rows: list[dict[str, object]] = []
    for family, cols in FEATURE_FAMILIES.items():
        solo_pred_lab = fit_predict_prob(train_label, test_label, cols, "y_lab")
        solo_pred_nuis = fit_predict_prob(train_nuis, test_nuis, cols, "y_nuis")
        keep = [col for col in feature_cols if col not in cols]
        drop_pred_lab = fit_predict_prob(train_label, test_label, keep, "y_lab")
        drop_pred_nuis = fit_predict_prob(train_nuis, test_nuis, keep, "y_nuis")

        delta_lab = full_lab_auc - float(roc_auc_score(test_label["y_lab"], drop_pred_lab))
        delta_nuis = full_nuis_auc - float(roc_auc_score(test_nuis["y_nuis"], drop_pred_nuis))
        lab_ci = bootstrap_delta_ci(
            test_label["y_lab"].to_numpy(),
            full_pred_lab,
            drop_pred_lab,
            seed=SEED + len(family_rows) * 13 + 1,
        )
        nuis_ci = bootstrap_delta_ci(
            test_nuis["y_nuis"].to_numpy(),
            full_pred_nuis,
            drop_pred_nuis,
            seed=SEED + len(family_rows) * 13 + 2,
        )

        family_rows.append(
            {
                "family": family,
                "n_features": len(cols),
                "solo_lab_auc": float(roc_auc_score(test_label["y_lab"], solo_pred_lab)),
                "solo_nuis_auc": float(roc_auc_score(test_nuis["y_nuis"], solo_pred_nuis)),
                "drop_lab_auc": float(roc_auc_score(test_label["y_lab"], drop_pred_lab)),
                "drop_nuis_auc": float(roc_auc_score(test_nuis["y_nuis"], drop_pred_nuis)),
                "delta_lab": float(delta_lab),
                "delta_nuis": float(delta_nuis),
                "delta_lab_ci_low": lab_ci[0],
                "delta_lab_ci_high": lab_ci[1],
                "delta_nuis_ci_low": nuis_ci[0],
                "delta_nuis_ci_high": nuis_ci[1],
                "harm_ratio": float(delta_nuis / max(abs(delta_lab), 1e-6)),
            }
        )

    family_df = pd.DataFrame(family_rows).sort_values("delta_nuis", ascending=False).reset_index(drop=True)
    family_df.to_csv(AUDIT_ROOT / "feature_governance_family.csv", index=False, encoding="utf-8-sig")

    feature_rows: list[dict[str, object]] = []
    feature_pred_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature in feature_cols:
        keep = [col for col in feature_cols if col != feature]
        drop_pred_lab = fit_predict_prob(train_label, test_label, keep, "y_lab")
        drop_pred_nuis = fit_predict_prob(train_nuis, test_nuis, keep, "y_nuis")
        feature_pred_cache[feature] = (drop_pred_lab, drop_pred_nuis)
        feature_rows.append(
            {
                "feature": feature,
                "family": feature_to_family.get(feature, "unknown"),
                "delta_lab": full_lab_auc - float(roc_auc_score(test_label["y_lab"], drop_pred_lab)),
                "delta_nuis": full_nuis_auc - float(roc_auc_score(test_nuis["y_nuis"], drop_pred_nuis)),
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values("delta_nuis", ascending=False).reset_index(drop=True)
    feature_df.to_csv(AUDIT_ROOT / "feature_governance_feature.csv", index=False, encoding="utf-8-sig")

    ci_features = set(feature_df.head(8)["feature"].tolist())
    ci_features.update({"ps_alpha", "ps_deviation_variance", "cross_noise_ratio"})
    ci_rows: list[dict[str, object]] = []
    for idx, feature in enumerate(sorted(ci_features)):
        drop_pred_lab, drop_pred_nuis = feature_pred_cache[feature]
        lab_ci = bootstrap_delta_ci(
            test_label["y_lab"].to_numpy(),
            full_pred_lab,
            drop_pred_lab,
            seed=SEED + 101 + idx * 7,
        )
        nuis_ci = bootstrap_delta_ci(
            test_nuis["y_nuis"].to_numpy(),
            full_pred_nuis,
            drop_pred_nuis,
            seed=SEED + 102 + idx * 7,
        )
        point = feature_df.loc[feature_df["feature"] == feature].iloc[0]
        ci_rows.append(
            {
                "feature": feature,
                "family": point["family"],
                "delta_lab": float(point["delta_lab"]),
                "delta_nuis": float(point["delta_nuis"]),
                "delta_lab_ci_low": lab_ci[0],
                "delta_lab_ci_high": lab_ci[1],
                "delta_nuis_ci_low": nuis_ci[0],
                "delta_nuis_ci_high": nuis_ci[1],
            }
        )

    feature_ci_df = pd.DataFrame(ci_rows).sort_values("delta_nuis", ascending=False).reset_index(drop=True)
    feature_ci_df.to_csv(AUDIT_ROOT / "feature_governance_feature_ci.csv", index=False, encoding="utf-8-sig")

    summary = {
        "label_task": {
            "train_split": "train_core",
            "test_split": "id_test",
            "full_auc": full_lab_auc,
            "n_test": int(len(test_label)),
        },
        "nuisance_task": {
            "task": "real_only_jpeg_subsampling_444_vs_420",
            "train_split": "train_core",
            "test_split": "id_test",
            "full_auc": full_nuis_auc,
            "n_test": int(len(test_nuis)),
            "counts_test": test_nuis["jpeg_sampling"].value_counts().to_dict(),
        },
    }
    with open(AUDIT_ROOT / "feature_governance_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stringify_keys(summary), fh, indent=2, ensure_ascii=False)

    return {
        "family_df": family_df,
        "feature_df": feature_df,
        "feature_ci_df": feature_ci_df,
        "summary": summary,
    }


def keep_only_ablation() -> pd.DataFrame:
    df, feature_cols = load_features_with_raw_paths()
    sets = {
        "full33": feature_cols,
        "keep_only": CURRENT_KEEP_CANDIDATE_FEATURES,
        "color_only": FEATURE_FAMILIES["color"],
        "keep_no_frs": [name for name in CURRENT_KEEP_CANDIDATE_FEATURES if name != "frs_mid_variance"],
    }

    rows: list[dict[str, object]] = []
    train_label = df[df["split_role"] == "train_core"].copy()
    test_label = df[df["split_role"] == "id_test"].copy()
    ytr_lab = (train_label["label"] == "ai").astype(int)
    yte_lab = (test_label["label"] == "ai").astype(int)

    for set_name, cols in sets.items():
        model = logistic_pipe()
        model.fit(train_label[cols], ytr_lab)
        pred = model.predict_proba(test_label[cols])[:, 1]
        rows.append(
            {
                "set_name": set_name,
                "task": "label_train_core_to_id_test",
                "auc": float(roc_auc_score(yte_lab, pred)),
                "n_features": len(cols),
            }
        )

    real_df = df[df["label"] == "nature"].copy()
    real_sampling = sampling_map(real_df["raw_path"])
    real_df["jpeg_sampling"] = real_df["raw_path"].map(real_sampling)
    real_df = real_df[real_df["jpeg_sampling"].isin(["4:4:4", "4:2:0"])].copy()
    train_nuis = real_df[real_df["split_role"] == "train_core"].copy()
    test_nuis = real_df[real_df["split_role"] == "id_test"].copy()
    ytr_nuis = (train_nuis["jpeg_sampling"] == "4:4:4").astype(int)
    yte_nuis = (test_nuis["jpeg_sampling"] == "4:4:4").astype(int)

    for set_name, cols in sets.items():
        model = logistic_pipe()
        model.fit(train_nuis[cols], ytr_nuis)
        pred = model.predict_proba(test_nuis[cols])[:, 1]
        rows.append(
            {
                "set_name": set_name,
                "task": "nuisance_444_vs_420_train_core_to_id_test",
                "auc": float(roc_auc_score(yte_nuis, pred)),
                "n_features": len(cols),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(AUDIT_ROOT / "keep_only_ablation.csv", index=False, encoding="utf-8-sig")
    return out


def read_bgr(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def exact_residue_center_crop(img: np.ndarray, crop_size: int, residue: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < crop_size + residue or w < crop_size + residue:
        raise ValueError(f"LOW_SUPPORT: {img.shape[:2]}")

    def pick(length: int) -> int:
        center = length // 2 - crop_size // 2
        candidates = np.arange(residue, length - crop_size + 1, 8)
        return int(candidates[np.argmin(np.abs(candidates - center))])

    x0 = pick(w)
    y0 = pick(h)
    return img[y0 : y0 + crop_size, x0 : x0 + crop_size]


def universal_chroma420(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    for idx in (1, 2):
        channel = out[:, :, idx]
        blurred = cv2.sepFilter2D(
            channel,
            -1,
            kernel,
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        down = cv2.resize(
            blurred,
            (channel.shape[1] // 2, channel.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        up = cv2.resize(
            down,
            (channel.shape[1], channel.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        out[:, :, idx] = up
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def proxy_ps_alpha(arr: np.ndarray) -> float:
    y = arr[:, :, 0].astype(np.float64)
    n = y.shape[0]
    f = np.fft.fft2(y) / (n * n)
    p = np.abs(np.fft.fftshift(f)) ** 2
    coords = np.arange(n) - (n // 2)
    radius = np.floor(
        np.sqrt((coords[:, None] ** 2 + coords[None, :] ** 2).astype(np.float64)) + 0.5
    ).astype(np.int32)
    ring = np.bincount(radius.ravel(), weights=p.ravel()) / np.maximum(
        np.bincount(radius.ravel()),
        1,
    )
    lo = 20
    hi = min(64, len(ring) - 1)
    r = np.arange(lo, hi + 1, dtype=np.float64)
    c = ring[lo : hi + 1].astype(np.float64)
    noise_floor = 1.0 / (n * n)
    if np.all(c < noise_floor):
        return 0.0
    slope, _ = np.polyfit(np.log(r), np.log(c + noise_floor), 1)
    return float(-slope)


def proxy_cross_noise_ratio(arr: np.ndarray) -> float:
    r_y = convolve2d(arr[:, :, 0].astype(np.float64), PROXY_KERNEL, mode="valid")
    r_cb = convolve2d(arr[:, :, 2].astype(np.float64), PROXY_KERNEL, mode="valid")
    return (float(np.mean(np.abs(r_y))) + 1e-3) / (float(np.mean(np.abs(r_cb))) + 1e-3)


def proxy_srm_square3_energy_cr(arr: np.ndarray) -> float:
    r = convolve2d(arr[:, :, 1].astype(np.float64) - 128.0, PROXY_KERNEL, mode="valid")
    return float(np.mean(r**2))


def proxy_lbp_nonuniform_ratio(channel: np.ndarray) -> float:
    c_int = channel.astype(np.int16)
    centre = c_int[1:-1, 1:-1]
    code = np.zeros_like(centre, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(LBP_OFFSETS):
        neighbour = c_int[
            1 + dy : 1 + dy + centre.shape[0],
            1 + dx : 1 + dx + centre.shape[1],
        ]
        code |= ((neighbour >= centre).astype(np.uint8) << bit)
    return float(LBP_NONUNIFORM_LUT[code.ravel()].mean())


def extract_proxy_features(arr: np.ndarray) -> dict[str, float]:
    return {
        "ps_alpha": proxy_ps_alpha(arr),
        "cross_noise_ratio": proxy_cross_noise_ratio(arr),
        "srm_square3_energy_cr": proxy_srm_square3_energy_cr(arr),
        "lbp_nonuniform_ratio_cr": proxy_lbp_nonuniform_ratio(arr[:, :, 1]),
        "lbp_nonuniform_ratio_cb": proxy_lbp_nonuniform_ratio(arr[:, :, 2]),
    }


def proxy_variant_record(
    row_dict: dict[str, object],
    crop_size: int,
    residue: int,
    variant: ProxyVariant,
) -> dict[str, object]:
    path = Path(str(row_dict["raw_path"]))
    bgr = read_bgr(path)
    orientation = read_orientation(path)
    bgr = apply_orientation(bgr, orientation)
    bgr = exact_residue_center_crop(bgr, crop_size=crop_size, residue=residue)
    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    if variant.apply_chroma420:
        arr = universal_chroma420(arr)
    features = extract_proxy_features(arr)
    return {
        "variant": variant.name,
        "raw_path": str(row_dict["raw_path"]),
        "relative_path": str(row_dict["relative_path"]).replace("\\", "/"),
        "label": str(row_dict["label"]),
        "generator": str(row_dict["generator"]),
        "jpeg_sampling": row_dict.get("jpeg_sampling"),
        **features,
    }


def proxy_risk_audit(meta: pd.DataFrame, crop_size: int, residue: int) -> dict[str, object]:
    accepted = meta[meta["S"] >= (crop_size + residue)].copy()
    real_only = accepted[accepted["label"] == "nature"].copy()
    real_sampling = sampling_map(real_only["raw_path"])
    real_only["jpeg_sampling"] = real_only["raw_path"].map(real_sampling)
    accepted = accepted.merge(
        real_only[["raw_path", "jpeg_sampling"]],
        on="raw_path",
        how="left",
    )

    rng = np.random.default_rng(SEED)
    label_parts: list[pd.DataFrame] = []
    for _, group in accepted.groupby(["generator", "label"]):
        take = min(PROXY_LABEL_PER_GROUP, len(group))
        draw = group.iloc[rng.choice(len(group), size=take, replace=False)]
        label_parts.append(draw)
    label_sample = pd.concat(label_parts, ignore_index=True)
    label_sample.to_csv(AUDIT_ROOT / "proxy_label_sample.csv", index=False, encoding="utf-8-sig")

    nuisance_pool = accepted[
        (accepted["label"] == "nature") & (accepted["jpeg_sampling"].isin(["4:4:4", "4:2:0"]))
    ].copy()
    nuisance_parts: list[pd.DataFrame] = []
    for sampling in ("4:4:4", "4:2:0"):
        group = nuisance_pool[nuisance_pool["jpeg_sampling"] == sampling]
        take = min(PROXY_NUISANCE_PER_CLASS, len(group))
        draw = group.iloc[rng.choice(len(group), size=take, replace=False)]
        nuisance_parts.append(draw)
    nuisance_sample = pd.concat(nuisance_parts, ignore_index=True)
    nuisance_sample.to_csv(
        AUDIT_ROOT / "proxy_nuisance_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_cols = [
        "ps_alpha",
        "cross_noise_ratio",
        "srm_square3_energy_cr",
        "lbp_nonuniform_ratio_cr",
        "lbp_nonuniform_ratio_cb",
    ]

    all_records: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    nuisance_auc_rows: list[dict[str, object]] = []

    for variant in PROXY_VARIANTS:
        with ThreadPoolExecutor(max_workers=RAW_WORKERS) as ex:
            label_records = list(
                ex.map(
                    lambda row: proxy_variant_record(row._asdict(), crop_size, residue, variant),
                    label_sample.itertuples(index=False),
                )
            )
        label_df = pd.DataFrame(label_records)
        all_records.extend(label_records)

        X_label = label_df[feature_cols]
        y_label = (label_df["label"] == "ai").astype(int).to_numpy()
        groups = label_df["generator"].to_numpy()
        pred_logo = np.full(len(label_df), np.nan, dtype=float)
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        for train_idx, test_idx in splitter.split(X_label, y_label, groups):
            model = logistic_pipe()
            model.fit(X_label.iloc[train_idx], y_label[train_idx])
            pred_logo[test_idx] = model.predict_proba(X_label.iloc[test_idx])[:, 1]
        label_logo_auc = float(roc_auc_score(y_label, pred_logo))

        with ThreadPoolExecutor(max_workers=RAW_WORKERS) as ex:
            nuisance_records = list(
                ex.map(
                    lambda row: proxy_variant_record(row._asdict(), crop_size, residue, variant),
                    nuisance_sample.itertuples(index=False),
                )
            )
        nuisance_df = pd.DataFrame(nuisance_records)
        all_records.extend(nuisance_records)

        X_nuis = nuisance_df[feature_cols]
        y_nuis = (nuisance_df["jpeg_sampling"] == "4:4:4").astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        model = logistic_pipe()
        nuisance_prob = cross_val_predict(
            model,
            X_nuis,
            y_nuis,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        nuisance_auc = float(roc_auc_score(y_nuis, nuisance_prob))

        variant_rows.append(
            {
                "variant": variant.name,
                "crop_size": crop_size,
                "residue": residue,
                "label_logo_auc": label_logo_auc,
                "nuisance_auc": nuisance_auc,
                "n_label_sample": int(len(label_df)),
                "n_nuisance_sample": int(len(nuisance_df)),
            }
        )

        for feature in feature_cols:
            auc = roc_auc_score(y_nuis, nuisance_df[feature].to_numpy(dtype=float))
            nuisance_auc_rows.append(
                {
                    "variant": variant.name,
                    "feature": feature,
                    "nuisance_auc_abs": float(max(auc, 1.0 - auc)),
                }
            )

    pd.DataFrame(variant_rows).to_csv(
        AUDIT_ROOT / "proxy_variant_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(nuisance_auc_rows).sort_values(
        ["variant", "nuisance_auc_abs"],
        ascending=[True, False],
    ).to_csv(
        AUDIT_ROOT / "proxy_feature_nuisance_auc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_records).to_csv(
        AUDIT_ROOT / "proxy_variant_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "variant_df": pd.DataFrame(variant_rows),
        "nuisance_feature_df": pd.DataFrame(nuisance_auc_rows),
    }


def proxy_crop_compare(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for crop_size, residue in ((248, 4), (252, 4), (252, 1)):
        result = proxy_risk_audit(meta, crop_size=crop_size, residue=residue)
        for row in result["variant_df"].to_dict(orient="records"):
            rows.append(
                {
                    "variant": row["variant"],
                    "crop_size": crop_size,
                    "residue": residue,
                    "label_logo_auc": row["label_logo_auc"],
                    "nuisance_auc": row["nuisance_auc"],
                    "n_label_sample": row["n_label_sample"],
                    "n_nuisance_sample": row["n_nuisance_sample"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT_ROOT / "proxy_crop_compare.csv", index=False, encoding="utf-8-sig")
    return out


def build_report(
    snapshot: dict[str, object],
    geometry: dict[str, object],
    governance: dict[str, object],
    proxy: dict[str, object],
    keep_only: pd.DataFrame,
    crop_compare: pd.DataFrame,
) -> None:
    rec = geometry["recommendation"]
    family_df = governance["family_df"]
    feature_ci_df = governance["feature_ci_df"]
    proxy_df = proxy["variant_df"]
    proxy_nuis = proxy["nuisance_feature_df"]

    lines = [
        "# Spec v4 Audit Bundle",
        "",
        "## Snapshot",
        "",
        f"- Raw files on disk: `{snapshot['raw_file_count']}`",
        f"- Metadata rows in parquet: `{snapshot['metadata_rows']}`",
        f"- Metadata rows matched to current raw snapshot: `{snapshot['matched_metadata_rows']}`",
        f"- Metadata-only stale rows: `{snapshot['stale_metadata_rows']}`",
        f"- Raw snapshot hash: `{snapshot['raw_snapshot_sha256']}`",
        "",
        "## Geometry Recommendation",
        "",
        f"- Recommended crop: `{rec['crop_size']}x{rec['crop_size']}`",
        f"- Recommended residue: `({rec['residue']},{rec['residue']})`",
        f"- Support threshold: `S >= {rec['threshold']}`",
        f"- Coverage total: `{rec['coverage_total']:.6f}`",
        f"- Accepted AI rate: `{rec['accepted_ai']:.6f}`",
        f"- Accepted real rate: `{rec['accepted_real']:.6f}`",
        f"- Support-gate MI: `{rec['mi_bits']:.6f} bit`",
        f"- Mean retained area / inscribed-square area: `{rec['mean_retained_area_vs_inscribed_square']:.6f}`",
        f"- Mean center Linf drift: `{rec['mean_center_linf']:.6f}` px",
        "",
        "## Governance Findings",
        "",
        f"- Current 33-feature label AUC (`train_core -> id_test`): `{governance['summary']['label_task']['full_auc']:.6f}`",
        f"- Current 33-feature real-only subsampling nuisance AUC (`train_core -> id_test`): `{governance['summary']['nuisance_task']['full_auc']:.6f}`",
        "",
        "Keep-only ablation:",
        "",
    ]

    for row in keep_only.itertuples(index=False):
        lines.append(
            "- "
            f"`{row.set_name}` / `{row.task}`: auc=`{row.auc:.6f}`, n_features=`{row.n_features}`"
        )

    lines.extend(
        [
            "",
        "Top family-level nuisance contributors:",
        "",
        ]
    )

    for row in family_df.itertuples(index=False):
        lines.append(
            "- "
            f"`{row.family}`: delta_lab=`{row.delta_lab:.6f}`, "
            f"delta_nuis=`{row.delta_nuis:.6f}`, "
            f"solo_nuis_auc=`{row.solo_nuis_auc:.6f}`, "
            f"harm_ratio=`{row.harm_ratio:.3f}`"
        )

    lines.extend(
        [
            "",
            "Top feature-level CI bundle:",
            "",
        ]
    )
    for row in feature_ci_df.itertuples(index=False):
        lines.append(
            "- "
            f"`{row.feature}` ({row.family}): "
            f"delta_lab=`{row.delta_lab:.6f}` "
            f"[{row.delta_lab_ci_low:.6f}, {row.delta_lab_ci_high:.6f}], "
            f"delta_nuis=`{row.delta_nuis:.6f}` "
            f"[{row.delta_nuis_ci_low:.6f}, {row.delta_nuis_ci_high:.6f}]"
        )

    lines.extend(
        [
            "",
            "Proxy crop compare:",
            "",
        ]
    )
    for row in crop_compare.itertuples(index=False):
        lines.append(
            "- "
            f"`C={row.crop_size}, r={row.residue}, {row.variant}`: "
            f"label_logo_auc=`{row.label_logo_auc:.6f}`, nuisance_auc=`{row.nuisance_auc:.6f}`"
        )

    lines.extend(
        [
            "",
            "## Proxy Raw Audit",
            "",
        ]
    )
    for row in proxy_df.itertuples(index=False):
        lines.append(
            "- "
            f"`{row.variant}`: label_logo_auc=`{row.label_logo_auc:.6f}`, "
            f"nuisance_auc=`{row.nuisance_auc:.6f}`, "
            f"n_label_sample=`{row.n_label_sample}`, "
            f"n_nuisance_sample=`{row.n_nuisance_sample}`"
        )

    lines.extend(
        [
            "",
            "Univariate nuisance AUCs on proxy real-only task:",
            "",
        ]
    )
    for row in proxy_nuis.sort_values(["variant", "nuisance_auc_abs"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"- `{row.variant}` / `{row.feature}`: nuisance_auc_abs=`{row.nuisance_auc_abs:.6f}`"
        )

    (AUDIT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_audit_root()

    raw_rels, raw_rel_set = current_raw_snapshot()
    metadata_rows = int(pd.read_parquet(METADATA_PARQUET, columns=["relative_path"]).shape[0])
    meta = load_current_metadata(raw_rel_set)
    snapshot = {
        "raw_file_count": len(raw_rels),
        "metadata_rows": metadata_rows,
        "matched_metadata_rows": int(len(meta)),
        "stale_metadata_rows": int(metadata_rows - len(meta)),
        "raw_snapshot_sha256": snapshot_hash(raw_rels),
    }
    with open(AUDIT_ROOT / "snapshot_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stringify_keys(snapshot), fh, indent=2, ensure_ascii=False)

    input_mode_audit(meta)
    geometry = geometry_audit(meta)
    governance = feature_governance_audit()
    keep_only = keep_only_ablation()
    proxy = proxy_risk_audit(
        meta,
        crop_size=int(geometry["recommended_crop"]),
        residue=int(geometry["recommended_residue"]),
    )
    crop_compare = proxy_crop_compare(meta)
    build_report(snapshot, geometry, governance, proxy, keep_only, crop_compare)


if __name__ == "__main__":
    main()
