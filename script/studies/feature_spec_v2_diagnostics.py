from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
STUDY_SCRIPT = ROOT / "script" / "studies" / "feature_spec_v2_validation.py"
STUDY_DIR = ROOT / "audit_output" / "studies" / "feature_spec_v2_validation_20260325"
DIAG_DIR = STUDY_DIR / "diagnostics"


def load_study_module():
    spec = importlib.util.spec_from_file_location("feature_spec_v2_validation", STUDY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def auc_to_dprime(auc: float) -> float:
    return math.sqrt(2.0) * NormalDist().inv_cdf(float(auc))


def auc_to_tpr_at_fpr5(auc: float) -> float:
    dprime = auc_to_dprime(float(auc))
    return NormalDist().cdf(dprime - NormalDist().inv_cdf(0.95))


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_auc_mapping() -> pd.DataFrame:
    set_metrics = pd.read_csv(STUDY_DIR / "feature_set_metrics.csv")
    targets = [
        ("gate_clean_075", 0.75),
        ("v4_control_minimal", float(set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "label_logo_clean") & (set_metrics["feature_set"] == "control_minimal")]["auc"].iloc[0])),
        ("v4_cfa_xy_only", float(set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "label_logo_clean") & (set_metrics["feature_set"] == "cfa_xy_only")]["auc"].iloc[0])),
        ("v4_control_plus_cfa_xy", float(set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "label_logo_clean") & (set_metrics["feature_set"] == "control_plus_cfa_xy")]["auc"].iloc[0])),
        ("v4_crsrm_only", float(set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "label_logo_clean") & (set_metrics["feature_set"] == "crsrm_only")]["auc"].iloc[0])),
    ]
    rows = []
    for name, auc in targets:
        rows.append(
            {
                "name": name,
                "auc": auc,
                "dprime": auc_to_dprime(auc),
                "tpr_at_fpr5_gaussian": auc_to_tpr_at_fpr5(auc),
            }
        )
    return pd.DataFrame(rows)


def build_shift_redundancy() -> tuple[pd.DataFrame, dict[str, float]]:
    set_metrics = pd.read_csv(STUDY_DIR / "feature_set_metrics.csv")
    set_gate = pd.read_csv(STUDY_DIR / "feature_set_gate_summary.csv", decimal=",")
    set_clean = set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "label_logo_clean")][["feature_set", "auc"]].rename(columns={"auc": "clean_auc"})
    set_resize = set_metrics[(set_metrics["preprocess_version"] == "v4_exact") & (set_metrics["task"] == "xdeg_resize50_bilinear")][["feature_set", "auc"]].rename(columns={"auc": "resize50_auc"})
    merged = set_clean.merge(set_resize, on="feature_set").merge(
        set_gate[set_gate["preprocess_version"] == "v4_exact"][["feature_set", "resize50_mean_feature_shift", "resize50_max_feature_shift"]],
        on="feature_set",
    )
    merged["xdeg_gap"] = merged["clean_auc"] - merged["resize50_auc"]
    stats = {
        "corr_mean_shift_vs_xdeg_gap": float(merged["resize50_mean_feature_shift"].corr(merged["xdeg_gap"])),
        "corr_max_shift_vs_xdeg_gap": float(merged["resize50_max_feature_shift"].corr(merged["xdeg_gap"])),
        "corr_mean_shift_vs_resize50_auc": float(merged["resize50_mean_feature_shift"].corr(merged["resize50_auc"])),
    }
    return merged.sort_values("resize50_mean_feature_shift"), stats


def logo_predictions(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    preds = np.zeros(len(df), dtype=float)
    for group in sorted(df["generator_norm"].unique()):
        train = df[df["generator_norm"] != group]
        test_mask = (df["generator_norm"] == group).to_numpy()
        pipe = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000, solver="liblinear"))
        pipe.fit(train[feature_cols], train["target"])
        preds[test_mask] = pipe.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
    return preds


def build_control_generalization(module) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = module.load_common_tables()
    label_df = module.build_label_sample(common, smoke=False)
    clean = module.build_feature_table(label_df, preprocess_version="v4_exact", degradation="clean", workers=8)
    features = list(module.CONTROL_MINIMAL_KEYS)
    clean["target"] = (clean["label_norm"] == "ai").astype(int)

    pipe = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000, solver="liblinear"))
    y = clean["target"].to_numpy()
    pred = cross_val_predict(
        pipe,
        clean[features],
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]

    rows = [
        {
            "scope": "pooled_random_cv",
            "generator_norm": "ALL",
            "n_rows": int(len(clean)),
            "auc": float(roc_auc_score(y, pred)),
        }
    ]

    for group in sorted(clean["generator_norm"].unique()):
        sub = clean[clean["generator_norm"] == group].copy()
        y_sub = sub["target"].to_numpy()
        pred_sub = cross_val_predict(
            pipe,
            sub[features],
            y_sub,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        rows.append(
            {
                "scope": "within_generator_cv",
                "generator_norm": group,
                "n_rows": int(len(sub)),
                "auc": float(roc_auc_score(y_sub, pred_sub)),
            }
        )

    rows.append(
        {
            "scope": "logo",
            "generator_norm": "ALL",
            "n_rows": int(len(clean)),
            "auc": float(module.evaluate_logo_auc(clean[features + ["target", "generator_norm"]], features, "target")),
        }
    )

    corr = clean[features].corr().round(6).reset_index().rename(columns={"index": "feature"})
    return pd.DataFrame(rows), corr


def build_cross_noise_diagnostics(module) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = module.load_common_tables()
    nuisance_df = module.build_real_nuisance_sample(common, smoke=False)
    feat = module.build_feature_table(nuisance_df, preprocess_version="v4_exact", degradation="clean", workers=8)
    feat["target_420"] = (feat["jpeg_subsampling"] == "4:2:0").astype(int)

    pooled = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "generator_norm": "ALL",
                "auc_predict_420": float(roc_auc_score(feat["target_420"], feat["cross_noise_ratio"])),
                "mean_420": float(feat.loc[feat["target_420"] == 1, "cross_noise_ratio"].mean()),
                "mean_444": float(feat.loc[feat["target_420"] == 0, "cross_noise_ratio"].mean()),
                "median_420": float(feat.loc[feat["target_420"] == 1, "cross_noise_ratio"].median()),
                "median_444": float(feat.loc[feat["target_420"] == 0, "cross_noise_ratio"].median()),
            }
        ]
    )

    by_gen = []
    for group, sub in feat.groupby("generator_norm"):
        if sub["target_420"].nunique() < 2:
            continue
        by_gen.append(
            {
                "scope": "per_generator",
                "generator_norm": group,
                "auc_predict_420": float(roc_auc_score(sub["target_420"], sub["cross_noise_ratio"])),
                "mean_420": float(sub.loc[sub["target_420"] == 1, "cross_noise_ratio"].mean()),
                "mean_444": float(sub.loc[sub["target_420"] == 0, "cross_noise_ratio"].mean()),
                "median_420": float(sub.loc[sub["target_420"] == 1, "cross_noise_ratio"].median()),
                "median_444": float(sub.loc[sub["target_420"] == 0, "cross_noise_ratio"].median()),
            }
        )

    return pooled, pd.DataFrame(by_gen)


def build_logo_bootstrap(module) -> pd.DataFrame:
    common = module.load_common_tables()
    label_df = module.build_label_sample(common, smoke=False)
    clean = module.build_feature_table(label_df, preprocess_version="v4_exact", degradation="clean", workers=8)
    clean["target"] = (clean["label_norm"] == "ai").astype(int)

    set_defs = {
        "control_minimal": list(module.CONTROL_MINIMAL_KEYS),
        "control_with_wavelet": list(module.CONTROL_WITH_WAVELET_KEYS),
        "cfa_xy_only": list(module.CFA_XY_KEYS),
        "control_plus_cfa_xy": list(module.CONTROL_WITH_WAVELET_KEYS + module.CFA_XY_KEYS),
    }
    y = clean["target"].to_numpy()
    preds = {name: logo_predictions(clean, cols) for name, cols in set_defs.items()}
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(42)

    comparisons = [
        ("control_plus_cfa_xy", "control_minimal"),
        ("cfa_xy_only", "control_minimal"),
        ("control_plus_cfa_xy", "cfa_xy_only"),
        ("control_with_wavelet", "control_minimal"),
    ]
    rows = []
    for a, b in comparisons:
        diffs = []
        for _ in range(1000):
            boot = np.concatenate(
                [
                    rng.choice(idx0, size=len(idx0), replace=True),
                    rng.choice(idx1, size=len(idx1), replace=True),
                ]
            )
            diffs.append(roc_auc_score(y[boot], preds[a][boot]) - roc_auc_score(y[boot], preds[b][boot]))
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        rows.append(
            {
                "set_a": a,
                "set_b": b,
                "delta_auc": float(roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b])),
                "ci95_low": float(np.quantile(diffs_arr, 0.025)),
                "ci95_high": float(np.quantile(diffs_arr, 0.975)),
                "approx_two_sided_p": float(2.0 * min((diffs_arr <= 0).mean(), (diffs_arr >= 0).mean())),
            }
        )
    return pd.DataFrame(rows)


def build_chance_corrected_ratio() -> pd.DataFrame:
    set_metrics = pd.read_csv(STUDY_DIR / "feature_set_metrics.csv")
    rows = []
    for preprocess_version, feature_set in [
        ("v4_exact", "crsrm_only"),
        ("v4_exact", "cfa_xy_only"),
        ("v4_exact", "control_minimal"),
        ("v4_exact", "control_plus_cfa_xy"),
        ("v4_exact", "wavelet_parent_only"),
        ("old_v1", "crsrm_only"),
        ("old_v1", "cfa_xy_only"),
    ]:
        sub = set_metrics[(set_metrics["preprocess_version"] == preprocess_version) & (set_metrics["feature_set"] == feature_set)]
        clean_auc = float(sub[sub["task"] == "label_logo_clean"]["auc"].iloc[0])
        nat_auc = float(sub[sub["task"] == "real_jpeg_444_vs_420"]["auc"].iloc[0])
        rows.append(
            {
                "preprocess_version": preprocess_version,
                "feature_set": feature_set,
                "clean_auc": clean_auc,
                "natural_nuisance_auc": nat_auc,
                "chance_corrected_ratio": float(abs(clean_auc - 0.5) / (abs(nat_auc - 0.5) + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    module = load_study_module()

    auc_map = build_auc_mapping()
    save_frame(auc_map, DIAG_DIR / "auc_sla_mapping.csv")

    shift_df, shift_stats = build_shift_redundancy()
    save_frame(shift_df, DIAG_DIR / "resize_shift_redundancy.csv")

    control_df, control_corr = build_control_generalization(module)
    save_frame(control_df, DIAG_DIR / "control_minimal_generalization.csv")
    save_frame(control_corr, DIAG_DIR / "control_minimal_correlation_matrix.csv")

    cross_pooled, cross_by_gen = build_cross_noise_diagnostics(module)
    save_frame(cross_pooled, DIAG_DIR / "cross_noise_ratio_pooled.csv")
    save_frame(cross_by_gen, DIAG_DIR / "cross_noise_ratio_by_generator.csv")

    bootstrap = build_logo_bootstrap(module)
    save_frame(bootstrap, DIAG_DIR / "logo_bootstrap_comparisons.csv")

    ratio_df = build_chance_corrected_ratio()
    save_frame(ratio_df, DIAG_DIR / "signal_nuisance_ratio.csv")

    summary = {
        "auc_sla_mapping_file": str((DIAG_DIR / "auc_sla_mapping.csv").resolve()),
        "shift_redundancy_file": str((DIAG_DIR / "resize_shift_redundancy.csv").resolve()),
        "control_generalization_file": str((DIAG_DIR / "control_minimal_generalization.csv").resolve()),
        "cross_noise_pooled_file": str((DIAG_DIR / "cross_noise_ratio_pooled.csv").resolve()),
        "cross_noise_by_generator_file": str((DIAG_DIR / "cross_noise_ratio_by_generator.csv").resolve()),
        "logo_bootstrap_file": str((DIAG_DIR / "logo_bootstrap_comparisons.csv").resolve()),
        "signal_nuisance_ratio_file": str((DIAG_DIR / "signal_nuisance_ratio.csv").resolve()),
        "shift_redundancy": shift_stats,
    }
    (DIAG_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
