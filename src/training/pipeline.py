"""Training and evaluation pipeline for feature extraction v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import (
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    CFA_GATED_KEYS,
    CFA_VALIDITY_QUANTILE,
    ECE_BINS,
    FEATURE_FAMILY_MAP,
    FEATURE_SET_COLUMNS,
    MODEL_SPECS,
    TARGET_FPR,
)
from .feature_audit import run_feature_shortcut_audit
from .metrics import auc_bootstrap_ci, safe_logit, split_metrics, threshold_at_target_fpr
from src.feature_extraction import CONDITIONAL_CFA_KEYS, FEATURE_VERSION as FEATURE_VERSION_EXPECTED
from src.feature_extraction.constants import PREPROCESS_VERSION_EXPECTED


LABEL_MAP = {
    "ai": 1,
    "fake": 1,
    "nature": 0,
    "real": 0,
}


@dataclass
class FittedCandidate:
    candidate_name: str
    model_name: str
    feature_set: str
    feature_columns: tuple[str, ...]
    cfa_threshold: float
    model: Any
    platt: LogisticRegression


def load_training_table(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = {
        "split_role",
        "label",
        "feature_version",
        "preprocess_version",
        "status",
    }
    missing = sorted(expected_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Training table missing required columns: {missing}")
    if frame["status"].ne("ok").any():
        bad = int(frame["status"].ne("ok").sum())
        raise ValueError(f"Training table contains {bad} non-ok rows.")
    feature_versions = frame["feature_version"].dropna().astype(str).unique().tolist()
    if feature_versions != [FEATURE_VERSION_EXPECTED]:
        raise ValueError(
            f"Expected only feature_version={FEATURE_VERSION_EXPECTED!r}, got {feature_versions}"
        )
    preprocess_versions = frame["preprocess_version"].dropna().astype(str).unique().tolist()
    if preprocess_versions != [PREPROCESS_VERSION_EXPECTED]:
        raise ValueError(
            f"Expected only preprocess_version={PREPROCESS_VERSION_EXPECTED!r}, got {preprocess_versions}"
        )
    labels = frame["label"].astype(str).map(LABEL_MAP)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), "label"].astype(str).unique().tolist())
        raise ValueError(f"Unknown label values: {unknown}")
    prepared = frame.copy()
    prepared["y"] = labels.astype(np.int32)
    return prepared


def add_conditional_cfa_gates(
    frame: pd.DataFrame,
    *,
    quantile: float = CFA_VALIDITY_QUANTILE,
) -> tuple[pd.DataFrame, float]:
    prepared = frame.copy()
    train_validity = prepared.loc[prepared["split_role"] == "train_core", "cfa_validity_score"].astype(float)
    threshold = float(train_validity.quantile(quantile))
    return apply_conditional_cfa_threshold(prepared, threshold=threshold), threshold


def apply_conditional_cfa_threshold(
    frame: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    prepared = frame.copy()
    gate = prepared["cfa_validity_score"].astype(float) >= threshold
    for source, target in zip(CONDITIONAL_CFA_KEYS, CFA_GATED_KEYS):
        prepared[target] = prepared[source].astype(float) * gate.astype(np.float64)
    prepared["cfa_gate_active"] = gate.astype(np.int8)
    return prepared


def build_model(model_name: str) -> Any:
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=80,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")


def raw_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    prob = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return safe_logit(prob)


def fit_platt(raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    platt = LogisticRegression(max_iter=1000, random_state=42)
    platt.fit(raw.reshape(-1, 1), y)
    return platt


def apply_platt(platt: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return np.asarray(platt.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=np.float64)


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = ("train_core", "calibration", "val", "id_test", "ood_eval")
    splits = {name: frame.loc[frame["split_role"] == name].reset_index(drop=True) for name in required}
    missing = [name for name, split in splits.items() if split.empty]
    if missing:
        raise ValueError(f"Missing required split_role slices: {missing}")
    return splits


def build_cfa_gate_coverage(
    frame: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    prepared = apply_conditional_cfa_threshold(frame, threshold=threshold)
    coverage = (
        prepared.groupby(["split_role", "label"], as_index=False)
        .agg(
            gate_rate=("cfa_gate_active", "mean"),
            gate_active_count=("cfa_gate_active", "sum"),
            rows=("cfa_gate_active", "size"),
        )
        .sort_values(["split_role", "label"], ignore_index=True)
    )
    return coverage


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)


def train_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, FittedCandidate], float]:
    return train_candidates_with_feature_sets(frame)


def train_candidates_with_feature_sets(
    frame: pd.DataFrame,
    *,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    model_names: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, FittedCandidate], float]:
    prepared, cfa_threshold = add_conditional_cfa_gates(frame)
    splits = split_frames(prepared)
    train_df = splits["train_core"]
    cal_df = splits["calibration"]
    val_df = splits["val"]

    feature_sets = FEATURE_SET_COLUMNS if feature_sets is None else feature_sets
    model_specs = MODEL_SPECS
    if model_names is not None:
        allowed = set(model_names)
        model_specs = tuple(spec for spec in MODEL_SPECS if spec.name in allowed)
        if not model_specs:
            raise ValueError(f"No active MODEL_SPECS matched model_names={sorted(allowed)!r}")

    candidate_rows: list[dict[str, Any]] = []
    fitted: dict[str, FittedCandidate] = {}

    for feature_set, columns in feature_sets.items():
        x_train = _matrix(train_df, columns)
        y_train = train_df["y"].to_numpy(dtype=np.int32)
        x_cal = _matrix(cal_df, columns)
        y_cal = cal_df["y"].to_numpy(dtype=np.int32)
        x_val = _matrix(val_df, columns)
        y_val = val_df["y"].to_numpy(dtype=np.int32)

        for spec in model_specs:
            model = build_model(spec.name)
            model.fit(x_train, y_train)
            platt = fit_platt(raw_scores(model, x_cal), y_cal)
            val_prob = apply_platt(platt, raw_scores(model, x_val))
            threshold = threshold_at_target_fpr(y_val, val_prob, target_fpr=TARGET_FPR)
            metrics = split_metrics(y_val, val_prob, threshold=threshold, n_bins=ECE_BINS)
            candidate_name = f"{feature_set}__{spec.name}"
            candidate_rows.append(
                {
                    "candidate_name": candidate_name,
                    "feature_set": feature_set,
                    "model_name": spec.name,
                    "model_family": spec.family,
                    "feature_count": len(columns),
                    "cfa_threshold": cfa_threshold,
                    "val_auc": metrics["auc"],
                    "val_brier": metrics["brier"],
                    "val_ece": metrics["ece"],
                    "val_threshold": metrics["threshold"],
                    "val_tpr": metrics["tpr"],
                    "val_fpr": metrics["fpr"],
                    "val_precision": metrics["precision"],
                    "val_accuracy": metrics["accuracy"],
                }
            )
            fitted[candidate_name] = FittedCandidate(
                candidate_name=candidate_name,
                model_name=spec.name,
                feature_set=feature_set,
                feature_columns=columns,
                cfa_threshold=cfa_threshold,
                model=model,
                platt=platt,
            )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values(
        ["val_auc", "val_brier", "feature_count"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    return candidate_frame, fitted, cfa_threshold


def predict_candidate(
    candidate: FittedCandidate,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(frame, candidate.feature_columns)
    raw = raw_scores(candidate.model, x)
    prob = apply_platt(candidate.platt, raw)
    return raw, prob


def evaluate_selected_candidate(
    frame: pd.DataFrame,
    candidate: FittedCandidate,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prepared = apply_conditional_cfa_threshold(frame, threshold=candidate.cfa_threshold)
    splits = split_frames(prepared)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for split_name in ("val", "id_test", "ood_eval"):
        split_df = splits[split_name].copy()
        raw, prob = predict_candidate(candidate, split_df)
        y_true = split_df["y"].to_numpy(dtype=np.int32)
        metrics = split_metrics(y_true, prob, threshold=threshold, n_bins=ECE_BINS)
        auc_low, auc_high = auc_bootstrap_ci(
            y_true,
            prob,
            n_boot=BOOTSTRAP_ROUNDS,
            seed=BOOTSTRAP_SEED,
        )
        metrics_row = {
            "split": split_name,
            "candidate_name": candidate.candidate_name,
            "feature_set": candidate.feature_set,
            "model_name": candidate.model_name,
            "feature_count": len(candidate.feature_columns),
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            **metrics,
        }
        metric_rows.append(metrics_row)
        pred_df = split_df.loc[:, ["source_file_path", "generator", "label", "split_role"]].copy()
        pred_df["y_true"] = y_true
        pred_df["raw_score"] = raw
        pred_df["prob_ai"] = prob
        pred_df["threshold"] = threshold
        pred_df["pred_ai"] = (prob >= threshold).astype(np.int32)
        prediction_frames[split_name] = pred_df

    metric_frame = pd.DataFrame(metric_rows)
    return metric_frame, prediction_frames


def evaluate_selected_by_generator(
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generator, group in prediction_frame.groupby("generator", sort=True):
        metrics = split_metrics(
            group["y_true"].to_numpy(dtype=np.int32),
            group["prob_ai"].to_numpy(dtype=np.float64),
            threshold=float(group["threshold"].iloc[0]),
            n_bins=ECE_BINS,
        )
        rows.append(
            {
                "generator": str(generator),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("generator", ignore_index=True)


def selected_model_feature_importance(candidate: FittedCandidate) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate.model_name == "lightgbm":
        importances = np.asarray(candidate.model.feature_importances_, dtype=np.float64)
    elif candidate.model_name == "logreg":
        clf = candidate.model.named_steps["clf"]
        importances = np.abs(np.asarray(clf.coef_[0], dtype=np.float64))
    else:
        raise ValueError(f"Unsupported model_name={candidate.model_name!r} for feature importance.")

    feature_frame = pd.DataFrame(
        {
            "feature": list(candidate.feature_columns),
            "importance": importances,
        }
    )
    feature_frame["family"] = feature_frame["feature"].map(FEATURE_FAMILY_MAP).fillna("unknown")
    feature_frame = feature_frame.sort_values("importance", ascending=False, ignore_index=True)

    family_frame = (
        feature_frame.groupby("family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    return feature_frame, family_frame


def save_training_artifacts(
    *,
    output_dir: Path | str,
    candidate_frame: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    generator_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    family_importance: pd.DataFrame,
    cfa_gate_coverage: pd.DataFrame | None,
    summary: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "candidate_metrics_csv": str((root / "candidate_val_metrics.csv").resolve()),
        "selected_metrics_csv": str((root / "selected_model_metrics.csv").resolve()),
        "generator_metrics_csv": str((root / "selected_model_ood_by_generator.csv").resolve()),
        "feature_importance_csv": str((root / "selected_model_feature_importance.csv").resolve()),
        "family_importance_csv": str((root / "selected_model_family_importance.csv").resolve()),
        "summary_json": str((root / "summary.json").resolve()),
    }
    candidate_frame.to_csv(root / "candidate_val_metrics.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(root / "selected_model_metrics.csv", index=False, encoding="utf-8-sig")
    generator_metrics.to_csv(root / "selected_model_ood_by_generator.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(root / "selected_model_feature_importance.csv", index=False, encoding="utf-8-sig")
    family_importance.to_csv(root / "selected_model_family_importance.csv", index=False, encoding="utf-8-sig")
    if cfa_gate_coverage is not None:
        coverage_path = root / "cfa_gate_coverage.csv"
        cfa_gate_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
        files["cfa_gate_coverage_csv"] = str(coverage_path.resolve())
    for split_name, pred_df in prediction_frames.items():
        pred_path = root / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        files[f"predictions_{split_name}_csv"] = str(pred_path.resolve())
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return files


def save_model_parameters(
    *,
    output_dir: Path | str,
    candidate: FittedCandidate,
    threshold: float,
    feature_table_path: Path | str,
    feature_version: str,
    preprocess_version: str,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    files = {
        "model_joblib": str((root / "selected_model.joblib").resolve()),
        "platt_joblib": str((root / "selected_platt_calibrator.joblib").resolve()),
        "threshold_json": str((root / "selected_threshold.json").resolve()),
        "feature_schema_json": str((root / "selected_feature_schema.json").resolve()),
        "manifest_json": str((root / "model_manifest.json").resolve()),
    }

    joblib.dump(candidate.model, files["model_joblib"])
    joblib.dump(candidate.platt, files["platt_joblib"])

    threshold_payload = {
        "threshold": float(threshold),
        "target_fpr": float(TARGET_FPR),
        "cfa_threshold": float(candidate.cfa_threshold),
    }
    Path(files["threshold_json"]).write_text(
        json.dumps(threshold_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    schema_payload = {
        "candidate_name": candidate.candidate_name,
        "feature_set": candidate.feature_set,
        "model_name": candidate.model_name,
        "feature_columns": list(candidate.feature_columns),
    }
    Path(files["feature_schema_json"]).write_text(
        json.dumps(schema_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_table_path": str(Path(feature_table_path).resolve()),
        "feature_version": str(feature_version),
        "preprocess_version": str(preprocess_version),
        "candidate_name": candidate.candidate_name,
        "feature_set": candidate.feature_set,
        "model_name": candidate.model_name,
        "threshold": float(threshold),
        "target_fpr": float(TARGET_FPR),
        "cfa_threshold": float(candidate.cfa_threshold),
    }
    Path(files["manifest_json"]).write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return files


def run_training_baseline(
    feature_table_path: Path | str,
    *,
    output_dir: Path | str,
    model_output_dir: Path | str | None = None,
) -> dict[str, Any]:
    frame = load_training_table(feature_table_path)
    candidate_frame, fitted, cfa_threshold = train_candidates(frame)
    best = candidate_frame.iloc[0]
    selected = fitted[str(best["candidate_name"])]
    threshold = float(best["val_threshold"])
    selected_metrics, prediction_frames = evaluate_selected_candidate(
        frame,
        selected,
        threshold=threshold,
    )
    generator_metrics = evaluate_selected_by_generator(prediction_frames["ood_eval"])
    feature_importance, family_importance = selected_model_feature_importance(selected)
    gate_coverage = build_cfa_gate_coverage(frame, threshold=cfa_threshold)
    summary = {
        "feature_table_path": str(Path(feature_table_path).resolve()),
        "rows": int(len(frame)),
        "feature_version": str(frame["feature_version"].iloc[0]),
        "preprocess_version": str(frame["preprocess_version"].iloc[0]),
        "candidate_count": int(len(candidate_frame)),
        "selected_candidate": str(best["candidate_name"]),
        "selected_feature_set": str(best["feature_set"]),
        "selected_model_name": str(best["model_name"]),
        "selected_val_auc": float(best["val_auc"]),
        "selected_val_brier": float(best["val_brier"]),
        "selected_val_ece": float(best["val_ece"]),
        "selected_threshold": threshold,
        "cfa_threshold": cfa_threshold,
        "split_role_counts": {str(k): int(v) for k, v in frame["split_role"].value_counts().to_dict().items()},
    }
    files = save_training_artifacts(
        output_dir=output_dir,
        candidate_frame=candidate_frame,
        selected_metrics=selected_metrics,
        prediction_frames=prediction_frames,
        generator_metrics=generator_metrics,
        feature_importance=feature_importance,
        family_importance=family_importance,
        cfa_gate_coverage=gate_coverage,
        summary=summary,
    )

    if model_output_dir is not None:
        model_files = save_model_parameters(
            output_dir=model_output_dir,
            candidate=selected,
            threshold=threshold,
            feature_table_path=feature_table_path,
            feature_version=str(frame["feature_version"].iloc[0]),
            preprocess_version=str(frame["preprocess_version"].iloc[0]),
        )
        summary["model_files"] = model_files

    summary["files"] = files
    Path(files["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_training_with_feature_audit(
    feature_table_path: Path | str,
    *,
    audit_output_root: Path | str,
    metadata_csv_path: Path | str,
    model_output_dir: Path | str,
) -> dict[str, Any]:
    frame = load_training_table(feature_table_path)

    root = Path(audit_output_root)
    phase1_output_dir = root / "phase1_feature_audit"
    phase2_output_dir = root / "phase2_training_eval"

    phase1_summary = run_feature_shortcut_audit(
        frame,
        feature_table_path=feature_table_path,
        metadata_csv_path=metadata_csv_path,
        output_dir=phase1_output_dir,
    )
    phase2_summary = run_training_baseline(
        feature_table_path,
        output_dir=phase2_output_dir,
        model_output_dir=model_output_dir,
    )

    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_table_path": str(Path(feature_table_path).resolve()),
        "metadata_csv_path": str(Path(metadata_csv_path).resolve()),
        "audit_output_root": str(root.resolve()),
        "phase1_output_dir": str(phase1_output_dir.resolve()),
        "phase2_output_dir": str(phase2_output_dir.resolve()),
        "model_output_dir": str(Path(model_output_dir).resolve()),
        "phase1": phase1_summary,
        "phase2": phase2_summary,
        "claim_scope": "empirical",
    }
    summary_path = root / "two_phase_summary.json"
    summary["summary_json"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
