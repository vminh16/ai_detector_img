"""Phase-closing benchmark, nuisance audit, degradation suite, and ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import (
    ABLATION_FEATURE_SET_COLUMNS,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    DEGRADATION_EVAL_SPLITS,
    DEGRADATION_SPECS,
    FEATURE_SET_COLUMNS,
)
from .degradation import build_nuisance_eval_frame, extract_degraded_feature_frame
from .metrics import auc_bootstrap_ci, split_metrics
from .pipeline import (
    FittedCandidate,
    apply_conditional_cfa_threshold,
    build_cfa_gate_coverage,
    evaluate_selected_by_generator,
    load_training_table,
    predict_candidate,
    save_model_parameters,
    selected_model_feature_importance,
    train_candidates_with_feature_sets,
)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _write_json(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.resolve())


def _pooled_eval(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["split_role"].isin(DEGRADATION_EVAL_SPLITS)].reset_index(drop=True)


def _candidate_threshold_map(candidate_frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(row["candidate_name"]): float(row["val_threshold"])
        for row in candidate_frame.to_dict(orient="records")
    }


def _auc_abs(value: float) -> float:
    if np.isnan(value):
        return float("nan")
    return float(max(value, 1.0 - value))


def evaluate_candidate_across_splits(
    frame: pd.DataFrame,
    candidate: FittedCandidate,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prepared = apply_conditional_cfa_threshold(frame, threshold=candidate.cfa_threshold)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    split_order = list(DEGRADATION_EVAL_SPLITS) + ["pooled_eval"]
    for split_name in split_order:
        if split_name == "pooled_eval":
            split_df = _pooled_eval(prepared)
        else:
            split_df = prepared.loc[prepared["split_role"] == split_name].reset_index(drop=True)
        if split_df.empty:
            continue
        raw, prob = predict_candidate(candidate, split_df)
        y_true = split_df["y"].to_numpy(dtype=np.int32)
        metrics = split_metrics(y_true, prob, threshold=threshold, n_bins=15)
        auc_low, auc_high = auc_bootstrap_ci(
            y_true,
            prob,
            n_boot=BOOTSTRAP_ROUNDS,
            seed=BOOTSTRAP_SEED,
        )
        metric_rows.append(
            {
                "split": split_name,
                "candidate_name": candidate.candidate_name,
                "feature_set": candidate.feature_set,
                "model_name": candidate.model_name,
                "feature_count": len(candidate.feature_columns),
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                **metrics,
            }
        )
        pred_df = split_df.loc[:, ["source_file_path", "generator", "label", "split_role", "cfa_gate_active"]].copy()
        pred_df["y_true"] = y_true
        pred_df["raw_score"] = raw
        pred_df["prob_ai"] = prob
        pred_df["threshold"] = float(threshold)
        pred_df["pred_ai"] = (prob >= threshold).astype(np.int32)
        pred_df["evaluation_split"] = split_name
        prediction_frames[split_name] = pred_df
    return pd.DataFrame(metric_rows), prediction_frames


def evaluate_multiple_candidates_clean(
    frame: pd.DataFrame,
    *,
    fitted_candidates: dict[str, FittedCandidate],
    threshold_map: dict[str, float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, candidate in fitted_candidates.items():
        metrics, _ = evaluate_candidate_across_splits(frame, candidate, threshold=threshold_map[name])
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def evaluate_model_level_auc_nat(
    frame: pd.DataFrame,
    *,
    fitted_candidates: dict[str, FittedCandidate],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuisance = build_nuisance_eval_frame(frame)
    label_summary = (
        nuisance.groupby(["split_role", "jpeg_subsampling_live"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["split_role", "jpeg_subsampling_live"], ignore_index=True)
    )
    rows: list[dict[str, Any]] = []
    for name, candidate in fitted_candidates.items():
        prepared = apply_conditional_cfa_threshold(nuisance, threshold=candidate.cfa_threshold)
        for split_name in list(DEGRADATION_EVAL_SPLITS) + ["pooled_eval"]:
            if split_name == "pooled_eval":
                split_df = _pooled_eval(prepared)
            else:
                split_df = prepared.loc[prepared["split_role"] == split_name].reset_index(drop=True)
            if split_df.empty:
                continue
            y_true = split_df["nuisance_is_420"].to_numpy(dtype=np.int32)
            _, prob = predict_candidate(candidate, split_df)
            auc = float("nan")
            if np.unique(y_true).size == 2:
                auc = float(roc_auc_score(y_true, prob))
            rows.append(
                {
                    "candidate_name": name,
                    "feature_set": candidate.feature_set,
                    "model_name": candidate.model_name,
                    "split": split_name,
                    "n_rows": int(len(split_df)),
                    "n_420": int(np.count_nonzero(y_true == 1)),
                    "n_444": int(np.count_nonzero(y_true == 0)),
                    "auc_nat_raw": auc,
                    "auc_nat_abs": _auc_abs(auc),
                    "pred_mean_420": float(np.mean(prob[y_true == 1])) if np.any(y_true == 1) else float("nan"),
                    "pred_mean_444": float(np.mean(prob[y_true == 0])) if np.any(y_true == 0) else float("nan"),
                    "pred_gap_420_minus_444": (
                        float(np.mean(prob[y_true == 1]) - np.mean(prob[y_true == 0]))
                        if np.any(y_true == 1) and np.any(y_true == 0)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows), label_summary


def materialize_degradation_tables(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    workers: int,
    force_rerun: bool,
    show_progress: bool,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_frame = _pooled_eval(frame)
    tables: dict[str, pd.DataFrame] = {}
    for spec in DEGRADATION_SPECS:
        csv_path = output_dir / f"{spec.name}.csv"
        loaded = None if force_rerun else _read_csv_if_exists(csv_path)
        if loaded is not None:
            tables[spec.name] = loaded
            continue
        degraded = extract_degraded_feature_frame(
            eval_frame,
            degradation_name=spec.name,
            workers=workers,
            show_progress=show_progress,
        )
        _write_csv(degraded, csv_path)
        tables[spec.name] = degraded
    return tables


def evaluate_degradation_suite(
    *,
    degraded_tables: dict[str, pd.DataFrame],
    fitted_candidates: dict[str, FittedCandidate],
    threshold_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[pd.DataFrame] = []
    for degradation_name, degraded in degraded_tables.items():
        coverage = build_cfa_gate_coverage(
            degraded.assign(y=degraded["y"].astype(np.int32)),
            threshold=next(iter(fitted_candidates.values())).cfa_threshold,
        )
        coverage["degradation_name"] = degradation_name
        coverage_rows.append(coverage)
        for name, candidate in fitted_candidates.items():
            metrics, _ = evaluate_candidate_across_splits(
                degraded.assign(y=degraded["y"].astype(np.int32)),
                candidate,
                threshold=threshold_map[name],
            )
            metrics["degradation_name"] = degradation_name
            metric_rows.append(metrics)
    metrics_frame = pd.concat(metric_rows, axis=0, ignore_index=True) if metric_rows else pd.DataFrame()
    coverage_frame = pd.concat(coverage_rows, axis=0, ignore_index=True) if coverage_rows else pd.DataFrame()
    return metrics_frame, coverage_frame


def build_degradation_gap_summary(
    clean_metrics: pd.DataFrame,
    degradation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    base = clean_metrics.loc[:, ["candidate_name", "split", "auc", "brier", "ece", "tpr", "fpr"]].rename(
        columns={
            "auc": "clean_auc",
            "brier": "clean_brier",
            "ece": "clean_ece",
            "tpr": "clean_tpr",
            "fpr": "clean_fpr",
        }
    )
    merged = degradation_metrics.merge(base, on=["candidate_name", "split"], how="left")
    merged["auc_gap"] = merged["auc"] - merged["clean_auc"]
    merged["brier_gap"] = merged["brier"] - merged["clean_brier"]
    merged["ece_gap"] = merged["ece"] - merged["clean_ece"]
    merged["tpr_gap"] = merged["tpr"] - merged["clean_tpr"]
    merged["fpr_gap"] = merged["fpr"] - merged["clean_fpr"]
    keep_cols = [
        "candidate_name",
        "feature_set",
        "model_name",
        "degradation_name",
        "split",
        "clean_auc",
        "auc",
        "auc_gap",
        "clean_brier",
        "brier",
        "brier_gap",
        "clean_ece",
        "ece",
        "ece_gap",
        "clean_tpr",
        "tpr",
        "tpr_gap",
        "clean_fpr",
        "fpr",
        "fpr_gap",
    ]
    return merged.loc[:, keep_cols].sort_values(
        ["candidate_name", "degradation_name", "split"],
        ignore_index=True,
    )


def build_branch_closure_summary(
    *,
    ablation_candidate_frame: pd.DataFrame,
    ablation_clean_metrics: pd.DataFrame,
    auc_nat_metrics: pd.DataFrame,
    degradation_gap_summary: pd.DataFrame,
) -> pd.DataFrame:
    clean_pooled = ablation_clean_metrics.loc[ablation_clean_metrics["split"] == "pooled_eval", [
        "candidate_name",
        "feature_set",
        "model_name",
        "auc",
        "brier",
        "ece",
        "tpr",
        "fpr",
    ]].rename(
        columns={
            "auc": "clean_pooled_auc",
            "brier": "clean_pooled_brier",
            "ece": "clean_pooled_ece",
            "tpr": "clean_pooled_tpr",
            "fpr": "clean_pooled_fpr",
        }
    )
    nat_pooled = auc_nat_metrics.loc[auc_nat_metrics["split"] == "pooled_eval", [
        "candidate_name",
        "auc_nat_raw",
        "auc_nat_abs",
        "pred_gap_420_minus_444",
    ]]
    xdeg_summary = (
        degradation_gap_summary.loc[degradation_gap_summary["split"] == "pooled_eval"]
        .groupby("candidate_name", as_index=False)
        .agg(
            worst_xdeg_auc=("auc", "min"),
            mean_xdeg_auc=("auc", "mean"),
            worst_auc_gap=("auc_gap", "min"),
            mean_auc_gap=("auc_gap", "mean"),
        )
    )
    summary = (
        ablation_candidate_frame.loc[:, ["candidate_name", "feature_set", "model_name", "val_auc", "val_brier", "val_ece"]]
        .merge(clean_pooled, on=["candidate_name", "feature_set", "model_name"], how="left")
        .merge(nat_pooled, on="candidate_name", how="left")
        .merge(xdeg_summary, on="candidate_name", how="left")
        .sort_values(["val_auc", "clean_pooled_auc"], ascending=[False, False], ignore_index=True)
    )
    summary["uses_research_wavelet"] = summary["feature_set"].isin({"always_on_plus_wavelet", "full_v2", "full_v2_minus_content_adaptive_y_srm", "full_v2_minus_dark_textured_hetero", "full_v2_minus_conditional_cfa"})
    summary["uses_research_ysrm"] = summary["feature_set"].isin({"always_on_plus_ysrm", "full_v2", "full_v2_minus_wavelet_decay", "full_v2_minus_conditional_cfa", "full_v2_minus_dark_textured_hetero"})
    summary["uses_conditional_cfa"] = summary["feature_set"].isin({"always_on_plus_cfa_gated", "full_v2", "full_v2_minus_wavelet_decay", "full_v2_minus_content_adaptive_y_srm", "full_v2_minus_dark_textured_hetero"})
    return summary


def run_training_phase_closure(
    feature_table_path: Path | str,
    *,
    output_root: Path | str,
    model_output_dir: Path | str | None = None,
    workers: int = 1,
    force_rerun: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    frame = load_training_table(feature_table_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    phase1_dir = root / "phase1_clean_benchmark"
    phase2_dir = root / "phase2_model_nuisance"
    phase3_dir = root / "phase3_degradation_suite"
    phase4_dir = root / "phase4_family_ablation"
    phase5_dir = root / "phase5_phase_closure"

    baseline_candidate_frame, baseline_fitted, _ = train_candidates_with_feature_sets(
        frame,
        feature_sets=FEATURE_SET_COLUMNS,
    )
    best = baseline_candidate_frame.iloc[0]
    selected = baseline_fitted[str(best["candidate_name"])]
    selected_threshold = float(best["val_threshold"])
    selected_clean_metrics, selected_prediction_frames = evaluate_candidate_across_splits(
        frame,
        selected,
        threshold=selected_threshold,
    )
    selected_ood_by_generator = evaluate_selected_by_generator(selected_prediction_frames["ood_eval"])
    selected_feature_importance, selected_family_importance = selected_model_feature_importance(selected)
    clean_cfa_gate_coverage = build_cfa_gate_coverage(frame, threshold=selected.cfa_threshold)

    phase1_files = {
        "candidate_val_metrics_csv": _write_csv(baseline_candidate_frame, phase1_dir / "candidate_val_metrics.csv"),
        "selected_model_metrics_csv": _write_csv(selected_clean_metrics, phase1_dir / "selected_model_metrics.csv"),
        "selected_model_ood_by_generator_csv": _write_csv(
            selected_ood_by_generator,
            phase1_dir / "selected_model_ood_by_generator.csv",
        ),
        "selected_model_feature_importance_csv": _write_csv(
            selected_feature_importance,
            phase1_dir / "selected_model_feature_importance.csv",
        ),
        "selected_model_family_importance_csv": _write_csv(
            selected_family_importance,
            phase1_dir / "selected_model_family_importance.csv",
        ),
        "clean_cfa_gate_coverage_csv": _write_csv(clean_cfa_gate_coverage, phase1_dir / "clean_cfa_gate_coverage.csv"),
    }

    if model_output_dir is not None:
        phase1_files["model_files"] = save_model_parameters(
            output_dir=model_output_dir,
            candidate=selected,
            threshold=selected_threshold,
            feature_table_path=feature_table_path,
            feature_version=str(frame["feature_version"].iloc[0]),
            preprocess_version=str(frame["preprocess_version"].iloc[0]),
        )

    selected_model_name = selected.model_name
    ablation_candidate_frame, ablation_fitted, _ = train_candidates_with_feature_sets(
        frame,
        feature_sets=ABLATION_FEATURE_SET_COLUMNS,
        model_names=(selected_model_name,),
    )
    ablation_threshold_map = _candidate_threshold_map(ablation_candidate_frame)
    ablation_clean_metrics = evaluate_multiple_candidates_clean(
        frame,
        fitted_candidates=ablation_fitted,
        threshold_map=ablation_threshold_map,
    )
    auc_nat_metrics, nuisance_label_summary = evaluate_model_level_auc_nat(
        frame,
        fitted_candidates=ablation_fitted,
    )
    phase2_files = {
        "model_level_auc_nat_csv": _write_csv(auc_nat_metrics, phase2_dir / "model_level_auc_nat.csv"),
        "nuisance_label_summary_csv": _write_csv(nuisance_label_summary, phase2_dir / "nuisance_label_summary.csv"),
    }

    degraded_tables = materialize_degradation_tables(
        frame,
        output_dir=phase3_dir / "degraded_feature_tables",
        workers=workers,
        force_rerun=force_rerun,
        show_progress=show_progress,
    )
    degradation_metrics, degradation_cfa_coverage = evaluate_degradation_suite(
        degraded_tables=degraded_tables,
        fitted_candidates=ablation_fitted,
        threshold_map=ablation_threshold_map,
    )
    degradation_gap_summary = build_degradation_gap_summary(ablation_clean_metrics, degradation_metrics)
    phase3_files = {
        "degradation_metrics_csv": _write_csv(degradation_metrics, phase3_dir / "degradation_metrics.csv"),
        "degradation_gap_summary_csv": _write_csv(
            degradation_gap_summary,
            phase3_dir / "degradation_gap_summary.csv",
        ),
        "degradation_cfa_gate_coverage_csv": _write_csv(
            degradation_cfa_coverage,
            phase3_dir / "degradation_cfa_gate_coverage.csv",
        ),
    }

    branch_closure_summary = build_branch_closure_summary(
        ablation_candidate_frame=ablation_candidate_frame,
        ablation_clean_metrics=ablation_clean_metrics,
        auc_nat_metrics=auc_nat_metrics,
        degradation_gap_summary=degradation_gap_summary,
    )
    phase4_files = {
        "ablation_candidate_val_metrics_csv": _write_csv(
            ablation_candidate_frame,
            phase4_dir / "ablation_candidate_val_metrics.csv",
        ),
        "ablation_clean_metrics_csv": _write_csv(ablation_clean_metrics, phase4_dir / "ablation_clean_metrics.csv"),
        "branch_closure_summary_csv": _write_csv(branch_closure_summary, phase4_dir / "branch_closure_summary.csv"),
    }

    best_branch = (
        branch_closure_summary.sort_values(
            ["clean_pooled_auc", "mean_xdeg_auc", "worst_xdeg_auc", "auc_nat_abs"],
            ascending=[False, False, False, True],
            ignore_index=True,
        ).iloc[0]
        if not branch_closure_summary.empty
        else None
    )
    closure_summary = {
        "feature_table_path": str(Path(feature_table_path).resolve()),
        "rows": int(len(frame)),
        "feature_version": str(frame["feature_version"].iloc[0]),
        "preprocess_version": str(frame["preprocess_version"].iloc[0]),
        "selected_clean_candidate": str(best["candidate_name"]),
        "selected_clean_feature_set": str(best["feature_set"]),
        "selected_clean_model_name": str(best["model_name"]),
        "selected_clean_val_auc": float(best["val_auc"]),
        "selected_clean_threshold": float(selected_threshold),
        "required_audits_completed": True,
        "best_branch_candidate": None if best_branch is None else str(best_branch["candidate_name"]),
        "best_branch_feature_set": None if best_branch is None else str(best_branch["feature_set"]),
        "best_branch_clean_pooled_auc": None if best_branch is None else float(best_branch["clean_pooled_auc"]),
        "best_branch_worst_xdeg_auc": None if best_branch is None else float(best_branch["worst_xdeg_auc"]),
        "best_branch_auc_nat_abs": None if best_branch is None else float(best_branch["auc_nat_abs"]),
        "phase1_files": phase1_files,
        "phase2_files": phase2_files,
        "phase3_files": phase3_files,
        "phase4_files": phase4_files,
    }
    phase5_files = {
        "phase_closure_summary_json": _write_json(closure_summary, phase5_dir / "phase_closure_summary.json"),
    }
    closure_summary["phase5_files"] = phase5_files
    _write_json(closure_summary, root / "summary.json")
    return closure_summary
