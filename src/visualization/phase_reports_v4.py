"""Visualization helpers for the v4 pipeline phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATUS_ORDER = ["ACCEPTED", "LOW_SUPPORT", "UNSUPPORTED_INPUT", "DECODE_ERROR"]
STATUS_COLORS = {
    "ACCEPTED": "#2a9d8f",
    "LOW_SUPPORT": "#f4a261",
    "UNSUPPORTED_INPUT": "#e76f51",
    "DECODE_ERROR": "#264653",
}


def _ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(path: Path | str, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    for column in ["width", "height", "support", "crop_origin_x", "crop_origin_y"]:
        if column in manifest.columns:
            manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    return manifest


def preprocessing_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    total = int(len(manifest))
    accepted = int((manifest["status"] == "ACCEPTED").sum())
    by_label = (
        manifest.groupby(["label", "status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    )
    rates = (
        by_label.div(by_label.sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
        .round(6)
        .to_dict(orient="index")
    )
    summary = {
        "n_total": total,
        "n_accepted": accepted,
        "acceptance_rate": float(accepted / total) if total else 0.0,
        "status_counts": manifest["status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict(),
        "status_rates_by_label": rates,
        "input_mode_counts": manifest["input_mode"].fillna("UNKNOWN").value_counts().to_dict(),
        "input_format_counts": manifest["input_format"].fillna("UNKNOWN").value_counts().to_dict(),
        "generator_count": int(manifest["generator"].nunique(dropna=True)),
    }
    if "support" in manifest:
        accepted_support = manifest.loc[manifest["status"] == "ACCEPTED", "support"].dropna()
        if not accepted_support.empty:
            summary["accepted_support"] = {
                "min": float(accepted_support.min()),
                "median": float(accepted_support.median()),
                "max": float(accepted_support.max()),
            }
    return summary


def _plot_status_by_label(manifest: pd.DataFrame, output_path: Path) -> None:
    table = (
        manifest.groupby(["label", "status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(table), dtype=float)
    y = np.arange(len(table))
    for status in STATUS_ORDER:
        values = table[status].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=STATUS_COLORS[status], label=status)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(table.index.tolist())
    ax.set_xlabel("Image count")
    ax.set_title("Preprocessing v4 status by label")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_support_distribution(manifest: pd.DataFrame, output_path: Path, threshold: int = 252) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [("nature", "#457b9d"), ("ai", "#e63946")]:
        values = manifest.loc[manifest["label"] == label, "support"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        ax.hist(values, bins=40, alpha=0.5, label=label, color=color)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold}")
    ax.set_xlabel("Support = min(height, width)")
    ax.set_ylabel("Count")
    ax.set_title("Support distribution before exact crop gate")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_status_by_generator(manifest: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    table = manifest.groupby(["generator", "status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).head(top_n).index]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(table) + 1)))
    bottom = np.zeros(len(table), dtype=float)
    x = np.arange(len(table))
    for status in STATUS_ORDER:
        values = table[status].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=STATUS_COLORS[status], label=status)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(table.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Top generators by preprocessing status")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_crop_origin_scatter(manifest: pd.DataFrame, output_path: Path) -> None:
    accepted = manifest.loc[manifest["status"] == "ACCEPTED"].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        accepted["crop_origin_x"],
        accepted["crop_origin_y"],
        s=6,
        alpha=0.35,
        color="#1d3557",
        edgecolors="none",
    )
    ax.set_xlabel("crop_origin_x")
    ax.set_ylabel("crop_origin_y")
    ax.set_title("Accepted crop origins (all should satisfy x,y ≡ 4 mod 8)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_patch_gallery(manifest: pd.DataFrame, output_path: Path, per_mode: int = 4) -> None:
    accepted = manifest.loc[(manifest["status"] == "ACCEPTED") & (manifest["saved_patch"] == True)].copy()
    if accepted.empty:
        return
    rng = np.random.default_rng(42)
    selected_frames: list[pd.DataFrame] = []
    for mode in ["RGB", "RGBA"]:
        subset = accepted.loc[accepted["input_mode"] == mode]
        if subset.empty:
            continue
        take = min(per_mode, len(subset))
        selected_frames.append(subset.sample(n=take, random_state=42))
    if not selected_frames:
        take = min(2 * per_mode, len(accepted))
        selected_frames = [accepted.sample(n=take, random_state=42)]
    gallery = pd.concat(selected_frames, ignore_index=True)
    n = len(gallery)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes_arr.flat, gallery.iterrows()):
        patch = np.load(row["output_path"])
        ax.imshow(patch)
        ax.set_title(f"{row['generator']} | {row['input_mode']}")
        ax.axis("off")
    fig.suptitle("Accepted patch gallery after preprocessing v4", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_preprocessing_report(
    manifest_path: Path | str,
    output_root: Path | str,
    *,
    threshold: int = 252,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    out_root = _ensure_dir(output_root)
    figures_root = _ensure_dir(out_root / "plots")
    summary = preprocessing_summary(manifest)
    _save_json(out_root / "preprocessing_summary.json", summary)
    _plot_status_by_label(manifest, figures_root / "status_by_label.png")
    _plot_support_distribution(manifest, figures_root / "support_distribution.png", threshold=threshold)
    _plot_status_by_generator(manifest, figures_root / "status_by_generator.png")
    _plot_crop_origin_scatter(manifest, figures_root / "crop_origin_scatter.png")
    _plot_patch_gallery(manifest, figures_root / "patch_gallery.png")
    summary["figure_dir"] = str(figures_root.resolve())
    return summary


def render_feature_phase_report(features_csv: Path | str, output_root: Path | str) -> dict[str, Any]:
    out_root = _ensure_dir(output_root)
    path = Path(features_csv)
    if not path.exists():
        payload = {
            "available": False,
            "reason": f"missing feature dataset: {path}",
        }
        _save_json(out_root / "feature_phase_status.json", payload)
        return payload

    df = pd.read_csv(path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    payload = {
        "available": True,
        "n_rows": int(len(df)),
        "n_numeric_features": int(len(numeric_cols)),
        "label_counts": df.get("label", pd.Series(dtype=str)).value_counts().to_dict(),
    }
    _save_json(out_root / "feature_phase_status.json", payload)

    if numeric_cols:
        missing = df[numeric_cols].isna().mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(missing.index[::-1], missing.to_numpy()[::-1], color="#457b9d")
        ax.set_xlabel("Missing rate")
        ax.set_title("Top 20 feature missing rates")
        fig.tight_layout()
        fig.savefig(out_root / "top_missing_features.png", dpi=160)
        plt.close(fig)
    return payload


def render_model_phase_report(artifact_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    out_root = _ensure_dir(output_root)
    root = Path(artifact_root)
    eval_path = root / "evaluation_metrics.csv"
    if not eval_path.exists():
        payload = {
            "available": False,
            "reason": f"missing evaluation_metrics.csv under {root}",
        }
        _save_json(out_root / "model_phase_status.json", payload)
        return payload

    eval_df = pd.read_csv(eval_path)
    payload = {
        "available": True,
        "n_rows": int(len(eval_df)),
        "columns": eval_df.columns.tolist(),
    }
    _save_json(out_root / "model_phase_status.json", payload)

    numeric_cols = [col for col in eval_df.columns if pd.api.types.is_numeric_dtype(eval_df[col])]
    plot_cols = [col for col in numeric_cols if col.lower() in {"auc", "roc_auc", "pr_auc", "tpr", "fpr"}]
    if plot_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        eval_df[plot_cols].plot(ax=ax, marker="o")
        ax.set_title("Model evaluation metrics overview")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(out_root / "evaluation_metrics_overview.png", dpi=160)
        plt.close(fig)
    return payload
