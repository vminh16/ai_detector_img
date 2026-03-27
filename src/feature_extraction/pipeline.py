"""High-level feature extraction pipeline for v2."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cfa import extract_conditional_cfa_features
from .color import extract_color_features
from .constants import DEFAULT_CONFIG, FeatureExtractionConfig, feature_keys
from .frequency import extract_frequency_features
from .hetero import extract_dark_hetero_features
from .microtexture import extract_microtexture_features
from .spatial import extract_spatial_features
from .types import BASE_COLUMNS, FeatureExtractionResult, FeatureExtractionStatus
from .views import FeatureContext
from .wavelet import extract_wavelet_features
from .worker import extract_all_features

logger = logging.getLogger(__name__)


def load_feature_manifest(
    manifest_path: Path | str,
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
    max_files: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    accepted = manifest.loc[(manifest["status"] == "ACCEPTED") & (manifest["saved_patch"].astype(bool))].copy()
    accepted = accepted.rename(columns={"file_path": "source_file_path", "output_path": "patch_path"})
    accepted["dataset_name"] = config.dataset_name
    accepted["feature_version"] = config.feature_version
    accepted["preprocess_version"] = accepted["preprocess_version"].fillna(config.preprocess_version_expected)
    accepted = assign_split_roles(accepted, config=config)
    if max_files is not None:
        accepted = stratified_sample_rows(accepted, max_rows=max_files, seed=config.split_seed)
    return accepted.reset_index(drop=True)


def assign_split_roles(
    manifest: pd.DataFrame,
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    frame = manifest.copy()
    frame["split_role"] = ""
    ood_mask = frame["generator"].isin(config.ood_generators)
    frame.loc[ood_mask, "split_role"] = "ood_eval"

    rng = np.random.default_rng(config.split_seed)
    id_mask = frame["generator"].isin(config.id_generators)
    groups = frame.loc[id_mask].groupby(["generator", "label"], sort=True)

    for _, group in groups:
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = int(round(n_total * config.id_test_fraction))
        n_val = int(round(n_total * config.val_fraction))
        remainder = n_total - n_test - n_val
        n_cal = int(round(remainder * config.calibration_fraction_of_remainder))

        pointer = 0
        frame.loc[indices[pointer : pointer + n_test], "split_role"] = "id_test"
        pointer += n_test
        frame.loc[indices[pointer : pointer + n_val], "split_role"] = "val"
        pointer += n_val
        frame.loc[indices[pointer : pointer + n_cal], "split_role"] = "calibration"
        pointer += n_cal
        frame.loc[indices[pointer:], "split_role"] = "train_core"

    unresolved = int((frame["split_role"] == "").sum())
    if unresolved:
        raise ValueError(f"{unresolved} rows were left without split_role.")
    return frame


def stratified_sample_rows(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(frame):
        return frame.copy()
    weights = frame.groupby(["generator", "label"]).size()
    target_counts: dict[tuple[str, str], int] = {}
    allocated = 0
    for key, count in weights.items():
        target = max(1, int(round(max_rows * (count / len(frame)))))
        target_counts[key] = min(int(count), target)
        allocated += target_counts[key]
    keys = list(target_counts)
    idx = 0
    while allocated > max_rows:
        key = keys[idx % len(keys)]
        if target_counts[key] > 1:
            target_counts[key] -= 1
            allocated -= 1
        idx += 1
    while allocated < max_rows:
        key = keys[idx % len(keys)]
        group_count = int(weights[key])
        if target_counts[key] < group_count:
            target_counts[key] += 1
            allocated += 1
        idx += 1

    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    for key, group in frame.groupby(["generator", "label"], sort=True):
        n_take = target_counts[key]
        choice = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        sampled_parts.append(frame.loc[np.sort(choice)])
    sampled = pd.concat(sampled_parts, axis=0).sort_values(["generator", "label", "patch_path"]).head(max_rows)
    return sampled.reset_index(drop=True)


def extract_feature_vector(
    patch: np.ndarray,
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    ctx = FeatureContext(patch)
    features: dict[str, float] = {}
    features.update(extract_frequency_features(ctx))
    features.update(extract_color_features(ctx))
    features.update(extract_spatial_features(ctx))
    if config.include_conditional:
        features.update(extract_conditional_cfa_features(ctx))
    if config.include_research:
        features.update(extract_wavelet_features(ctx))
        features.update(extract_dark_hetero_features(ctx))
        features.update(extract_microtexture_features(ctx))
    return features


def _task_from_row(row_id: int, row: pd.Series, config: FeatureExtractionConfig) -> tuple[Any, ...]:
    return (
        row_id,
        str(row["source_file_path"]),
        str(row["patch_path"]),
        str(row["generator"]),
        str(row["label"]),
        str(row["split_role"]),
        str(row["dataset_name"]),
        str(row["preprocess_version"]),
        config.feature_version,
        config.include_conditional,
        config.include_research,
    )


def build_tasks(
    manifest: pd.DataFrame,
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
) -> list[tuple[Any, ...]]:
    return [_task_from_row(idx, row, config) for idx, (_, row) in enumerate(manifest.iterrows())]


def results_to_frame(
    results: list[FeatureExtractionResult],
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    names = feature_keys(config)
    rows = [result.manifest_row(names) for result in sorted(results, key=lambda item: item.row_id)]
    return pd.DataFrame(rows, columns=BASE_COLUMNS + list(names) + ["status", "error"])


def save_feature_table(frame: pd.DataFrame, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path.resolve()


def summarise_feature_table(frame: pd.DataFrame, *, config: FeatureExtractionConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    ok_mask = frame["status"] == FeatureExtractionStatus.OK.value
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "ok_rows": int(ok_mask.sum()),
        "error_rows": int((~ok_mask).sum()),
        "feature_count": len(feature_keys(config)),
        "split_role_counts": frame["split_role"].value_counts().to_dict(),
        "generator_counts": frame["generator"].value_counts().to_dict(),
    }
    if "cfa_validity_score" in frame.columns and ok_mask.any():
        valid = frame.loc[ok_mask, "cfa_validity_score"].astype(float)
        summary["cfa_validity_score"] = {
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "q10": float(valid.quantile(0.10)),
            "q50": float(valid.quantile(0.50)),
            "q90": float(valid.quantile(0.90)),
        }
    return summary


def run_feature_pipeline(
    manifest: pd.DataFrame,
    *,
    config: FeatureExtractionConfig = DEFAULT_CONFIG,
    workers: int | None = None,
    chunksize: int = 32,
    show_progress: bool = True,
) -> list[FeatureExtractionResult]:
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    tasks = build_tasks(manifest, config=config)
    results: list[FeatureExtractionResult] = []
    if workers <= 1:
        iterator = (extract_all_features(task) for task in tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Extract features v2", unit="img", dynamic_ncols=True)
        for result in iterator:
            results.append(result)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(extract_all_features, tasks, chunksize=chunksize)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Extract features v2", unit="img", dynamic_ncols=True)
        for result in iterator:
            results.append(result)
    return results
