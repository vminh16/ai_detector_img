"""Process-safe worker entrypoint for feature extraction v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .cfa import extract_conditional_cfa_features
from .color import extract_color_features
from .constants import DEFAULT_CONFIG, FeatureExtractionConfig, feature_keys
from .frequency import extract_frequency_features
from .hetero import extract_dark_hetero_features
from .microtexture import extract_microtexture_features
from .spatial import extract_spatial_features
from .types import FeatureExtractionResult, FeatureExtractionStatus
from .views import FeatureContext
from .wavelet import extract_wavelet_features


ALL_FEATURE_KEYS = feature_keys(DEFAULT_CONFIG)


def _extract_with_config(patch: np.ndarray, config: FeatureExtractionConfig) -> dict[str, float]:
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


def extract_all_features(task: tuple[Any, ...]) -> FeatureExtractionResult:
    (
        row_id,
        source_file_path,
        patch_path,
        generator,
        label,
        split_role,
        dataset_name,
        preprocess_version,
        feature_version,
        include_conditional,
        include_research,
    ) = task
    config = FeatureExtractionConfig(
        feature_version=feature_version,
        include_conditional=bool(include_conditional),
        include_research=bool(include_research),
    )
    try:
        patch = np.load(Path(patch_path), allow_pickle=False)
        features = _extract_with_config(patch, config)
        return FeatureExtractionResult(
            row_id=row_id,
            source_file_path=source_file_path,
            patch_path=patch_path,
            generator=generator,
            label=label,
            split_role=split_role,
            dataset_name=dataset_name,
            preprocess_version=preprocess_version,
            feature_version=feature_version,
            status=FeatureExtractionStatus.OK,
            features=features,
        )
    except Exception as exc:
        empty = {key: np.nan for key in feature_keys(config)}
        return FeatureExtractionResult(
            row_id=row_id,
            source_file_path=source_file_path,
            patch_path=patch_path,
            generator=generator,
            label=label,
            split_role=split_role,
            dataset_name=dataset_name,
            preprocess_version=preprocess_version,
            feature_version=feature_version,
            status=FeatureExtractionStatus.ERROR,
            error=f"{type(exc).__name__}: {exc}",
            features=empty,
        )
