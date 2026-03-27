"""
inference/explainer.py
======================
Produce top-N feature contributors from LightGBM gain-based importance.

This is a *global* explanation (same ranking for every image).
"""
from __future__ import annotations

import numpy as np

from inference.artifact_loader import Artifacts
from inference.schemas import FeatureContributor


def top_contributors(
    artifacts: Artifacts,
    n: int = 3,
) -> list[FeatureContributor]:
    """Return the top-*n* features by normalised gain.

    Falls back to an empty list when importance is unavailable.
    """
    if artifacts.feature_importance is None or artifacts.importance_names is None:
        return []

    imp = artifacts.feature_importance
    names = artifacts.importance_names

    indices = np.argsort(imp)[::-1][:n]

    return [
        FeatureContributor(
            feature=names[i],
            importance=round(float(imp[i]), 6),
        )
        for i in indices
    ]
