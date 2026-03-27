"""
inference/artifact_loader.py
============================
Load champion model, Platt calibrator, and supporting artifacts
into memory once at startup.

This module is **import-time safe** — it only reads files when
``load_artifacts()`` is explicitly called.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from inference.config import InferenceConfig
from inference.errors import ModelLoadError


@dataclass
class Artifacts:
    """Container for all loaded inference-time artefacts."""
    champion_model: Any       # lightgbm.Booster
    calibrator: Any           # sklearn LogisticRegression (Platt)
    config: InferenceConfig

    # Derived at load time
    feature_importance: np.ndarray | None = None  # per-feature gain share
    importance_names: list[str] | None = None


def load_artifacts(config: InferenceConfig) -> Artifacts:
    """Load all required artefacts from disk.

    Parameters
    ----------
    config : fully initialised ``InferenceConfig`` (already contains
             threshold_lock, winsor_caps, and feature_schema data).

    Returns
    -------
    Artifacts
        Ready-to-use container.

    Raises
    ------
    ModelLoadError
        If any required file is missing or corrupt.
    """
    model_path: Path = config.champion_model_path
    calib_path: Path = config.calibrator_path

    for p in (model_path, calib_path):
        if not p.exists():
            raise ModelLoadError(f"Required artifact missing: {p}")

    try:
        champion = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Cannot load champion model: {exc}") from exc

    try:
        calibrator = joblib.load(calib_path)
    except Exception as exc:
        raise ModelLoadError(f"Cannot load Platt calibrator: {exc}") from exc

    # Pre-compute normalised feature importance (gain-based)
    feat_imp: np.ndarray | None = None
    imp_names: list[str] | None = None
    try:
        raw_imp = np.array(champion.feature_importance(importance_type="gain"), dtype=np.float64)
        total = raw_imp.sum()
        if total > 0:
            feat_imp = raw_imp / total
        else:
            feat_imp = raw_imp
        imp_names = list(config.feature_order)
    except Exception:
        pass  # importance is optional (falls back gracefully)

    return Artifacts(
        champion_model=champion,
        calibrator=calibrator,
        config=config,
        feature_importance=feat_imp,
        importance_names=imp_names,
    )
