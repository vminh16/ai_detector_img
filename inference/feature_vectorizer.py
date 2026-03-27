"""
inference/feature_vectorizer.py
===============================
Wraps the 4 existing feature-group modules into a single extraction
call that returns the 33-dim feature vector in canonical order.

Also applies the GBDT-specific transforms:
  1. Winsorisation (clip to training [p1, p99] bounds)
  2. No imputation — LightGBM handles NaN natively

The log1p / sentinel-zero transforms learned at training time are
**not** applied here for the GBDT champion because LightGBM tree
splits are invariant to monotone transforms.  They would only matter
for the SVM/RF candidates which are not the champion.
"""
from __future__ import annotations

import numpy as np

from inference.config import InferenceConfig
from inference.errors import FeatureExtractionError

# Existing extractors (accept 256×256×3 uint8 YCrCb ndarray)
from src.feature_extraction.frequency import extract_frequency_features
from src.feature_extraction.color import extract_color_features
from src.feature_extraction.microtexture import extract_microtexture_features
from src.feature_extraction.spatial import extract_spatial_features


def extract_features(arr: np.ndarray) -> dict[str, float]:
    """Extract all 33 features from a preprocessed YCrCb array.

    Parameters
    ----------
    arr : np.ndarray
        256×256×3 ``uint8`` YCrCb image produced by ``preprocess_image``.

    Returns
    -------
    dict[str, float]
        Feature name → value mapping (may contain ``NaN`` for spatial
        features when the noise estimate is degenerate).
    """
    try:
        features: dict[str, float] = {}
        features.update(extract_frequency_features(arr))
        features.update(extract_color_features(arr))
        features.update(extract_microtexture_features(arr))
        features.update(extract_spatial_features(arr))
        return features
    except Exception as exc:
        raise FeatureExtractionError(
            f"Feature extraction failed: {exc}"
        ) from exc


def winsorise(
    features: dict[str, float],
    config: InferenceConfig,
) -> dict[str, float]:
    """Clip feature values to training-time [p1, p99] bounds."""
    caps = config.winsor_caps
    out: dict[str, float] = {}
    for name, val in features.items():
        if name in caps:
            lo, hi = caps[name]
            if np.isnan(val):
                out[name] = val  # preserve NaN for native handling
            else:
                out[name] = float(np.clip(val, lo, hi))
        else:
            out[name] = val
    return out


def build_vector(
    features: dict[str, float],
    config: InferenceConfig,
) -> np.ndarray:
    """Assemble features into a 1-D array in canonical order.

    Parameters
    ----------
    features : dict from ``extract_features`` (optionally winsorised).
    config : provides ``feature_order`` (33 canonical names).

    Returns
    -------
    np.ndarray of shape ``(33,)`` with ``float64`` dtype.
        Missing keys become ``NaN``.
    """
    order = config.feature_order
    vec = np.array(
        [features.get(name, np.nan) for name in order],
        dtype=np.float64,
    )
    return vec


def featurise(
    arr: np.ndarray,
    config: InferenceConfig,
) -> tuple[dict[str, float], np.ndarray]:
    """Full featurisation: extract → winsorise → vectorise.

    Returns ``(features_dict, vector_1d)``.
    """
    raw = extract_features(arr)
    clipped = winsorise(raw, config)
    vec = build_vector(clipped, config)
    return clipped, vec
