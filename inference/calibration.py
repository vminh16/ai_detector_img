"""
inference/calibration.py
========================
Platt-scaling calibration layer.

Converts the champion LightGBM raw margin (logit) into a calibrated
probability P(AI-generated | features) via a pre-trained sigmoid
(``LogisticRegression`` fitted on the calibration split).
"""
from __future__ import annotations

import numpy as np

from inference.artifact_loader import Artifacts


def raw_predict(vector: np.ndarray, artifacts: Artifacts) -> float:
    """Return the raw LightGBM margin (un-calibrated logit).

    Parameters
    ----------
    vector : 1-D array of shape ``(33,)`` in canonical feature order.
    """
    # LightGBM Booster.predict expects 2-D input
    pred = artifacts.champion_model.predict(vector.reshape(1, -1))
    return float(pred[0])


def calibrate(raw_score: float, artifacts: Artifacts) -> float:
    """Apply Platt scaling: logit → P(AI) ∈ [0, 1].

    The calibrator is a ``LogisticRegression`` that was fitted on
    (raw_score, label) pairs from the held-out calibration split.
    """
    x = np.array([[raw_score]], dtype=np.float64)
    prob = artifacts.calibrator.predict_proba(x)[0, 1]
    return float(prob)


def predict(vector: np.ndarray, artifacts: Artifacts) -> tuple[float, float]:
    """Full prediction: raw score + calibrated probability.

    Returns ``(raw_score, calibrated_prob)``.
    """
    raw = raw_predict(vector, artifacts)
    prob = calibrate(raw, artifacts)
    return raw, prob
