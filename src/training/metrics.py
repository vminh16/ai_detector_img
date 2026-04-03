"""Metrics for training and evaluation v2."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(acc - conf)
    return float(ece)


def threshold_at_target_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    target_fpr: float,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    negatives = y_prob[y_true == 0]
    if negatives.size == 0:
        return 1.0
    candidates = np.unique(y_prob)
    best = float(np.nextafter(np.max(candidates), np.inf))
    for threshold in np.sort(candidates):
        fpr = float(np.mean(negatives >= threshold))
        if fpr <= target_fpr:
            best = float(threshold)
            break
    return best


def split_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    n_bins: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    positives = y_true == 1
    negatives = y_true == 0
    pred_pos = y_pred == 1
    pred_neg = y_pred == 0

    tp = int(np.count_nonzero(positives & pred_pos))
    tn = int(np.count_nonzero(negatives & pred_neg))
    fp = int(np.count_nonzero(negatives & pred_pos))
    fn = int(np.count_nonzero(positives & pred_neg))

    auc = float("nan")
    if np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    tpr = float(tp / max(1, int(np.count_nonzero(positives))))
    fpr = float(fp / max(1, int(np.count_nonzero(negatives))))
    precision = float(tp / max(1, tp + fp))
    recall = tpr
    accuracy = float((tp + tn) / max(1, y_true.size))

    return {
        "auc": auc,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "threshold": float(threshold),
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_pos": int(np.count_nonzero(positives)),
        "n_neg": int(np.count_nonzero(negatives)),
        "n_total": int(y_true.size),
    }


def auc_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n = y_true.size
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        y_sample = y_true[sample_idx]
        if np.unique(y_sample).size < 2:
            continue
        scores.append(float(roc_auc_score(y_sample, y_prob[sample_idx])))
    if not scores:
        return float("nan"), float("nan")
    lower = float(np.quantile(scores, 0.025))
    upper = float(np.quantile(scores, 0.975))
    return lower, upper


def safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

