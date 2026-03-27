"""Shared numerical helpers for feature extraction v2."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d
from scipy.signal.windows import tukey

from .constants import EPS, PATCH_SIZE, SOBEL_X, SOBEL_Y


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.ravel(a).astype(np.float64)
    b_flat = np.ravel(b).astype(np.float64)
    std_a = float(np.std(a_flat))
    std_b = float(np.std(b_flat))
    if std_a < EPS or std_b < EPS:
        return 0.0
    cov = float(np.mean((a_flat - np.mean(a_flat)) * (b_flat - np.mean(b_flat))))
    return cov / (std_a * std_b + EPS)


def safe_skew_kurt(values: np.ndarray, *, min_count: int = 32) -> tuple[float, float]:
    x = np.ravel(values).astype(np.float64)
    if x.size < min_count:
        return 0.0, 0.0
    mu = float(np.mean(x))
    diff = x - mu
    var = float(np.mean(diff ** 2))
    sigma = np.sqrt(var)
    if sigma < EPS:
        return 0.0, 0.0
    m3 = float(np.mean(diff ** 3))
    m4 = float(np.mean(diff ** 4))
    skew = m3 / (sigma ** 3 + EPS)
    kurt = m4 / (sigma ** 4 + EPS) - 3.0
    return skew, kurt


def linear_fit_with_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x64 = np.ravel(x).astype(np.float64)
    y64 = np.ravel(y).astype(np.float64)
    if x64.size < 2 or float(np.std(x64)) < EPS or float(np.std(y64)) < EPS:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x64, y64, 1)
    fitted = slope * x64 + intercept
    ss_res = float(np.sum((y64 - fitted) ** 2))
    ss_tot = float(np.sum((y64 - np.mean(y64)) ** 2))
    r2 = 0.0 if ss_tot < EPS else max(0.0, 1.0 - ss_res / (ss_tot + EPS))
    return float(slope), float(r2)


def monotone_violation_rate(x: np.ndarray, y: np.ndarray, *, bins: int = 8) -> float:
    x64 = np.ravel(x).astype(np.float64)
    y64 = np.ravel(y).astype(np.float64)
    if x64.size < max(16, bins * 2):
        return 0.0
    edges = np.quantile(x64, np.linspace(0.0, 1.0, bins + 1))
    summaries: list[float] = []
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (x64 >= left) & (x64 <= right)
        else:
            mask = (x64 >= left) & (x64 < right)
        if int(np.count_nonzero(mask)) == 0:
            continue
        summaries.append(float(np.mean(y64[mask])))
    if len(summaries) < 2:
        return 0.0
    diffs = np.diff(np.asarray(summaries, dtype=np.float64))
    return float(np.mean(diffs < 0.0))


def center_crop_to_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    h, w = array.shape[:2]
    if target_h > h or target_w > w:
        raise ValueError(f"target shape {shape} exceeds array shape {array.shape}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return array[top : top + target_h, left : left + target_w]


def box_mean_valid(image: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    ii = np.pad(x, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
    return summed / float(window * window)


def box_var_valid(image: np.ndarray, window: int) -> np.ndarray:
    mean = box_mean_valid(image, window)
    mean_sq = box_mean_valid(np.asarray(image, dtype=np.float64) ** 2, window)
    return np.maximum(mean_sq - mean ** 2, 0.0)


def trimmed_mean(values: np.ndarray, *, trim_fraction: float = 0.05) -> float:
    x = np.sort(np.ravel(values).astype(np.float64))
    if x.size == 0:
        return 0.0
    cut = int(x.size * trim_fraction)
    if cut * 2 >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[cut : x.size - cut]))


def convolve_valid(image: np.ndarray, kernel: tuple[tuple[float, ...], ...] | np.ndarray) -> np.ndarray:
    return convolve2d(np.asarray(image, dtype=np.float32), np.asarray(kernel, dtype=np.float32), mode="valid")


def sobel_magnitude_valid(image: np.ndarray) -> np.ndarray:
    gx = convolve_valid(image, SOBEL_X)
    gy = convolve_valid(image, SOBEL_Y)
    return np.sqrt(gx ** 2 + gy ** 2, dtype=np.float32)


@lru_cache(maxsize=1)
def tukey_window_2d() -> np.ndarray:
    win = tukey(PATCH_SIZE, alpha=0.25).astype(np.float32)
    return np.outer(win, win).astype(np.float32)
