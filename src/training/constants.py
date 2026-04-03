"""Constants and feature-set definitions for training v2."""

from __future__ import annotations

from dataclasses import dataclass

from src.feature_extraction import (
    ALL_FEATURE_KEYS,
    ALWAYS_ON_KEYS,
    CONDITIONAL_CFA_KEYS,
    CONTROL_COLOR_KEYS,
    CONTROL_FREQUENCY_KEYS,
    CONTROL_SPATIAL_KEYS,
    FFT_MIDBAND_KEYS,
    HETERO_KEYS,
    VALIDITY_KEYS,
    WAVELET_KEYS,
    YSRM_KEYS,
)

CONTROL_MINIMAL_KEYS = CONTROL_COLOR_KEYS + CONTROL_SPATIAL_KEYS + CONTROL_FREQUENCY_KEYS
CFA_GATED_KEYS = tuple(f"{name}_gated" for name in CONDITIONAL_CFA_KEYS)

FEATURE_SET_COLUMNS: dict[str, tuple[str, ...]] = {
    "control_minimal": CONTROL_MINIMAL_KEYS,
    "always_on": ALWAYS_ON_KEYS,
    "always_on_plus_cfa_raw": ALWAYS_ON_KEYS + CONDITIONAL_CFA_KEYS + VALIDITY_KEYS,
    "always_on_plus_cfa_gated": ALWAYS_ON_KEYS + CFA_GATED_KEYS + VALIDITY_KEYS,
    "full_v2": tuple(ALL_FEATURE_KEYS),
}

FEATURE_FAMILY_MAP: dict[str, str] = {}
for name in CONTROL_COLOR_KEYS:
    FEATURE_FAMILY_MAP[name] = "control_color"
for name in CONTROL_SPATIAL_KEYS:
    FEATURE_FAMILY_MAP[name] = "control_spatial"
for name in CONTROL_FREQUENCY_KEYS:
    FEATURE_FAMILY_MAP[name] = "control_frequency"
for name in FFT_MIDBAND_KEYS:
    FEATURE_FAMILY_MAP[name] = "fft_midband"
for name in CONDITIONAL_CFA_KEYS + VALIDITY_KEYS:
    FEATURE_FAMILY_MAP[name] = "conditional_cfa"
for name in WAVELET_KEYS:
    FEATURE_FAMILY_MAP[name] = "wavelet_decay"
for name in HETERO_KEYS:
    FEATURE_FAMILY_MAP[name] = "dark_textured_hetero"
for name in YSRM_KEYS:
    FEATURE_FAMILY_MAP[name] = "content_adaptive_y_srm"

TARGET_FPR = 0.05
ECE_BINS = 15
BOOTSTRAP_ROUNDS = 300
BOOTSTRAP_SEED = 42
CFA_VALIDITY_QUANTILE = 0.75


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str


MODEL_SPECS = (
    ModelSpec(name="logreg", family="linear"),
    ModelSpec(name="lightgbm", family="tree"),
)
