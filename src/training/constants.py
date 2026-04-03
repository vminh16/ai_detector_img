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

ABLATION_FEATURE_SET_COLUMNS: dict[str, tuple[str, ...]] = {
    "always_on": ALWAYS_ON_KEYS,
    "always_on_plus_cfa_gated": ALWAYS_ON_KEYS + CFA_GATED_KEYS + VALIDITY_KEYS,
    "always_on_plus_wavelet": ALWAYS_ON_KEYS + WAVELET_KEYS,
    "always_on_plus_ysrm": ALWAYS_ON_KEYS + YSRM_KEYS,
    "full_v2": tuple(ALL_FEATURE_KEYS),
    "full_v2_minus_conditional_cfa": tuple(
        key for key in ALL_FEATURE_KEYS if key not in set(CONDITIONAL_CFA_KEYS + VALIDITY_KEYS)
    ),
    "full_v2_minus_wavelet_decay": tuple(key for key in ALL_FEATURE_KEYS if key not in set(WAVELET_KEYS)),
    "full_v2_minus_content_adaptive_y_srm": tuple(key for key in ALL_FEATURE_KEYS if key not in set(YSRM_KEYS)),
    "full_v2_minus_dark_textured_hetero": tuple(key for key in ALL_FEATURE_KEYS if key not in set(HETERO_KEYS)),
}

FEATURE_FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "control_color": CONTROL_COLOR_KEYS,
    "control_spatial": CONTROL_SPATIAL_KEYS,
    "control_frequency": CONTROL_FREQUENCY_KEYS,
    "fft_midband": FFT_MIDBAND_KEYS,
    "conditional_cfa": CONDITIONAL_CFA_KEYS + VALIDITY_KEYS,
    "wavelet_decay": WAVELET_KEYS,
    "dark_textured_hetero": HETERO_KEYS,
    "content_adaptive_y_srm": YSRM_KEYS,
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

JPEG_SUBSAMPLING_LABELS = {
    0: "4:4:4",
    1: "4:2:2",
    2: "4:2:0",
}

DEGRADATION_EVAL_SPLITS = ("val", "id_test", "ood_eval")


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    description: str


DEGRADATION_SPECS = (
    DegradationSpec(name="jpeg95_420", description="JPEG quality 95, subsampling 4:2:0"),
    DegradationSpec(name="jpeg90_420", description="JPEG quality 90, subsampling 4:2:0"),
    DegradationSpec(
        name="resize75_bilinear",
        description="Resize to 75% with bilinear interpolation, then restore to 248x248",
    ),
    DegradationSpec(
        name="resize50_bilinear",
        description="Resize to 50% with bilinear interpolation, then restore to 248x248",
    ),
    DegradationSpec(
        name="resize50_jpeg90_420",
        description="Resize to 50% and restore with bilinear interpolation, then JPEG quality 90 4:2:0",
    ),
)
DEGRADATION_SPEC_MAP = {spec.name: spec for spec in DEGRADATION_SPECS}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str


MODEL_SPECS = (
    ModelSpec(name="logreg", family="linear"),
    ModelSpec(name="lightgbm", family="tree"),
)
