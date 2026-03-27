"""Constants and config for feature extraction v2."""

from __future__ import annotations

from dataclasses import dataclass

PATCH_SIZE = 248
RGB_CHANNELS = 3

MID_BAND_MIN = 0.12
MID_BAND_MAX = 0.32
MID_BAND_SPLIT = 0.22
VALIDITY_HIGH_MIN = 0.32
VALIDITY_HIGH_MAX = 0.46

FFT_TOTAL_MIN = 0.02
FFT_TOTAL_MAX = 0.46
FFT_RADIAL_BINS = 24

LOCAL_WINDOW = 9
HETERO_RESIDUAL_WINDOW = 7
MASK_MIN_PIXELS = 64
EPS = 1e-8

EDGE3_KERNEL = (
    (0.0, -1.0, 0.0),
    (-1.0, 4.0, -1.0),
    (0.0, -1.0, 0.0),
)
SQUARE3_KERNEL = (
    (-1.0, 2.0, -1.0),
    (2.0, -4.0, 2.0),
    (-1.0, 2.0, -1.0),
)
SQUARE5_KERNEL = (
    (0.0, 0.0, 0.25, 0.0, 0.0),
    (0.0, 0.0, -0.5, 0.0, 0.0),
    (0.25, -0.5, 1.0, -0.5, 0.25),
    (0.0, 0.0, -0.5, 0.0, 0.0),
    (0.0, 0.0, 0.25, 0.0, 0.0),
)
SOBEL_X = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)
SOBEL_Y = (
    (-1.0, -2.0, -1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 1.0),
)

FEATURE_VERSION = "v2_rgb248_exact_multibranch"
DATASET_NAME = "GenImage"
PREPROCESS_VERSION_EXPECTED = "v4_rgb248_r4_exact"

ID_GENERATORS = ("ADM", "Midjourney", "SDv14", "VQDM", "Wukong")
OOD_GENERATORS = ("GLIDE", "SDv15")
SPLIT_SEED = 42
ID_TEST_FRACTION = 0.10
VAL_FRACTION = 0.10
CALIBRATION_FRACTION_OF_REMAINDER = 0.05

CONTROL_COLOR_KEYS = (
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
)
CONTROL_SPATIAL_KEYS = (
    "spatial_snr_ratio",
    "skew_noise_y",
    "kurt_noise_y",
)
CONTROL_FREQUENCY_KEYS = ("frs_mid_variance",)
FFT_MIDBAND_KEYS = (
    "fft_mid_logenergy",
    "fft_mid_flatness",
    "fft_mid_ring_var",
    "fft_mid_inner_outer_ratio",
    "fft_mid_anisotropy_hv",
    "fft_mid_anisotropy_diag",
)
CONDITIONAL_CFA_KEYS = (
    "cfa_rg_pi_xy",
    "cfa_bg_pi_xy",
    "cfa_rgb_pi_xy_mean",
    "cfa_rgb_pi_xy_gap",
)
VALIDITY_KEYS = ("cfa_validity_score",)
WAVELET_KEYS = (
    "wav_energy_l1",
    "wav_energy_l2",
    "wav_energy_l3",
    "wav_decay_alpha",
    "wav_ratio_l1_l2",
    "wav_ratio_l2_l3",
)
HETERO_KEYS = (
    "lochet_dark_flat_slope",
    "lochet_dark_flat_r2",
    "lochet_dark_flat_cv",
    "lochet_dark_edge_flat_logratio",
    "lochet_dark_monotone_violation",
)
YSRM_KEYS = (
    "ysrm_midtex_edge3_energy",
    "ysrm_midtex_edge3_mar",
    "ysrm_midtex_square3_energy",
    "ysrm_midtex_square3_mar",
    "ysrm_midtex_square5_energy",
    "ysrm_midtex_square5_mar",
)

ALWAYS_ON_KEYS = CONTROL_COLOR_KEYS + CONTROL_SPATIAL_KEYS + CONTROL_FREQUENCY_KEYS + FFT_MIDBAND_KEYS
RESEARCH_KEYS = WAVELET_KEYS + HETERO_KEYS + YSRM_KEYS


@dataclass(frozen=True)
class FeatureExtractionConfig:
    feature_version: str = FEATURE_VERSION
    dataset_name: str = DATASET_NAME
    preprocess_version_expected: str = PREPROCESS_VERSION_EXPECTED
    patch_size: int = PATCH_SIZE
    include_conditional: bool = True
    include_research: bool = True
    split_seed: int = SPLIT_SEED
    id_generators: tuple[str, ...] = ID_GENERATORS
    ood_generators: tuple[str, ...] = OOD_GENERATORS
    id_test_fraction: float = ID_TEST_FRACTION
    val_fraction: float = VAL_FRACTION
    calibration_fraction_of_remainder: float = CALIBRATION_FRACTION_OF_REMAINDER


DEFAULT_CONFIG = FeatureExtractionConfig()


def feature_keys(config: FeatureExtractionConfig = DEFAULT_CONFIG) -> tuple[str, ...]:
    keys: list[str] = list(ALWAYS_ON_KEYS)
    if config.include_conditional:
        keys.extend(CONDITIONAL_CFA_KEYS)
        keys.extend(VALIDITY_KEYS)
    if config.include_research:
        keys.extend(RESEARCH_KEYS)
    return tuple(keys)
