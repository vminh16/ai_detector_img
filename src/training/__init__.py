"""Public API for training baseline v2."""

from .constants import (
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    CFA_GATED_KEYS,
    CFA_VALIDITY_QUANTILE,
    CONTROL_MINIMAL_KEYS,
    ECE_BINS,
    FEATURE_SET_COLUMNS,
    MODEL_SPECS,
    TARGET_FPR,
)
from .metrics import auc_bootstrap_ci, expected_calibration_error, safe_logit, split_metrics, threshold_at_target_fpr
from .pipeline import (
    add_conditional_cfa_gates,
    evaluate_selected_by_generator,
    evaluate_selected_candidate,
    load_training_table,
    run_training_baseline,
    split_frames,
    train_candidates,
)

__all__ = [
    "BOOTSTRAP_ROUNDS",
    "BOOTSTRAP_SEED",
    "CFA_GATED_KEYS",
    "CFA_VALIDITY_QUANTILE",
    "CONTROL_MINIMAL_KEYS",
    "ECE_BINS",
    "FEATURE_SET_COLUMNS",
    "MODEL_SPECS",
    "TARGET_FPR",
    "add_conditional_cfa_gates",
    "auc_bootstrap_ci",
    "evaluate_selected_by_generator",
    "evaluate_selected_candidate",
    "expected_calibration_error",
    "load_training_table",
    "run_training_baseline",
    "safe_logit",
    "split_frames",
    "split_metrics",
    "threshold_at_target_fpr",
    "train_candidates",
]
