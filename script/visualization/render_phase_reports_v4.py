from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import (  # noqa: E402
    render_feature_phase_report,
    render_model_phase_report,
    render_preprocessing_report,
)

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_v4_rgb248_r4_exact"
FEATURES_PATH = PROJECT_ROOT / "features" / "features_dataset_v4_rgb248_r4_exact.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "04_artifacts_v4_rgb248_r4_exact"
VIS_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "spec_v4_20260319" / "visualization_v4_rgb248_r4_exact"


def main() -> None:
    render_preprocessing_report(PROCESSED_ROOT / "manifest.csv", VIS_ROOT / "preprocessing", threshold=252)
    render_feature_phase_report(FEATURES_PATH, VIS_ROOT / "feature_extraction")
    render_model_phase_report(MODEL_ROOT, VIS_ROOT / "model_analysis")
    print(f"Phase reports rendered under {VIS_ROOT}")


if __name__ == "__main__":
    main()
