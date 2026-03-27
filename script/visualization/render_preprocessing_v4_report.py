from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import render_preprocessing_report  # noqa: E402

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_v4_rgb248_r4_exact"
MANIFEST_PATH = PROCESSED_ROOT / "manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "spec_v4_20260319" / "visualization_v4_rgb248_r4_exact" / "preprocessing"


def main() -> None:
    summary = render_preprocessing_report(MANIFEST_PATH, OUTPUT_ROOT, threshold=252)
    print(f"Preprocessing report rendered -> {OUTPUT_ROOT}")
    print(summary)


if __name__ == "__main__":
    main()
