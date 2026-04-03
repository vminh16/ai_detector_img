from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_training_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("features/feature_extraction_v2_rgb248_exact.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit_output/validation/training_v2_baseline_20260403"),
    )
    args = parser.parse_args()

    summary = run_training_baseline(
        args.feature_table.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
