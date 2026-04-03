from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_training_baseline, run_training_with_feature_audit


def main() -> None:
    default_run_name = f"training_v2_two_phase_{datetime.now().strftime('%Y%m%d')}"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("features/feature_extraction_v2_rgb248_exact.csv"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=default_run_name,
    )
    parser.add_argument(
        "--audit-output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("audit_output/data_audit/metadata/per_file_metadata.csv"),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
    )
    args = parser.parse_args()

    audit_output_root = (
        args.audit_output_root
        if args.audit_output_root is not None
        else Path("audit_output/validation") / args.run_name
    )
    model_output_dir = (
        args.model_output_dir
        if args.model_output_dir is not None
        else Path("models/param") / args.run_name
    )

    feature_table_path = (PROJECT_ROOT / args.feature_table).resolve() if not args.feature_table.is_absolute() else args.feature_table.resolve()
    metadata_csv_path = (PROJECT_ROOT / args.metadata_csv).resolve() if not args.metadata_csv.is_absolute() else args.metadata_csv.resolve()
    audit_output_root = (PROJECT_ROOT / audit_output_root).resolve() if not audit_output_root.is_absolute() else audit_output_root.resolve()
    model_output_dir = (PROJECT_ROOT / model_output_dir).resolve() if not model_output_dir.is_absolute() else model_output_dir.resolve()

    if args.phase2_only:
        summary = run_training_baseline(
            feature_table_path,
            output_dir=audit_output_root / "phase2_training_eval",
            model_output_dir=model_output_dir,
        )
    else:
        summary = run_training_with_feature_audit(
            feature_table_path,
            audit_output_root=audit_output_root,
            metadata_csv_path=metadata_csv_path,
            model_output_dir=model_output_dir,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
