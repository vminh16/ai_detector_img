from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import (  # noqa: E402
    DEFAULT_CONFIG,
    ALL_FEATURE_KEYS,
    load_feature_manifest,
    results_to_frame,
    run_feature_pipeline,
    summarise_feature_table,
)


AUDIT_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "feature_extraction_v2_core"
SUMMARY_PATH = AUDIT_ROOT / "validation_summary.json"
SEQUENTIAL_CSV = AUDIT_ROOT / "sequential_sample.csv"
PARALLEL_CSV = AUDIT_ROOT / "parallel_sample.csv"
COMPARISON_CSV = AUDIT_ROOT / "sequential_parallel_compare.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed_v4_rgb248_r4_exact" / "manifest.csv"


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(
        right,
        on=["source_file_path", "patch_path", "generator", "label", "split_role"],
        suffixes=("_seq", "_par"),
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for key in ALL_FEATURE_KEYS:
        diff = np.abs(merged[f"{key}_seq"] - merged[f"{key}_par"])
        rows.append(
            {
                "feature": key,
                "max_abs_diff": float(np.nanmax(diff)),
                "mean_abs_diff": float(np.nanmean(diff)),
                "allclose_1e_8": bool(np.allclose(merged[f"{key}_seq"], merged[f"{key}_par"], atol=1e-8, rtol=1e-8, equal_nan=True)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = load_feature_manifest(MANIFEST_PATH, config=DEFAULT_CONFIG, max_files=48)

    sequential_results = run_feature_pipeline(manifest, config=DEFAULT_CONFIG, workers=1, show_progress=False)
    sequential_frame = results_to_frame(sequential_results, config=DEFAULT_CONFIG)
    sequential_frame.to_csv(SEQUENTIAL_CSV, index=False, encoding="utf-8-sig")

    parallel_results = run_feature_pipeline(manifest, config=DEFAULT_CONFIG, workers=2, chunksize=4, show_progress=False)
    parallel_frame = results_to_frame(parallel_results, config=DEFAULT_CONFIG)
    parallel_frame.to_csv(PARALLEL_CSV, index=False, encoding="utf-8-sig")

    comparison = compare_frames(sequential_frame, parallel_frame)
    comparison.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "sample_rows": int(len(manifest)),
        "feature_count": int(len(ALL_FEATURE_KEYS)),
        "sequential_summary": summarise_feature_table(sequential_frame, config=DEFAULT_CONFIG),
        "parallel_summary": summarise_feature_table(parallel_frame, config=DEFAULT_CONFIG),
        "parallel_match_allclose_1e_8": bool(comparison["allclose_1e_8"].all()),
        "max_feature_abs_diff": float(comparison["max_abs_diff"].max()),
        "files": {
            "sequential_csv": str(SEQUENTIAL_CSV),
            "parallel_csv": str(PARALLEL_CSV),
            "comparison_csv": str(COMPARISON_CSV),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
