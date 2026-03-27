from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    CROP_SIZE,
    DEFAULT_CONFIG,
    RESIDUE_X,
    RESIDUE_Y,
    SUPPORT_THRESHOLD,
    apply_orientation,
    apply_orientation_pil,
    build_output_path,
    preprocess_one,
    read_orientation,
    run_pipeline,
    save_manifest,
)

AUDIT_ROOT = PROJECT_ROOT / "audit_output" / "validation" / "spec_v4_20260319" / "implementation_checks"
SUMMARY_JSON = AUDIT_ROOT / "preprocessing_v4_validation_summary.json"
CASE_CSV = AUDIT_ROOT / "preprocessing_v4_case_results.csv"
TEMP_OUTPUT_ROOT = AUDIT_ROOT / "tmp_outputs"
FIXTURE_ROOT = AUDIT_ROOT / "batch_fixture"
FIXTURE_RAW_ROOT = FIXTURE_ROOT / "raw"
FIXTURE_PROCESSED_ROOT = FIXTURE_ROOT / "processed"
FIXTURE_MANIFEST = FIXTURE_ROOT / "manifest.csv"
METADATA_PARQUET = PROJECT_ROOT / "audit_output" / "data_audit" / "metadata" / "per_file_metadata.parquet"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
SAMPLE_PER_GROUP = 5
SEED = 42


def ensure_dirs() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    if TEMP_OUTPUT_ROOT.exists():
        shutil.rmtree(TEMP_OUTPUT_ROOT)
    TEMP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def live_raw_set() -> set[str]:
    return {
        str(path.relative_to(RAW_ROOT)).replace("\\", "/")
        for path in RAW_ROOT.rglob("*")
        if path.is_file()
    }


def load_current_metadata() -> pd.DataFrame:
    raw_set = live_raw_set()
    meta = pd.read_parquet(METADATA_PARQUET).copy()
    meta["relative_path"] = meta["relative_path"].str.replace("\\", "/", regex=False)
    meta = meta[meta["relative_path"].isin(raw_set)].copy()
    meta["raw_path"] = meta["relative_path"].map(lambda rel: str((RAW_ROOT / rel).resolve()))
    meta["support"] = meta[["width", "height"]].min(axis=1)
    return meta


def sample_group(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    take = min(n, len(df))
    return df.sample(n=take, random_state=SEED).copy()


def series_all_true(case_df: pd.DataFrame, column: str) -> bool:
    if column not in case_df:
        return True
    series = case_df[column]
    if series.empty:
        return True
    return bool(series.astype("boolean").fillna(True).all())


def manual_composite(rgba: np.ndarray, background_value: int = 128) -> np.ndarray:
    rgba_u16 = rgba.astype(np.uint16, copy=False)
    rgb = rgba_u16[..., :3]
    alpha = rgba_u16[..., 3:4]
    bg = np.uint16(background_value)
    return (((alpha * rgb) + ((255 - alpha) * bg) + 127) // 255).astype(np.uint8)


def manual_normalize(image: Image.Image) -> np.ndarray:
    if image.mode == "RGB":
        return np.array(image, dtype=np.uint8, copy=True)
    if image.mode == "RGBA":
        rgba = np.array(image, dtype=np.uint8, copy=True)
        return manual_composite(rgba)
    raise ValueError(f"unsupported mode for manual validation: {image.mode}")


def independent_expected_patch(path: Path, x0: int, y0: int) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        image = image.copy()
    image = apply_orientation_pil(image, read_orientation(path))
    rgb8 = manual_normalize(image)
    patch = rgb8[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE, :]
    return np.ascontiguousarray(patch)


def validate_case(path: Path, expected_status: str) -> dict[str, object]:
    result = preprocess_one(
        path,
        config=DEFAULT_CONFIG,
        input_root=RAW_ROOT,
        output_root=TEMP_OUTPUT_ROOT,
        save_patch=False,
    )
    row: dict[str, object] = {
        "file_path": str(path),
        "expected_status": expected_status,
        "actual_status": result.status.value,
        "input_format": result.input_format,
        "input_mode": result.input_mode,
        "width": result.width,
        "height": result.height,
        "support": result.support,
        "crop_origin_x": result.crop_origin_x,
        "crop_origin_y": result.crop_origin_y,
        "patch_shape": result.patch_shape,
        "saved_patch": result.saved_patch,
        "stale_output_removed": result.stale_output_removed,
        "error": result.error,
        "status_match": result.status.value == expected_status,
    }
    if expected_status == "ACCEPTED":
        expected_patch = independent_expected_patch(
            path,
            result.crop_origin_x,
            result.crop_origin_y,
        )
        row["patch_matches_exact_slice"] = bool(np.array_equal(result.patch, expected_patch))
        row["patch_dtype_ok"] = bool(result.patch is not None and result.patch.dtype == np.uint8)
        row["patch_shape_ok"] = bool(result.patch is not None and result.patch.shape == (CROP_SIZE, CROP_SIZE, 3))
        row["residue_x_ok"] = bool(result.crop_origin_x % 8 == RESIDUE_X)
        row["residue_y_ok"] = bool(result.crop_origin_y % 8 == RESIDUE_Y)
    else:
        row["patch_is_none"] = result.patch is None
    return row


def orientation_reference_checks() -> dict[str, bool]:
    base = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    pil_base = Image.fromarray(base)
    expected = {
        1: base,
        2: np.fliplr(base),
        3: np.rot90(base, k=2),
        4: np.flipud(base),
        5: np.transpose(base, (1, 0, 2)),
        6: np.rot90(base, k=3),
        7: np.flipud(np.fliplr(np.transpose(base, (1, 0, 2)))),
        8: np.rot90(base, k=1),
    }
    checks: dict[str, bool] = {}
    for orientation, expected_arr in expected.items():
        arr_ok = np.array_equal(apply_orientation(base.copy(), orientation), expected_arr)
        pil_ok = np.array_equal(
            np.array(apply_orientation_pil(pil_base.copy(), orientation)),
            expected_arr,
        )
        checks[f"orientation_{orientation}_ndarray"] = bool(arr_ok)
        checks[f"orientation_{orientation}_pil"] = bool(pil_ok)
    return checks


def alpha_formula_checks() -> dict[str, bool]:
    rgba = np.array(
        [
            [[255, 0, 0, 255], [255, 0, 0, 0]],
            [[0, 255, 0, 128], [0, 0, 255, 64]],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [[255, 0, 0], [128, 128, 128]],
            [[64, 192, 64], [96, 96, 160]],
        ],
        dtype=np.uint8,
    )
    actual = manual_composite(rgba)
    return {"synthetic_alpha_matches_expected": bool(np.array_equal(actual, expected))}


def path_collision_check() -> dict[str, object]:
    jpg_path = RAW_ROOT / "collision_test" / "nature" / "example.jpg"
    png_path = RAW_ROOT / "collision_test" / "nature" / "example.png"
    jpg_out = build_output_path(jpg_path, RAW_ROOT, TEMP_OUTPUT_ROOT)
    png_out = build_output_path(png_path, RAW_ROOT, TEMP_OUTPUT_ROOT)
    return {
        "jpg_output": str(jpg_out),
        "png_output": str(png_out),
        "collision_safe": str(jpg_out) != str(png_out),
    }


def stale_cleanup_check(gray_path: Path) -> dict[str, object]:
    output_path = build_output_path(gray_path, RAW_ROOT, TEMP_OUTPUT_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), np.zeros((2, 2, 3), dtype=np.uint8))
    result = preprocess_one(
        gray_path,
        config=DEFAULT_CONFIG,
        input_root=RAW_ROOT,
        output_root=TEMP_OUTPUT_ROOT,
        save_patch=True,
    )
    return {
        "status": result.status.value,
        "stale_output_removed": result.stale_output_removed,
        "output_exists_after": output_path.exists(),
    }


def build_case_table(meta: pd.DataFrame) -> pd.DataFrame:
    rgb_ok = sample_group(
        meta[
            (meta["image_mode"] == "RGB")
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] >= SUPPORT_THRESHOLD)
        ],
        SAMPLE_PER_GROUP,
    )
    rgba_ok = sample_group(
        meta[
            (meta["image_mode"] == "RGBA")
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] >= SUPPORT_THRESHOLD)
        ],
        SAMPLE_PER_GROUP,
    )
    gray_reject = sample_group(
        meta[
            (meta["image_mode"] == "L")
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
        ],
        SAMPLE_PER_GROUP,
    )
    low_support = sample_group(
        meta[
            (meta["image_mode"].isin(["RGB", "RGBA"]))
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] < SUPPORT_THRESHOLD)
        ],
        SAMPLE_PER_GROUP,
    )

    rows: list[dict[str, object]] = []
    for _, row in rgb_ok.iterrows():
        rows.append(validate_case(Path(row["raw_path"]), "ACCEPTED"))
    for _, row in rgba_ok.iterrows():
        rows.append(validate_case(Path(row["raw_path"]), "ACCEPTED"))
    for _, row in gray_reject.iterrows():
        rows.append(validate_case(Path(row["raw_path"]), "UNSUPPORTED_INPUT"))
    for _, row in low_support.iterrows():
        rows.append(validate_case(Path(row["raw_path"]), "LOW_SUPPORT"))
    return pd.DataFrame(rows)


def first_path(meta: pd.DataFrame, mask: pd.Series) -> Path:
    return Path(meta.loc[mask, "raw_path"].iloc[0])


def batch_smoke_check(meta: pd.DataFrame) -> dict[str, object]:
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    FIXTURE_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    fixture_paths = [
        first_path(
            meta,
            (meta["image_mode"] == "RGB")
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] >= SUPPORT_THRESHOLD),
        ),
        first_path(
            meta,
            (meta["image_mode"] == "RGBA")
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] >= SUPPORT_THRESHOLD),
        ),
        first_path(
            meta,
            (meta["image_mode"] == "L")
            & (meta["format_detected"].isin(["JPEG", "PNG"])),
        ),
        first_path(
            meta,
            (meta["image_mode"].isin(["RGB", "RGBA"]))
            & (meta["format_detected"].isin(["JPEG", "PNG"]))
            & (meta["support"] < SUPPORT_THRESHOLD),
        ),
    ]

    for source in fixture_paths:
        rel = source.relative_to(RAW_ROOT)
        dst = FIXTURE_RAW_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)

    results = run_pipeline(
        FIXTURE_RAW_ROOT,
        FIXTURE_PROCESSED_ROOT,
        config=DEFAULT_CONFIG,
        workers=2,
        overwrite=True,
        show_progress=False,
    )
    save_manifest(results, FIXTURE_MANIFEST)

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1

    output_files = sorted(
        str(path.relative_to(FIXTURE_PROCESSED_ROOT)).replace("\\", "/")
        for path in FIXTURE_PROCESSED_ROOT.rglob("*.npy")
    )
    return {
        "n_results": len(results),
        "status_counts": status_counts,
        "accepted_saved_patch": sum(result.saved_patch for result in results),
        "all_batch_patches_cleared": all(result.patch is None for result in results),
        "output_files": output_files,
        "manifest_exists": FIXTURE_MANIFEST.exists(),
        "manifest_rows": int(len(pd.read_csv(FIXTURE_MANIFEST))),
    }


def main() -> None:
    ensure_dirs()
    meta = load_current_metadata()
    case_df = build_case_table(meta)
    case_df.to_csv(CASE_CSV, index=False, encoding="utf-8-sig")

    gray_sample = meta.loc[meta["image_mode"] == "L", "raw_path"].iloc[0]
    summary = {
        "config": {
            "crop_size": CROP_SIZE,
            "residue": [RESIDUE_X, RESIDUE_Y],
            "support_threshold": SUPPORT_THRESHOLD,
        },
        "case_counts": case_df["expected_status"].value_counts().sort_index().to_dict(),
        "all_status_match": bool(case_df["status_match"].all()),
        "all_exact_slice_checks_pass": series_all_true(case_df, "patch_matches_exact_slice"),
        "all_patch_shape_checks_pass": series_all_true(case_df, "patch_shape_ok"),
        "all_residue_checks_pass": (
            series_all_true(case_df, "residue_x_ok")
            and series_all_true(case_df, "residue_y_ok")
        ),
        "all_reject_patch_checks_pass": series_all_true(case_df, "patch_is_none"),
        "orientation_reference_checks": orientation_reference_checks(),
        "alpha_formula_checks": alpha_formula_checks(),
        "path_collision_check": path_collision_check(),
        "stale_cleanup_check": stale_cleanup_check(Path(gray_sample)),
        "batch_smoke_check": batch_smoke_check(meta),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if TEMP_OUTPUT_ROOT.exists():
        try:
            shutil.rmtree(TEMP_OUTPUT_ROOT)
        except OSError:
            pass


if __name__ == "__main__":
    main()
