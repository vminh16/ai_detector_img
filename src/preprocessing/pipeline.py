"""Preprocessing v4 core pipeline."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import (
    COUNT_EXTENSIONS,
    DEFAULT_CONFIG,
    DISCOVERY_EXTENSIONS,
    SUPPORTED_FORMATS,
    PreprocessConfig,
)
from .decode import decode_image, normalize_mode_to_rgb
from .errors import DecodeImageError, LowSupportError, UnsupportedInputError
from .geometry import crop_exact_residue
from .orientation import apply_orientation, apply_orientation_pil, read_orientation
from .types import MANIFEST_COLUMNS, PreprocessResult, PreprocessStatus

logger = logging.getLogger(__name__)


def discover_images(root: Path | str) -> list[Path]:
    """Recursively collect candidate image files under root."""

    root_path = Path(root)
    found: list[Path] = []
    for dirpath, _, filenames in os.walk(root_path):
        dir_path = Path(dirpath)
        for filename in filenames:
            if Path(filename).suffix.lower() in DISCOVERY_EXTENSIONS:
                found.append(dir_path / filename)
    return sorted(found)


def scan_dataset_tree(root: Path | str) -> dict[str, Any]:
    root_path = Path(root).resolve()
    images = discover_images(root_path)
    by_subdir: dict[str, int] = {}
    for path in images:
        key = str(path.parent.relative_to(root_path))
        by_subdir[key] = by_subdir.get(key, 0) + 1
    return {
        "root": str(root_path),
        "total_images": len(images),
        "by_subdir": dict(sorted(by_subdir.items())),
    }


def count_dataset(root: Path | str) -> list[dict[str, Any]]:
    root_path = Path(root)
    stats: list[dict[str, Any]] = []
    for generator_dir in sorted(root_path.iterdir()):
        if not generator_dir.is_dir():
            continue
        for label_dir in sorted(generator_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            count = sum(
                1
                for file_path in label_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in COUNT_EXTENSIONS
            )
            stats.append(
                {
                    "generator": generator_dir.name,
                    "label": label_dir.name,
                    "count": count,
                }
            )
    return stats


def _infer_label(filepath: Path) -> str:
    parts = [part.lower() for part in filepath.parts]
    if "ai" in parts:
        return "ai"
    if "nature" in parts or "real" in parts:
        return "nature"
    return "unknown"


def _infer_generator(filepath: Path, input_root: Path | None) -> str:
    if input_root is None:
        return filepath.parent.name or "unknown"
    try:
        rel = filepath.relative_to(input_root)
    except ValueError:
        return filepath.parent.name or "unknown"
    return rel.parts[0] if rel.parts else "unknown"


def build_output_path(
    filepath: Path | str,
    input_root: Path | str | None,
    output_root: Path | str,
    *,
    output_ext: str = ".npy",
) -> Path:
    """Build a collision-safe output path by preserving the source extension."""

    source = Path(filepath).resolve()
    output_root_path = Path(output_root).resolve()
    if input_root is not None:
        input_root_path = Path(input_root).resolve()
        try:
            rel = source.relative_to(input_root_path)
        except ValueError:
            rel = Path(source.name)
    else:
        rel = Path(source.name)
    return output_root_path / rel.parent / f"{rel.name}{output_ext}"


def _save_patch(filepath: Path, patch: np.ndarray) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(filepath), patch)


def _remove_stale_output(filepath: Path) -> bool:
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def preprocess_one(
    filepath: Path | str,
    *,
    config: PreprocessConfig = DEFAULT_CONFIG,
    input_root: Path | str | None = None,
    output_root: Path | str | None = None,
    save_patch: bool = True,
    return_patch: bool = True,
) -> PreprocessResult:
    """Run champion v4 preprocessing on a single file."""

    path = Path(filepath).resolve()
    input_root_path = Path(input_root).resolve() if input_root is not None else None
    output_path = ""
    output_target: Path | None = None
    if output_root is not None:
        output_target = build_output_path(
            path,
            input_root_path,
            output_root,
            output_ext=config.output_ext,
        )
        output_path = str(output_target)

    result = PreprocessResult(
        file_path=str(path),
        output_path=output_path,
        generator=_infer_generator(path, input_root_path),
        label=_infer_label(path),
        status=PreprocessStatus.DECODE_ERROR,
        preprocess_version=config.preprocess_version,
        support_threshold=config.support_threshold,
        crop_size=config.crop_size,
        residue_x=config.residue_x,
        residue_y=config.residue_y,
    )

    try:
        orientation = read_orientation(path)
        decoded = decode_image(path)
        result.input_format = decoded.format_name
        result.input_mode = decoded.mode
        result.orientation = orientation
        result.orientation_applied = orientation not in (None, 1)

        if decoded.format_name not in SUPPORTED_FORMATS:
            raise UnsupportedInputError(
                f"decoded format {decoded.format_name or 'UNKNOWN'} is outside JPEG/PNG contract"
            )

        image = apply_orientation_pil(decoded.image, orientation)
        width, height = image.size
        result.width = width
        result.height = height
        result.support = min(height, width)
        normalized = normalize_mode_to_rgb(
            image,
            background_value=config.alpha_background_value,
        )

        height, width = normalized.rgb8.shape[:2]
        result.status = PreprocessStatus.ACCEPTED
        result.normalized_mode = normalized.normalized_mode
        result.alpha_composited = normalized.alpha_composited
        result.width = width
        result.height = height
        result.support = min(height, width)

        if result.support < config.support_threshold:
            raise LowSupportError(
                f"support={result.support} is below threshold={config.support_threshold}"
            )

        patch, origin = crop_exact_residue(
            normalized.rgb8,
            crop_size=config.crop_size,
            residue_x=config.residue_x,
            residue_y=config.residue_y,
        )
        result.crop_origin_x = origin.x
        result.crop_origin_y = origin.y
        result.patch_shape = f"{patch.shape[0]}x{patch.shape[1]}x{patch.shape[2]}"
        result.patch_dtype = str(patch.dtype)

        if output_target is not None and save_patch:
            _save_patch(output_target, patch)
            result.saved_patch = True
        if return_patch:
            result.patch = patch
        return result

    except LowSupportError as exc:
        result.status = PreprocessStatus.LOW_SUPPORT
        result.error = str(exc)
    except UnsupportedInputError as exc:
        result.status = PreprocessStatus.UNSUPPORTED_INPUT
        result.error = str(exc)
    except DecodeImageError as exc:
        result.status = PreprocessStatus.DECODE_ERROR
        result.error = str(exc)
    except Exception as exc:
        result.status = PreprocessStatus.DECODE_ERROR
        result.error = f"{type(exc).__name__}: {exc}"

    if output_target is not None:
        result.stale_output_removed = _remove_stale_output(output_target)
    return result


def results_to_frame(results: Iterable[PreprocessResult]) -> pd.DataFrame:
    rows = [result.manifest_row() for result in results]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def save_manifest(results: Iterable[PreprocessResult], path: Path | str) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(results).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path.resolve()


def summarise_results(results: Iterable[PreprocessResult]) -> dict[str, int]:
    result_list = list(results)
    return {
        "total_scanned": len(result_list),
        "accepted": sum(result.accepted for result in result_list),
        "low_support": sum(result.status == PreprocessStatus.LOW_SUPPORT for result in result_list),
        "unsupported_input": sum(result.status == PreprocessStatus.UNSUPPORTED_INPUT for result in result_list),
        "decode_error": sum(result.status == PreprocessStatus.DECODE_ERROR for result in result_list),
        "saved_patch": sum(result.saved_patch for result in result_list),
        "stale_output_removed": sum(result.stale_output_removed for result in result_list),
        "orientation_applied": sum(result.orientation_applied for result in result_list),
        "alpha_composited": sum(result.alpha_composited for result in result_list),
    }


def print_summary(summary: dict[str, int], output_root: Path | str = "") -> None:
    logger.info("=" * 60)
    logger.info("PREPROCESSING V4 COMPLETE")
    for key, value in summary.items():
        logger.info("  %-24s: %s", key, value)
    if output_root:
        logger.info("  %-24s: %s", "output_root", output_root)
    logger.info("=" * 60)


def run_pipeline(
    input_root: Path | str,
    output_root: Path | str,
    *,
    config: PreprocessConfig = DEFAULT_CONFIG,
    workers: int = 8,
    overwrite: bool = True,
    show_progress: bool = True,
    log_failures: bool = True,
) -> list[PreprocessResult]:
    """Run champion v4 preprocessing on every discovered image."""

    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    if input_root_path == output_root_path:
        raise ValueError("input_root and output_root must differ.")

    output_root_path.mkdir(parents=True, exist_ok=True)
    images = discover_images(input_root_path)
    logger.info("Discovered %d candidate files under %s", len(images), input_root_path)
    if not images:
        return []

    results: list[PreprocessResult] = []
    t0 = time.perf_counter()

    def submit_path(path: Path) -> PreprocessResult:
        return preprocess_one(
            path,
            config=config,
            input_root=input_root_path,
            output_root=output_root_path,
            save_patch=True,
            return_patch=False,
        )

    skipped: list[PreprocessResult] = []
    todo: list[Path] = []
    for path in images:
        output_path = build_output_path(
            path,
            input_root_path,
            output_root_path,
            output_ext=config.output_ext,
        )
        if overwrite or not output_path.exists():
            todo.append(path)
            continue
        skipped.append(
            PreprocessResult(
                file_path=str(path.resolve()),
                output_path=str(output_path),
                generator=_infer_generator(path, input_root_path),
                label=_infer_label(path),
                status=PreprocessStatus.ACCEPTED,
                preprocess_version=config.preprocess_version,
                error="skipped_existing_output",
            )
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(submit_path, path): path for path in todo}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Preprocess v4",
                unit="img",
                dynamic_ncols=True,
            )
        for future in iterator:
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                output_path = build_output_path(
                    path,
                    input_root_path,
                    output_root_path,
                    output_ext=config.output_ext,
                )
                result = PreprocessResult(
                    file_path=str(path.resolve()),
                    output_path=str(output_path),
                    generator=_infer_generator(path, input_root_path),
                    label=_infer_label(path),
                    status=PreprocessStatus.DECODE_ERROR,
                    preprocess_version=config.preprocess_version,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            if log_failures and not result.accepted:
                logger.warning("%s -> %s (%s)", path.name, result.status.value, result.error)

    elapsed = time.perf_counter() - t0
    logger.info("Preprocessing v4 finished in %.1f s", elapsed)
    return skipped + results


__all__ = [
    "apply_orientation",
    "build_output_path",
    "count_dataset",
    "discover_images",
    "preprocess_one",
    "print_summary",
    "read_orientation",
    "results_to_frame",
    "run_pipeline",
    "save_manifest",
    "scan_dataset_tree",
    "summarise_results",
]
