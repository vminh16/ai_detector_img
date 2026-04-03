"""Degradation suite and nuisance-label helpers for training v2."""

from __future__ import annotations

import __main__
import io
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from tqdm import tqdm

from src.feature_extraction import ALL_FEATURE_KEYS, DEFAULT_CONFIG, extract_feature_vector

from .constants import DEGRADATION_SPEC_MAP, JPEG_SUBSAMPLING_LABELS

BASE_METADATA_COLUMNS = (
    "source_file_path",
    "patch_path",
    "generator",
    "label",
    "y",
    "split_role",
    "dataset_name",
    "preprocess_version",
    "feature_version",
)


def _ensure_uint8_rgb_patch(patch: np.ndarray) -> np.ndarray:
    array = np.asarray(patch)
    if array.dtype != np.uint8:
        raise TypeError(f"Expected uint8 patch, got {array.dtype!r}")
    if array.shape != (248, 248, 3):
        raise ValueError(f"Expected patch shape (248, 248, 3), got {array.shape!r}")
    return array


def _pil_rgb(patch: np.ndarray) -> Image.Image:
    return Image.fromarray(_ensure_uint8_rgb_patch(patch), mode="RGB")


def _jpeg_roundtrip(patch: np.ndarray, *, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    image = _pil_rgb(patch)
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return np.asarray(decoded, dtype=np.uint8)


def _resize_roundtrip(patch: np.ndarray, *, scale: float) -> np.ndarray:
    image = _pil_rgb(patch)
    target = max(1, int(round(image.width * float(scale))))
    down = image.resize((target, target), resample=Image.Resampling.BILINEAR)
    up = down.resize((image.width, image.height), resample=Image.Resampling.BILINEAR)
    return np.asarray(up, dtype=np.uint8)


def apply_degradation(patch: np.ndarray, degradation_name: str) -> np.ndarray:
    name = str(degradation_name)
    if name not in DEGRADATION_SPEC_MAP:
        raise KeyError(f"Unknown degradation_name={name!r}")
    patch = _ensure_uint8_rgb_patch(patch)
    if name == "jpeg95_420":
        return _jpeg_roundtrip(patch, quality=95)
    if name == "jpeg90_420":
        return _jpeg_roundtrip(patch, quality=90)
    if name == "resize75_bilinear":
        return _resize_roundtrip(patch, scale=0.75)
    if name == "resize50_bilinear":
        return _resize_roundtrip(patch, scale=0.50)
    if name == "resize50_jpeg90_420":
        resized = _resize_roundtrip(patch, scale=0.50)
        return _jpeg_roundtrip(resized, quality=90)
    raise KeyError(f"Unsupported degradation_name={name!r}")


@lru_cache(maxsize=131072)
def detect_jpeg_subsampling(source_path: str) -> str | None:
    path = Path(source_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            if str(image.format).upper() != "JPEG":
                return None
            code = JpegImagePlugin.get_sampling(image)
    except Exception:
        return None
    return JPEG_SUBSAMPLING_LABELS.get(int(code), "unknown")


def annotate_real_jpeg_subsampling(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    values = [detect_jpeg_subsampling(str(path)) for path in annotated["source_file_path"]]
    annotated["jpeg_subsampling_live"] = values
    return annotated


def build_nuisance_eval_frame(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_real_jpeg_subsampling(frame)
    real_only = annotated["y"].eq(0)
    valid = real_only & annotated["jpeg_subsampling_live"].isin(["4:4:4", "4:2:0"])
    subset = annotated.loc[valid].copy()
    subset["nuisance_is_420"] = subset["jpeg_subsampling_live"].eq("4:2:0").astype(np.int32)
    return subset.reset_index(drop=True)


def _process_pool_safe() -> bool:
    main_file = getattr(__main__, "__file__", "")
    if not main_file:
        return False
    return Path(main_file).name != "<stdin>"


def _degradation_task_from_row(row_id: int, row: pd.Series, degradation_name: str) -> tuple[Any, ...]:
    return (
        int(row_id),
        str(row["source_file_path"]),
        str(row["patch_path"]),
        str(row["generator"]),
        str(row["label"]),
        int(row["y"]),
        str(row["split_role"]),
        str(row["dataset_name"]),
        str(row["preprocess_version"]),
        str(row["feature_version"]),
        str(degradation_name),
    )


def build_degradation_tasks(frame: pd.DataFrame, *, degradation_name: str) -> list[tuple[Any, ...]]:
    tasks: list[tuple[Any, ...]] = []
    for idx, row in enumerate(frame.itertuples(index=False)):
        tasks.append(
            (
                int(idx),
                str(row.source_file_path),
                str(row.patch_path),
                str(row.generator),
                str(row.label),
                int(row.y),
                str(row.split_role),
                str(row.dataset_name),
                str(row.preprocess_version),
                str(row.feature_version),
                str(degradation_name),
            )
        )
    return tasks


def _extract_degraded_row(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        row_id,
        source_file_path,
        patch_path,
        generator,
        label,
        y,
        split_role,
        dataset_name,
        preprocess_version,
        feature_version,
        degradation_name,
    ) = task
    try:
        clean_patch = np.load(Path(patch_path), allow_pickle=False)
        degraded_patch = apply_degradation(clean_patch, degradation_name)
        features = extract_feature_vector(degraded_patch, config=DEFAULT_CONFIG)
        row = {
            "row_id": int(row_id),
            "source_file_path": source_file_path,
            "patch_path": patch_path,
            "generator": generator,
            "label": label,
            "y": int(y),
            "split_role": split_role,
            "dataset_name": dataset_name,
            "preprocess_version": preprocess_version,
            "feature_version": feature_version,
            "degradation_name": degradation_name,
            "status": "ok",
            "error": "",
        }
        row.update(features)
        return row
    except Exception as exc:
        row = {
            "row_id": int(row_id),
            "source_file_path": source_file_path,
            "patch_path": patch_path,
            "generator": generator,
            "label": label,
            "y": int(y),
            "split_role": split_role,
            "dataset_name": dataset_name,
            "preprocess_version": preprocess_version,
            "feature_version": feature_version,
            "degradation_name": degradation_name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        row.update({name: np.nan for name in ALL_FEATURE_KEYS})
        return row


def extract_degraded_feature_frame(
    frame: pd.DataFrame,
    *,
    degradation_name: str,
    workers: int = 1,
    chunksize: int = 32,
    show_progress: bool = True,
) -> pd.DataFrame:
    tasks = build_degradation_tasks(frame, degradation_name=degradation_name)
    rows: list[dict[str, Any]] = []
    if workers <= 1 or not _process_pool_safe():
        iterator = (_extract_degraded_row(task) for task in tasks)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(tasks),
                desc=f"Degrade {degradation_name}",
                unit="img",
                dynamic_ncols=True,
            )
        rows.extend(iterator)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_extract_degraded_row, tasks, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"Degrade {degradation_name}",
                    unit="img",
                    dynamic_ncols=True,
                )
            rows.extend(iterator)
    degraded = pd.DataFrame(rows).sort_values("row_id", ignore_index=True)
    return degraded.drop(columns=["row_id"])
