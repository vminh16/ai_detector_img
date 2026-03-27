"""
src/extract_worker.py
=====================
Unified feature extraction worker for multiprocessing.

Reads one .npy file from disk, feeds it to all 4 feature groups,
and returns a flat dict of 33 features + metadata.

This module exists as a standalone file (not inline in a notebook)
because Windows ProcessPoolExecutor uses 'spawn' start method,
which requires worker functions to be importable from a module.
"""
from __future__ import annotations

import numpy as np

from src.feature_extraction.frequency import (
    FEATURE_KEYS as FREQ_KEYS,
    extract_frequency_features,
)
from src.feature_extraction.color import (
    FEATURE_KEYS as COLOR_KEYS,
    extract_color_features,
)
from src.feature_extraction.microtexture import (
    FEATURE_KEYS as MICRO_KEYS,
    extract_microtexture_features,
)
from src.feature_extraction.spatial import (
    FEATURE_KEYS as SPATIAL_KEYS,
    extract_spatial_features,
)

ALL_FEATURE_KEYS: list[str] = (
    list(FREQ_KEYS) + list(COLOR_KEYS) + list(MICRO_KEYS) + list(SPATIAL_KEYS)
)
"""All 33 feature keys in canonical order."""


def extract_all_features(task: tuple[str, str, str]) -> dict:
    """Process one .npy file through all 4 feature groups.

    Designed to run inside a child process via ProcessPoolExecutor.

    Parameters
    ----------
    task : (npy_path, generator, label)

    Returns
    -------
    dict
        Flat record with: file_path, generator, label,
        33 feature columns, status, error.
    """
    npy_path, generator, label = task
    record: dict = {
        "file_path": npy_path,
        "generator": generator,
        "label": label,
    }
    try:
        # Single disk read — feeds all 4 extractors
        arr = np.load(npy_path, allow_pickle=False)

        # Group 1: Frequency (6 features, guaranteed finite)
        record.update(extract_frequency_features(arr))

        # Group 2: Color (9 features, guaranteed finite)
        record.update(extract_color_features(arr))

        # Group 3: Microtexture (10 features, guaranteed finite)
        record.update(extract_microtexture_features(arr))

        # Group 4: Spatial (8 features, may contain NaN)
        record.update(extract_spatial_features(arr))

        record["status"] = "ok"
        record["error"] = ""

    except Exception as exc:
        for key in ALL_FEATURE_KEYS:
            record[key] = np.nan
        record["status"] = "error"
        record["error"] = str(exc)

    return record
