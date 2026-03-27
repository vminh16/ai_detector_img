"""
src/dataset_tools/strip_metadata.py
=====================
Strip **all** metadata from images and correct EXIF orientation.

Uses **OpenCV as the sole image backend** — no PIL/Pillow decode or encode.
EXIF orientation is read from raw file bytes via the ``struct`` module so
that no second image library is involved.

Pipeline (per file)
-------------------
1. Parse EXIF orientation tag from file header bytes.
2. Decode pixels with ``cv2.imread`` (BGR).
3. Apply orientation transform (rotate / flip / transpose) if needed.
4. Write clean pixels with ``cv2.imwrite`` — this naturally strips every
   byte of EXIF, ICC, XMP, IPTC, Software, and all other metadata.

Public API
----------
- ``read_orientation``      – EXIF orientation tag (1-8) from raw bytes
- ``orientation_transform``  – map tag → list of OpenCV operations
- ``apply_orientation``      – correct an image array in-place
- ``build_output_path``      – mirror a path under a new root
- ``write_clean``            – write an image with format-specific params
- ``process_file``           – full single-file pipeline (read → orient → write)
- ``discover_images``        – recursive image discovery
- ``run_strip_pipeline``     – batch orchestrator with thread pool
- ``summarise_results``      – aggregate per-action counts
"""
from __future__ import annotations

import logging
import os
import struct
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# ===================================================================
# Constants
# ===================================================================

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

#: Default JPEG quality (1-100).  94-95 is near-lossless.
DEFAULT_JPEG_QUALITY: int = 94

#: Default PNG compression level (0-9).  3 is a good speed/size balance.
DEFAULT_PNG_COMPRESSION: int = 3

_ORIENTATION_TAG: int = 0x0112

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger(__name__)


# ===================================================================
# EXIF orientation – raw byte parsing (no PIL)
# ===================================================================

def _parse_ifd_orientation(
    data: bytes,
    endian: str,
    ifd_offset: int,
) -> int | None:
    """Walk IFD0 entries and return orientation value (1-8) or *None*."""
    if ifd_offset + 2 > len(data):
        return None
    num_entries = struct.unpack_from(endian + "H", data, ifd_offset)[0]
    pos = ifd_offset + 2
    for _ in range(num_entries):
        if pos + 12 > len(data):
            return None
        tag = struct.unpack_from(endian + "H", data, pos)[0]
        if tag == _ORIENTATION_TAG:
            type_id = struct.unpack_from(endian + "H", data, pos + 2)[0]
            if type_id == 3:          # SHORT
                val = struct.unpack_from(endian + "H", data, pos + 8)[0]
            elif type_id == 4:        # LONG
                val = struct.unpack_from(endian + "I", data, pos + 8)[0]
            else:
                return None
            return val if 1 <= val <= 8 else None
        pos += 12
    return None


def _parse_tiff_header(data: bytes) -> int | None:
    """Parse TIFF byte-order + IFD0 and return orientation."""
    if len(data) < 8:
        return None
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return None
    magic = struct.unpack_from(endian + "H", data, 2)[0]
    if magic != 42:
        return None
    ifd_offset = struct.unpack_from(endian + "I", data, 4)[0]
    return _parse_ifd_orientation(data, endian, ifd_offset)


def _read_jpeg_orientation(filepath: Path) -> int | None:
    """Extract EXIF orientation from a JPEG file's APP1 segment."""
    try:
        with open(filepath, "rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0:1] != b"\xff":
                    return None
                mtype = marker[1]
                if mtype == 0xFF:
                    continue
                if mtype in (0xDA, 0xD9, 0x00):
                    return None
                raw_len = fh.read(2)
                if len(raw_len) < 2:
                    return None
                seg_len = struct.unpack(">H", raw_len)[0]
                if mtype == 0xE1:
                    payload = fh.read(seg_len - 2)
                    if payload[:6] == b"Exif\x00\x00":
                        return _parse_tiff_header(payload[6:])
                    return None
                fh.seek(seg_len - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None
    return None


def _read_tiff_orientation(filepath: Path) -> int | None:
    """Extract orientation from a TIFF file's IFD0."""
    try:
        with open(filepath, "rb") as fh:
            header = fh.read(64 * 1024)
        return _parse_tiff_header(header)
    except OSError:
        return None


def read_orientation(filepath: Path) -> int | None:
    """Read EXIF orientation tag from *filepath*.

    Supports JPEG (APP1 Exif) and TIFF (IFD0).  Returns an ``int`` in
    ``[1, 8]`` or ``None`` when no orientation is found.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return _read_jpeg_orientation(filepath)
    if ext in (".tif", ".tiff"):
        return _read_tiff_orientation(filepath)
    return None


# ===================================================================
# Orientation → OpenCV transforms
# ===================================================================

def orientation_transform(orientation: int | None) -> list[str]:
    """Map EXIF orientation tag to a list of named OpenCV operations.

    Returns an empty list for orientation 1 (normal) or ``None``.
    Possible operation names:

    - ``"flip_h"``   – ``cv2.flip(img, 1)``
    - ``"flip_v"``   – ``cv2.flip(img, 0)``
    - ``"rot_180"``  – ``cv2.rotate(img, ROTATE_180)``
    - ``"rot_cw"``   – ``cv2.rotate(img, ROTATE_90_CLOCKWISE)``
    - ``"rot_ccw"``  – ``cv2.rotate(img, ROTATE_90_COUNTERCLOCKWISE)``
    - ``"transpose"``– ``cv2.transpose(img)``
    """
    _MAP: dict[int, list[str]] = {
        1: [],
        2: ["flip_h"],
        3: ["rot_180"],
        4: ["flip_v"],
        5: ["transpose"],
        6: ["rot_cw"],
        7: ["transpose", "flip_v"],
        8: ["rot_ccw"],
    }
    if orientation is None or orientation not in _MAP:
        return []
    return _MAP[orientation]


def apply_orientation(img: np.ndarray, orientation: int | None) -> np.ndarray:
    """Apply EXIF orientation correction to a BGR image array.

    Returns the (possibly transformed) array.  Does **not** modify the
    input when no correction is needed.
    """
    _DISPATCH: dict[str, Any] = {
        "flip_h":    lambda i: cv2.flip(i, 1),
        "flip_v":    lambda i: cv2.flip(i, 0),
        "rot_180":   lambda i: cv2.rotate(i, cv2.ROTATE_180),
        "rot_cw":    lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE),
        "rot_ccw":   lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "transpose": lambda i: cv2.transpose(i),
    }
    ops = orientation_transform(orientation)
    for name in ops:
        img = _DISPATCH[name](img)
    return img


# ===================================================================
# Path helpers
# ===================================================================

def build_output_path(
    filepath: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Mirror *filepath* under *output_root* preserving the subtree."""
    return output_root / filepath.relative_to(input_root)


# ===================================================================
# Write helper
# ===================================================================

def write_clean(
    filepath: Path,
    img: np.ndarray,
    *,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> bool:
    """Write *img* to *filepath* with format-appropriate parameters.

    ``cv2.imwrite`` produces files that contain **zero** EXIF / ICC /
    XMP / IPTC — exactly what we need.

    Returns ``True`` on success, ``False`` on failure.
    """
    ext = filepath.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, jpeg_quality]
    else:
        params = []
    # Use imencode + tofile to support Unicode paths on Windows.
    success, buf = cv2.imencode(ext, img, params)
    if success:
        buf.tofile(str(filepath))
    return success


# ===================================================================
# Single-file pipeline
# ===================================================================

def process_file(
    filepath: Path,
    input_root: Path,
    output_root: Path,
    *,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> dict[str, Any]:
    """Read → orient → write for one image.

    Returns a status dict with keys:

    - ``file``        – str, source path
    - ``orientation`` – int | None, raw EXIF value
    - ``rotated``     – bool, whether orientation correction was applied
    - ``action``      – ``"processed"`` | ``"failed"``
    - ``error``       – str | None
    """
    filepath = Path(filepath)
    rec: dict[str, Any] = {
        "file": str(filepath),
        "orientation": None,
        "rotated": False,
        "action": None,
        "error": None,
    }

    try:
        # 1. Parse EXIF orientation from raw bytes
        ori = read_orientation(filepath)
        rec["orientation"] = ori

        # 2. Decode with OpenCV (BGR, no metadata carried)
        #    Use np.fromfile + imdecode to support Unicode paths on Windows.
        raw = np.fromfile(str(filepath), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            rec["action"] = "failed"
            rec["error"] = "cv2.imread returned None (corrupt or unsupported)"
            return rec

        # 3. Apply orientation correction
        if ori is not None and ori != 1:
            img = apply_orientation(img, ori)
            rec["rotated"] = True

        # 4. Build output path & write clean
        dst = build_output_path(filepath, input_root, output_root)
        dst.parent.mkdir(parents=True, exist_ok=True)

        ok = write_clean(
            dst, img,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )
        if not ok:
            rec["action"] = "failed"
            rec["error"] = f"cv2.imwrite failed for {dst}"
        else:
            rec["action"] = "processed"

    except Exception as exc:
        rec["action"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"

    return rec


# ===================================================================
# Discovery
# ===================================================================

def discover_images(root: Path) -> list[Path]:
    """Recursively collect all image files under *root*."""
    root = Path(root)
    found: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                found.append(dp / fn)
    return found


# ===================================================================
# Batch pipeline
# ===================================================================

def run_strip_pipeline(
    input_root: Path | str,
    output_root: Path | str,
    *,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
    workers: int = 8,
    overwrite: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Discover + process all images from *input_root* into *output_root*.

    Parameters
    ----------
    input_root : Path
        Source directory (e.g. ``data/raw_cleaned``).
    output_root : Path
        Destination directory (e.g. ``data/cleaned_v2``).
    jpeg_quality : int
        JPEG output quality (1-100).
    png_compression : int
        PNG compression level (0-9).
    workers : int
        Number of I/O threads.
    overwrite : bool
        If ``False``, skip files whose output already exists.
    show_progress : bool
        Display a tqdm progress bar.

    Returns
    -------
    list[dict]
        Per-file status records.
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    if input_root == output_root:
        raise ValueError("input_root and output_root must be different.")

    output_root.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_root)
    logger.info("Discovered %d images in %s", len(images), input_root)

    if not images:
        logger.warning("No images found — nothing to do.")
        return []

    # Optional skip for already-processed files
    if not overwrite:
        todo: list[Path] = []
        skipped_count = 0
        for fp in images:
            dst = build_output_path(fp, input_root, output_root)
            if dst.exists():
                skipped_count += 1
            else:
                todo.append(fp)
        if skipped_count:
            logger.info(
                "Skipping %d files (output exists, overwrite=False).",
                skipped_count,
            )
        images = todo

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                process_file, fp, input_root, output_root,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
            ): fp
            for fp in images
        }
        iterator = as_completed(futs)
        if show_progress:
            iterator = tqdm(
                iterator, total=len(futs),
                desc="Stripping metadata", unit="img", dynamic_ncols=True,
            )
        for fut in iterator:
            fp = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "file": str(fp),
                    "orientation": None,
                    "rotated": False,
                    "action": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(rec)
            if rec["action"] == "failed":
                logger.warning("FAIL %s — %s", fp.name, rec.get("error"))

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline finished in %.1f s", elapsed)
    return results


# ===================================================================
# Summary
# ===================================================================

def summarise_results(results: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate pipeline results into a summary dict.

    Keys: ``total``, ``processed``, ``rotated``, ``no_orientation``,
    ``failed``.
    """
    total = len(results)
    processed = sum(1 for r in results if r["action"] == "processed")
    rotated = sum(1 for r in results if r.get("rotated"))
    no_ori = sum(
        1 for r in results
        if r["action"] == "processed" and r["orientation"] is None
    )
    failed = sum(1 for r in results if r["action"] == "failed")
    return {
        "total_scanned": total,
        "total_processed": processed,
        "total_rotated_by_orientation": rotated,
        "total_no_orientation": no_ori,
        "total_failed": failed,
    }


def print_summary(summary: dict[str, int], output_root: Path | str = "") -> None:
    """Log a formatted summary block."""
    sep = "=" * 60
    logger.info(sep)
    logger.info("STRIP-METADATA PIPELINE COMPLETE")
    for key, val in summary.items():
        logger.info("  %-35s: %d", key, val)
    if output_root:
        logger.info("  %-35s: %s", "output_location", output_root)
    logger.info(sep)


