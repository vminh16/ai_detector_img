"""
src/preprocess.py
=================
Library module for dataset preprocessing.

Functions
---------
- ``infer_label``        – path-based real / fake label inference
- ``discover_images``    – recursive image file discovery
- ``convert_png_to_jpg`` – single PNG → clean JPEG conversion
- ``process_file``       – single-file dispatch (convert or copy)
- ``run_cleaning``       – full pipeline orchestrator (thread-pool)
- ``summarise_results``  – aggregate action counts from pipeline results
- ``count_dataset``      – quick per-generator × label file count
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

# ===================================================================
# Constants
# ===================================================================

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)

_LABEL_FAKE: frozenset[str] = frozenset(
    {"fake", "ai", "generated", "synthetic", "gen", "diffusion"}
)
_LABEL_REAL: frozenset[str] = frozenset(
    {"real", "nature", "genuine", "authentic", "natural", "original", "orig"}
)

_BG_COLOR = (255, 255, 255)

logger = logging.getLogger(__name__)

# ===================================================================
# Label inference
# ===================================================================


def infer_label(filepath: Path, root: Path) -> str:
    """Infer ``'real'``, ``'fake'``, or ``'unknown'`` from folder hierarchy.

    Scans every directory component between *root* and *filepath*.
    """
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return "unknown"

    for part in rel.parts[:-1]:
        low = part.lower()
        if low in _LABEL_FAKE:
            return "fake"
        if low in _LABEL_REAL:
            return "real"
    return "unknown"


# ===================================================================
# Discovery
# ===================================================================


def discover_images(root: Path) -> list[Path]:
    """Recursively find all image files under *root*."""
    found: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                found.append(dp / fn)
    return found


# ===================================================================
# Conversion helpers
# ===================================================================


def convert_png_to_jpg(
    src: Path,
    dst: Path,
    quality: int = 94,
) -> None:
    """Open a PNG, flatten alpha → white, save as clean JPEG.

    Parameters
    ----------
    src : Path
        Source PNG file.
    dst : Path
        Destination JPEG path (should end in ``.jpg``).
    quality : int
        JPEG quality 1-95 (default 94).
    """
    with Image.open(src) as img:
        if img.mode in ("RGBA", "LA", "PA", "RGBa"):
            background = Image.new("RGB", img.size, _BG_COLOR)
            rgba = img.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[3])
            out_img = background
        elif img.mode != "RGB":
            out_img = img.convert("RGB")
        else:
            out_img = img
        out_img.save(
            dst, format="JPEG", quality=quality,
            optimize=True, subsampling="4:2:0",
        )


def _build_output_path(
    filepath: Path,
    input_root: Path,
    output_root: Path,
    convert_png: bool,
) -> Path:
    """Mirror *filepath* under *output_root*, changing extension if needed."""
    rel = filepath.relative_to(input_root)
    if convert_png:
        rel = rel.with_suffix(".jpg")
    return output_root / rel


# ===================================================================
# Single-file processing
# ===================================================================


def process_file(
    filepath: Path,
    input_root: Path,
    output_root: Path,
    quality: int = 94,
) -> dict[str, Any]:
    """Process a single image file.  Returns a status record.

    Returns
    -------
    dict with keys ``file``, ``label``, ``action``, ``error``.
    """
    label = infer_label(filepath, input_root)
    ext = filepath.suffix.lower()
    rec: dict[str, Any] = {
        "file": str(filepath),
        "label": label,
        "action": None,
        "error": None,
    }

    needs_convert = label == "fake" and ext == ".png"
    dst = _build_output_path(filepath, input_root, output_root, needs_convert)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if needs_convert:
            convert_png_to_jpg(filepath, dst, quality)
            rec["action"] = "converted"
        else:
            shutil.copy2(filepath, dst)
            if label == "fake":
                rec["action"] = "copied_fake_jpeg"
            elif label == "unknown":
                rec["action"] = "copied_unknown"
            else:
                rec["action"] = "copied_real"
    except Exception as exc:
        rec["action"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"

    return rec


# ===================================================================
# Pipeline orchestration
# ===================================================================


def run_cleaning(
    input_root: Path,
    output_root: Path,
    *,
    quality: int = 94,
    workers: int = 8,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Discover images in *input_root* and clean them into *output_root*.

    Parameters
    ----------
    input_root : Path
        Root of the raw dataset.
    output_root : Path
        Destination root (will be created if missing).
    quality : int
        JPEG quality for converted PNGs.
    workers : int
        Number of I/O threads.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    list[dict]
        Per-file status records with keys ``file``, ``label``,
        ``action``, ``error``.
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    if input_root == output_root:
        raise ValueError("Input and output directories must differ.")

    output_root.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_root)
    logger.info("Discovered %d image files in %s", len(images), input_root)
    if not images:
        logger.warning("No images found – nothing to do.")
        return []

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(process_file, fp, input_root, output_root, quality): fp
            for fp in images
        }
        iterator = as_completed(futs)
        if show_progress:
            iterator = tqdm(
                iterator, total=len(futs), desc="Cleaning",
                unit="img", dynamic_ncols=True,
            )
        for fut in iterator:
            fp = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "file": str(fp), "label": "?",
                    "action": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(rec)
            if rec["action"] == "failed":
                logger.warning("  FAIL %s – %s", fp.name, rec.get("error"))

    elapsed = time.perf_counter() - t0
    logger.info("Cleaning done in %.1f s", elapsed)
    return results


# ===================================================================
# Summary helpers
# ===================================================================


def summarise_results(results: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate action counts from pipeline results.

    Returns
    -------
    dict mapping action names to counts, plus ``'total'`` and ``'elapsed'``.
    """
    counts = dict(Counter(r["action"] for r in results))
    counts["total"] = len(results)
    return counts


def count_dataset(root: Path) -> list[dict[str, Any]]:
    """Quick per-generator × label file count under *root*.

    Returns a list of dicts with keys ``generator``, ``label``, ``count``.
    """
    root = Path(root)
    stats: list[dict[str, Any]] = []
    for gen_dir in sorted(root.iterdir()):
        if not gen_dir.is_dir():
            continue
        for label_dir in sorted(gen_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            count = sum(
                1 for f in label_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            stats.append({
                "generator": gen_dir.name,
                "label": label_dir.name,
                "count": count,
            })
    return stats
