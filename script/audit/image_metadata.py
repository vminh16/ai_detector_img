#!/usr/bin/env python3
"""
audit_image_metadata.py
=======================
Dataset image metadata auditor for ML shortcut-learning detection.

Recursively scans an image directory, extracts per-file metadata
(EXIF, ICC profile, format, dimensions, compression artefacts),
infers contextual labels from the folder hierarchy, and exports:

  1. per_file_metadata  (.csv / .parquet / both)
  2. metadata_summary.json
  3. metadata_report.md
  4. suspicious_shortcuts.csv
  5. folder_structure_summary.csv

Designed for large datasets (60 k+ images) with concurrent I/O,
minimal pixel decoding, and robust error handling.

Usage
-----
    python audit_image_metadata.py --root_dir ./data/raw --out_dir ./audit_output
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, IptcImagePlugin
from tqdm import tqdm

# Optional: pyarrow for parquet support
try:
    import pyarrow as _pa  # noqa: F401

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

# ===================================================================
# Constants
# ===================================================================

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif",
     ".heic", ".heif", ".avif"}
)

# EXIF tag IDs – main IFD
_MAIN_EXIF_TAGS: dict[int, str] = {
    270:   "image_description",
    271:   "make",
    272:   "model",
    274:   "orientation",
    305:   "software",
    306:   "datetime",
    315:   "artist",
    316:   "host_computer",
    33432: "copyright",
}
# EXIF sub-IFD (0x8769) tags
_SUB_EXIF_TAGS: dict[int, str] = {
    36867: "datetime_original",
}
_ALL_EXIF_FIELDS: list[str] = list(_MAIN_EXIF_TAGS.values()) + list(
    _SUB_EXIF_TAGS.values()
)

# Heuristic word sets for path inference
_LABEL_REAL: frozenset[str] = frozenset(
    {"real", "nature", "genuine", "authentic", "natural", "original", "orig"}
)
_LABEL_FAKE: frozenset[str] = frozenset(
    {"fake", "ai", "generated", "synthetic", "gen", "diffusion"}
)
_SPLIT_WORDS: frozenset[str] = frozenset(
    {"train", "val", "validation", "test", "eval", "dev", "holdout", "calibration"}
)
_KNOWN_GENERATORS: frozenset[str] = frozenset(
    {"midjourney", "stablediffusion", "sd", "sdv14", "sdv15", "sdxl",
     "dalle", "adm", "glide", "vqdm", "wukong", "imagen",
     "biggan", "stylegan", "progan"}
)

# Default OOD hold-out generators (normalised lowercase, no separators)
_DEFAULT_OOD_GENERATORS: frozenset[str] = frozenset({"sdv15", "glide"})

# PIL mode helpers
_MODE_CHANNELS: dict[str, int] = {
    "1": 1, "L": 1, "P": 1, "I": 1, "F": 1,
    "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
    "RGBA": 4, "CMYK": 4, "RGBa": 4,
    "LA": 2, "PA": 2,
    "I;16": 1, "I;16B": 1, "I;16L": 1,
}
_MODE_BIT_DEPTH: dict[str, int] = {
    "1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8,
    "CMYK": 8, "YCbCr": 8, "LAB": 8, "HSV": 8,
    "I": 32, "F": 32, "LA": 8, "PA": 8, "RGBa": 8,
    "I;16": 16, "I;16B": 16, "I;16L": 16,
}
_ALPHA_MODES: frozenset[str] = frozenset({"RGBA", "LA", "PA", "RGBa"})
_GRAY_MODES: frozenset[str] = frozenset(
    {"L", "1", "I", "F", "LA", "I;16", "I;16B", "I;16L"}
)

# IJG standard JPEG luminance quantisation table (quality 50)
_JPEG_STD_LUM_QT: list[int] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
]

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("audit_metadata")


def _artifact_dirs(out_dir: Path) -> dict[str, Path]:
    """Stable subdirectories under audit_output for long-term hygiene."""
    data_audit = out_dir / "data_audit"
    paths = {
        "metadata": data_audit / "metadata",
        "dataset_profile": data_audit / "dataset_profile",
        "duplicates": data_audit / "duplicates",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# ===================================================================
# CLI
# ===================================================================

def build_cli(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit image dataset metadata for shortcut-learning risks.",
    )
    ap.add_argument("--root_dir", required=True,
                    help="Root folder of the image dataset.")
    ap.add_argument("--out_dir", default="./audit_output",
                    help="Output folder (created if absent).")
    ap.add_argument("--workers", type=int, default=8,
                    help="I/O worker threads (default: 8).")
    ap.add_argument("--hash", action="store_true",
                    help="Compute SHA-256 hash per file (slower).")
    ap.add_argument("--format", default="parquet",
                    choices=["csv", "parquet", "both"],
                    help="Per-file export format (default: parquet).")
    ap.add_argument("--sample_exif_text", type=int, default=20,
                    help="Max unique EXIF text sample values per field.")
    ap.add_argument("--max_errors", type=int, default=0,
                    help="Abort after N file errors (0 = unlimited).")
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--ood_generators", nargs="*", default=None,
                    help=("OOD hold-out generator names (lowercase). "
                          "Default: sdv15 glide"))
    return ap.parse_args(argv)


# ===================================================================
# File discovery
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
# Path inference
# ===================================================================

def _norm(token: str) -> str:
    """Strip separators for fuzzy generator matching."""
    return token.replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def infer_from_path(filepath: Path, root: Path) -> dict[str, Any]:
    """Heuristically infer label / generator / split from folder hierarchy."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        rel = Path(filepath.name)

    parts_raw = list(rel.parts[:-1])  # exclude filename
    parts_lower = [p.lower() for p in parts_raw]

    info: dict[str, Any] = {
        "relative_path": str(rel),
        "path_depth": len(rel.parts),
        "path_parts_json": json.dumps(parts_raw),
        "inferred_label": None,
        "inferred_generator": None,
        "inferred_source": None,
        "inferred_split": None,
    }
    for part in parts_lower:
        norm = _norm(part)
        if info["inferred_label"] is None:
            if part in _LABEL_REAL:
                info["inferred_label"] = "real"
            elif part in _LABEL_FAKE:
                info["inferred_label"] = "fake"
        if info["inferred_generator"] is None:
            for gen in _KNOWN_GENERATORS:
                if norm == gen or norm.startswith(gen):
                    info["inferred_generator"] = part
                    break
        if info["inferred_split"] is None and part in _SPLIT_WORDS:
            info["inferred_split"] = part
    return info


# ===================================================================
# JPEG quality estimation
# ===================================================================

def _estimate_jpeg_quality(qt: dict[int, list[int]] | None) -> int | None:
    """Estimate JPEG quality from the luminance quantisation table (IJG)."""
    if not qt or 0 not in qt:
        return None
    table = list(qt[0])
    if len(table) != 64:
        return None
    scales: list[float] = []
    for i in range(64):
        base = _JPEG_STD_LUM_QT[i]
        if base > 0 and table[i] > 0:
            scales.append(table[i] * 100.0 / base)
    if not scales:
        return None
    avg = sum(scales) / len(scales)
    q = round(5000.0 / avg) if avg >= 100.0 else round((200.0 - avg) / 2.0)
    return max(1, min(100, q))


# ===================================================================
# Single-file metadata extraction
# ===================================================================

# Fields populated by the image-reading branch (used for defaulting on error)
_IMG_FIELDS: list[str] = [
    "format_detected", "width", "height", "aspect_ratio", "pixel_count",
    "image_mode", "n_channels", "has_alpha", "is_grayscale", "bit_depth",
    "frame_count", "has_exif", "exif_tag_count",
    *_ALL_EXIF_FIELDS,
    "has_icc_profile", "icc_profile_size", "has_xmp", "has_iptc",
    "jpeg_quantization_present", "jpeg_quantization_table_count",
    "quality_estimate", "jpeg_subsampling", "jpeg_progressive",
]


def _safe_exif_str(val: Any) -> str | None:
    """Coerce an EXIF tag value to a clean string or None."""
    if val is None:
        return None
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip("\x00 ") or None
        except Exception:
            return None
    s = str(val).strip()
    return s if s else None


def extract_metadata(
    filepath: Path, root: Path, compute_hash: bool,
) -> dict[str, Any]:
    """Extract all audit-relevant metadata from one image file."""
    rec: dict[str, Any] = {}

    # ---- File info ----
    rec["file_path"] = str(filepath)
    rec["file_name"] = filepath.name
    rec["extension"] = filepath.suffix.lower()
    try:
        st = filepath.stat()
        rec["file_size_bytes"] = st.st_size
        rec["modified_time"] = datetime.fromtimestamp(st.st_mtime).isoformat()
    except OSError:
        rec["file_size_bytes"] = None
        rec["modified_time"] = None

    # Optional hash
    if compute_hash:
        try:
            h = hashlib.sha256()
            with open(filepath, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            rec["sha256"] = h.hexdigest()
        except OSError:
            rec["sha256"] = None
    else:
        rec["sha256"] = None

    # ---- Path inference ----
    rec.update(infer_from_path(filepath, root))

    # ---- Image metadata via Pillow (no pixel decode) ----
    try:
        with Image.open(filepath) as img:
            # Structure
            rec["format_detected"] = img.format
            rec["width"] = img.width
            rec["height"] = img.height
            rec["aspect_ratio"] = round(img.width / max(img.height, 1), 4)
            rec["pixel_count"] = img.width * img.height
            rec["image_mode"] = img.mode
            rec["n_channels"] = _MODE_CHANNELS.get(img.mode, len(img.getbands()))
            rec["has_alpha"] = img.mode in _ALPHA_MODES
            rec["is_grayscale"] = img.mode in _GRAY_MODES
            rec["bit_depth"] = _MODE_BIT_DEPTH.get(img.mode)
            try:
                rec["frame_count"] = getattr(img, "n_frames", 1) or 1
            except Exception:
                rec["frame_count"] = 1

            # EXIF
            exif = None
            try:
                exif = img.getexif()
            except Exception:
                pass

            rec["has_exif"] = bool(exif) and len(exif) > 0
            rec["exif_tag_count"] = len(exif) if exif else 0

            exif_ifd = None
            if exif:
                try:
                    exif_ifd = exif.get_ifd(0x8769)
                except Exception:
                    pass

            for tag_id, field in _MAIN_EXIF_TAGS.items():
                val = exif.get(tag_id) if exif else None
                if val is None and exif_ifd:
                    val = exif_ifd.get(tag_id)
                rec[field] = _safe_exif_str(val)

            for tag_id, field in _SUB_EXIF_TAGS.items():
                val = exif_ifd.get(tag_id) if exif_ifd else None
                if val is None and exif:
                    val = exif.get(tag_id)
                rec[field] = _safe_exif_str(val)

            # ICC profile
            icc = img.info.get("icc_profile")
            rec["has_icc_profile"] = (
                isinstance(icc, (bytes, bytearray)) and len(icc) > 0
            )
            rec["icc_profile_size"] = (
                len(icc) if isinstance(icc, (bytes, bytearray)) and icc else None
            )

            # XMP
            xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
            rec["has_xmp"] = (
                bool(xmp)
                and (len(xmp) > 0 if isinstance(xmp, (bytes, str)) else False)
            )

            # IPTC
            try:
                iptc = IptcImagePlugin.getiptcinfo(img)
            except Exception:
                iptc = None
            rec["has_iptc"] = bool(iptc)

            # JPEG compression clues
            qt = getattr(img, "quantization", None)
            rec["jpeg_quantization_present"] = bool(qt)
            rec["jpeg_quantization_table_count"] = len(qt) if qt else None
            rec["quality_estimate"] = _estimate_jpeg_quality(qt) if qt else None
            rec["jpeg_subsampling"] = None  # Pillow does not reliably expose this
            rec["jpeg_progressive"] = (
                bool(img.info.get("progressive") or img.info.get("progression"))
                if img.format == "JPEG"
                else None
            )

    except Exception as exc:
        for f in _IMG_FIELDS:
            rec.setdefault(f, None)
        rec["_error"] = f"{type(exc).__name__}: {exc}"

    return rec


# ===================================================================
# Concurrent orchestration
# ===================================================================

def process_all(
    images: list[Path],
    root: Path,
    compute_hash: bool,
    workers: int,
    max_errors: int,
) -> tuple[list[dict[str, Any]], int]:
    """Extract metadata from all *images* with a thread pool.

    Returns ``(records, error_count)``.
    """
    records: list[dict[str, Any]] = []
    err_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(extract_metadata, fp, root, compute_hash): fp
            for fp in images
        }
        with tqdm(
            total=len(futs), desc="Scanning images", unit="img", dynamic_ncols=True,
        ) as pbar:
            for fut in as_completed(futs):
                fp = futs[fut]
                try:
                    rec = fut.result()
                    records.append(rec)
                    if "_error" in rec:
                        err_count += 1
                        logger.warning("  %s – %s", fp.name, rec["_error"])
                except Exception as exc:
                    err_count += 1
                    records.append({
                        "file_path": str(fp),
                        "file_name": fp.name,
                        "extension": fp.suffix.lower(),
                        "_error": f"{type(exc).__name__}: {exc}",
                    })
                    logger.error("  Unhandled – %s: %s", fp.name, exc)
                pbar.update(1)
                if 0 < max_errors <= err_count:
                    logger.error(
                        "Reached --max_errors=%d. Cancelling remaining.", max_errors,
                    )
                    for f in futs:
                        f.cancel()
                    break

    return records, err_count


# ===================================================================
# Folder structure summary
# ===================================================================

def build_folder_summary(images: list[Path], root: Path) -> pd.DataFrame:
    """Per-folder file count, depth, extension distribution."""
    buckets: dict[str, dict[str, Any]] = {}
    for fp in images:
        try:
            folder = str(fp.parent.relative_to(root))
        except ValueError:
            folder = str(fp.parent)
        if folder not in buckets:
            buckets[folder] = {
                "folder": folder,
                "file_count": 0,
                "depth": len(Path(folder).parts),
                "ext": collections.Counter(),
            }
        buckets[folder]["file_count"] += 1
        buckets[folder]["ext"][fp.suffix.lower()] += 1

    rows = [
        {
            "folder": b["folder"],
            "file_count": b["file_count"],
            "depth": b["depth"],
            "extension_distribution": json.dumps(dict(b["ext"].most_common())),
        }
        for b in buckets.values()
    ]
    return pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)


# ===================================================================
# Aggregate summary
# ===================================================================

def _safe_key(k: Any) -> str:
    """JSON-safe dict key from possibly-NaN pandas value."""
    if k is None or (isinstance(k, float) and k != k):
        return "<None>"
    return str(k)


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """High-level aggregate statistics."""
    has_err = "_error" in df.columns
    n_total = len(df)
    n_fail = int(df["_error"].notna().sum()) if has_err else 0

    s: dict[str, Any] = {
        "total_files_scanned": n_total,
        "valid_images": n_total - n_fail,
        "failed_images": n_fail,
        "scan_timestamp": datetime.now().isoformat(),
    }

    # Categorical distributions
    for col, key in [("extension", "by_extension"),
                     ("format_detected", "by_format")]:
        if col in df.columns:
            s[key] = {
                _safe_key(k): int(v)
                for k, v in df[col].value_counts(dropna=False).items()
            }

    # Numeric summaries
    for col in ("width", "height", "file_size_bytes", "pixel_count"):
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(num):
                s[f"{col}_stats"] = {
                    "min": float(num.min()),
                    "max": float(num.max()),
                    "mean": round(float(num.mean()), 2),
                    "median": float(num.median()),
                    "std": round(float(num.std()), 2) if len(num) > 1 else 0.0,
                }

    # Boolean rates
    for col in ("has_exif", "has_icc_profile", "has_alpha", "is_grayscale",
                "has_xmp", "has_iptc", "jpeg_quantization_present",
                "jpeg_progressive"):
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > 0:
                s[f"{col}_rate"] = round(
                    float(df.loc[valid, col].astype(float).mean()), 4,
                )

    # Label / generator distributions
    for col, key in [("inferred_label", "by_inferred_label"),
                     ("inferred_generator", "by_inferred_generator")]:
        if col in df.columns:
            s[key] = {
                _safe_key(k): int(v)
                for k, v in df[col].value_counts(dropna=False).items()
            }

    return s


# ===================================================================
# Shortcut risk analysis
# ===================================================================

def _risk_level(diff: float) -> str:
    if diff > 0.30:
        return "HIGH"
    if diff > 0.10:
        return "MEDIUM"
    return "LOW"


def analyse_shortcuts(
    df: pd.DataFrame, sample_n: int,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Heuristic shortcut risk analysis.

    Returns ``(alerts_df, text_samples_dict)``.
    """
    alerts: list[dict[str, str]] = []
    text_samples: dict[str, list] = {}
    lbl_col = "inferred_label"
    has_labels = lbl_col in df.columns and df[lbl_col].notna().any()

    if not has_labels:
        alerts.append({
            "feature": "label_inference",
            "risk_level": "INFO",
            "description": ("Could not infer labels from paths – "
                            "per-class shortcut analysis skipped."),
            "detail": "",
        })
        return pd.DataFrame(alerts), text_samples

    groups: dict[str, pd.DataFrame] = dict(
        list(df.groupby(lbl_col, dropna=True))
    )
    if len(groups) < 2:
        alerts.append({
            "feature": "label_balance",
            "risk_level": "WARNING",
            "description": (f"Only one label detected "
                            f"({list(groups.keys())}). "
                            f"Inter-class analysis skipped."),
            "detail": "",
        })
        return pd.DataFrame(alerts), text_samples

    labels = sorted(groups.keys())

    # ---- Binary features ----
    for col in ("has_exif", "has_icc_profile", "has_alpha", "is_grayscale",
                "has_xmp", "has_iptc", "jpeg_quantization_present",
                "jpeg_progressive"):
        if col not in df.columns:
            continue
        rates: dict[str, float] = {}
        for lbl in labels:
            g = groups[lbl]
            valid = g[col].notna()
            if valid.sum() > 0:
                rates[lbl] = float(g.loc[valid, col].astype(float).mean())
        if len(rates) >= 2:
            vals = list(rates.values())
            diff = abs(max(vals) - min(vals))
            detail = "; ".join(f"{k}={v:.4f}" for k, v in rates.items())
            alerts.append({
                "feature": col,
                "risk_level": _risk_level(diff),
                "description": f"Rate delta={diff:.4f}",
                "detail": detail,
            })

    # ---- Categorical features ----
    for col in ("extension", "format_detected", "image_mode"):
        if col not in df.columns:
            continue
        dists = {
            lbl: groups[lbl][col].value_counts(
                normalize=True, dropna=False,
            ).to_dict()
            for lbl in labels
        }
        all_cats = {c for d in dists.values() for c in d}
        max_diff, worst = 0.0, None
        for cat in all_cats:
            ps = [dists[lbl].get(cat, 0.0) for lbl in labels]
            d = abs(max(ps) - min(ps))
            if d > max_diff:
                max_diff, worst = d, cat
        detail = (
            "; ".join(
                f"{lbl}:{worst}={dists[lbl].get(worst, 0):.4f}" for lbl in labels
            )
            if worst is not None
            else ""
        )
        alerts.append({
            "feature": col,
            "risk_level": _risk_level(max_diff),
            "description": f"Max category delta={max_diff:.4f} ('{worst}')",
            "detail": detail,
        })

    # ---- Numeric features ----
    for col in ("width", "height", "file_size_bytes", "pixel_count",
                "quality_estimate"):
        if col not in df.columns:
            continue
        medians: dict[str, float] = {}
        for lbl in labels:
            s = pd.to_numeric(groups[lbl][col], errors="coerce").dropna()
            if len(s):
                medians[lbl] = float(s.median())
        if len(medians) >= 2:
            vals = list(medians.values())
            mean_v = sum(vals) / len(vals)
            rel_diff = abs(max(vals) - min(vals)) / max(mean_v, 1e-9)
            detail = "; ".join(f"{k}_med={v:.1f}" for k, v in medians.items())
            alerts.append({
                "feature": col,
                "risk_level": _risk_level(rel_diff),
                "description": f"Relative median delta={rel_diff:.4f}",
                "detail": detail,
            })

    # ---- Text EXIF fields – presence & exclusive values ----
    text_fields = [
        "software", "make", "model", "artist",
        "host_computer", "image_description", "copyright",
    ]
    for field in text_fields:
        if field not in df.columns:
            continue

        # Collect samples
        vc = df[field].dropna().value_counts()
        text_samples[field] = [
            {"value": str(v), "count": int(c)}
            for v, c in vc.head(sample_n).items()
        ]

        # Presence bias
        pres = {lbl: float(groups[lbl][field].notna().mean()) for lbl in labels}
        diff = abs(max(pres.values()) - min(pres.values()))
        if diff > 0.05:
            detail = "; ".join(f"{k}_pres={v:.4f}" for k, v in pres.items())
            alerts.append({
                "feature": f"{field} (presence)",
                "risk_level": _risk_level(diff),
                "description": f"Presence delta={diff:.4f}",
                "detail": detail,
            })

        # Exclusive-value check
        for lbl in labels:
            top3 = groups[lbl][field].dropna().value_counts().head(3)
            for val, cnt in top3.items():
                if cnt < 10:
                    continue
                others = sum(
                    int(groups[l][field].eq(val).sum())
                    for l in labels
                    if l != lbl
                )
                if others == 0:
                    alerts.append({
                        "feature": f"{field} (exclusive value)",
                        "risk_level": "HIGH",
                        "description": (
                            f"'{str(val)[:80]}' appears {cnt}x "
                            f"ONLY in label='{lbl}'"
                        ),
                        "detail": f"field={field}",
                    })

    # ---- Resolution pattern ----
    if "width" in df.columns and "height" in df.columns:
        for lbl in labels:
            g = groups[lbl]
            mask = g["width"].notna() & g["height"].notna()
            gv = g[mask]
            if gv.empty:
                continue
            res = (
                gv["width"].astype(int).astype(str)
                + "x"
                + gv["height"].astype(int).astype(str)
            )
            for rv, cnt in res.value_counts().head(3).items():
                ratio = cnt / len(gv)
                if ratio < 0.80:
                    continue
                for other_lbl in labels:
                    if other_lbl == lbl:
                        continue
                    og = groups[other_lbl]
                    omask = og["width"].notna() & og["height"].notna()
                    ogv = og[omask]
                    if ogv.empty:
                        continue
                    ores = (
                        ogv["width"].astype(int).astype(str)
                        + "x"
                        + ogv["height"].astype(int).astype(str)
                    )
                    other_ratio = float((ores == rv).mean())
                    if other_ratio < 0.20:
                        alerts.append({
                            "feature": "resolution_pattern",
                            "risk_level": "HIGH",
                            "description": (
                                f"{rv} dominates '{lbl}' ({ratio:.1%}) "
                                f"but rare in '{other_lbl}' ({other_ratio:.1%})"
                            ),
                            "detail": f"label={lbl}",
                        })

    # Sort by severity
    out = pd.DataFrame(alerts)
    if not out.empty:
        _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "WARNING": 3, "INFO": 4}
        out["_s"] = out["risk_level"].map(_order).fillna(9)
        out = out.sort_values("_s").drop(columns="_s").reset_index(drop=True)
    return out, text_samples


# ===================================================================
# Patch 1: Per-generator / per-label breakdown
# ===================================================================

def _group_stats(g: pd.DataFrame) -> dict[str, Any]:
    """Compute standard audit stats for a slice of the master DataFrame."""
    row: dict[str, Any] = {"total_files": len(g)}
    for col in ("width", "height"):
        if col in g.columns:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(s):
                row[f"{col}_min"] = float(s.min())
                row[f"{col}_median"] = float(s.median())
                row[f"{col}_max"] = float(s.max())
    # Format distribution (rates)
    if "extension" in g.columns:
        vc = g["extension"].value_counts(normalize=True)
        for ext, rate in vc.items():
            row[f"pct{ext}"] = round(float(rate) * 100, 2)
    # JPEG quality
    if "quality_estimate" in g.columns:
        q = pd.to_numeric(g["quality_estimate"], errors="coerce").dropna()
        if len(q):
            row["quality_mean"] = round(float(q.mean()), 2)
            row["quality_std"] = round(float(q.std()), 2) if len(q) > 1 else 0.0
            row["quality_median"] = float(q.median())
    # Boolean rates
    for col in ("has_exif", "has_icc_profile", "has_alpha", "is_grayscale"):
        if col in g.columns:
            valid = g[col].notna()
            if valid.sum() > 0:
                row[f"{col}_rate"] = round(
                    float(g.loc[valid, col].astype(float).mean()), 4,
                )
    return row


def build_generator_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (generator, label) pair with audit stats."""
    gen_col = "inferred_generator"
    lbl_col = "inferred_label"
    if gen_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for gen, gdf in df.groupby(gen_col, dropna=False):
        gen_str = _safe_key(gen)
        if lbl_col in df.columns and gdf[lbl_col].notna().any():
            for lbl, ldf in gdf.groupby(lbl_col, dropna=False):
                r = _group_stats(ldf)
                r["generator"] = gen_str
                r["label"] = _safe_key(lbl)
                rows.append(r)
        else:
            r = _group_stats(gdf)
            r["generator"] = gen_str
            r["label"] = "<all>"
            rows.append(r)
    out = pd.DataFrame(rows)
    cols_first = ["generator", "label", "total_files"]
    rest = [c for c in out.columns if c not in cols_first]
    return out[cols_first + rest].sort_values(
        ["generator", "label"]
    ).reset_index(drop=True)


def build_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per inferred_label with audit stats."""
    lbl_col = "inferred_label"
    if lbl_col not in df.columns or not df[lbl_col].notna().any():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for lbl, ldf in df.groupby(lbl_col, dropna=False):
        r = _group_stats(ldf)
        r["label"] = _safe_key(lbl)
        rows.append(r)
    out = pd.DataFrame(rows)
    cols_first = ["label", "total_files"]
    rest = [c for c in out.columns if c not in cols_first]
    return out[cols_first + rest].sort_values("label").reset_index(drop=True)


# ===================================================================
# Patch 2: OOD flag / eval_group
# ===================================================================

def tag_ood(df: pd.DataFrame, ood_set: frozenset[str]) -> None:
    """Add ``is_ood_generator`` and ``eval_group`` columns **in-place**."""
    gen_col = "inferred_generator"
    if gen_col not in df.columns:
        df["is_ood_generator"] = None
        df["eval_group"] = "UNKNOWN"
        return
    normed = df[gen_col].fillna("").apply(_norm).str.lower()
    df["is_ood_generator"] = normed.isin(ood_set)
    df["eval_group"] = "UNKNOWN"
    df.loc[df[gen_col].notna() & ~df["is_ood_generator"], "eval_group"] = "ID"
    df.loc[df["is_ood_generator"], "eval_group"] = "OOD"


def build_ood_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table filtered to OOD generators only."""
    if "eval_group" not in df.columns:
        return pd.DataFrame()
    ood = df[df["eval_group"] == "OOD"]
    if ood.empty:
        return pd.DataFrame()
    return build_generator_summary(ood)


# ===================================================================
# Patch 3: Quality & format histograms
# ===================================================================

def build_quality_histograms(
    df: pd.DataFrame, n_bins: int = 10,
) -> dict[str, Any]:
    """Return quality_estimate histograms: global + per-generator."""
    out: dict[str, Any] = {}
    if "quality_estimate" not in df.columns:
        return out
    q_all = pd.to_numeric(df["quality_estimate"], errors="coerce").dropna()
    if q_all.empty:
        return out
    bins = list(range(0, 101, 100 // n_bins))  # 0,10,20,...,100
    if bins[-1] != 100:
        bins.append(100)
    labels_b = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    # Global
    cuts = pd.cut(q_all, bins=bins, labels=labels_b, include_lowest=True,
                  right=True)
    out["global"] = cuts.value_counts().sort_index().to_dict()
    # Per generator
    gen_col = "inferred_generator"
    if gen_col in df.columns:
        per_gen: dict[str, dict] = {}
        for gen, gdf in df.groupby(gen_col, dropna=False):
            gq = pd.to_numeric(gdf["quality_estimate"], errors="coerce").dropna()
            if gq.empty:
                continue
            cuts_g = pd.cut(gq, bins=bins, labels=labels_b,
                            include_lowest=True, right=True)
            per_gen[_safe_key(gen)] = cuts_g.value_counts().sort_index().to_dict()
        out["per_generator"] = per_gen
    return out


def build_format_distributions(df: pd.DataFrame) -> dict[str, Any]:
    """Format / extension distributions: global + per-generator."""
    out: dict[str, Any] = {}
    for col in ("extension", "format_detected"):
        if col not in df.columns:
            continue
        global_d = df[col].value_counts(dropna=False).to_dict()
        out[f"{col}_global"] = {_safe_key(k): int(v) for k, v in global_d.items()}
        gen_col = "inferred_generator"
        if gen_col in df.columns:
            per_gen: dict[str, dict] = {}
            for gen, gdf in df.groupby(gen_col, dropna=False):
                gd = gdf[col].value_counts(dropna=False).to_dict()
                per_gen[_safe_key(gen)] = {_safe_key(k): int(v)
                                           for k, v in gd.items()}
            out[f"{col}_per_generator"] = per_gen
    return out


# ===================================================================
# Patch 4: Duplicate audit (exact-hash)
# ===================================================================

def audit_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find exact-match duplicates via sha256. Returns (dup_df, dup_summary)."""
    summary: dict[str, Any] = {"dedup_enabled": False}
    empty_df = pd.DataFrame(
        columns=["sha256", "dup_group", "file_path",
                 "inferred_generator", "inferred_label"],
    )
    if "sha256" not in df.columns or df["sha256"].isna().all():
        summary["note"] = "Hash not computed (--hash not set). Dedup skipped."
        return empty_df, summary

    summary["dedup_enabled"] = True
    hash_col = df["sha256"].dropna()
    dup_hashes = hash_col[hash_col.duplicated(keep=False)]
    if dup_hashes.empty:
        summary["total_duplicate_groups"] = 0
        summary["total_duplicate_files"] = 0
        return empty_df, summary

    dup_idx = dup_hashes.index
    need_cols = ["sha256", "file_path"]
    for c in ("inferred_generator", "inferred_label"):
        if c in df.columns:
            need_cols.append(c)
    dup_df = df.loc[dup_idx, need_cols].copy()
    dup_df.sort_values("sha256", inplace=True)
    # Group id
    dup_df["dup_group"] = (dup_df["sha256"] != dup_df["sha256"].shift()).cumsum()

    n_groups = dup_df["dup_group"].nunique()
    n_files = len(dup_df)
    summary["total_duplicate_groups"] = int(n_groups)
    summary["total_duplicate_files"] = int(n_files)

    # Cross analyses
    gen_col = "inferred_generator"
    lbl_col = "inferred_label"
    cross_gen = 0
    cross_label = 0
    intra_gen = 0
    for _, grp in dup_df.groupby("dup_group"):
        if gen_col in grp.columns:
            gens = grp[gen_col].dropna().nunique()
            if gens > 1:
                cross_gen += 1
            elif gens == 1:
                intra_gen += 1
        if lbl_col in grp.columns:
            lbls = grp[lbl_col].dropna().nunique()
            if lbls > 1:
                cross_label += 1

    summary["cross_generator_dup_groups"] = cross_gen
    summary["intra_generator_dup_groups"] = intra_gen
    summary["cross_label_dup_groups"] = cross_label
    return dup_df.reset_index(drop=True), summary


# ===================================================================
# Patch 5: Symmetry audit checklist
# ===================================================================

def build_symmetry_checklist() -> str:
    """Generate a static Markdown checklist describing pipeline symmetry."""
    return """# Symmetry Audit Checklist

This checklist is auto-generated by `audit_image_metadata.py` to certify
that the metadata extraction pipeline treats **real** and **fake** images
identically.  It should be reviewed and attached to the project report.

## Decode & I/O

- [x] Same decode library (Pillow / PIL) used for all classes.
- [x] Same `Image.open()` call path — no class-conditional branches.
- [x] File discovery is purely extension-based; label/class does **not**
      influence which files are scanned.
- [x] No EXIF orientation auto-rotate applied (raw header read only).
- [x] No ICC profile rendering applied (presence flag only).

## Metadata Extraction

- [x] Identical metadata fields extracted for every image regardless of
      its inferred label, generator, or format.
- [x] EXIF / ICC / XMP / IPTC presence is recorded for **audit only**;
      none of these fields are used as model features.
- [x] `quality_estimate` is derived from the JPEG quantisation table
      using a deterministic IJG-based formula — no label-aware logic.

## Error Handling

- [x] Corrupt or unreadable files are logged identically for both classes
      (same `_error` column, same `logger.warning` path).
- [x] A single file failure never halts the pipeline (unless
      `--max_errors` threshold is explicitly reached).
- [x] Error count is reported in the final summary.

## Path Inference

- [x] Label is inferred **after** all metadata has been extracted —
      inference result cannot influence the extraction logic.
- [x] Generator name is inferred from folder hierarchy only; it does not
      change which metadata fields are read.

## Output

- [x] All outputs are audit artefacts.  No file in `out_dir` is designed
      to be consumed as a model feature vector.
- [x] Shortcut alerts flag metadata asymmetries between classes **for
      human review**, not for automated feature selection.

## Concurrency & Determinism

- [x] Thread pool processes files in arrival order per `as_completed`;
      no label-based scheduling.
- [x] SHA-256 hash (when enabled) is computed on raw file bytes —
      identical for any reader.

---
*Auto-generated — review before including in the final report.*
"""


# ===================================================================
# Markdown report
# ===================================================================

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def build_report(
    summary: dict[str, Any],
    shortcuts: pd.DataFrame,
    text_samples: dict[str, list],
    folder_df: pd.DataFrame,
) -> str:
    """Return the Markdown audit report as a string."""
    L: list[str] = []
    L.append("# Image Dataset – Metadata Audit Report\n")
    L.append(f"**Generated:** {summary.get('scan_timestamp', '')}\n")

    # 1. Overview
    L.append("## 1. Overview\n")
    L.append(_md_table(
        ["Metric", "Value"],
        [
            ["Total files scanned", summary["total_files_scanned"]],
            ["Valid images", summary["valid_images"]],
            ["Failed images", summary["failed_images"]],
        ],
    ))
    L.append("")

    # 2. Extensions
    L.append("## 2. File Extensions\n")
    ext = summary.get("by_extension", {})
    L.append(_md_table(
        ["Extension", "Count"],
        [[k, v] for k, v in sorted(ext.items(), key=lambda x: -x[1])],
    ))
    L.append("")

    # 3. Format
    L.append("## 3. Detected Formats\n")
    fmt = summary.get("by_format", {})
    L.append(_md_table(
        ["Format", "Count"],
        [[k, v] for k, v in sorted(fmt.items(), key=lambda x: -x[1])],
    ))
    L.append("")

    # 4. Dimensions & size
    L.append("## 4. Dimension & Size Statistics\n")
    for col in ("width", "height", "file_size_bytes", "pixel_count"):
        key = f"{col}_stats"
        if key in summary:
            st = summary[key]
            L.append(
                f"- **{col}**: min={st['min']:.0f}  median={st['median']:.0f}  "
                f"mean={st['mean']:.0f}  max={st['max']:.0f}  std={st['std']:.0f}"
            )
    L.append("")

    # 5. Boolean rates
    L.append("## 5. Metadata Presence Rates\n")
    bool_rows = []
    for col in ("has_exif", "has_icc_profile", "has_alpha", "is_grayscale",
                "has_xmp", "has_iptc", "jpeg_quantization_present",
                "jpeg_progressive"):
        key = f"{col}_rate"
        if key in summary and summary[key] is not None:
            bool_rows.append([col, f"{summary[key]:.4f}"])
    L.append(_md_table(["Feature", "Rate"], bool_rows))
    L.append("")

    # 6. Labels
    if "by_inferred_label" in summary:
        L.append("## 6. Inferred Labels\n")
        L.append(_md_table(
            ["Label", "Count"],
            [[k, v] for k, v in sorted(
                summary["by_inferred_label"].items(), key=lambda x: -x[1],
            )],
        ))
        L.append("")

    # 7. Generators
    if "by_inferred_generator" in summary:
        L.append("## 7. Inferred Generators\n")
        L.append(_md_table(
            ["Generator", "Count"],
            [[k, v] for k, v in sorted(
                summary["by_inferred_generator"].items(), key=lambda x: -x[1],
            )],
        ))
        L.append("")

    # 8. Folder structure (top 20)
    L.append("## 8. Folder Structure (top 20 by file count)\n")
    top_f = folder_df.nlargest(20, "file_count")
    L.append(_md_table(
        ["Folder", "Files", "Depth"],
        [[r["folder"], r["file_count"], r["depth"]] for _, r in top_f.iterrows()],
    ))
    L.append("")

    # 9. Text samples
    if text_samples:
        L.append("## 9. Top EXIF Text Values\n")
        for field, samples in text_samples.items():
            if not samples:
                continue
            L.append(f"### {field}\n")
            L.append(_md_table(
                ["Value", "Count"],
                [[s["value"][:80], s["count"]] for s in samples[:15]],
            ))
            L.append("")

    # 10. Shortcut alerts
    L.append("## 10. Shortcut Risk Alerts\n")
    if shortcuts.empty:
        L.append("_No alerts._\n")
    else:
        L.append(_md_table(
            ["Risk", "Feature", "Description", "Detail"],
            [
                [f"**{r['risk_level']}**", r["feature"],
                 r["description"], r.get("detail", "")]
                for _, r in shortcuts.iterrows()
            ],
        ))
        L.append("")

    return "\n".join(L)


# ===================================================================
# Export
# ===================================================================

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for ``json.dump``."""
    if hasattr(obj, "item"):  # numpy scalar
        v = obj.item()
        return None if isinstance(v, float) and v != v else v
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return str(obj)


def export_all(
    df: pd.DataFrame,
    summary: dict[str, Any],
    shortcuts: pd.DataFrame,
    text_samples: dict[str, list],
    folder_df: pd.DataFrame,
    report_md: str,
    out_dir: Path,
    fmt: str,
    *,
    generator_summary: pd.DataFrame | None = None,
    class_summary: pd.DataFrame | None = None,
    ood_summary: pd.DataFrame | None = None,
    quality_histograms: dict[str, Any] | None = None,
    format_distributions: dict[str, Any] | None = None,
    dup_df: pd.DataFrame | None = None,
    dup_summary: dict[str, Any] | None = None,
    symmetry_md: str | None = None,
) -> list[str]:
    """Write every output artefact. Returns list of created file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = _artifact_dirs(out_dir)
    created: list[str] = []

    write_csv = fmt in ("csv", "both")
    write_pq = fmt in ("parquet", "both")

    if write_pq and not _HAS_PYARROW:
        logger.warning("pyarrow not installed – parquet skipped, using CSV instead.")
        write_pq = False
        write_csv = True

    if write_csv:
        p = dirs["metadata"] / "per_file_metadata.csv"
        df.to_csv(p, index=False)
        created.append(str(p))
    if write_pq:
        p = dirs["metadata"] / "per_file_metadata.parquet"
        df.to_parquet(p, index=False, engine="pyarrow")
        created.append(str(p))

    # Summary JSON (include text samples + histograms + format dists)
    summary_out: dict[str, Any] = {
        **summary,
        "exif_text_samples": text_samples,
    }
    if quality_histograms:
        summary_out["quality_histograms"] = quality_histograms
    if format_distributions:
        summary_out["format_distributions"] = format_distributions
    if dup_summary:
        summary_out["duplicate_audit"] = dup_summary

    p = dirs["metadata"] / "metadata_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False,
                  default=_json_default)
    created.append(str(p))

    # Markdown report
    p = dirs["metadata"] / "metadata_report.md"
    with open(p, "w", encoding="utf-8") as f:
        f.write(report_md)
    created.append(str(p))

    # Shortcuts
    p = dirs["metadata"] / "suspicious_shortcuts.csv"
    shortcuts.to_csv(p, index=False)
    created.append(str(p))

    # Folder summary
    p = dirs["dataset_profile"] / "folder_structure_summary.csv"
    folder_df.to_csv(p, index=False)
    created.append(str(p))

    # --- New artefacts (Patches 1-5) ---

    # Generator breakdown
    if generator_summary is not None and not generator_summary.empty:
        p = dirs["dataset_profile"] / "generator_summary.csv"
        generator_summary.to_csv(p, index=False)
        created.append(str(p))

    # Class breakdown
    if class_summary is not None and not class_summary.empty:
        p = dirs["dataset_profile"] / "class_summary.csv"
        class_summary.to_csv(p, index=False)
        created.append(str(p))

    # OOD summary
    if ood_summary is not None and not ood_summary.empty:
        p = dirs["dataset_profile"] / "ood_summary.csv"
        ood_summary.to_csv(p, index=False)
        created.append(str(p))

    # Quality histograms
    if quality_histograms:
        p = dirs["dataset_profile"] / "quality_histograms.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(quality_histograms, f, indent=2, default=_json_default)
        created.append(str(p))

    # Format distributions
    if format_distributions:
        p = dirs["dataset_profile"] / "format_distributions.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(format_distributions, f, indent=2, default=_json_default)
        created.append(str(p))

    # Duplicate report
    if dup_df is not None and not dup_df.empty:
        p = dirs["duplicates"] / "duplicates_report.csv"
        dup_df.to_csv(p, index=False)
        created.append(str(p))
    if dup_summary:
        p = dirs["duplicates"] / "duplicate_summary.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dup_summary, f, indent=2, default=_json_default)
        created.append(str(p))

    # Symmetry checklist
    if symmetry_md:
        p = dirs["dataset_profile"] / "symmetry_audit_checklist.md"
        with open(p, "w", encoding="utf-8") as f:
            f.write(symmetry_md)
        created.append(str(p))

    return created


# ===================================================================
# Main
# ===================================================================

def main(argv: list[str] | None = None) -> None:
    args = build_cli(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format=_LOG_FMT,
    )

    root = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not root.is_dir():
        logger.error("Root directory does not exist: %s", root)
        sys.exit(1)

    logger.info("Audit started  – root: %s", root)
    logger.info("Config: workers=%d  hash=%s  format=%s  max_errors=%d",
                args.workers, args.hash, args.format, args.max_errors)
    t0 = time.perf_counter()

    # 1. Discover
    images = discover_images(root)
    logger.info("Discovered %d image files.", len(images))
    if not images:
        logger.warning("No image files found. Nothing to do.")
        sys.exit(0)

    # 2. Extract
    records, err_count = process_all(
        images, root, args.hash, args.workers, args.max_errors,
    )

    # 3. DataFrame
    df = pd.DataFrame(records)
    if "_error" not in df.columns:
        df["_error"] = None

    # 3a. OOD tagging (Patch 2)
    ood_set = (
        frozenset(_norm(g) for g in args.ood_generators)
        if args.ood_generators is not None
        else _DEFAULT_OOD_GENERATORS
    )
    tag_ood(df, ood_set)
    logger.info("OOD hold-out generators: %s", ", ".join(sorted(ood_set)))

    # 4. Summary
    summary = compute_summary(df)

    # 5. Folder structure
    folder_df = build_folder_summary(images, root)

    # 6. Shortcut analysis
    shortcuts, text_samples = analyse_shortcuts(df, args.sample_exif_text)

    # 6a. Generator & class breakdown (Patch 1)
    gen_sum = build_generator_summary(df)
    cls_sum = build_class_summary(df)

    # 6b. OOD summary (Patch 2)
    ood_sum = build_ood_summary(df)

    # 6c. Quality histograms & format distributions (Patch 3)
    q_hist = build_quality_histograms(df)
    fmt_dist = build_format_distributions(df)

    # 6d. Duplicate audit (Patch 4)
    dup_df, dup_summary = audit_duplicates(df)

    # 6e. Symmetry checklist (Patch 5)
    symmetry_md = build_symmetry_checklist()

    # 7. Report
    report_md = build_report(summary, shortcuts, text_samples, folder_df)

    # 8. Export
    created = export_all(
        df, summary, shortcuts, text_samples, folder_df,
        report_md, out_dir, args.format,
        generator_summary=gen_sum,
        class_summary=cls_sum,
        ood_summary=ood_sum,
        quality_histograms=q_hist,
        format_distributions=fmt_dist,
        dup_df=dup_df,
        dup_summary=dup_summary,
        symmetry_md=symmetry_md,
    )

    elapsed = time.perf_counter() - t0
    sep = "=" * 60
    logger.info(sep)
    logger.info("AUDIT COMPLETE")
    logger.info("  Total scanned  : %d", summary["total_files_scanned"])
    logger.info("  Valid images   : %d", summary["valid_images"])
    logger.info("  Failed images  : %d", summary["failed_images"])
    logger.info("  Elapsed        : %.1f s", elapsed)
    logger.info("  Output files:")
    for fp in created:
        logger.info("    -> %s", fp)
    logger.info(sep)


if __name__ == "__main__":
    main()
