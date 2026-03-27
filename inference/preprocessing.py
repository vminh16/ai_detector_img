"""
inference/preprocessing.py
==========================
Hardened preprocessing engine adapted for byte-buffer input
(no file paths — the API receives raw bytes).

Pipeline:  decode → EXIF orient → pad (reflect) → center-crop 256×256
           (with grid misalignment) → JPEG bottleneck → BGR→YCrCb

Reuses the proven algorithms from ``src/preprocessing/pipeline.py``:
same constants, same grid-misalignment logic, same JPEG bottleneck,
same BT.601 YCrCb conversion.
"""
from __future__ import annotations

import hashlib
import struct

import cv2
import numpy as np

from inference.config import InferenceConfig
from inference.errors import InvalidImageError, PreprocessingError


# ── EXIF orientation from raw bytes ─────────────────────────────

_ORIENTATION_TAG: int = 0x0112


def _parse_ifd_orientation(data: bytes, endian: str, ifd_offset: int) -> int | None:
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
            if type_id == 3:
                val = struct.unpack_from(endian + "H", data, pos + 8)[0]
            elif type_id == 4:
                val = struct.unpack_from(endian + "I", data, pos + 8)[0]
            else:
                return None
            return val if 1 <= val <= 8 else None
        pos += 12
    return None


def _parse_tiff_header(data: bytes) -> int | None:
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


def parse_exif_orientation(payload: bytes) -> int | None:
    """Parse EXIF orientation from raw image bytes (JPEG APP1 or TIFF IFD0).

    Returns orientation ``int`` in ``[1, 8]`` or ``None``.
    """
    if len(payload) < 4:
        return None

    # JPEG: walk APP segments looking for Exif APP1
    if payload[:2] == b"\xff\xd8":
        pos = 2
        while pos < len(payload) - 4:
            if payload[pos] != 0xFF:
                return None
            mtype = payload[pos + 1]
            if mtype == 0xFF:
                pos += 1
                continue
            if mtype in (0xDA, 0xD9, 0x00):
                return None
            if pos + 4 > len(payload):
                return None
            seg_len = struct.unpack(">H", payload[pos + 2 : pos + 4])[0]
            if mtype == 0xE1:
                seg_data = payload[pos + 4 : pos + 2 + seg_len]
                if seg_data[:6] == b"Exif\x00\x00":
                    return _parse_tiff_header(seg_data[6:])
                return None
            pos += 2 + seg_len
        return None

    # TIFF-based (unlikely in API payloads, but supported)
    if payload[:2] in (b"II", b"MM"):
        return _parse_tiff_header(payload[:64 * 1024])

    return None


# ── Orientation transform dispatch (identical to training pipeline) ──

_ORIENTATION_OPS: dict[int, list[str]] = {
    1: [], 2: ["flip_h"], 3: ["rot_180"], 4: ["flip_v"],
    5: ["transpose"], 6: ["rot_cw"], 7: ["transpose", "flip_v"], 8: ["rot_ccw"],
}

_OP_FN = {
    "flip_h":    lambda i: cv2.flip(i, 1),
    "flip_v":    lambda i: cv2.flip(i, 0),
    "rot_180":   lambda i: cv2.rotate(i, cv2.ROTATE_180),
    "rot_cw":    lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE),
    "rot_ccw":   lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "transpose": lambda i: cv2.transpose(i),
}


def apply_orientation(img: np.ndarray, orientation: int | None) -> np.ndarray:
    if orientation is None or orientation not in _ORIENTATION_OPS:
        return img
    for name in _ORIENTATION_OPS[orientation]:
        img = _OP_FN[name](img)
    return img


# ── Core geometry + compression helpers ──────────────────────────

def decode_from_bytes(payload: bytes) -> np.ndarray:
    """Decode raw bytes → BGR uint8 using OpenCV (ignores EXIF orientation)."""
    arr = np.frombuffer(payload, dtype=np.uint8)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise InvalidImageError("Failed to decode image bytes.")
    return img


def pad_to_min_size(
    img: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, bool, int, int]:
    """Reflect-pad so both dims >= *min_size*.

    Returns ``(image, was_padded, pad_left, pad_top)``.
    """
    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img, False, 0, 0

    pad_top = max((min_size - h) // 2, 0)
    pad_bot = max(min_size - h - pad_top, 0)
    pad_left = max((min_size - w) // 2, 0)
    pad_right = max(min_size - w - pad_left, 0)

    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, True, pad_left, pad_top


def center_crop(
    img: np.ndarray,
    size: int,
    misalign_offset: int,
    pad_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Center-crop ``size×size`` with DCT grid misalignment."""
    h, w = img.shape[:2]
    x0 = w // 2 - size // 2
    y0 = h // 2 - size // 2

    orig_x0 = x0 - pad_origin[0]
    orig_y0 = y0 - pad_origin[1]

    if orig_x0 % 8 == 0 or orig_y0 % 8 == 0:
        x0 += misalign_offset
        y0 += misalign_offset

    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))

    return img[y0 : y0 + size, x0 : x0 + size]


def deterministic_q(key: str, q_min: int, q_max: int) -> int:
    """SHA-256 deterministic JPEG quality from *key*."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "little")
    return q_min + (val % (q_max - q_min + 1))


def jpeg_bottleneck(img: np.ndarray, quality: int) -> np.ndarray:
    """In-memory JPEG round-trip at *quality*."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise PreprocessingError("JPEG encode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise PreprocessingError("JPEG decode failed")
    return out


def bgr_to_ycrcb(img: np.ndarray) -> np.ndarray:
    """BGR uint8 → YCrCb uint8 (BT.601)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


# ── Full preprocessing pipeline ─────────────────────────────────

def preprocess_image(
    payload: bytes,
    image_hash: str,
    config: InferenceConfig,
) -> np.ndarray:
    """Run the full preprocessing pipeline on raw image bytes.

    Parameters
    ----------
    payload : raw uploaded image bytes
    image_hash : SHA-256 hex digest (used for deterministic JPEG Q)
    config : runtime configuration

    Returns
    -------
    np.ndarray
        256×256×3 ``uint8`` YCrCb array ready for feature extraction.

    Raises
    ------
    InvalidImageError
        If the bytes cannot be decoded.
    PreprocessingError
        If any preprocessing step fails.
    """
    try:
        # 1. Decode
        bgr = decode_from_bytes(payload)

        # 2. EXIF orientation
        orientation = parse_exif_orientation(payload)
        bgr = apply_orientation(bgr, orientation)

        # 3. Pad (reflect-101) to guarantee room for crop + misalignment
        bgr, _was_padded, pad_left, pad_top = pad_to_min_size(
            bgr, config.pad_min_size
        )

        # 4. Center-crop with grid misalignment
        bgr = center_crop(
            bgr,
            size=config.crop_size,
            misalign_offset=config.grid_misalign_offset,
            pad_origin=(pad_left, pad_top),
        )

        # 5. JPEG bottleneck (deterministic Q from image hash)
        q = deterministic_q(image_hash, config.jpeg_q_min, config.jpeg_q_max)
        bgr = jpeg_bottleneck(bgr, q)

        # 6. BGR → YCrCb
        ycrcb = bgr_to_ycrcb(bgr)

        return ycrcb

    except (InvalidImageError, PreprocessingError):
        raise
    except Exception as exc:
        raise PreprocessingError(f"Unexpected preprocessing error: {exc}") from exc
