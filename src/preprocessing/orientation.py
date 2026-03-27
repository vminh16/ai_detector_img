"""EXIF orientation utilities for preprocessing v4."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ORIENTATION_TAG: int = 0x0112
_PIL_TRANSPOSE = Image.Transpose if hasattr(Image, "Transpose") else Image


def _parse_ifd_orientation(data: bytes, endian: str, ifd_offset: int) -> int | None:
    if ifd_offset + 2 > len(data):
        return None
    num_entries = struct.unpack_from(endian + "H", data, ifd_offset)[0]
    pos = ifd_offset + 2
    for _ in range(num_entries):
        if pos + 12 > len(data):
            return None
        tag = struct.unpack_from(endian + "H", data, pos)[0]
        if tag == ORIENTATION_TAG:
            type_id = struct.unpack_from(endian + "H", data, pos + 2)[0]
            if type_id == 3:
                value = struct.unpack_from(endian + "H", data, pos + 8)[0]
            elif type_id == 4:
                value = struct.unpack_from(endian + "I", data, pos + 8)[0]
            else:
                return None
            return value if 1 <= value <= 8 else None
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


def _read_jpeg_orientation(filepath: Path) -> int | None:
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
    try:
        with open(filepath, "rb") as fh:
            header = fh.read(64 * 1024)
        return _parse_tiff_header(header)
    except OSError:
        return None


def _read_pillow_orientation(filepath: Path) -> int | None:
    try:
        with Image.open(filepath) as image:
            exif = image.getexif()
            if exif is None:
                return None
            value = exif.get(ORIENTATION_TAG)
    except (OSError, ValueError, UnidentifiedImageError):
        return None
    if isinstance(value, int) and 1 <= value <= 8:
        return value
    return None


def read_orientation(filepath: Path | str) -> int | None:
    """Return EXIF orientation in [1, 8] when present."""

    path = Path(filepath)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        orientation = _read_jpeg_orientation(path)
        if orientation is not None:
            return orientation
    elif ext in {".tif", ".tiff"}:
        orientation = _read_tiff_orientation(path)
        if orientation is not None:
            return orientation
    return _read_pillow_orientation(path)


_ORIENTATION_OPS_ARRAY: dict[int, list[str]] = {
    1: [],
    2: ["flip_h"],
    3: ["rot_180"],
    4: ["flip_v"],
    5: ["transpose"],
    6: ["rot_cw"],
    7: ["transpose", "flip_both"],
    8: ["rot_ccw"],
}

_OP_DISPATCH_ARRAY: dict[str, Any] = {
    "flip_h": lambda image: cv2.flip(image, 1),
    "flip_v": lambda image: cv2.flip(image, 0),
    "flip_both": lambda image: cv2.flip(image, -1),
    "rot_180": lambda image: cv2.rotate(image, cv2.ROTATE_180),
    "rot_cw": lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    "rot_ccw": lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "transpose": lambda image: cv2.transpose(image),
}

_ORIENTATION_OPS_PIL: dict[int, tuple[Any, ...]] = {
    1: (),
    2: (_PIL_TRANSPOSE.FLIP_LEFT_RIGHT,),
    3: (_PIL_TRANSPOSE.ROTATE_180,),
    4: (_PIL_TRANSPOSE.FLIP_TOP_BOTTOM,),
    5: (_PIL_TRANSPOSE.TRANSPOSE,),
    6: (_PIL_TRANSPOSE.ROTATE_270,),
    7: (_PIL_TRANSPOSE.TRANSVERSE,),
    8: (_PIL_TRANSPOSE.ROTATE_90,),
}


def apply_orientation(image: np.ndarray, orientation: int | None) -> np.ndarray:
    """Apply EXIF orientation to an ndarray image."""

    if orientation is None or orientation not in _ORIENTATION_OPS_ARRAY:
        return image
    out = image
    for op_name in _ORIENTATION_OPS_ARRAY[orientation]:
        out = _OP_DISPATCH_ARRAY[op_name](out)
    return out


def apply_orientation_pil(image: Image.Image, orientation: int | None) -> Image.Image:
    """Apply EXIF orientation to a PIL image."""

    if orientation is None or orientation not in _ORIENTATION_OPS_PIL:
        return image
    out = image
    for op in _ORIENTATION_OPS_PIL[orientation]:
        out = out.transpose(op)
    return out
