"""
Load 8-bit uncompressed grayscale BMPs (DUKE_FLORIDA_150 frame images).

Handles top-down DIB (negative biHeight) and bottom-up (positive biHeight).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BmpInfo:
    width: int
    height: int
    bpp: int
    top_down: bool


def read_bmp_info(path: str | Path) -> BmpInfo:
    path = Path(path)
    with path.open("rb") as f:
        hdr = f.read(54)
    if len(hdr) < 54:
        raise ValueError(f"Truncated BMP header: {path}")
    w = struct.unpack_from("<i", hdr, 18)[0]
    h_signed = struct.unpack_from("<i", hdr, 22)[0]
    bpp = struct.unpack_from("<H", hdr, 28)[0]
    if bpp != 8:
        raise ValueError(f"Expected 8 bpp, got {bpp}: {path}")
    top_down = h_signed < 0
    h = abs(h_signed)
    return BmpInfo(width=w, height=h, bpp=bpp, top_down=top_down)


def read_bmp_gray(path: str | Path) -> np.ndarray:
    """
    Return HxW uint8 grayscale array. Row 0 is top of image (screen coordinates).
    """
    path = Path(path)
    with path.open("rb") as f:
        data = f.read()
    if len(data) < 54:
        raise ValueError(f"File too small: {path}")

    off_bits = struct.unpack_from("<I", data, 10)[0]
    w = struct.unpack_from("<i", data, 18)[0]
    h_signed = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    if bpp != 8:
        raise ValueError(f"Expected 8 bpp, got {bpp}: {path}")

    top_down = h_signed < 0
    h = abs(h_signed)
    stride = (w + 3) & ~3
    need = off_bits + stride * h
    if len(data) < need:
        raise ValueError(f"BMP pixel data truncated: {path} need {need} have {len(data)}")

    raw = np.frombuffer(data, dtype=np.uint8, offset=off_bits, count=stride * h)
    raw = raw.reshape((h, stride))[:, :w].copy()

    if not top_down:
        raw = np.flipud(raw)
    return raw


def write_bmp_gray(path: str | Path, arr: np.ndarray) -> None:
    """
    Write an ``H×W`` ``uint8`` grayscale image as an 8-bit uncompressed BMP.

    Layout matches what ``read_bmp_gray`` expects: BITMAPINFOHEADER, 256×4 BGRA
    palette, top-down DIB (negative ``biHeight``), row stride padded to 4 bytes.
    Uses only ``struct`` / ``numpy`` (no Pillow).
    """
    path = Path(path)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"write_bmp_gray expects H×W array, got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    stride = (w + 3) & ~3
    pad = stride - w

    # 256-entry BGRA palette: value k -> (k, k, k, 0)
    palette = np.empty((256, 4), dtype=np.uint8)
    palette[:, 0] = np.arange(256, dtype=np.uint8)  # B
    palette[:, 1] = palette[:, 0]  # G
    palette[:, 2] = palette[:, 0]  # R
    palette[:, 3] = 0

    # Top-down: file row order matches arr[0] = top (same as read_bmp_gray with negative height)
    if pad:
        pad_buf = np.zeros((h, pad), dtype=np.uint8)
        pixels = np.hstack((arr, pad_buf))
    else:
        pixels = arr
    pixel_bytes = pixels.tobytes()

    off_bits = 14 + 40 + 256 * 4
    file_size = off_bits + len(pixel_bytes)

    file_hdr = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, off_bits)
    dib_hdr = struct.pack(
        "<IiiHHIIiiII",
        40,  # biSize
        w,
        -h,  # negative height => top-down
        1,  # biPlanes
        8,  # biBitCount
        0,  # biCompression BI_RGB
        len(pixel_bytes),  # biSizeImage
        0,
        0,  # ppm
        256,  # biClrUsed
        256,  # biClrImportant
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(file_hdr)
        f.write(dib_hdr)
        f.write(palette.tobytes())
        f.write(pixel_bytes)
