"""
Load span BMP sequences as T×H×W volumes and build semantic segmentation targets.

Classes (Goal 2 semantic head):
  0 = air (gray 255)
  1 = solid / non-line (gray 0)
  2 = comm (line type 0)
  3 = primary (1)
  4 = neutral (2)
  5 = secondary (3)
  6 = transmission (4)

Line pixels with invalid low-3-bit type (>4) are mapped to class 1 (solid).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.bmp_io import read_bmp_gray


def load_span_volume(span_dir: str | Path, *, max_frames: int | None = None) -> np.ndarray:
    """Return uint8 array [T, H, W] of raw BMP pixels."""
    span_dir = Path(span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frame_*.bmp under {span_dir}")
    frames = [read_bmp_gray(p) for p in paths]
    return np.stack(frames, axis=0)


def raw_to_semantic_labels(vol: np.ndarray) -> np.ndarray:
    """
    Map raw uint8 volume to int64 class map [T,H,W], values 0..6.
    """
    vol = vol.astype(np.uint8)
    out = np.zeros(vol.shape, dtype=np.int64)
    air = vol == 255
    solid = vol == 0
    out[air] = 0
    out[solid] = 1
    line = (vol >= 128) & (vol < 255)
    tc = (vol[line] & 7).astype(np.int64)
    idx = np.where(line)
    # class = 2 + type_code for types 0..4
    cls = np.where(tc <= 4, 2 + tc, 1)
    out[idx] = cls
    return out


def list_span_dirs(data_root: str | Path) -> list[Path]:
    """Immediate subdirectories of data_root that look like span folders (contain frame_*.bmp)."""
    data_root = Path(data_root)
    out: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if list(p.glob("frame_*.bmp")):
            out.append(p)
    return out
