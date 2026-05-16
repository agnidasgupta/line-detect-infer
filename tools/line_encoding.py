"""
Line voxel encoding per Lines_Detect_Goals.txt (labeled BMPs).

Each labeled line pixel has a unique grayscale value in [128, 254]:
  - Lower 3 bits (gray & 0x07): line type
        0 = comm, 1 = primary, 2 = neutral, 3 = secondary, 4 = transmission
  - Upper 5 bits (gray >> 3): compact instance id field (0–31)

Packed value: gray = (id_field << 3) | type_code

Note: Documentation sometimes refers to a "semantic" conductor id starting at 32;
that is a naming convention outside the 8-bit packing. The 5-bit id_field is what
is stored in the upper bits of the gray byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

LINE_TYPE_NAMES: Final[tuple[str, ...]] = (
    "comm",
    "primary",
    "neutral",
    "secondary",
    "transmission",
)


@dataclass(frozen=True)
class LineLabelDecode:
    gray: int
    type_code: int
    type_name: str
    id_field: int  # 5-bit field: gray >> 3
    valid_line_byte: bool  # True if 128 <= gray < 255 and type in 0..4


def decode_line_gray(gray: int) -> LineLabelDecode:
    """Decode one 8-bit gray label value."""
    g = int(gray) & 0xFF
    if 128 <= g < 255:
        t = g & 0x07
        id_field = (g >> 3) & 0x1F
        if t < len(LINE_TYPE_NAMES):
            name = LINE_TYPE_NAMES[t]
            valid_type = t <= 4
        else:
            name = f"reserved_type_{t}"
            valid_type = False
        return LineLabelDecode(
            gray=g,
            type_code=t,
            type_name=name,
            id_field=id_field,
            valid_line_byte=valid_type,
        )
    return LineLabelDecode(
        gray=g,
        type_code=-1,
        type_name="not_line_label",
        id_field=-1,
        valid_line_byte=False,
    )


def pack_line_gray(id_field: int, type_code: int) -> int:
    """Reconstruct gray from 5-bit id field and 3-bit type (must match dataset convention)."""
    tid = int(id_field) & 0x1F
    tc = int(type_code) & 0x07
    return (tid << 3) | tc


def line_label_mask(img: np.ndarray, *, strict_type: bool = False) -> np.ndarray:
    """
    Boolean mask: labeled line voxels (128 <= value < 255).

    If strict_type is True, exclude pixels whose low 3 bits are not in 0..4
    (non-comm..transmission per spec).
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    m = (img >= 128) & (img < 255)
    if strict_type:
        tc = img & 0x07
        m &= tc <= 4
    return m


def unique_line_grays_in_span(frames: list[np.ndarray]) -> dict[int, LineLabelDecode]:
    """Collect unique gray line labels across frames with decodes."""
    out: dict[int, LineLabelDecode] = {}
    for im in frames:
        m = line_label_mask(im, strict_type=False)
        if not np.any(m):
            continue
        vals = np.unique(im[m])
        for v in vals.tolist():
            if v not in out:
                out[int(v)] = decode_line_gray(int(v))
    return out
