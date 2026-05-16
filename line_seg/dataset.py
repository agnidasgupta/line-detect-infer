"""PyTorch Dataset: one span per item (variable T)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .volume import load_span_volume, raw_to_semantic_labels


class PoleSpanSegDataset(Dataset):
    """
    Each item is a full span: input ``[1, T, H, W]`` float (channel, then depth T), target ``[T, H, W]`` long.
    With ``batch_size=1``, ``DataLoader`` stacks to ``[1, 1, T, H, W]`` — the shape ``SpanUNet3D`` expects.

    Batch size should be **1** unless you add padding/collate for variable T.
    """

    def __init__(
        self,
        span_dirs: list[str | Path],
        *,
        max_frames: int | None = None,
        normalize: float = 255.0,
    ):
        self.span_dirs = [Path(p) for p in span_dirs]
        self.max_frames = max_frames
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.span_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        d = self.span_dirs[idx]
        vol = load_span_volume(d, max_frames=self.max_frames)
        x = torch.from_numpy(vol.astype(np.float32) / self.normalize).unsqueeze(0)
        y = torch.from_numpy(raw_to_semantic_labels(vol))
        return x, y, str(d.name)
