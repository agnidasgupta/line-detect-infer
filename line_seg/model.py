"""
Lightweight 3D UNet for T×H×W span volumes.

Decoder uses trilinear upsampling + conv to avoid off-by-one issues with odd H,W.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_groups(num_channels: int) -> int:
    """Pick a GroupNorm group count that divides num_channels (prefer 8, then smaller)."""
    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    """Conv–Norm–ReLU ×2. Uses GroupNorm so batch_size=1 does not destabilize training like BatchNorm."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        g = _group_norm_groups(c_out)
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpanUNet3D(nn.Module):
    """
    Semantic segmentation: logits [B, num_classes, T, H, W].

    ``in_channels``: usually 1 (raw / 255).
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1, base: int = 24):
        super().__init__()
        b = base
        self.enc1 = ConvBlock3D(in_channels, b)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = ConvBlock3D(b, b * 2)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.mid = ConvBlock3D(b * 2, b * 4)
        self.dec2 = ConvBlock3D(b * 2 + b * 4, b * 2)
        self.dec1 = ConvBlock3D(b + b * 2, b)
        self.out_conv = nn.Conv3d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        m = self.mid(p2)
        u2 = F.interpolate(m, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out_conv(d1)
