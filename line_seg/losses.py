"""Segmentation losses for Goal 2."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_ce_class_weights(s: str | None, num_classes: int) -> torch.Tensor | None:
    """
    Parse ``"w0,w1,...,w{n-1}"`` into a length-``num_classes`` float tensor (CPU).

    Raises ``ValueError`` on bad input (caller may convert to SystemExit).
    """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != num_classes:
        raise ValueError(f"expected {num_classes} comma-separated weights, got {len(parts)}")
    vals = [float(p) for p in parts]
    if any(v < 0 for v in vals):
        raise ValueError("CE class weights must be non-negative")
    return torch.tensor(vals, dtype=torch.float32)


def dice_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft multiclass Dice (one-vs-rest), mean over classes."""
    probs = F.softmax(logits, dim=1)
    loss = 0.0
    for c in range(num_classes):
        pt = probs[:, c]
        gt = (target == c).float()
        inter = (pt * gt).sum(dim=(1, 2, 3))
        union = pt.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
        loss = loss + (1.0 - (2.0 * inter + eps) / (union + eps)).mean()
    return loss / num_classes


def _weighted_focal_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multiclass focal-style loss: mean over voxels of
    ``w[y] * (1 - p_t)^gamma * (-log p_t)`` where ``p_t`` is softmax prob of the true class.

    For ``gamma == 0`` this matches weighted NLL (same as ``cross_entropy`` with ``weight``)
    up to numerical detail; use ``cross_entropy`` path in ``SegmentationLoss`` when gamma==0.
    """
    log_probs = F.log_softmax(logits, dim=1)
    if target.dtype != torch.long:
        target = target.long()
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp(min=eps, max=1.0 - eps)
    ce_noreduce = -log_pt
    w = class_weights.to(device=logits.device, dtype=logits.dtype)
    w_pix = w[target]
    mod = (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)
    return (w_pix * mod * ce_noreduce).mean()


class SegmentationLoss(nn.Module):
    """Cross-entropy (optionally focal) + optional multiclass Dice."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        *,
        line_class_ce_boost: float = 1.0,
        line_class_min: int = 2,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_gamma = float(focal_gamma)
        if class_weights is not None:
            w = class_weights.clone().float()
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
            w[line_class_min:] *= line_class_ce_boost
        self.register_buffer("class_weights", w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.class_weights.to(logits.device, dtype=logits.dtype)
        if self.focal_gamma > 0.0:
            ce = _weighted_focal_nll(logits, target, w, self.focal_gamma)
        else:
            ce = F.cross_entropy(logits, target.long(), weight=w)
        if self.dice_weight <= 0:
            return ce
        d = dice_loss_multiclass(logits, target.long(), num_classes=self.num_classes)
        return ce + self.dice_weight * d
