"""
Segmentation evaluation: confusion matrix, per-class IoU / precision / recall,
macro averages, and optional 3D connected-component metrics for merged "line" objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage


def confusion_matrix_to_text(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    title: str = "Confusion matrix (rows = true class, columns = predicted class)",
) -> str:
    """Human-readable ASCII confusion matrix: counts and row-normalized (recall per true class)."""
    cm = np.asarray(cm, dtype=np.int64)
    names = list(class_names)
    lines = [title, ""]
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_norm = np.divide(cm.astype(np.float64), np.maximum(row_sums, 1e-9))

    # Count block
    w = max(8, max(len(n) for n in names) if names else 8)
    header = " " * w + "".join(f"{j:>10}" for j in range(len(names)))
    lines.append("Counts:")
    lines.append(header)
    lines.append(" " * w + "".join(f"{names[j][:8]:>10}" for j in range(len(names))))
    for i, name in enumerate(names):
        row = f"{name[:w]:<{w}}" + "".join(f"{int(cm[i, j]):>10}" for j in range(len(names)))
        lines.append(row)
    lines.append("")
    lines.append("Row-normalized: for pixels with true label i, fraction predicted as j (row sums to 1).")
    lines.append("Diagonal entries are per-class recall; off-diagonals are confusion into other classes.")
    lines.append(header)
    lines.append(" " * w + "".join(f"{names[j][:8]:>10}" for j in range(len(names))))
    for i, name in enumerate(names):
        row = f"{name[:w]:<{w}}" + "".join(f"{row_norm[i, j]:>10.4f}" for j in range(len(names)))
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def training_history_diagnosis(rows: list[dict], *, val_loss_key: str = "val_loss", iou_key: str = "val_mean_iou") -> str:
    """
    Interpret epoch logs (no pixel CM): spikes, train/val gap, plateau, checkpoint sensitivity.
    Works with minimal logs that only have train_loss / val_loss / val_mean_iou.
    """
    if not rows:
        return "(No training history rows provided.)\n"
    lines: list[str] = ["=== Training-log diagnosis (from train_log.jsonl) ===", ""]
    epochs = [int(r["epoch"]) for r in rows]
    tl = np.array([float(r["train_loss"]) for r in rows])
    vl = np.array([float(r[val_loss_key]) for r in rows])
    ious = np.array([float(r[iou_key]) for r in rows if iou_key in r])

    lines.append(f"Epochs: {epochs[0]}–{epochs[-1]}  (n={len(rows)})")
    lines.append(f"Train loss:  start {tl[0]:.4f}  end {tl[-1]:.4f}  min {tl.min():.4f}")
    lines.append(f"Val loss:    start {vl[0]:.4f}  end {vl[-1]:.4f}  min {vl.min():.4f}  std {vl.std():.4f}")
    if len(ious) == len(rows):
        lines.append(f"Val mIoU:    start {ious[0]:.4f}  end {ious[-1]:.4f}  max {ious.max():.4f}  std {ious.std():.4f}")
        best_i = int(np.argmax(ious))
        best_l = int(np.argmin(vl))
        lines.append(f"Best val mIoU at epoch {epochs[best_i]} ({ious[best_i]:.4f}); best val loss at epoch {epochs[best_l]} ({vl[best_l]:.4f}).")
        if best_i != best_l:
            lines.append(
                "Note: lowest val loss and highest mIoU occur on different epochs — "
                "prefer checkpointing on val_mean_iou for segmentation (--checkpoint_metric val_mean_iou)."
            )

    # Spike detection: val loss above rolling median + k * MAD-ish (use simple z on diff)
    if len(vl) >= 5:
        med = float(np.median(vl))
        mad = float(np.median(np.abs(vl - med))) + 1e-9
        z = (vl - med) / (1.4826 * mad + 1e-9)
        spike_idx = np.where(z > 3.5)[0]
        if len(spike_idx) > 0:
            sp = ", ".join(f"epoch {epochs[int(i)]} (val_loss={vl[int(i)]:.4f})" for i in spike_idx)
            lines.append("")
            lines.append(f"Large val-loss spikes (robust z > 3.5 vs median): {sp}.")
            lines.append(
                "With batch_size=1, BatchNorm running stats are unstable; GroupNorm in the UNet "
                "reduces this. A small val set (~15% of spans) also increases metric variance."
            )

    gap = vl - tl
    lines.append("")
    lines.append(f"Mean(val_loss - train_loss) last 5 epochs: {gap[-5:].mean():.4f} (positive => val harder than train).")

    lines.append("")
    lines.append("Suggested next steps (implemented in current line_seg defaults / flags):")
    lines.append("- Use GroupNorm instead of BatchNorm in ConvBlock3D (batch_size=1).")
    lines.append("- Use ReduceLROnPlateau + mild gradient clipping.")
    lines.append("- Consider --checkpoint_metric val_mean_iou so best.pt tracks segmentation quality, not noisy CE+Dice.")
    lines.append("- Optional --line_class_ce_boost > 1 to up-weight rare line classes in CE.")
    lines.append("- Increase val_fraction or fixed val list for stabler curves if you have enough spans.")
    lines.append("")
    return "\n".join(lines)


def confusion_matrix_accumulate(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    num_classes: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add counts for one volume to a ``num_classes x num_classes`` confusion matrix
    (rows = true class, cols = predicted class).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy().ravel()
    else:
        pred = np.asarray(pred).ravel()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy().ravel()
    else:
        target = np.asarray(target).ravel()
    if out is None:
        out = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    np.add.at(out, (target[valid], pred[valid]), 1)
    return out


def confusion_matrix_accumulate_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    Add one volume's counts to a confusion matrix on **GPU** (rows=true, cols=pred).
    ``pred`` and ``target`` are shaped ``[T,H,W]`` or any shape broadcastable to flat indices.
    """
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    if idx.numel() == 0:
        return out
    cnt = torch.bincount(idx, minlength=num_classes * num_classes).to(device=out.device, dtype=out.dtype)
    out.add_(cnt.view(num_classes, num_classes))
    return out


@dataclass
class SegmentationScores:
    """Derived from a confusion matrix (rows=true, cols=pred)."""
    per_class_iou: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    mean_iou: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    pixel_accuracy: float


def scores_from_confusion(cm: np.ndarray, eps: float = 1e-9) -> SegmentationScores:
    """Per-class IoU, precision, recall; macro averages; overall pixel accuracy."""
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)  # sum over true -> per pred col
    true_sum = cm.sum(axis=1).astype(np.float64)  # per true row

    precision = tp / (pred_sum + eps)
    recall = tp / (true_sum + eps)
    union = true_sum + pred_sum - tp
    iou = tp / (union + eps)

    macro_p = float(np.nanmean(precision))
    macro_r = float(np.nanmean(recall))
    macro_f1 = float(2 * macro_p * macro_r / (macro_p + macro_r + eps)) if (macro_p + macro_r) > 0 else 0.0
    mean_iou = float(np.nanmean(iou))
    total = cm.sum()
    acc = float(tp.sum() / (total + eps))

    return SegmentationScores(
        per_class_iou=iou,
        per_class_precision=precision,
        per_class_recall=recall,
        mean_iou=mean_iou,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        pixel_accuracy=acc,
    )


def binary_line_masks(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    line_class_min: int = 2,
    line_class_max: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge semantic line classes into single binary 'any conductor' masks."""
    p = (pred >= line_class_min) & (pred <= line_class_max)
    t = (target >= line_class_min) & (target <= line_class_max)
    return p.astype(np.bool_), t.astype(np.bool_)


def line_object_detection_scores(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    line_class_min: int = 2,
    line_class_max: int = 6,
    iou_match_thresh: float = 0.5,
) -> dict[str, float]:
    """
    3D connected components on merged line masks; greedy IoU matching.

    Returns precision / recall / F1 for **instances** (blobs), plus binary pixel IoU
    for the merged line mask.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    pb, tb = binary_line_masks(pred, target, line_class_min=line_class_min, line_class_max=line_class_max)

    inter = np.logical_and(pb, tb).sum()
    union = np.logical_or(pb, tb).sum()
    line_pixel_iou = float(inter / (union + 1e-9))

    struct = ndimage.generate_binary_structure(3, 1)
    lp, np_ = ndimage.label(pb, structure=struct)
    lg, ng = ndimage.label(tb, structure=struct)

    if np_ == 0 and ng == 0:
        return {
            "line_obj_precision": 1.0,
            "line_obj_recall": 1.0,
            "line_obj_f1": 1.0,
            "line_obj_tp": 0.0,
            "line_obj_fp": 0.0,
            "line_obj_fn": 0.0,
            "line_pixel_iou": line_pixel_iou,
            "line_pred_instances": 0.0,
            "line_gt_instances": 0.0,
        }

    matched_gt = set()
    tp = 0
    for pid in range(1, np_ + 1):
        pm = lp == pid
        best_iou = 0.0
        best_gid = -1
        for gid in range(1, ng + 1):
            if gid in matched_gt:
                continue
            gm = lg == gid
            inter_b = np.logical_and(pm, gm).sum()
            uni_b = np.logical_or(pm, gm).sum()
            iou = float(inter_b / (uni_b + 1e-9))
            if iou > best_iou:
                best_iou = iou
                best_gid = gid
        if best_iou >= iou_match_thresh and best_gid >= 0:
            tp += 1
            matched_gt.add(best_gid)

    fp = np_ - tp
    fn = ng - len(matched_gt)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9) if (prec + rec) > 0 else 0.0

    return {
        "line_obj_precision": float(prec),
        "line_obj_recall": float(rec),
        "line_obj_f1": float(f1),
        "line_obj_tp": float(tp),
        "line_obj_fp": float(fp),
        "line_obj_fn": float(fn),
        "line_pixel_iou": line_pixel_iou,
        "line_pred_instances": float(np_),
        "line_gt_instances": float(ng),
    }


def aggregate_object_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Micro line-object precision/recall/F1: sum TP/FP/FN across all validation spans."""
    if not metrics_list:
        return {}
    tp = sum(m.get("line_obj_tp", 0.0) for m in metrics_list)
    fp = sum(m.get("line_obj_fp", 0.0) for m in metrics_list)
    fn = sum(m.get("line_obj_fn", 0.0) for m in metrics_list)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9) if (prec + rec) > 0 else 0.0
    pix = [m.get("line_pixel_iou", 0.0) for m in metrics_list]
    return {
        "line_obj_precision_micro": float(prec),
        "line_obj_recall_micro": float(rec),
        "line_obj_f1_micro": float(f1),
        "line_pixel_iou_mean_span": float(np.mean(pix)) if pix else 0.0,
        "line_pred_instances_total": sum(m.get("line_pred_instances", 0) for m in metrics_list),
        "line_gt_instances_total": sum(m.get("line_gt_instances", 0) for m in metrics_list),
    }
