#!/usr/bin/env python3
"""
Goal 2: semantic segmentation of span volumes (air / solid / line types).

Trains a 3D UNet and writes:
  - train_log.jsonl — per-epoch metrics (loss, IoU, precision/recall, object metrics)
  - metrics_history.json — same as list
  - history_loss.png, history_scores.png — training curves
  - After training: confusion_matrix.png, confusion_matrix.npy, **validation_report.txt**
    (ASCII confusion matrix + sklearn classification_report + JSON summary + log diagnosis)
    from the **best** checkpoint on the full validation set.

Usage (from DUKE_FLORIDA_150):

    python3 -m line_seg.train_goal2 --data_root . --out_dir ./goal2_runs/exp1 --epochs 20

Multi-GPU (NCCL): ``--nproc_per_node`` must equal **visible** GPU count (e.g. 2 for two A6000s).

    torchrun --standalone --nproc_per_node=2 -m line_seg.train_goal2 \\
      --data_root . --out_dir ./goal2_runs/exp_mg --epochs 30

Or let the script pick the count (re-launches via ``torch.distributed.run``):

    python3 -m line_seg.train_goal2 --multi_gpu --data_root . --out_dir ./goal2_runs/exp_mg --epochs 30

See README_GOAL2.md. For **inference** on new spans (optional labels), use ``python3 -m line_seg.infer_goal2``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_seg.dataset import PoleSpanSegDataset
from line_seg.eval_metrics import (
    aggregate_object_metrics,
    confusion_matrix_accumulate_torch,
    confusion_matrix_to_text,
    line_object_detection_scores,
    scores_from_confusion,
    training_history_diagnosis,
)
from line_seg.losses import SegmentationLoss, parse_ce_class_weights
from line_seg.model import SpanUNet3D
from line_seg.volume import list_span_dirs

CLASS_NAMES = [
    "air",
    "solid",
    "comm",
    "primary",
    "neutral",
    "secondary",
    "transmission",
]


def set_seed(s: int) -> None:
    """Same seed on every rank so DDP starts from identical weights."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _worker_init_fn(worker_id: int, base_seed: int, rank: int) -> None:
    ss = base_seed + worker_id + rank * 1000
    np.random.seed(ss)
    random.seed(ss)


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _setup_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    """
    NCCL distributed training when launched with torchrun (WORLD_SIZE > 1).
    Returns: (distributed, rank, local_rank, world_size, device).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training (WORLD_SIZE>1) requires CUDA; use torchrun with NCCL.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        n_cuda = torch.cuda.device_count()
        if local_rank >= n_cuda:
            raise RuntimeError(
                f"Distributed local_rank={local_rank} but only {n_cuda} CUDA device(s) are visible "
                f"(need LOCAL_RANK in 0..{n_cuda - 1}). Usually `--nproc_per_node` exceeds the GPU count.\n"
                f"Fix: torchrun --standalone --nproc_per_node={n_cuda} -m line_seg.train_goal2 ...\n"
                f"Or: python3 -m line_seg.train_goal2 --multi_gpu ...  (sets nproc to visible GPU count)\n"
                f"If CUDA_VISIBLE_DEVICES limits GPUs, nproc_per_node must match that count."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        ws = dist.get_world_size()
        return True, rank, local_rank, ws, torch.device("cuda", local_rank)
    if device_arg == "cpu" or not torch.cuda.is_available():
        return False, 0, 0, 1, torch.device("cpu")
    return False, 0, 0, 1, torch.device(device_arg)


def _teardown_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _reexec_with_torchrun_multi_gpu(argv_without_script: list[str], nproc: int) -> None:
    """Replace this process with torch.distributed.run using all ``nproc`` visible GPUs."""
    print(
        f"[train_goal2] Starting multi-GPU: re-exec with torch.distributed.run "
        f"--nproc_per_node={nproc} (visible CUDA devices).",
        file=sys.stderr,
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "line_seg.train_goal2",
        *argv_without_script,
    ]
    os.execvp(sys.executable, cmd)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module | None = None,
    *,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> dict:
    """Full pass: confusion matrix (GPU), loss, sklearn-style aggregates, line-object metrics."""
    model.eval()
    m = _unwrap_model(model)
    cm_t = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    n_batches = 0
    object_per_span: list[dict[str, float]] = []
    # Stream labels to CPU per batch (avoid torch.cat of huge GPU tensors → OOM)
    y_chunks: list[np.ndarray] = []
    p_chunks: list[np.ndarray] = []

    for x, y, _name in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                logits = m(x)
                if criterion is not None:
                    loss_sum_t += criterion(logits, y).float()
                    n_batches += 1
        else:
            logits = m(x)
            if criterion is not None:
                loss_sum_t += criterion(logits, y).float()
                n_batches += 1
        pred = logits.argmax(dim=1)[0].long()
        yt = y[0].long()
        confusion_matrix_accumulate_torch(pred, yt, num_classes, cm_t)
        # scipy CC: one contiguous CPU copy per span (frees GPU before next span)
        p_cpu = pred.detach().contiguous().cpu().numpy()
        y_cpu = yt.detach().contiguous().cpu().numpy()
        del logits, pred, yt
        object_per_span.append(line_object_detection_scores(p_cpu, y_cpu))
        p_chunks.append(p_cpu.reshape(-1))
        y_chunks.append(y_cpu.reshape(-1))

    y_flat = np.concatenate(y_chunks) if y_chunks else np.array([], dtype=np.int64)
    p_flat = np.concatenate(p_chunks) if p_chunks else np.array([], dtype=np.int64)
    cm = cm_t.cpu().numpy()
    scores = scores_from_confusion(cm)
    obj_micro = aggregate_object_metrics(object_per_span)

    loss_mean = (loss_sum_t / max(n_batches, 1)).item() if criterion is not None and n_batches > 0 else None
    if y_flat.size == 0:
        report_str = "(empty validation loader — no classification_report)\n"
    else:
        report_str = classification_report(
            y_flat,
            p_flat,
            labels=list(range(num_classes)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    out = {
        "loss": loss_mean,
        "confusion_matrix": cm,
        "mean_iou": scores.mean_iou,
        "macro_precision": scores.macro_precision,
        "macro_recall": scores.macro_recall,
        "macro_f1": scores.macro_f1,
        "pixel_accuracy": scores.pixel_accuracy,
        "per_class_iou": scores.per_class_iou.tolist(),
        "per_class_precision": scores.per_class_precision.tolist(),
        "per_class_recall": scores.per_class_recall.tolist(),
        **obj_micro,
    }
    out["classification_report_str"] = report_str
    return out


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    tick = np.arange(len(class_names))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "k", fontsize=8)
    ax.set_title("Validation confusion matrix (pixels)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_curves(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train_loss", marker="o", ms=3)
    ax.plot(epochs, [r["val_loss"] for r in rows], label="val_loss", marker="s", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Training vs validation loss")
    fig.tight_layout()
    fig.savefig(out_dir / "history_loss.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["val_mean_iou"] for r in rows], label="val mIoU", marker="o", ms=3)
    if rows and all("val_macro_precision" in r for r in rows):
        ax.plot(epochs, [r["val_macro_precision"] for r in rows], label="val macro P", alpha=0.85)
        ax.plot(epochs, [r["val_macro_recall"] for r in rows], label="val macro R", alpha=0.85)
        ax.plot(epochs, [r["val_macro_f1"] for r in rows], label="val macro F1", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Validation segmentation scores")
    fig.tight_layout()
    fig.savefig(out_dir / "history_scores.png", dpi=140)
    plt.close(fig)


def _span_paths_for_meta(span_list: list[Path], data_root: Path) -> list[str]:
    """Store span paths relative to data_root when possible (portable across machines)."""
    root = data_root.resolve()
    out: list[str] = []
    for s in span_list:
        p = Path(s).resolve()
        try:
            out.append(str(p.relative_to(root)))
        except ValueError:
            out.append(str(p))
    return out


def build_validation_report_text(
    final: dict,
    class_names: list[str],
    *,
    history: list[dict] | None = None,
    report_title: str | None = None,
) -> str:
    """Single text file: confusion matrix (ASCII) + sklearn report + JSON scalars + optional log diagnosis."""
    title = report_title or "Goal 2 — validation report (best checkpoint, full validation set)"
    summary = {
        "mean_iou": final["mean_iou"],
        "macro_precision": final["macro_precision"],
        "macro_recall": final["macro_recall"],
        "macro_f1": final["macro_f1"],
        "pixel_accuracy": final["pixel_accuracy"],
        "per_class_iou": dict(zip(class_names, final["per_class_iou"])),
        "per_class_precision": dict(zip(class_names, final["per_class_precision"])),
        "per_class_recall": dict(zip(class_names, final["per_class_recall"])),
        "line_object_micro_precision": final.get("line_obj_precision_micro"),
        "line_object_micro_recall": final.get("line_obj_recall_micro"),
        "line_object_micro_f1": final.get("line_obj_f1_micro"),
        "line_pixel_iou_mean_span": final.get("line_pixel_iou_mean_span"),
        "line_pred_instances_total": final.get("line_pred_instances_total"),
        "line_gt_instances_total": final.get("line_gt_instances_total"),
        "val_loss": final.get("loss"),
    }
    parts = [
        title,
        "",
        confusion_matrix_to_text(final["confusion_matrix"], class_names),
        "=== Sklearn classification_report (pixels; support = pixel count) ===",
        "",
        final["classification_report_str"].rstrip(),
        "",
        "=== Scalar summary (JSON) ===",
        "",
        json.dumps(summary, indent=2),
        "",
    ]
    if history:
        parts.append(training_history_diagnosis(history))
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Goal 2: 3D UNet semantic segmentation on pole spans",
        epilog="Inference on a separate input tree (labeled or unlabeled BMPs): "
        "python3 -m line_seg.infer_goal2 --weights ./goal2_runs/exp/best.pt "
        "--input_dir ./path/to/spans --output_dir ./inference_out --device cuda",
    )
    p.add_argument("--data_root", type=str, default=".", help="DUKE_FLORIDA_150 root (contains span folders)")
    p.add_argument("--out_dir", type=str, default="./goal2_runs/run", help="Checkpoints and logs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of spans for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_frames", type=int, default=None, help="Cap frames per span (debug)")
    p.add_argument(
        "--checkpoint_metric",
        type=str,
        default="val_mean_iou",
        choices=("val_loss", "val_mean_iou"),
        help="Save best.pt when this metric improves (min loss or max mIoU).",
    )
    p.add_argument(
        "--line_class_ce_boost",
        type=float,
        default=2.0,
        help="Multiply CE loss weight for line classes 2–6 (1.0 = uniform). Ignored if --ce_class_weights is set.",
    )
    p.add_argument(
        "--ce_class_weights",
        type=str,
        default=None,
        metavar="W0,W1,...,W6",
        help="Override per-class CE weights (7 comma-separated non-negative floats, order: air..transmission). "
        "When set, --line_class_ce_boost is ignored for CE.",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Focal modulation exponent on (1 - p_true). 0 = plain weighted CE. Typical: 1.0–2.0 for hard examples.",
    )
    p.add_argument("--grad_clip_norm", type=float, default=1.0, help="0 disables gradient clipping")
    p.add_argument("--lr_patience", type=int, default=4, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr_factor", type=float, default=0.5, help="LR multiply factor on plateau")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(8, CPU count); 0 loads in main thread).",
    )
    p.add_argument("--no_amp", action="store_true", help="Disable CUDA automatic mixed precision (fp16/bf16)")
    p.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="torch.backends.cudnn.benchmark=True (can help fixed shapes; variable T may not benefit).",
    )
    p.add_argument(
        "--torch_compile",
        action="store_true",
        help="Wrap the UNet with torch.compile (PyTorch 2+; may fail on some stacks).",
    )
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use every visible CUDA GPU: re-exec via torch.distributed.run with "
        "nproc_per_node=torch.cuda.device_count(). Ignored under torchrun (WORLD_SIZE>1). "
        "Single-GPU machine: no-op.",
    )
    args = p.parse_args()

    ws_env = int(os.environ.get("WORLD_SIZE", "1"))
    if args.multi_gpu and ws_env <= 1:
        if not torch.cuda.is_available():
            print("[warn] --multi_gpu ignored: CUDA not available")
        else:
            n_gpus = torch.cuda.device_count()
            if n_gpus <= 1:
                print("[warn] --multi_gpu ignored: only one visible CUDA device (check CUDA_VISIBLE_DEVICES)")
            else:
                argv_f = [a for a in sys.argv[1:] if a != "--multi_gpu"]
                _reexec_with_torchrun_multi_gpu(argv_f, n_gpus)

    distributed, rank, local_rank, world_size, device = _setup_distributed(args.device)
    is_main = rank == 0
    pin_memory = device.type == "cuda"
    use_amp = not args.no_amp and device.type == "cuda"
    nb = pin_memory
    nw = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 1)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    set_seed(args.seed)
    data_root = Path(args.data_root)
    spans = list_span_dirs(data_root)
    if len(spans) < 2:
        _teardown_distributed(distributed)
        raise SystemExit(f"Need at least 2 span directories under {data_root}, found {len(spans)}")

    rng = random.Random(args.seed)
    rng.shuffle(spans)
    n_val = max(1, int(len(spans) * args.val_fraction))
    val_spans = spans[:n_val]
    train_spans = spans[n_val:]
    if is_main:
        print(
            f"[data] train_spans={len(train_spans)} val_spans={len(val_spans)}  "
            f"world_size={world_size} device={device} amp={use_amp} workers={nw} pin_mem={pin_memory}"
        )

    train_ds = PoleSpanSegDataset(train_spans, max_frames=args.max_frames)
    val_ds = PoleSpanSegDataset(val_spans, max_frames=args.max_frames)
    if distributed and len(train_ds) < world_size:
        _teardown_distributed(distributed)
        raise SystemExit(
            f"train set ({len(train_ds)} spans) must be >= world_size ({world_size}); "
            "lower --nproc_per_node or add data."
        )
    train_sampler: DistributedSampler | None = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    worker_init = partial(_worker_init_fn, base_seed=args.seed, rank=rank) if nw > 0 else None
    prefetch = 4 if nw > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=nw > 0,
        prefetch_factor=prefetch,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=nw > 0,
        prefetch_factor=prefetch,
        worker_init_fn=worker_init,
    )

    num_classes = 7
    try:
        ce_override = parse_ce_class_weights(args.ce_class_weights, num_classes)
    except ValueError as e:
        _teardown_distributed(distributed)
        raise SystemExit(f"--ce_class_weights: {e}") from e

    base = SpanUNet3D(num_classes=num_classes, in_channels=1, base=args.base_channels).to(device)
    if args.torch_compile:
        if hasattr(torch, "compile"):
            base = torch.compile(base)  # type: ignore[assignment]
        elif is_main:
            print("[warn] --torch_compile ignored: torch.compile not available in this PyTorch build")
    if distributed:
        model = DDP(base, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        model = base
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SegmentationLoss(
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        class_weights=ce_override,
        line_class_ce_boost=args.line_class_ce_boost if ce_override is None else 1.0,
        focal_gamma=args.focal_gamma,
    ).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-6
    )
    # torch.cuda.amp (not torch.amp.GradScaler): some PyTorch wheels omit GradScaler on torch.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    log_path = out_dir / "train_log.jsonl"
    if is_main and log_path.exists():
        log_path.unlink()

    meta_paths = {
        "train_spans": _span_paths_for_meta(train_spans, data_root),
        "val_spans": _span_paths_for_meta(val_spans, data_root),
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "class_index_legend": {i: n for i, n in enumerate(CLASS_NAMES)},
        "checkpoint_metric": args.checkpoint_metric,
        "line_class_ce_boost": args.line_class_ce_boost,
        "ce_class_weights": ce_override.tolist() if ce_override is not None else None,
        "focal_gamma": float(args.focal_gamma),
        "dice_weight": args.dice_weight,
        "lr": args.lr,
        "data_root": str(data_root.resolve()),
        "distributed": distributed,
        "world_size": world_size,
        "amp": use_amp,
        "num_workers": nw,
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "torch_compile": bool(args.torch_compile),
    }
    if is_main:
        (out_dir / "run_meta.json").write_text(json.dumps(meta_paths, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    best_score = float("inf") if args.checkpoint_metric == "val_loss" else float("-inf")
    history: list[dict] = []

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            tr_loss_t = torch.zeros((), dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            for x, y, _name in train_loader:
                x = x.to(device, non_blocking=nb)
                y = y.to(device, non_blocking=nb)
                opt.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if args.grad_clip_norm and args.grad_clip_norm > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip_norm and args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    opt.step()
                tr_loss_t += loss.detach().float()
            n_batches_tr = torch.tensor([len(train_loader)], device=device, dtype=torch.float32)
            if distributed:
                dist.all_reduce(tr_loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_batches_tr, op=dist.ReduceOp.SUM)
            tr_loss = (tr_loss_t / n_batches_tr.clamp(min=1.0)).item()

            if distributed and rank != 0:
                val_stats = None
            else:
                val_stats = evaluate_loader(
                    model,
                    val_loader,
                    device,
                    num_classes,
                    criterion=criterion,
                    use_amp=use_amp,
                    non_blocking=nb,
                )
            if distributed:
                obj_list = [val_stats] if rank == 0 else [None]
                dist.broadcast_object_list(obj_list, src=0)
                val_stats = obj_list[0]
            assert val_stats is not None
            va_loss = val_stats["loss"]
            dt = time.perf_counter() - t0

            rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_mean_iou": val_stats["mean_iou"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_pixel_accuracy": val_stats["pixel_accuracy"],
                "val_per_class_iou": val_stats["per_class_iou"],
                "val_line_obj_precision_micro": val_stats.get("line_obj_precision_micro", 0.0),
                "val_line_obj_recall_micro": val_stats.get("line_obj_recall_micro", 0.0),
                "val_line_obj_f1_micro": val_stats.get("line_obj_f1_micro", 0.0),
                "val_line_pixel_iou_mean_span": val_stats.get("line_pixel_iou_mean_span", 0.0),
                "seconds": round(dt, 3),
            }
            if is_main:
                history.append(rec)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                    f"val_mIoU={val_stats['mean_iou']:.4f}  macroP={val_stats['macro_precision']:.4f}  "
                    f"macroR={val_stats['macro_recall']:.4f}  lineObjF1={val_stats.get('line_obj_f1_micro', 0):.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)"
                )

            if args.checkpoint_metric == "val_loss":
                cur_ck = va_loss
                improved = cur_ck < best_score - 1e-12
            else:
                cur_ck = val_stats["mean_iou"]
                improved = cur_ck > best_score + 1e-12
            if improved:
                best_score = cur_ck
                if is_main:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": _unwrap_model(model).state_dict(),
                            "num_classes": num_classes,
                            "base_channels": args.base_channels,
                            "checkpoint_metric": args.checkpoint_metric,
                            "best_score": float(best_score),
                            "world_size": world_size,
                        },
                        out_dir / "best.pt",
                    )

            scheduler.step(va_loss)

        if is_main:
            torch.save(
                {
                    "model_state": _unwrap_model(model).state_dict(),
                    "num_classes": num_classes,
                    "base_channels": args.base_channels,
                },
                out_dir / "last.pt",
            )

            (out_dir / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            plot_training_curves(history, out_dir)

            meta_paths["finished_epoch"] = args.epochs
            meta_paths["scheduler_note"] = "ReduceLROnPlateau monitors val_loss (min)"
            (out_dir / "run_meta.json").write_text(json.dumps(meta_paths, indent=2), encoding="utf-8")

            best_path = out_dir / "best.pt"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                _unwrap_model(model).load_state_dict(ckpt["model_state"])
            final = evaluate_loader(
                model,
                val_loader,
                device,
                num_classes,
                criterion=criterion,
                use_amp=use_amp,
                non_blocking=nb,
            )
            cm = final["confusion_matrix"]
            np.save(out_dir / "confusion_matrix.npy", cm)
            plot_confusion_matrix(cm, CLASS_NAMES, out_dir / "confusion_matrix.png")
            report_path = out_dir / "validation_report.txt"
            report_path.write_text(build_validation_report_text(final, CLASS_NAMES, history=history), encoding="utf-8")

            print(f"[ok] wrote {out_dir / 'best.pt'}, {out_dir / 'last.pt'}")
            print(f"[ok] wrote {out_dir / 'metrics_history.json'}, history_loss.png, history_scores.png")
            print(f"[ok] wrote {out_dir / 'confusion_matrix.png'}, {report_path}")
    finally:
        _teardown_distributed(distributed)


if __name__ == "__main__":
    main()
