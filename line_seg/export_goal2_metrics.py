#!/usr/bin/env python3
"""
Recompute / backfill evaluation artifacts for a Goal 2 run folder.

Requires: `best.pt`, `run_meta.json` under `--out_dir`. Span folders are resolved
with `--data_root` when `val_spans` entries are relative names.

    python3 -m line_seg.export_goal2_metrics --data_root . --out_dir ./goal2_runs/exp1

Writes: confusion_matrix.npy, confusion_matrix.png, **validation_report.txt**
(one file: ASCII CM + sklearn classification_report + JSON + training-log
diagnosis), and refreshes metrics_history.json / history plots from train_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_seg.dataset import PoleSpanSegDataset
from line_seg.losses import SegmentationLoss, parse_ce_class_weights
from line_seg.model import SpanUNet3D
from line_seg.train_goal2 import (
    CLASS_NAMES,
    _worker_init_fn,
    build_validation_report_text,
    evaluate_loader,
    plot_confusion_matrix,
    plot_training_curves,
)


def _load_history_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _resolve_val_span_paths(raw: list[str], data_root: Path) -> list[Path]:
    out: list[Path] = []
    root = data_root.resolve()
    for p in raw:
        pp = Path(p)
        out.append(pp.resolve() if pp.is_absolute() else (root / pp).resolve())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export Goal 2 metrics from best.pt + val split")
    p.add_argument("--data_root", type=str, default=".", help="DUKE_FLORIDA_150 root (for relative val_spans)")
    p.add_argument("--out_dir", type=str, required=True, help="Run folder with best.pt and run_meta.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default min(8, CPU count)).",
    )
    p.add_argument("--no_amp", action="store_true", help="Disable AMP for validation forward")
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument(
        "--line_class_ce_boost",
        type=float,
        default=None,
        help="CE line-class boost (default: read from run_meta.json if present, else 1.0). Ignored if explicit weights used.",
    )
    p.add_argument(
        "--ce_class_weights",
        type=str,
        default=None,
        metavar="W0,...,W6",
        help="Override per-class CE weights; default from run_meta.ce_class_weights if present.",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Focal gamma for val loss (default: run_meta.focal_gamma or 0).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    meta_path = out_dir / "run_meta.json"
    best_path = out_dir / "best.pt"
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}")
    if not best_path.exists():
        raise SystemExit(f"Missing {best_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    boost = args.line_class_ce_boost
    if boost is None:
        boost = float(meta.get("line_class_ce_boost", 1.0))
    val_paths = _resolve_val_span_paths(meta["val_spans"], data_root)
    num_classes = int(meta.get("num_classes", 7))
    device = torch.device(args.device)
    pin = device.type == "cuda"
    use_amp = not args.no_amp and pin
    nw = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 1)
    if pin:
        torch.set_float32_matmul_precision("high")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    base_ch = int(ckpt.get("base_channels", 24))
    model = SpanUNet3D(num_classes=num_classes, in_channels=1, base=base_ch).to(device)
    model.load_state_dict(ckpt["model_state"])

    val_ds = PoleSpanSegDataset(val_paths)
    worker_init = partial(_worker_init_fn, base_seed=0, rank=0) if nw > 0 else None
    prefetch = 4 if nw > 0 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        prefetch_factor=prefetch,
        worker_init_fn=worker_init,
    )
    if args.ce_class_weights is not None:
        try:
            ce_ov = parse_ce_class_weights(args.ce_class_weights, num_classes)
        except ValueError as e:
            raise SystemExit(f"--ce_class_weights: {e}") from e
    elif meta.get("ce_class_weights") is not None:
        ce_ov = torch.tensor(meta["ce_class_weights"], dtype=torch.float32)
        if ce_ov.numel() != num_classes:
            raise SystemExit("run_meta.ce_class_weights length must match num_classes")
    else:
        ce_ov = None

    focal_g = float(args.focal_gamma) if args.focal_gamma is not None else float(meta.get("focal_gamma", 0.0))

    criterion = SegmentationLoss(
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        class_weights=ce_ov,
        line_class_ce_boost=boost if ce_ov is None else 1.0,
        focal_gamma=focal_g,
    ).to(device)

    final = evaluate_loader(
        model,
        val_loader,
        device,
        num_classes,
        criterion=criterion,
        use_amp=use_amp,
        non_blocking=pin,
    )
    cm = final["confusion_matrix"]

    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, CLASS_NAMES, out_dir / "confusion_matrix.png")

    log_path = out_dir / "train_log.jsonl"
    history = _load_history_jsonl(log_path)
    (out_dir / "validation_report.txt").write_text(
        build_validation_report_text(final, CLASS_NAMES, history=history),
        encoding="utf-8",
    )

    if history:
        (out_dir / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_training_curves(history, out_dir)
    else:
        print(f"[warn] no {log_path}; skipped metrics_history.json and history plots")

    print(f"[ok] wrote {out_dir / 'validation_report.txt'} and confusion matrix artifacts")


if __name__ == "__main__":
    main()
