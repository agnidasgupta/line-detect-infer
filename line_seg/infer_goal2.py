#!/usr/bin/env python3
"""
Goal 2 inference: load ``best.pt`` / ``last.pt``, run ``SpanUNet3D`` on span folders under ``--input_dir``,
write predicted **labeled BMPs** (dataset gray encoding) per span, wall-clock timings, and optionally
``evaluation_report.txt`` (same layout as training ``validation_report.txt``) when **any** input span
contains line labels (``128 <= gray < 255`` in the **input** BMPs).

Input layout: ``--input_dir`` may be the project root (spans + ``line_seg`` + ``tools`` + …).
Only **immediate subdirectories** that contain at least one ``.bmp`` file anywhere underneath are
considered; others (e.g. code folders) are skipped. Among those, inference runs only where
``frame_*.bmp`` exists in that folder or in one **immediate** child directory (same layout as training,
with optional one-level nesting for frames). Spans may be unlabeled (only air/solid: 255 / 0) —
then evaluation is skipped.

Usage (from ``DUKE_FLORIDA_150`` root):

    python3 -m line_seg.infer_goal2 \\
      --weights ./goal2_runs/exp1/best.pt \\
      --input_dir ./path/to/spans_or_same_as_data_root \\
      --output_dir ./inference_runs/run1 \\
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_seg.eval_metrics import (
    aggregate_object_metrics,
    confusion_matrix_accumulate,
    line_object_detection_scores,
    scores_from_confusion,
)
from line_seg.model import SpanUNet3D
from line_seg.train_goal2 import CLASS_NAMES, build_validation_report_text
from line_seg.volume import load_span_volume, raw_to_semantic_labels
from tools.bmp_io import write_bmp_gray
from tools.line_encoding import pack_line_gray


def _directory_contains_bmp(root: Path) -> bool:
    """True if any file under ``root`` ends with ``.bmp`` (case-insensitive); first match wins."""
    root = Path(root)
    if not root.is_dir():
        return False
    for _, _, filenames in os.walk(root, topdown=True):
        for fn in filenames:
            if fn.lower().endswith(".bmp"):
                return True
    return False


def _resolve_frames_directory(span_root: Path) -> Path | None:
    """``frame_*.bmp`` folder: ``span_root`` or first sorted immediate subdirectory that has them."""
    span_root = Path(span_root)
    if list(span_root.glob("frame_*.bmp")):
        return span_root
    for sub in sorted(span_root.iterdir()):
        if sub.is_dir() and list(sub.glob("frame_*.bmp")):
            return sub
    return None


def list_infer_span_frame_dirs(data_root: str | Path) -> tuple[list[tuple[Path, Path]], dict[str, list[str]]]:
    """
    Discover spans under ``data_root`` when it mixes code and span folders (see module docstring).

    Returns ``(pairs, diagnostics)`` with diagnostics keys ``skipped_no_bmp`` and
    ``skipped_no_frame_bmps`` (sorted name lists). Kept in this module so inference runs even
    if an older ``line_seg/volume.py`` is deployed without these helpers.
    """
    data_root = Path(data_root)
    pairs: list[tuple[Path, Path]] = []
    skipped_no_bmp: list[str] = []
    skipped_no_frame: list[str] = []
    if not data_root.is_dir():
        return pairs, {"skipped_no_bmp": skipped_no_bmp, "skipped_no_frame_bmps": skipped_no_frame}

    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if not _directory_contains_bmp(p):
            skipped_no_bmp.append(p.name)
            continue
        frames = _resolve_frames_directory(p)
        if frames is None:
            skipped_no_frame.append(p.name)
            continue
        pairs.append((p, frames))

    diagnostics = {
        "skipped_no_bmp": sorted(skipped_no_bmp),
        "skipped_no_frame_bmps": sorted(skipped_no_frame),
    }
    return pairs, diagnostics


def span_volume_has_line_labels(vol: np.ndarray) -> bool:
    """True if BMPs contain dataset line labels (128–254), i.e. evaluation GT is available."""
    v = vol.astype(np.uint8)
    return bool(np.any((v >= 128) & (v < 255)))


def semantic_classes_to_label_uint8(pred: np.ndarray) -> np.ndarray:
    """
    Map int64 class map ``0..6`` to uint8 BMP pixels (air=255, solid=0, line = pack_line_gray(0, type)).
    """
    pred = np.asarray(pred, dtype=np.int64)
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred == 0] = 255
    out[pred == 1] = 0
    for c in range(2, 7):
        m = pred == c
        type_code = c - 2
        out[m] = pack_line_gray(0, type_code)
    return out


def _sorted_frame_paths(span_dir: Path) -> list[Path]:
    return sorted(span_dir.glob("frame_*.bmp"))


@torch.no_grad()
def _forward_span(
    model: torch.nn.Module,
    vol: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
) -> np.ndarray:
    """Return int64 ``[T,H,W]`` predicted class indices."""
    # Match DataLoader + PoleSpanSegDataset: __getitem__ yields [1,T,H,W], default_collate
    # stacks batch dim → [B,1,T,H,W]. A single unsqueeze(0) on [T,H,W] would be [1,T,H,W]
    # and Conv3d would treat T as C (e.g. 113 channels) → GroupNorm / channel errors.
    x = torch.from_numpy(vol.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
    else:
        logits = model(x)
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def _aggregate_eval_dict(
    pred_list: list[np.ndarray],
    gt_list: list[np.ndarray],
    num_classes: int,
) -> dict:
    """Build the same scalar/report keys as ``evaluate_loader`` (no val loss)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    object_rows: list[dict[str, float]] = []
    y_chunks: list[np.ndarray] = []
    p_chunks: list[np.ndarray] = []
    for pr, gt in zip(pred_list, gt_list):
        confusion_matrix_accumulate(pr, gt, num_classes, cm)
        object_rows.append(line_object_detection_scores(pr.astype(np.int64), gt.astype(np.int64)))
        p_chunks.append(pr.reshape(-1))
        y_chunks.append(gt.reshape(-1))
    y_flat = np.concatenate(y_chunks)
    p_flat = np.concatenate(p_chunks)
    scores = scores_from_confusion(cm)
    obj_micro = aggregate_object_metrics(object_rows)
    report_str = classification_report(
        y_flat,
        p_flat,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    out = {
        "loss": None,
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
        "classification_report_str": report_str,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Goal 2: run trained SpanUNet3D on span BMP folders")
    p.add_argument("--weights", type=str, required=True, help="Checkpoint .pt (e.g. goal2_runs/exp1/best.pt)")
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root (e.g. DUKE_FLORIDA_150): only immediate subdirs that contain .bmp are candidates; "
        "need frame_*.bmp at span root or one level down.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for per-span outputs, timings, and optional evaluation_report.txt",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_frames", type=int, default=None, help="Cap frames per span (same as training)")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP on CUDA forward")
    args = p.parse_args()

    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    input_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    span_pairs, span_diag = list_infer_span_frame_dirs(input_root)
    if not span_pairs:
        msg = (
            f"No runnable spans under {input_root}: need immediate subdirs with at least one .bmp, "
            f"and frame_*.bmp at span root or one subdirectory deep.\n"
            f"  skipped (no .bmp anywhere): {len(span_diag['skipped_no_bmp'])} -> {span_diag['skipped_no_bmp'][:20]}"
            f"{' ...' if len(span_diag['skipped_no_bmp']) > 20 else ''}\n"
            f"  skipped (.bmp present but no frame_*.bmp): {len(span_diag['skipped_no_frame_bmps'])} "
            f"-> {span_diag['skipped_no_frame_bmps'][:20]}{' ...' if len(span_diag['skipped_no_frame_bmps']) > 20 else ''}"
        )
        raise SystemExit(msg)

    print(
        f"[infer_goal2] span discovery: {len(span_pairs)} to run; "
        f"skipped_no_bmp={len(span_diag['skipped_no_bmp'])}, "
        f"skipped_no_frame_bmps={len(span_diag['skipped_no_frame_bmps'])}",
        flush=True,
    )

    ckpt_path = Path(args.weights).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" not in ckpt:
        raise SystemExit("Checkpoint missing 'model_state'; use a Goal 2 best.pt / last.pt from train_goal2")
    num_classes = int(ckpt.get("num_classes", 7))
    base_ch = int(ckpt.get("base_channels", 24))
    if num_classes != 7:
        raise SystemExit(f"Expected num_classes=7, got {num_classes}")

    model = SpanUNet3D(num_classes=num_classes, in_channels=1, base=base_ch).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    timings_path = out_root / "inference_timings.txt"
    manifest = {
        "weights": str(ckpt_path),
        "input_dir": str(input_root),
        "output_dir": str(out_root),
        "n_spans": len(span_pairs),
        "span_discovery": span_diag,
        "span_frame_dirs": [
            {"span_root": str(sr.resolve()), "frames_dir": str(fd.resolve())} for sr, fd in span_pairs
        ],
        "max_frames": args.max_frames,
        "device": str(device),
        "amp": use_amp,
    }

    labeled_preds: list[np.ndarray] = []
    labeled_gts: list[np.ndarray] = []
    labeled_span_names: list[str] = []
    span_summary_rows: list[dict] = []
    n_inference_ok = 0
    n_inference_failed = 0
    n_loaded_with_line_bytes = 0  # 128–254 in BMP input (same rule as training GT)

    with timings_path.open("w", encoding="utf-8") as tlog:
        tlog.write("# span_name\tseconds\tstatus\tinput_line_bytes_128_254\n")
        for span_root, frames_dir in span_pairs:
            name = span_root.name
            span_out = out_root / name
            pred_dir = span_out / "predicted_bmp"
            pred_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            status = "ok"
            has_lbl: bool | None = None
            summary: dict = {
                "span": name,
                "frames_dir": str(frames_dir.resolve()),
            }
            try:
                vol = load_span_volume(frames_dir, max_frames=args.max_frames)
                if vol.ndim != 3:
                    raise ValueError(f"expected volume [T,H,W], got shape {vol.shape}")
                summary["volume_THW"] = [int(x) for x in vol.shape]
                summary["input_gray_min"] = int(vol.min())
                summary["input_gray_max"] = int(vol.max())
                has_lbl = span_volume_has_line_labels(vol)
                summary["input_had_line_labels_128_254"] = has_lbl
                if has_lbl:
                    n_loaded_with_line_bytes += 1

                gt_sem = raw_to_semantic_labels(vol).astype(np.int64)
                pred_sem = _forward_span(model, vol, device, use_amp=use_amp)
                pred_u8 = semantic_classes_to_label_uint8(pred_sem)

                frame_paths = _sorted_frame_paths(frames_dir)
                if len(frame_paths) != pred_u8.shape[0]:
                    raise RuntimeError(f"frame count mismatch: {len(frame_paths)} paths vs T={pred_u8.shape[0]}")
                for fp, frame in zip(frame_paths, pred_u8):
                    write_bmp_gray(pred_dir / fp.name, frame)

                np.save(span_out / "predicted_semantic_classes.npy", pred_sem.astype(np.int8))
                elapsed = time.perf_counter() - t0
                (span_out / "meta.json").write_text(
                    json.dumps(
                        {
                            "span": name,
                            "frames_dir": str(frames_dir.resolve()),
                            "shape_THW": list(pred_sem.shape),
                            "input_had_line_labels": has_lbl,
                            "seconds": round(elapsed, 4),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if has_lbl:
                    labeled_preds.append(pred_sem.copy())
                    labeled_gts.append(gt_sem.copy())
                    labeled_span_names.append(name)
                summary["inference"] = "ok"
                n_inference_ok += 1
            except Exception as e:  # noqa: BLE001 — record and continue other spans
                status = f"error: {e}"
                summary["inference"] = status
                n_inference_failed += 1
            summary["input_had_line_labels_128_254"] = has_lbl
            span_summary_rows.append(summary)
            dt = time.perf_counter() - t0
            line_flag = "" if has_lbl is None else str(bool(has_lbl))
            tlog.write(f"{name}\t{dt:.4f}\t{status}\t{line_flag}\n")
            tlog.flush()

    manifest["span_inference_summary"] = span_summary_rows
    manifest["n_spans_inference_ok"] = n_inference_ok
    manifest["n_spans_inference_failed"] = n_inference_failed
    manifest["n_spans_loaded_with_line_bytes_128_254"] = n_loaded_with_line_bytes
    manifest["labeled_spans_for_eval"] = labeled_span_names
    manifest["n_labeled_spans_eval"] = len(labeled_span_names)
    (out_root / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if labeled_preds:
        final = _aggregate_eval_dict(labeled_preds, labeled_gts, num_classes)
        title = "Goal 2 — inference evaluation (aggregate over input spans that contained line labels in BMPs)"
        report = build_validation_report_text(
            final,
            CLASS_NAMES,
            history=None,
            report_title=title,
        )
        footer = (
            "\n=== Inference note ===\n"
            f"Spans aggregated: {len(labeled_preds)}  ({', '.join(labeled_span_names)})\n"
            "Unlabeled spans (no 128–254 gray in input) are excluded from this aggregate.\n"
        )
        (out_root / "evaluation_report.txt").write_text(report.rstrip() + footer, encoding="utf-8")
        np.save(out_root / "evaluation_confusion_matrix.npy", final["confusion_matrix"])
    elif n_inference_ok == 0:
        print(
            "[infer_goal2] evaluation_report.txt not written: no span completed inference "
            f"(failures={n_inference_failed}). See inference_timings.txt. "
            "If BMPs actually contain line labels (128–254), the probe below still reflects loaded volumes.",
            flush=True,
        )
        if n_loaded_with_line_bytes > 0:
            print(
                f"[infer_goal2] Note: {n_loaded_with_line_bytes} span(s) had 128–254 gray in input after load, "
                "but the model forward or I/O failed before evaluation could run.",
                flush=True,
            )
    else:
        print(
            "[infer_goal2] No completed span had line-label bytes (128–254 in input BMPs); "
            "skipped evaluation_report.txt",
            flush=True,
        )

    print(f"[ok] wrote outputs under {out_root}")
    print(f"[ok] timings: {timings_path}")


if __name__ == "__main__":
    main()
