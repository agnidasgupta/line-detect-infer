#!/usr/bin/env python3
"""
Goal 2 — export ``best.pt`` (or any ``SpanUNet3D`` checkpoint) to ONNX for Unity Sentis.

Mirrors the pole sidecar export style (FP32 + optional FP16, sidecar JSON, opset 17,
optional onnxruntime parity check). Single-stage model: input ``volume`` ``[1, 1, T, H, W]``
float in ``[0, 1]`` (raw uint8 / 255) → output ``class_logits`` ``[1, num_classes, T, H, W]``.

Usage (from ``/notebooks/DUKE_FLORIDA_150``)::

    python3 -m line_seg.export_goal2_onnx \\
        --weights ./goal2_runs/exp_focal/best.pt \\
        --out_dir ./onnx_export_goal2 \\
        --opset 17 \\
        --fp16 \\
        --verify

The script writes:

    <out_dir>/line_seg_span_unet3d.onnx          # FP32
    <out_dir>/line_seg_span_unet3d_fp16.onnx     # FP16 (with --fp16)
    <out_dir>/line_seg_sidecar.json              # Unity-side metadata
    <out_dir>/export_log.txt                     # what was exported, with timings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_seg.model import SpanUNet3D  # noqa: E402
from line_seg.train_goal2 import CLASS_NAMES  # noqa: E402

INPUT_NAME = "volume"
OUTPUT_NAME = "class_logits"


def _load_checkpoint(weights: Path, device: torch.device) -> dict:
    if not weights.is_file():
        raise SystemExit(f"Checkpoint not found: {weights}")
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)
    if "model_state" not in ckpt:
        raise SystemExit(
            f"Checkpoint {weights} missing 'model_state'; expected a Goal 2 best.pt / last.pt"
        )
    return ckpt


def _build_model(ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, int, int]:
    num_classes = int(ckpt.get("num_classes", 7))
    base_ch = int(ckpt.get("base_channels", 24))
    if num_classes != 7:
        print(
            f"[warn] checkpoint num_classes={num_classes} (expected 7 for Goal 2); "
            "exporting anyway — Unity sidecar will use the reported number.",
            flush=True,
        )
    model = SpanUNet3D(num_classes=num_classes, in_channels=1, base=base_ch).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, num_classes, base_ch


def _export_fp32(
    model: torch.nn.Module,
    device: torch.device,
    out_path: Path,
    *,
    opset: int,
    sample_T: int,
    sample_H: int,
    sample_W: int,
) -> None:
    """torch.onnx.export with dynamic ``T,H,W`` axes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, sample_T, sample_H, sample_W, device=device, dtype=torch.float32)
    dynamic_axes = {
        INPUT_NAME: {2: "T", 3: "H", 4: "W"},
        OUTPUT_NAME: {2: "T", 3: "H", 4: "W"},
    }
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
    )


def _convert_fp16(in_path: Path, out_path: Path) -> None:
    """Convert FP32 ONNX to FP16 in-place using ``onnxconverter_common``."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "FP16 conversion requires onnx and onnxconverter-common: "
            "pip install onnx onnxconverter-common"
        ) from e
    model = onnx.load(in_path.as_posix())
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(fp16_model, out_path.as_posix())


def _verify_with_onnxruntime(
    onnx_path: Path,
    torch_model: torch.nn.Module,
    device: torch.device,
    *,
    sample_T: int,
    sample_H: int,
    sample_W: int,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Compare PyTorch and onnxruntime outputs on a synthetic span volume."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Verification requires onnxruntime: pip install onnxruntime"
        ) from e

    rng = np.random.default_rng(0)
    x_np = rng.random((1, 1, sample_T, sample_H, sample_W), dtype=np.float32)

    sess = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x_np})[0]

    with torch.no_grad():
        x_t = torch.from_numpy(x_np).to(device)
        torch_out = torch_model(x_t).detach().cpu().numpy()

    if torch_out.shape != ort_out.shape:
        raise SystemExit(
            f"shape mismatch: torch {torch_out.shape} vs onnx {ort_out.shape}"
        )

    diff = np.abs(torch_out - ort_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    pred_torch = torch_out.argmax(axis=1)
    pred_onnx = ort_out.argmax(axis=1)
    same = float((pred_torch == pred_onnx).mean())
    if not np.allclose(torch_out, ort_out, rtol=rtol, atol=atol):
        print(
            f"[warn] FP32 ONNX outputs differ slightly from PyTorch: "
            f"max|Δ|={max_abs:.3e}, mean|Δ|={mean_abs:.3e}, argmax_match={same:.4f}",
            flush=True,
        )
    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "argmax_match": same}


def _write_sidecar(
    out_path: Path,
    *,
    weights: Path,
    onnx_fp32: Path,
    onnx_fp16: Path | None,
    num_classes: int,
    base_channels: int,
    opset: int,
    sample_T: int,
    sample_H: int,
    sample_W: int,
    verify_metrics: dict[str, float] | None,
) -> None:
    sidecar = {
        "format_version": 1,
        "model_arch": "SpanUNet3D",
        "model_base": base_channels,
        "in_channels": 1,
        "num_classes": num_classes,
        "class_names": list(CLASS_NAMES),
        "input_normalization": "raw_uint8 / 255.0  (float in [0, 1])",
        "input_layout": "NCDHW with N=1, C=1, D=T (frames), H, W",
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_dynamic_axes": {INPUT_NAME: {"2": "T", "3": "H", "4": "W"}},
        "output_dynamic_axes": {OUTPUT_NAME: {"2": "T", "3": "H", "4": "W"}},
        "sample_export_shape": {"N": 1, "C": 1, "T": sample_T, "H": sample_H, "W": sample_W},
        "post_processing": {
            "argmax_dim": 1,
            "class_to_bmp_uint8": {
                "0": 255,
                "1": 0,
                "2": "pack_line_gray(0, 0)",
                "3": "pack_line_gray(0, 1)",
                "4": "pack_line_gray(0, 2)",
                "5": "pack_line_gray(0, 3)",
                "6": "pack_line_gray(0, 4)",
            },
            "pack_line_gray": "(id_field << 3) | type_code  with id_field=0",
        },
        "source_checkpoint": str(weights),
        "fp32_onnx": onnx_fp32.name,
        "fp16_onnx": onnx_fp16.name if onnx_fp16 is not None else None,
        "opset": opset,
        "precision": "fp32_and_fp16" if onnx_fp16 is not None else "fp32",
        "unity_runtime_notes": {
            "tensor_shape": "Use Sentis Tensor<float> with TensorShape(1, 1, T, H, W).",
            "input_range": "Float in [0, 1]; divide raw uint8 frame pixels by 255.",
            "argmax": "Apply argmax along axis 1 (class dim) on output to get per-voxel class index 0..6.",
            "labelled_bmp_writeback": "Use class_to_bmp_uint8 to recover dataset gray encoding (matches infer_goal2.py).",
            "backend": "Use BackendType.GPUCompute on supported GPUs.",
        },
        "verification": verify_metrics,
    }
    out_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export Goal 2 SpanUNet3D to ONNX (FP32 + optional FP16) for Unity Sentis.",
        epilog="See DUKE_FLORIDA_150/unity_export/LINE_SEG_UNITY_INTEGRATION.md for the C# side.",
    )
    p.add_argument("--weights", type=str, required=True, help="Goal 2 checkpoint .pt (e.g. best.pt)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for ONNX + sidecar")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17, matches pole sidecar)")
    p.add_argument("--fp16", action="store_true", help="Also write an FP16 ONNX variant")
    p.add_argument("--verify", action="store_true", help="Run onnxruntime parity check on FP32 output")
    p.add_argument("--sample_T", type=int, default=64, help="Dummy T (frames) for export tracing")
    p.add_argument("--sample_H", type=int, default=224, help="Dummy H for export tracing")
    p.add_argument("--sample_W", type=int, default=128, help="Dummy W for export tracing")
    p.add_argument("--device", type=str, default="cpu", help="Device for tracing (cpu is fine)")
    args = p.parse_args()

    device = torch.device(args.device)
    weights = Path(args.weights).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_H % 4 != 0 or args.sample_W % 4 != 0:
        raise SystemExit(
            f"sample_H and sample_W must be divisible by 4 (encoder pools (1,2,2) twice); "
            f"got H={args.sample_H}, W={args.sample_W}"
        )

    log: list[str] = []
    log.append(f"weights: {weights}")
    log.append(f"out_dir: {out_dir}")
    log.append(f"opset: {args.opset}")
    log.append(f"sample_TxHxW: {args.sample_T}x{args.sample_H}x{args.sample_W}")

    print(f"[export] loading {weights}", flush=True)
    ckpt = _load_checkpoint(weights, device)
    model, num_classes, base_ch = _build_model(ckpt, device)
    log.append(f"num_classes: {num_classes}")
    log.append(f"base_channels: {base_ch}")

    onnx_fp32 = out_dir / "line_seg_span_unet3d.onnx"
    print(f"[export] writing FP32 ONNX → {onnx_fp32}", flush=True)
    t0 = time.perf_counter()
    _export_fp32(
        model,
        device,
        onnx_fp32,
        opset=args.opset,
        sample_T=args.sample_T,
        sample_H=args.sample_H,
        sample_W=args.sample_W,
    )
    log.append(f"fp32_export_seconds: {time.perf_counter() - t0:.3f}")

    onnx_fp16: Path | None = None
    if args.fp16:
        onnx_fp16 = out_dir / "line_seg_span_unet3d_fp16.onnx"
        print(f"[export] writing FP16 ONNX → {onnx_fp16}", flush=True)
        t0 = time.perf_counter()
        _convert_fp16(onnx_fp32, onnx_fp16)
        log.append(f"fp16_convert_seconds: {time.perf_counter() - t0:.3f}")

    verify_metrics: dict[str, float] | None = None
    if args.verify:
        print("[export] running onnxruntime parity check (FP32)", flush=True)
        t0 = time.perf_counter()
        verify_metrics = _verify_with_onnxruntime(
            onnx_fp32,
            model,
            device,
            sample_T=args.sample_T,
            sample_H=args.sample_H,
            sample_W=args.sample_W,
        )
        log.append(f"verify_seconds: {time.perf_counter() - t0:.3f}")
        log.append(f"verify_metrics: {json.dumps(verify_metrics)}")
        print(f"[export] verify ok: {verify_metrics}", flush=True)

    sidecar_path = out_dir / "line_seg_sidecar.json"
    _write_sidecar(
        sidecar_path,
        weights=weights,
        onnx_fp32=onnx_fp32,
        onnx_fp16=onnx_fp16,
        num_classes=num_classes,
        base_channels=base_ch,
        opset=args.opset,
        sample_T=args.sample_T,
        sample_H=args.sample_H,
        sample_W=args.sample_W,
        verify_metrics=verify_metrics,
    )
    log.append(f"sidecar: {sidecar_path}")

    (out_dir / "export_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"[ok] wrote {onnx_fp32}", flush=True)
    if onnx_fp16 is not None:
        print(f"[ok] wrote {onnx_fp16}", flush=True)
    print(f"[ok] wrote {sidecar_path}", flush=True)


if __name__ == "__main__":
    main()
