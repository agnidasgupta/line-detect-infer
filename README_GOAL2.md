# Goal 2 — Semantic segmentation of span volumes (line types)

**Goal 2** (from `Lines_Detect_Goals.txt`): given span frame BMPs, predict **which voxels are conductors** and their **line type** (comm / primary / neutral / secondary / transmission), with temporal consistency handled by a 3D model over the whole span.

This folder implements a **first training stack**:

| Piece | Role |
|-------|------|
| `line_seg/volume.py` | Load `T×H×W` uint8 volume; map raw pixels → **7-class** semantic labels |
| `line_seg/dataset.py` | PyTorch `Dataset` — one span per item (variable `T`) |
| `line_seg/model.py` | `SpanUNet3D` — lightweight 3D UNet, spatial pooling only on `H,W` in the encoder |
| `line_seg/losses.py` | Cross-entropy + multiclass Dice |
| `line_seg/eval_metrics.py` | Confusion matrix → per-class IoU / P / R; merged-line **object** metrics (3D CC + IoU match) |
| `line_seg/train_goal2.py` | Training loop (`batch_size=1` recommended), logs + plots + final sklearn report |
| `line_seg/infer_goal2.py` | Inference: checkpoint + `--input_dir` / `--output_dir`, predicted BMPs, timings, optional eval report |

**Not yet included** (future work): instance IDs across frames, B&W-only inference without gray labels, Goal A binary “has lines” classifier.

---

## Class index (semantic)

| Index | Meaning |
|-------|---------|
| 0 | Air (`gray == 255`) |
| 1 | Solid / non-line (`gray == 0`) |
| 2 | Comm (line `type_code == 0`) |
| 3 | Primary (1) |
| 4 | Neutral (2) |
| 5 | Secondary (3) |
| 6 | Transmission (4) |

Line voxels use `128 <= gray < 255`; `type_code = gray & 0x07`. Invalid types `>4` are mapped to class **1** (solid).

---

## Setup

From `DUKE_FLORIDA_150/` (dataset root):

```bash
python3 -m pip install -r requirements_baseline.txt
python3 -m pip install -r requirements_goal2.txt
```

Use a CUDA machine for reasonable speed.

---

## Train

```bash
cd /path/to/DUKE_FLORIDA_150
python3 -m line_seg.train_goal2 --data_root . --out_dir ./goal2_runs/exp1 --epochs 30 --lr 1e-3
# Optional: --checkpoint_metric val_loss  (default is val_mean_iou for best.pt)
#           --line_class_ce_boost 1.0    (default 2.0 up-weights line classes 2–6 in CE; ignored if --ce_class_weights set)
#           --ce_class_weights 1,1,1,1,1,8,1   (optional explicit 7-class CE weights)
#           --focal_gamma 2.0            (optional; 0 = plain weighted CE)
#           --num_workers 4              (default min(8, CPU count); async BMP loading)
#           --no_amp                     (disable mixed precision; AMP is on by default on CUDA)
```

**Multi-GPU** (DistributedDataParallel + NCCL; validation and file I/O on rank 0 only):

`--nproc_per_node` must match **visible** GPU count (e.g. **2** on a dual-A6000 machine, not 4).

```bash
torchrun --standalone --nproc_per_node=2 -m line_seg.train_goal2 \
  --data_root . --out_dir ./goal2_runs/exp4 --epochs 30 --lr 1e-3
```

**Auto GPU count** (re-launches with `torch.distributed.run`; no need to set `nproc`):

```bash
python3 -m line_seg.train_goal2 --multi_gpu \
  --data_root . --out_dir ./goal2_runs/exp4 --epochs 30 --lr 1e-3
```

Requires **at least as many training spans as GPUs** (each rank must have data). `--device` is ignored under `torchrun`; each process binds `cuda:LOCAL_RANK`.

Each GPU processes different training spans each step (`batch_size=1` per process). **TF32** and `torch.set_float32_matmul_precision("high")` are enabled on CUDA; **cudnn.benchmark** stays off by default because span depth `T` varies (enable with `--cudnn_benchmark` if you cap `T`).

**Performance (single or multi-GPU):** `pin_memory` + `non_blocking` H2D copies, **DataLoader `prefetch_factor=4`** when `num_workers>0`, **GPU confusion matrix** (`torch.bincount`) during validation with **per-span CPU chunks** for sklearn (no giant `torch.cat` on GPU), **AMP** (fp16) on training forward/loss. Under DDP, **train loss is all-reduced** so logged `train_loss` is the global mean over all spans and GPUs. Optional **`--torch_compile`** wraps the UNet in `torch.compile` (PyTorch 2+). **Gradient accumulation** is not used (throughput scales with GPU count when using `torchrun`).

Outputs:

- `best.pt` / `last.pt` — weights + metadata
- `train_log.jsonl` — one JSON object per epoch: `train_loss`, `val_loss`, `val_mean_iou`, macro precision/recall/F1, pixel accuracy, per-class IoU list, **line object** micro P/R/F1 (merged classes 2–6, 3D connected components, IoU match ≥ 0.5), mean per-span line pixel IoU
- `metrics_history.json` — same metrics as a JSON array (easy plotting)
- `history_loss.png` — train vs validation loss per epoch
- `history_scores.png` — validation mIoU and macro P/R/F1 per epoch
- After the last epoch, using **`best.pt`**: `confusion_matrix.npy`, `confusion_matrix.png` (pixel-level, 7×7), **`validation_report.txt`** — one file with ASCII confusion matrix, sklearn `classification_report`, JSON scalar summary (object-line metrics), and training-log diagnosis
- `run_meta.json` — train/val span names and class name legend

**Validation split**: random 15% of spans (see `--val_fraction`).

### Backfill metrics (existing run folder)

If a run used an older script or you only need reports after the fact:

```bash
python3 -m line_seg.export_goal2_metrics --data_root . --out_dir ./goal2_runs/exp1
```

Reads `run_meta.json` + `best.pt`, runs full validation, writes `confusion_matrix.*`, **`validation_report.txt`**, and refreshes `metrics_history.json` / history plots from `train_log.jsonl` when present. Relative `val_spans` paths are joined to `--data_root`.

### Inference (`infer_goal2`)

Run a saved checkpoint on a span tree or **whole project root**: child folders without any `.bmp` under them (e.g. `line_seg`, `tools`) are skipped; only dirs with **`frame_*.bmp`** at the span root or one level down are run. Writes per-span **`predicted_bmp/`** (dataset gray encoding from predicted classes), **`inference_timings.txt`**, and **`evaluation_report.txt`** (same structure as **`validation_report.txt`**) only if some input BMPs contain line labels (`128–254`); otherwise evaluation is omitted.

```bash
python3 -m line_seg.infer_goal2 \
  --weights ./goal2_runs/exp1/best.pt \
  --input_dir ./holdout_or_unlabeled_root \
  --output_dir ./inference_runs/run1 \
  --device cuda
```

---

## Design notes

- **Variable T**: each span has a different number of frames; the dataloader uses **`batch_size=1`**. To batch multiple spans, add padding/collate (not implemented here). The UNet uses **GroupNorm** (not BatchNorm) so single-volume batches do not destabilize statistics.
- **Instance identity** (same conductor across frames with a stable id) is **not** modeled; this head is **semantic only**. Add an instance embedding head + discriminative loss later (see `POWER_LINE_DETECTION_PLAN.md` Phase 2).
- **Inference without gray labels**: train a second model on **synthetic B&W** (e.g. map lines → black) when that pipeline is ready.

---

## Related

- Line gray encoding: `tools/line_encoding.py`, `README_CATENARY_BASELINE.md`
- RANSAC **per-line** sag: `python3 -m tools.run_baseline --span SPAN --per_line_id --out_dir OUT`
