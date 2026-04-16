# Goal 2 — Semantic segmentation of span volumes (line types)

**Goal 2** (from `Lines_Detect_Goals.txt`): given span frame BMPs, predict **which voxels are conductors** and their **line type** (comm / primary / neutral / secondary / transmission), with temporal consistency handled by a 3D model over the whole span.

This folder implements a **first training stack**:

| Piece | Role |
|-------|------|
| `line_seg/volume.py` | Load `T×H×W` uint8 volume; map raw pixels → **7-class** semantic labels |
| `line_seg/dataset.py` | PyTorch `Dataset` — one span per item (variable `T`) |
| `line_seg/model.py` | `SpanUNet3D` — lightweight 3D UNet, spatial pooling only on `H,W` in the encoder |
| `line_seg/losses.py` | Cross-entropy + multiclass Dice |
| `line_seg/train_goal2.py` | Training loop (`batch_size=1` recommended) |

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
```

Outputs:

- `best.pt` / `last.pt` — weights + metadata
- `train_log.jsonl` — one JSON object per epoch (`train_loss`, `val_loss`, `val_mean_iou`)
- `run_meta.json` — train/val span names

**Validation split**: random 15% of spans (see `--val_fraction`).

---

## Design notes

- **Variable T**: each span has a different number of frames; the dataloader uses **`batch_size=1`**. To batch multiple spans, add padding/collate (not implemented here).
- **Instance identity** (same conductor across frames with a stable id) is **not** modeled; this head is **semantic only**. Add an instance embedding head + discriminative loss later (see `POWER_LINE_DETECTION_PLAN.md` Phase 2).
- **Inference without gray labels**: train a second model on **synthetic B&W** (e.g. map lines → black) when that pipeline is ready.

---

## Related

- Line gray encoding: `tools/line_encoding.py`, `README_CATENARY_BASELINE.md`
- RANSAC **per-line** sag: `python3 -m tools.run_baseline --span SPAN --per_line_id --out_dir OUT`
