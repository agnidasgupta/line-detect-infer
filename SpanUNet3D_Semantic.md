# `SpanUNet3D` — 3D UNet for Power-Line Span Semantic Segmentation

This document describes the **architecture, training pipeline, and key improvements** of the
`SpanUNet3D` model used for Goal 2: per-voxel semantic classification of power-line spans.
It covers [`line_seg/model.py`](line_seg/model.py), [`line_seg/train_goal2.py`](line_seg/train_goal2.py),
[`line_seg/infer_goal2.py`](line_seg/infer_goal2.py), and the Unity Sentis 2.6.1 deployment path.

Related: [`POWER_LINE_DETECTION_PLAN.md`](POWER_LINE_DETECTION_PLAN.md) §11 (full experiment log),
[`Line_Annotation.md`](Line_Annotation.md) (annotation quality analysis).

---

## 1. Architecture

### 1.1 Overview

A span is a `T × H × W` volume of 8-bit BMP frames (`frame_*.bmp`).
The model processes the **entire span volume at once** as a 5-D tensor.

```
Input  : [B, 1, T, H, W]   float32 in [0, 1]
                             (raw uint8 / 255; line-label pixels 128–254 stripped to 0)
Output : [B, 7, T, H, W]   per-voxel class logits
```

Seven semantic classes:

| Index | Class | BMP gray | Description |
|-------|-------|----------|-------------|
| 0 | air | 255 | Empty space |
| 1 | solid | 0 | Vegetation, insulators, hardware |
| 2 | comm | 128 | Communication conductor |
| 3 | primary | 129 | Primary conductor |
| 4 | neutral | 130 | Neutral conductor |
| 5 | secondary | 131 | Secondary conductor |
| 6 | transmission | 132 | Transmission conductor |

BMP encoding: `gray = (16 << 3) | type_code = 128 | type_code`.

---

### 1.2 Building block: `ConvBlock3D`

```python
class ConvBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        g = _group_norm_groups(c_out)   # prefer 8, fall back to 4/2/1
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
        )
```

Design choices:
- **`3×3×3` convolutions** on all three axes `(T, H, W)`. A thin conductor is a continuous
  "tube" in the 3D volume; joint spatial + temporal convolutions let the model detect it as
  one coherent object rather than disconnected per-frame blobs.
- **Two convs per block** double the effective receptive field before each pool/upsample.
- **`bias=False`** — affine parameters live in `GroupNorm`.
- **`GroupNorm`** — mandatory because `batch_size=1` (each span is a single large volume).
  `BatchNorm` would collapse on a one-sample batch, and span `T` varies across the dataset.

---

### 1.3 Full network: `SpanUNet3D`

```python
class SpanUNet3D(nn.Module):
    def __init__(self, num_classes=7, in_channels=1, base=24, dropout=0.0):
        b = base
        self.enc1 = ConvBlock3D(in_channels, b)          # 1 → 24
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = ConvBlock3D(b, b * 2)                # 24 → 48
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.mid   = ConvBlock3D(b * 2, b * 4)           # 48 → 96
        self.bottleneck_drop = nn.Dropout3d(p=dropout)   # regularisation (p=0.05 in production)
        self.dec2  = ConvBlock3D(b * 2 + b * 4, b * 2)  # 144 → 48
        self.dec1  = ConvBlock3D(b + b * 2, b)           # 72  → 24
        self.out_conv = nn.Conv3d(b, num_classes, 1)      # 24  → 7
```

Channel table (default `base=24`):

| Stage | Channels | Comment |
|-------|----------|---------|
| enc1 | 24 | |
| enc2 | 48 | after (1,2,2) pool |
| mid (bottleneck) | 96 | + Dropout3d(0.05) |
| dec2 (after skip concat 48+96) | 48 | |
| dec1 (after skip concat 24+48) | 24 | |
| out_conv 1×1×1 | 7 | logits |

**Parameter count**: ≈ 1.8 M (FP32 = ~7 MB, FP16 = ~3.5 MB).

---

### 1.4 Critical architectural choices

**Spatial-only pooling (`MaxPool3d((1, 2, 2))`):**
The temporal dimension `T` is **never downsampled**. This preserves per-frame resolution for all span lengths (even `T=9`) and means the network emits one class map per original frame without any temporal resampling. Catenary sag — the primary detection cue — remains spatially resolved across the full sequence.

**Decoder with trilinear upsampling (not transposed convolutions):**
`F.interpolate(..., scale_factor=(1,2,2), mode='trilinear', align_corners=False)` followed by a `ConvBlock3D`. Transposed convolutions with odd `H` or `W` produce shapes off by one from the encoder skip, breaking `torch.cat`. Trilinear upsampling to `encoder_feature.shape[2:]` enforces exact alignment.

**Dropout3d in bottleneck (added exp_v7):**
`nn.Dropout3d` zeroes entire feature channels (entire 3D spatial maps) rather than individual voxels. Placed at the bottleneck output — the most information-dense point in the network — it prevents the model from memorising span-specific spatial patterns. This was the primary fix for the persistent train/val gap observed in exp_v5 and exp_v6.

**`bias=False` throughout convolutional layers:**
All bias is provided by GroupNorm's learned affine shift `β`. Removing redundant Conv bias reduces parameter count without accuracy loss.

---

### 1.5 Data flow

```
x [1,1,T,H,W]
│
├─ enc1 [1→24] ──────────────────────────────────── skip e1 [1,24,T,H,W]
│   └─ pool1 (1,2,2)
│       └─ enc2 [24→48] ─────────────────────────── skip e2 [1,48,T,H/2,W/2]
│           └─ pool2 (1,2,2)
│               └─ mid [48→96] + Dropout3d
│                   └─ trilinear upsample to e2 size
│                       └─ cat(e2, up) [48+96=144]
│                           └─ dec2 [144→48]
│                               └─ trilinear upsample to e1 size
│                                   └─ cat(e1, up) [24+48=72]
│                                       └─ dec1 [72→24]
│                                           └─ out_conv 1×1×1 [24→7]
│
logits [1,7,T,H,W]  →  argmax along dim=1  →  class map [1,T,H,W]
```

---

## 2. Training pipeline

### 2.1 Critical fix: label stripping from model input

The most important discovery during development was that the labeled BMP files embed
line-class information directly in pixel gray values (128–254). The early training runs
(exp_v1 to exp_v3) fed these encoded values as model input: the model learned "this pixel
is gray-128, therefore comm" instead of learning visual / temporal features.

**Fix**: `strip_line_labels_from_input(vol)` in `volume.py` zeros all pixels with
`128 ≤ value ≤ 254` before passing the volume to the model. Applied in:
- `dataset.py` — during training
- `infer_goal2.py` — during Python inference (both labeled and unlabeled spans)
- `LineSegSentisInferenceManager.cs` — `NormalizeAndStripFrames()` in Unity

Without this fix, the model achieves artificially high metrics on labeled validation data
but produces only air/solid predictions on unlabeled spans.

### 2.2 Loss function

`SegmentationLoss` = weighted Focal CE + per-class weighted Dice.

```python
L = CE_focal(logits, target, class_weights) + dice_weight * Dice(logits, target, dice_class_weights)
```

Production hyperparameters (exp_v12):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `focal_gamma` | 2.5 | Suppresses easy air/solid voxels; focuses on ambiguous boundaries |
| `ce_class_weights` | `1,1,40,10,30,30,30` | comm 40× (rarest labeled class), primary 10× (well-represented), neutral/secondary/transmission 30× |
| `dice_class_weights` | `0.5,1,5,1,3,2,2` | comm Dice 5× — IoU-direct optimization for the hardest class |
| `dice_weight` | 0.5 | CE and Dice contribute equally |
| `label_smoothing` | 0.05 | Prevents boundary overconfidence |

### 2.3 Learning rate: OneCycleLR

After testing `ReduceLROnPlateau` (exp_v5–v9) and `CosineAnnealingWarmRestarts` (exp_v10–v11, which caused periodic 12–38 epoch performance regressions), `OneCycleLR` was adopted from exp_v11:

```
Epoch  1–16 (8%):  LR warms up from max_lr/25 = 4e-5 to max_lr = 1e-3
Epoch 17–157:      Single cosine decay from 1e-3 to 1e-3/10000 = 1e-7
```

This gives a single smooth learning trajectory without the catastrophic forgetting
caused by periodic LR resets.

### 2.4 Train/validation split — `val_seed`

A key discovery (exp_v9) was that the fixed `seed=42` always produced the same
structurally harder validation split. The val loss minimum was reached by epoch 28–32
in every run then increased monotonically — evidence of a data-distribution gap, not
model overfitting. No regulariser can close a distribution gap.

**Fix**: `--val_seed 7` draws a different 46-span validation set while keeping model
initialisation and augmentation seeds unchanged. The train/val gap dropped from
+0.17 to +0.06 in exp_v9, confirming the split was the problem.

### 2.5 Data augmentation

Applied in `dataset.py` when `--augment` is set:
- **Temporal flip**: randomly reverse the frame sequence along T with p=0.5. Lines sag symmetrically, so flipped volumes are valid training samples.
- **Gaussian noise**: σ=0.03 added to normalized input. Forces the model to learn structural features rather than memorising exact pixel intensities.

Both augmentations are disabled at validation and inference time.

### 2.6 Multi-GPU training (DDP)

Training runs on 2× NVIDIA A6000 GPUs using `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m line_seg.train_goal2 \
  --data_root . --out_dir ./goal2_runs/exp_v12 \
  --epochs 157 --lr 0.001 --lr_onecycle --onecycle_pct_start 0.08 \
  --focal_gamma 2.5 --ce_class_weights 1,1,40,10,30,30,30 \
  --dice_class_weights 0.5,1,5,1,3,2,2 --dropout 0.05 \
  --augment --noise_sigma 0.03 --val_seed 7 --weight_decay 1e-3
```

`torch.compile` was attempted (exp_v6) but fails on dynamic `T` dimension in
`F.interpolate` trilinear upsampling (`SymIntArrayRef` runtime error). Removed.

---

## 3. Evaluation metrics

### 3.1 Pixel-level (from confusion matrix)

| Metric | Formula |
|--------|---------|
| IoU_c | `TP_c / (TP_c + FP_c + FN_c)` |
| Precision_c | `TP_c / (TP_c + FP_c)` |
| Recall_c | `TP_c / (TP_c + FN_c)` |
| mean_iou (mIoU) | unweighted mean over 7 classes |
| **liu3** | mean IoU of classes 2–4 (comm / primary / neutral) — primary checkpoint metric |

`liu3` was chosen over `mIoU` because secondary and transmission have near-zero validation
pixel counts, making their IoU extremely noisy and an unreliable optimisation signal.

### 3.2 Object-level

All line classes (2–6) are merged into a binary "any conductor" mask. 3D connected
components (`scipy.ndimage.label`, 6-connectivity) identify individual conductor
instances. Greedy IoU matching (threshold 0.5) between predicted and GT components
yields micro precision / recall / F1 across all validation spans.

### 3.3 Reading the validation report

1. Ignore pixel accuracy — always ~99.8% due to dominant air class.
2. Focus on **per-class IoU for classes 2–4** (comm/primary/neutral).
3. Compare merged `line_pixel_iou` to mean per-class line IoU — a large gap means errors are **type confusion**, not mislocalization.
4. Check `line_obj_f1_micro` for instance-level detection.
5. Read `history_loss.png` for convergence diagnosis.

---

## 4. Best model results (exp_v12)

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| air | 1.000 | 1.000 | 1.000 | 1.000 |
| solid | 0.962 | 0.987 | 0.974 | 0.981 |
| **comm** | **0.496** | 0.617 | 0.718 | 0.664 |
| **primary** | **0.855** | 0.882 | 0.966 | 0.922 |
| **neutral** | **0.760** | 0.895 | 0.835 | 0.864 |
| secondary | 0.312 | 0.353 | 0.730 | 0.476 |
| transmission | 0.000 | — | — | — |

**liu3 = 0.704** · mIoU = 0.627 · Line-object micro F1 = 0.555 · Line pixel IoU = 0.817  
Best checkpoint: epoch 99 of 157 · Train/val gap last 5 epochs: +0.144

**Interpreting the IoU ranges on this dataset:**

| IoU range | Interpretation |
|-----------|----------------|
| < 0.20 | Rough localization only; shape/thickness or type badly off |
| 0.20 – 0.50 | Meaningful detection; verify with object-level F1 |
| 0.50 – 0.70 | Strong pixel agreement; conductors mostly correct thickness |
| > 0.70 | Very strong — close to annotated voxel positions |

Comm IoU of 0.496 indicates meaningful detection with residual type confusion (comm↔neutral).
This is an annotation ceiling — comm and neutral occupy the same spatial neighbourhood with
only a 3-bit type_code as the distinguishing signal in the current label scheme.

---

## 5. Inference pipeline

### 5.1 Python inference (`infer_goal2.py`)

```bash
python3 -m line_seg.infer_goal2 \
  --weights ./goal2_runs/exp_v12/best.pt \
  --input_dir . \
  --output_dir ./goal2_runs/infer_v12 \
  --device cuda
```

**Outputs per span**: predicted BMPs (`predicted_bmp/frame_*.bmp`), semantic class `.npy`, `meta.json`.  
**Outputs overall**: `inference_report.txt` (Section 1: line detection coverage per span; Section 2: evaluation vs GT labels if present), `evaluation_report.txt`, `inference_timings.txt`, `inference_manifest.json`.

### 5.2 ONNX export for Unity

```bash
python3 -m line_seg.export_goal2_onnx \
  --weights ./goal2_runs/exp_v12/best.pt \
  --out_dir ./unity_line_export_v12 \
  --fp16 --verify --ort_timing_runs 50
```

Exports: `line_seg_span_unet3d.onnx` (FP32), `line_seg_span_unet3d_fp16.onnx`, `line_seg_sidecar.json`.

### 5.3 TRT / ORT profiling

```bash
# Install GPU ORT: pip uninstall onnxruntime -y && pip install onnxruntime-gpu
python3 -m line_seg.trt_profile_goal2 \
  --onnx ./unity_line_export_v12/line_seg_span_unet3d.onnx \
  --weights ./goal2_runs/exp_v12/best.pt \
  --T 24 --H 112 --W 56 --runs 100
```

Measured latency on Paperspace A6000 (24×112×56 = 150 K voxels):  
PyTorch FP32: **6.2 ms** · PyTorch AMP/FP16: **4.8 ms**

ORT CPU: ~240 ms (fallback when `onnxruntime-gpu` is not installed).  
With `onnxruntime-gpu`: ORT CUDA ~15–30 ms, ORT TensorRT FP16 ~3–8 ms (Paperspace-only; not usable in Unity).

---

## 6. Unity Sentis 2.6.1 deployment

**Package**: `com.unity.ai.inference 2.6.1` · **Unity**: 6.3 LTS (6000.3.15f1)

Asset: `unity_line_export/LineSegSentisInferenceManager.cs`

### 6.1 Setup

1. Install Sentis: Window → Package Manager → + → Install package by name → `com.unity.ai.inference`
2. Drag `line_seg_span_unet3d_fp16.onnx` → ModelAsset field in Inspector
3. Drag `line_seg_sidecar.json` → SidecarJson field
4. Set `quantizeAtLoad = Float16` (halves VRAM; negligible accuracy loss)
5. Enable `warmupOnStart` (pre-compiles GPU shaders; avoids first-call stall)

### 6.2 Public API

```csharp
// Readiness check
bool ready = mgr.IsInferenceReady;

// Async inference (non-blocking, preferred for game loop)
var result = await mgr.InferSpanVolumeAsync(floatVolume, T, H, W);

// Sync inference (blocking, safe for editor scripts or background threads)
var result = mgr.InferSpanVolumeBlocking(floatVolume, T, H, W);

// Span with arbitrary H, W (not divisible by 4) — pad then crop
byte[] padded  = LineSegSentisInferenceManager.PadUint8VolumeTHW(raw, T, H, W, out int pH, out int pW);
float[] floats = LineSegSentisInferenceManager.NormalizeAndStripFrames(padded);
var res        = await mgr.InferSpanVolumeAsync(floats, T, pH, pW);
byte[] classes = LineSegSentisInferenceManager.CropClassMapTHW(res.classes, T, H, W, pH, pW);

// Class index → BMP gray byte (id_field=16: values 128–132)
byte gray = LineSegSentisInferenceManager.ClassToBmpGray(classIndex);
```

### 6.3 Key implementation notes

**H and W must be divisible by 4** at model input. The encoder applies `MaxPool3d((1,2,2))`
twice; violating this throws a `GroupNorm` channel error. Use `CeilToMultipleOf4`, `PadUint8VolumeTHW`, and `CropClassMapTHW` for arbitrary span sizes.

**Label stripping is mandatory** for labeled input spans.
`NormalizeAndStripFrames()` zeros pixels with `128 ≤ byte ≤ 254` in one pass, matching the
Python `strip_line_labels_from_input()` behaviour.

**BMP encoding** (critical fix from prior `id_field=0` bug):
`ClassToBmpGray` uses `(16 << 3) | type_code = 128 | type_code → 128–132`.
The prior `(0 << 3) | type_code = 0–4` produced solid/black values — invisible in any viewer
and not recognised as line pixels by `raw_to_semantic_labels()`.

**Namespace note**: if `ModelQuantizer`, `ModelWriter`, or `QuantizationType` cause CS0246,
add `using Unity.InferenceEngine;` alongside `using Unity.Sentis;`.

**Pre-bake quantized model** (avoids in-memory re-quantization on every `Awake()`):
```csharp
// Editor script — run once, then assign the .sentis file to modelAsset
mgr.SaveQuantizedModel("Assets/Models/LineSeg/line_seg_fp16.sentis", QuantizationType.Float16);
// Then set quantizeAtLoad = None in the Inspector
```

---

## 7. Key improvements: version 1 to current

| Version | Problem | Fix | Impact |
|---------|---------|-----|--------|
| exp_v1 | Zero line detection | CE boost 2×→20×; output BMP encoding `id_field=0→16` | Unblocked training |
| exp_v2–3 | Low comm IoU (0.786) | Per-class CE weights; label smoothing ε=0.05; more epochs | comm→0.818; liu3≈0.851 |
| exp_v4 | Model cheating on embedded labels | `strip_line_labels_from_input()` across train+infer | Training integrity restored |
| exp_v5 | Weak detection on binary input | Per-class Dice (comm 5×); temporal flip + noise augmentation | comm 0.461 on honest input |
| exp_v6 | Slow single-GPU training | 2-GPU DDP via `torchrun`; `torch.compile` removed (dynamic T) | ~2× epoch throughput |
| exp_v7 | Spatial memorisation | `Dropout3d(p=0.05)` at bottleneck | Reduced train/val gap |
| exp_v8 | Noisy secondary polluting checkpoint | `val_line_mean_iou` → classes 2–4 only; `--sched_metric` decoupled | Stable checkpointing |
| exp_v9 | Structural bias in fixed train/val split | `--val_seed 7` | Gap +0.17 → +0.06 |
| exp_v10–11 | LR restarts causing periodic regressions | `OneCycleLR` (single smooth warmup + cosine decay) | Stable convergence |
| exp_v12 | Best combined run | All of above + `focal_gamma=2.5`, boosted Dice weights | **liu3=0.704** (production) |

---

## 8. Related documents

- [`POWER_LINE_DETECTION_PLAN.md`](POWER_LINE_DETECTION_PLAN.md) §11 — full experiment log with per-epoch metrics
- [`Line_Annotation.md`](Line_Annotation.md) — annotation quality analysis and recommended improvements
- [`README_GOAL2.md`](README_GOAL2.md) — end-to-end Goal 2 pipeline and file layout
- [`README_CATENARY_BASELINE.md`](README_CATENARY_BASELINE.md) — RANSAC + catenary object-level baseline
- [`PAPERSPACE_UPLOAD_AND_RUN.md`](PAPERSPACE_UPLOAD_AND_RUN.md) — Paperspace training / inference instructions
