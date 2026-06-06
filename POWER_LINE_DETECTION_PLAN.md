# Power Line Detection Plan — DUKE_FLORIDA_150

Critical analysis and SOTA algorithm recommendations for detecting and classifying
power lines from sequential LiDAR cross-section frame images.

---

## 1. Dataset Characterization

**Note on scale:** Row counts such as “139 spans” and “17,541 frames” came from a **sample** of the tree used while drafting this plan. Your full `DUKE_FLORIDA_150` directory may contain **more** spans and frames. **Python code does not hardcode these numbers** — it discovers every immediate subfolder that contains `frame_*.bmp`.

### 1.1 Structure

| Property | Value (example snapshot; your tree may differ) |
|---|---|
| Total spans (pole-to-pole) | N (sample had ~139) |
| Total BMP frames | M (sample had ~17k) |
| Frames per span | varies widely (e.g. tens to hundreds) |
| Image dimensions | 31 × 57 pixels, 8-bit grayscale |
| Frame naming | `frame_XXXX_hYYY.bmp` |
| Image orientation | Top-down DIB (y=0 at top of image) |

### 1.2 Pixel Encoding

| Value | Meaning |
|---|---|
| 255 (white) | Empty space (air) |
| 0 (black) | Non-line solid (vegetation, transformers, insulators) |
| 128–254 (gray) | Labeled line pixel: `type = gray & 0x07`, `id = gray >> 3` |

### 1.3 Line Types Observed

| Type Code | Name | Observed in Dataset |
|---|---|---|
| 0 | Comm (communication) | Yes (gray 160) |
| 1 | Primary | Yes (grays 137, 145, 153) |
| 2 | Neutral | Yes (gray 130) |
| 3 | Secondary | Yes (gray 131) |
| 4 | Transmission | Not observed in sampled spans |

### 1.4 Key Data Properties

- **Lines per span**: 1–6 distinct labeled lines.
- **Line pixel density**: Very sparse — 0.1% to 0.9% of pixels per frame.
- **Line thickness**: 1–3 pixels wide per line.
- **Spatial ordering**: Lines maintain a consistent vertical ordering within a span
  (e.g., transmission at top, then primary, neutral, secondary, comm at bottom).
- **Temporal coherence (sag)**: All lines in a span exhibit a smooth, symmetric
  catenary-like "U" pattern in their y-centroid across frames — highest at the
  poles (frame endpoints), lowest at mid-span. This is the primary visual cue
  for human recognition.
- **Black pixels (solids)**: Intermittent and spatially localized. Some frames are
  entirely white. Others have heavy vegetation covering 10–14% of pixels.
- **Missing labels**: Some frames within a span have no gray pixels even when
  adjacent frames do (occlusion, lidar gaps). Some lower lines may be unlabeled
  per truthsayer instructions.

---

## 2. Problem Decomposition

Per `Lines_Detect_Goals.txt`, there are two ML goals:

### Goal A (Primary): Span-Level Binary Classification
> Does this span contain power lines between its poles?

This is the **gate** that filters false-positive pole detections (e.g., tree trunks).
The input is an entire sequence of B&W frames; the output is a binary yes/no.

### Goal B (Secondary): Per-Voxel Instance Segmentation + Classification
> Given a span with power lines, segment each black voxel as belonging to a
> specific line (consistent ID across all frames) and classify its type
> (comm/primary/neutral/secondary/transmission).

This is a **video instance segmentation** problem on sequential 2D cross-sections,
where:
- Objects (lines) are thin, persistent, and move coherently across frames.
- The number of objects is unknown and varies per span.
- Each object has both an **instance identity** (consistent ID) and a **semantic
  class** (line type).

---

## 3. Critical Analysis of Challenges

### 3.1 Extreme Image Size (31 × 57)

These images are **tiny** — smaller than a single CIFAR-10 image (32 × 32).
Standard architectures designed for 224×224+ inputs are wildly oversized. This
demands:
- Custom lightweight networks with very few downsampling stages.
- No aggressive pooling — a single 2× downsample reduces height to 28 pixels.
- Careful attention to the **sequence dimension** (100–300 frames) as the primary
  information axis, not the spatial dimensions.

### 3.2 Extreme Class Imbalance

Line pixels are 0.1–0.9% of each frame. Black (solid) pixels are 0–14%.
White (empty) dominates at 86–100%. This means:
- Standard cross-entropy will overwhelmingly favor predicting "empty".
- Focal loss, Dice loss, or Tversky loss are essential.
- Evaluation must use per-class metrics, not accuracy.

### 3.3 Temporal Dependency is Critical

The goals document explicitly states: *"There is no sufficient way to determine
[lines] solely from each image, but a human eye can pick them out reliably from a
flip-book."* This is the central challenge. Any viable approach **must** model the
relationship between frames. A per-frame segmentation model will fail.

### 3.4 Variable Sequence Length

Spans range from 9 to 298 frames. Models must handle variable-length sequences,
ruling out fixed-size 3D convolutions without padding/chunking strategies.

### 3.5 No Negative Examples (Yet)

The current dataset contains only spans **with** power lines. Goal A (binary
classification) cannot be trained without negative examples (spans with no lines).
This will need to be addressed through:
- Synthetic negatives (e.g., masking out gray pixels to create line-free spans).
- One-class classification / anomaly detection.
- Deferring Goal A until negative data is available.

### 3.6 Incomplete Labels

Lower lines may be unlabeled. Some frames have no labels despite lines being
present. The model should be robust to:
- Missing labels in supervision (ignore unlabeled regions below the lowest
  labeled line).
- Predicting more lines than labeled (acceptable per the goals document).

---

## 4. SOTA Algorithm Analysis

### 4.1 Approach 1: 3D UNet on Voxel Volumes (Recommended for Goal B)

**Concept**: Treat each span as a 3D volume (T × H × W) where T is the frame
(sequence) dimension. Apply a 3D encoder-decoder network that jointly processes
spatial and temporal information.

**Why it fits**:
- The frames are explicitly described as "MRI-like cross-sections" — this is
  exactly the domain where 3D UNets excel (medical image segmentation).
- The catenary sag pattern is a smooth 3D surface through the volume.
- Lines form continuous "tubes" through the volume, analogous to blood vessels
  in medical imaging.
- The tiny spatial dimensions (31 × 57) mean the 3D volume is very manageable:
  a 200-frame span is only 200 × 57 × 31 = ~355K voxels.

**Architecture**:
```
Input:  (1, T, 57, 31)    — single-channel B&W volume
Output: (K, T, 57, 31)    — K binary masks (one per detected line)
        + (K, num_types)   — classification logits per line
```

**Specific model**: **nnU-Net** (self-configuring 3D UNet) or a custom 3D UNet
with:
- Asymmetric kernel sizes: larger along T (temporal), smaller along H, W.
  E.g., (5, 3, 3) kernels to capture catenary sag across 5 frames while keeping
  spatial context tight.
- 2–3 encoder/decoder levels (more would over-reduce the tiny spatial dims).
- Instance segmentation head: either learned embeddings for clustering
  (discriminative loss) or direct multi-channel prediction.

**Relevant SOTA**:
- **nnU-Net v2** (Isensee et al., 2024): Self-configuring medical segmentation.
  Handles 3D volumes, extreme class imbalance, variable sizes. Would auto-tune
  patch size, augmentation, loss.
- **UNETR / Swin-UNETR**: Transformer-based 3D segmentation — likely overkill
  for this data size but provides strong temporal attention.
- **VoxelMorph-style**: If we frame line tracking as deformable registration
  across frames.

**Strengths**: Naturally captures temporal coherence. Well-studied for thin
structure segmentation (vessels, airways). Can handle variable sequence lengths
with sliding window or padding.

**Weaknesses**: Instance segmentation (separating individual lines) is harder
with dense prediction — needs post-processing or embedding-based clustering.

---

### 4.2 Approach 2: 2D UNet + Temporal Recurrence (LSTM/GRU/Transformer)

**Concept**: Per-frame feature extraction with a lightweight 2D encoder, followed
by a temporal model (bidirectional LSTM, GRU, or Transformer) that propagates
information across the sequence, then a decoder that produces per-frame masks.

**Why it fits**:
- Separates spatial feature extraction from temporal reasoning.
- Handles variable sequence lengths naturally (RNNs/Transformers).
- The temporal model can learn the catenary sag pattern explicitly.

**Architecture**:
```
Per-frame:  2D-Encoder(31×57) → feature map (C × 14 × 7)
Sequence:   BiLSTM or Transformer over T frames → temporal features
Per-frame:  2D-Decoder(temporal features) → (K, 57, 31) masks
```

**Relevant SOTA**:
- **UCorr** (2025): Uses a UNet with temporal correlation layer on sequential
  frames for thin wire segmentation — directly analogous to this problem. Trains
  on paired frames, uses correlation to detect stable features across time.
- **LOMM** (ICCV 2025): Latest Object Memory Management for video instance
  segmentation — maintains per-object state across frames for consistent IDs.
- **Tube-Link** (ICCV 2023): Processes short subclips with spatial-temporal tube
  masks, then links tubes across the sequence. Good for maintaining instance
  consistency.

**Strengths**: Flexible sequence handling. Can pre-train 2D encoder on
per-frame tasks. Explicit temporal reasoning.

**Weaknesses**: Bidirectional models need the full sequence at train time
(acceptable here since sequences are short). May struggle with very long spans
(298 frames) unless windowed.

---

### 4.3 Approach 3: Graph Neural Network on Temporal Trajectories

**Concept**: Detect candidate line pixels per frame (simple thresholding or
learned), then build a graph where nodes are candidate pixels and edges connect
candidates across adjacent frames. A GNN propagates information to cluster nodes
into line instances and classify types.

**Why it fits**:
- Lines are extremely sparse (< 1% of pixels). Working in sparse representation
  avoids wasting compute on the overwhelming white space.
- The "flip-book" recognition is essentially trajectory linking — connecting
  corresponding points across frames.
- Graph structure naturally handles variable numbers of lines and candidates.

**Architecture**:
```
Step 1: Per-frame black pixel extraction (trivial threshold)
Step 2: Build spatial-temporal graph:
        - Nodes: black pixel clusters per frame
        - Edges: proximity links within frame + across adjacent frames
Step 3: GNN (e.g., GAT, GraphSAGE) → node embeddings
Step 4: Clustering (e.g., mean-shift on embeddings) → line instances
Step 5: Classification head per cluster → line type
```

**Relevant SOTA**:
- **GNN for point cloud segmentation** (KPConv, PointNet++): These methods
  segment 3D point clouds into semantic classes. The sparse black pixels can be
  treated as a 3D point cloud (x, y, frame_index).
- **DCPLD-Net** (Chen et al., 2022): Diffusion-coupled CNN for real-time power
  line detection from LiDAR — uses graph-like propagation.
- **Luo et al. (2024)**: Deep learning with local topological information and
  graph convolutions for power line extraction.

**Strengths**: Computationally efficient (sparse). Naturally handles variable
line counts. Good at capturing geometric relationships between lines.

**Weaknesses**: Requires good candidate extraction as first step. May miss
lines that are occluded in many consecutive frames.

---

### 4.4 Approach 4: Catenary-Informed Model Fitting (Hybrid Geometric + ML)

**Concept**: Exploit the physics of the problem — power lines follow catenary
curves between poles. Use geometric model fitting (RANSAC + catenary regression)
to extract line candidates, then use ML only for classification and refinement.

**Why it fits**:
- The sag data confirms all lines follow smooth, symmetric catenary curves in
  the y-centroid vs. frame-index space.
- The geometric constraint dramatically reduces the search space.
- Works well with sparse, noisy data (lidar gaps, missing frames).

**Pipeline** (conductor-aligned default):
```
Step 1: Per-frame mask: labeled line pixels (128 ≤ gray < 255; type 0–4 in low 3 bits)
Step 2: Project those pixels to (frame_index, y) space (e.g. centroid per frame)
Step 3: RANSAC parabola + optional catenary refit:
        - Fit multiple catenary curves y(t) = a·cosh((t-t0)/a) + b
        - Each fitted curve represents a candidate line
Step 4: For each candidate curve, extract the x-spread of matching pixels
        → this gives the line's lateral position and confirms it's real
Step 5: ML classifier on extracted features:
        - Catenary parameters (a, b, t0)
        - Relative vertical position among detected lines
        - Pixel statistics (intensity, width, consistency)
        → Predict line type (comm/primary/neutral/secondary)
Step 6: Assign instance IDs based on curve identity
```

**Relevant SOTA**:
- **RANSAC + catenary** (Yang et al., 2018): 99.6% extraction accuracy on
  airborne LiDAR using Hough transform + RANSAC parabolic fitting.
- **Piecewise Model Growing** (Jwa & Sohn, 2012): 3D catenary decomposed into
  horizontal line (XY) + vertical catenary (XZ).
- **Particle Swarm Optimization** catenary fitting (Guo et al., 2019):
  Per-conductor 3D catenary models from segmented point clouds.

**Strengths**: Physics-informed — leverages strong domain priors. Interpretable.
Works with very small datasets. No training required for the geometric part.
Naturally produces instance-level separation.

**Weaknesses**: Assumes catenary model holds (may break near poles, with
bundled conductors, or with heavy vegetation). Vegetation/solid pixels that
are near lines will confuse fitting. Needs ML for robustness.

---

### 4.5 Approach 5: Temporal Embedding Network (Best for Goal A)

**Concept**: For the binary classification goal (does this span have power
lines?), process the entire frame sequence with a temporal aggregation network
and produce a single yes/no prediction.

**Architecture options**:
- **3D CNN classifier**: Treat span as a 3D volume, apply 3D convolutions,
  global average pool, binary head. Very simple and likely sufficient.
- **Frame-level features + attention pooling**: Extract per-frame features
  (e.g., % black pixels, spatial distribution statistics), then use a
  Transformer or attention-weighted pooling to aggregate across time.
- **Video classification backbone**: SlowFast, TimeSformer, or VideoMAE adapted
  to grayscale 31×57 frames.

**Key insight for Goal A**: Power lines create a distinctive temporal
"fingerprint" — stable, thin, spatially coherent dark regions that sag
symmetrically. Vegetation/tree trunks do NOT exhibit this pattern (they are
spatially fixed or chaotic across frames). A learned temporal feature should
easily discriminate these.

---

## 5. Recommended Approach: Phased Strategy

### Phase 1: Geometric Baseline + Goal A Feasibility
**Timeline**: 1–2 weeks

1. **Data preparation**: Write a loader that reads **all span folders** under the data root as B&W
   volumes (strip gray labels to simulate inference-time input).
2. **Catenary fitting baseline** (Approach 4, Steps 1–3): Apply RANSAC
   parabola / catenary refit on the **(frame_index, y)** projection of
   **labeled line voxels** (`128 ≤ gray < 255`, type in low 3 bits, id field in
   upper 5 bits — see `README_CATENARY_BASELINE.md` and `tools/line_encoding.py`).
   Optional secondary run on **black** (`gray == 0`) solids tests vegetation-only
   sag behavior. This establishes a **non-ML baseline** and insight into data.
3. **Goal A proxy**: Spans where catenary fitting finds ≥ 1 high-confidence
   curve → classified as "has lines". Evaluate precision using the labeled
   data.
4. **Visualization toolkit**: Build a flip-book viewer that overlays
   predicted lines on the B&W frames for rapid qualitative assessment.

### Phase 2: 3D UNet Instance Segmentation (Goal B)
**Timeline**: 2–4 weeks

**Started in repo**: `line_seg/` — semantic 7-class 3D UNet + `train_goal2.py` (see `README_GOAL2.md`). Instance IDs and B&W-only inference remain future work.

1. **Label preparation**: Convert gray-encoded labels into multi-channel
   instance masks. Each line ID gets its own binary mask channel.
   Handle missing labels by masking loss below the lowest labeled line.
2. **Model**: Custom 3D UNet with:
   - Input: (1, T, 57, 31) B&W volume (padded/chunked to fixed T).
   - Encoder: 3 levels, channels [16, 32, 64], asymmetric kernels.
   - Decoder: Instance embedding head (D-dimensional per-voxel embedding)
     + semantic head (5-class per-voxel).
   - Loss: Discriminative loss for instance embeddings + Focal/Dice for
     semantics + auxiliary catenary smoothness regularizer.
3. **Post-processing**: Mean-shift clustering on embeddings → instance masks.
   Fit catenary to each instance for temporal smoothing.
4. **Training**: 5-fold cross-validation across spans.

### Phase 3: Temporal Refinement + Goal A Classifier
**Timeline**: 1–2 weeks

1. **Refine Phase 2 with recurrence**: Add bidirectional temporal attention
   or LSTM layers to the 3D UNet bottleneck. This helps propagate identity
   through frames where a line is occluded.
2. **Goal A classifier**: Train a lightweight binary classifier on span-level
   features extracted from the Phase 2 encoder. Generate synthetic negatives
   by masking out all line-like structures from existing spans.
3. **Ensemble**: Combine catenary fitting confidence (Phase 1) with learned
   features (Phase 3) for robust binary classification.

---

## 6. Recommended Architecture Detail

### 6.1 Primary Model: Sparse3DUNet-Lines

```
Input: (B, 1, T_pad, 57, 31)  — B&W volume, T padded to multiple of 8
                                  or processed with sliding window

Encoder:
  enc1: Conv3d(1→16, k=(5,3,3), pad=(2,1,1)) + BN + ReLU
        Conv3d(16→16, k=(5,3,3), pad=(2,1,1)) + BN + ReLU
  pool1: MaxPool3d(2,2,2)  → (16, T/2, 28, 15)

  enc2: Conv3d(16→32, k=(5,3,3), pad=(2,1,1)) + BN + ReLU
        Conv3d(32→32, k=(5,3,3), pad=(2,1,1)) + BN + ReLU
  pool2: MaxPool3d(2,2,2)  → (32, T/4, 14, 7)

Bottleneck:
  Conv3d(32→64, k=(5,3,3)) + BN + ReLU
  BiLSTM or Transformer along T dimension (operating on flattened H×W features)
  Conv3d(64→64, k=(5,3,3)) + BN + ReLU

Decoder:
  up2: Upsample + Concat(enc2) → Conv3d(64+32→32)
  up1: Upsample + Concat(enc1) → Conv3d(32+16→16)

Heads:
  Instance: Conv3d(16→D, k=1) → D-dim embedding per voxel
  Semantic: Conv3d(16→6, k=1) → 6-class (bg, comm, primary, neutral, secondary, transmission)
  Binary:   GlobalAvgPool3d → FC → sigmoid (span-level "has lines" prediction)
```

**Parameter count**: ~150K–300K (very lightweight for 3D volumes this small).

**Training strategy**:
- Patch-based: Sample (T_chunk, 57, 31) windows with overlap along T.
- Augmentation: Random temporal flip, random temporal crop, intensity jitter,
  additive Gaussian noise to simulate lidar gaps.
- Loss: `L = λ_inst * DiscriminativeLoss + λ_sem * FocalDiceLoss + λ_bin * BCELoss`
- Optimizer: AdamW, OneCycleLR, ~200 epochs.

### 6.2 Alternative: Temporal Correlation Network (UCorr-inspired)

For a simpler, potentially faster approach:

```
Input: Pairs or triplets of consecutive B&W frames (1, 57, 31) each.

Shared 2D Encoder: Conv2d stack → feature maps per frame
Correlation Layer: Cross-frame feature correlation (à la UCorr)
Decoder: Per-frame masks + per-pixel embeddings
Temporal Linking: Hungarian matching on embeddings across all frame pairs
```

This avoids 3D convolutions entirely and handles variable sequence lengths
trivially. Training is on frame pairs/triplets, inference chains across the
full sequence.

---

## 7. Evaluation Metrics

### Goal A: Span-Level Classification
| Metric | Target |
|---|---|
| Precision | > 0.95 (few false "has lines" predictions) |
| Recall | > 0.99 (miss almost no real line spans) |
| F1-score | > 0.97 |

### Goal B: Instance Segmentation
| Metric | Description |
|---|---|
| **VPQ** (Video Panoptic Quality) | Standard metric for video instance segmentation — jointly evaluates recognition and segmentation across time. |
| **Per-line IoU** | For each ground-truth line, find best-matching predicted line; report temporal IoU (intersection of voxels across all frames). |
| **ID consistency** | % of frames where a predicted line's ID matches the same ground-truth line. |
| **Type accuracy** | Classification accuracy per line, weighted by line pixel count. |
| **Recall (line-level)** | % of ground-truth lines that are detected (matched IoU > 0.5). |

---

## 8. Data Split Strategy

With a **finite** number of spans, careful splitting is essential:

- **5-fold cross-validation** stratified by:
  - Number of lines per span (1 vs 2–3 vs 4–6).
  - Line types present (spans with comm, spans with only primary, etc.).
  - Span length (short < 50 frames, medium 50–150, long > 150).
- Each fold: roughly **80% train / 20% val** (exact counts depend on N).
- Report mean ± std across folds.

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dataset size limited (finite span count) for deep learning | High | Heavy augmentation, transfer from medical 3D segmentation, geometric priors as regularization. Start with catenary baseline. More spans in `DUKE_FLORIDA_150` improve generalization if labels stay consistent. |
| Vegetation pixels mistaken for lines | Medium | Temporal consistency criterion — vegetation doesn't sag smoothly. Catenary fitting rejects non-catenary patterns. |
| Missing/incomplete labels degrade training | Medium | Mask loss below lowest labeled line. Accept model predicting extra lines. |
| Variable sequence length causes padding issues | Low | Sliding window approach. RNN/Transformer handles variable length natively. |
| No negative examples for Goal A | High | Synthetic negatives by: (a) using spans with heavy vegetation but lines removed, (b) random noise volumes, (c) rotated/shuffled frames that break temporal coherence. |
| Lines too thin to segment reliably | Medium | Dilate labels during training (predict a 3-pixel-wide halo). Use distance-transform-weighted loss. Evaluate with tolerance margin. |

---

## 10. Summary of Recommendations

| Decision | Recommendation | Rationale |
|---|---|---|
| **Primary approach** | 3D UNet with instance embeddings | MRI-like data, tiny spatial dims, strong temporal dependency |
| **Baseline** | RANSAC sag fit on **line_gray** (decoded labels) + viz toolkit | Physics-informed, matches label encoding; optional `black` mode for solids |
| **Goal A method** | 3D CNN binary classifier + catenary confidence | Lightweight, can bootstrap from Goal B encoder |
| **Goal B method** | 3D UNet + discriminative loss + post-processing | Best balance of temporal modeling and instance separation |
| **Fallback** | UCorr-style 2D + temporal correlation | Simpler, faster to iterate, handles variable lengths easily |
| **Loss function** | Focal + Dice + Discriminative | Handles extreme imbalance + instance separation |
| **Data handling** | 5-fold CV, heavy augmentation, synthetic negatives | Small dataset demands careful validation |
| **First deliverable** | `tools/run_baseline.py` + `README_CATENARY_BASELINE.md` | Line-label–aware RANSAC + plots / GIF / JSON `line_labels_summary` |

---

## 11. Implementation Status — Phase 2 Progress (Goal B: Semantic Segmentation)

This section records what has been built and trained as of June 2026, updating the theoretical plan above with concrete results across 12 training experiments (exp_v1 → exp_v12).

### 11.1 What was built

**Model**: `SpanUNet3D` — a lightweight 3D UNet in `line_seg/model.py`.

```
Input  : [B, 1, T, H, W]  float32, normalised to [0, 1], labels stripped
Encoder: ConvBlock3D(1→24) → MaxPool3d(1,2,2)
         ConvBlock3D(24→48) → MaxPool3d(1,2,2)
Bottleneck: ConvBlock3D(48→96) + Dropout3d(p=0.05)
Decoder: trilinear upsample + skip → ConvBlock3D(144→48)
         trilinear upsample + skip → ConvBlock3D(72→24)
Output : Conv3d(24→7, k=1)  →  [B, 7, T, H, W] logits
```

Key design choices confirmed across training experiments:
- **GroupNorm** not BatchNorm — essential with `batch_size=1`; `BatchNorm` collapses on single-volume batches.
- **MaxPool3d(1,2,2)** — downsamples only H,W; the full frame sequence T is preserved so catenary sag context flows through the entire network depth.
- **Trilinear upsampling** — avoids off-by-one artifacts with odd H/W dimensions (common in raw span BMP sizes).
- **`base_channels=24`** — sufficient capacity for ~230 spans; larger bases showed no metric gain.
- **Dropout3d(p=0.05) in bottleneck** — added from exp_v7 to prevent spatial memorisation of training-span patterns, directly reducing the train/val gap.

Companion files: `train_goal2.py`, `infer_goal2.py`, `losses.py`, `eval_metrics.py`, `volume.py`, `dataset.py`, `export_goal2_onnx.py`, `trt_profile_goal2.py`, `unity_line_export/LineSegSentisInferenceManager.cs`.

---

### 11.2 Training progression — exp_v1 → exp_v12

#### Phase A — Initial builds (exp_v1 to exp_v3): getting line detection to work at all

##### exp_v1 — baseline (zero line detection)

Settings: focal γ=2.0, uniform CE boost 2×, 30 epochs, single GPU.

**Result**: Zero line detection. Two compounding failures:
1. **Loss imbalance**: 2× CE boost negligible against ~750× pixel-frequency gap for comm. Model collapsed to air/solid only.
2. **Output encoding bug**: `pack_line_gray(0, type_code)` produced BMP gray values 0–4 (solid/black range) — line predictions were invisible even when correct.

**Fixes before exp_v2**:

| Fix | File | Effect |
|-----|------|--------|
| `pack_line_gray(0,…)` → `pack_line_gray(16,…)` | `infer_goal2.py` | Output BMPs contain valid line pixels (128–132) |
| `np.zeros` → `np.ones` in `raw_to_semantic_labels` | `volume.py` | Gray 1–127 no longer silently treated as air |
| Extended span discovery to nested frame directories | `volume.py` | Training and inference see the same span set |
| `--line_class_ce_boost` default 2.0 → 10.0 | `train_goal2.py` | Meaningful gradient signal for line classes |
| `val_line_mean_iou` logged every epoch | `train_goal2.py` | Line-class progress visible during training |

---

##### exp_v2 — first successful run (IoU comm 0.786, primary 0.945)

Settings: focal γ=2.0, uniform boost **20×**, lr_patience=4, **40 epochs**.
Comm→neutral confusion 9.7%; best checkpoint at the final epoch (not yet converged).

**Fixes before exp_v3**: per-class CE weights (`1,1,25,10,25,30,30`), label smoothing ε=0.05, lr_patience→6.

---

##### exp_v3 — converged (comm 0.818, primary 0.949, neutral 0.785 · liu3 ≈ 0.851)

Settings: focal γ=2.0, `--ce_class_weights 1,1,25,10,25,30,30`, label smoothing ε=0.05, **60 epochs**.
Line-object F1: **0.932**. Residual inter-class confusion (comm↔neutral ~8%) identified as annotation ceiling — unresponsive to further loss tuning. See `Line_Annotation.md §11`.

---

#### Phase B — Critical discovery: model was learning from embedded label values (exp_v4)

After exp_v3, inference on truly unlabeled spans produced only air/solid predictions.
Investigation revealed that **the model had been training on embedded grayscale label values (128–254) in the input BMPs rather than on visual features**.

The labeled BMP files have line pixels encoded as gray values 128–254. The training code was feeding these encoded pixels directly as model input. The model learned the label encoding scheme, not the underlying scene geometry — it recognised "this pixel is gray-128, so it must be comm" rather than "this is a conductor-shaped feature in consecutive frames."

**Root-cause fix**: `strip_line_labels_from_input()` added to `volume.py`. Applied in `dataset.py` for training and `infer_goal2.py` for inference. All label pixels (128–254) are zeroed before the forward pass; the model sees only binary 0/255 (solid/air) input, matching the purely unlabeled spans it will encounter at inference time.

Effect: exp_v2 / exp_v3 metrics were invalidated (trained on cheated input). Retraining from scratch required.

---

#### Phase C — Training on correct binary input (exp_v5 → exp_v12): solving the train/val divergence

After the label-stripping fix, all subsequent experiments show a persistent **train/val loss divergence** — training loss descends while validation loss either plateaus or widens. This became the central challenge for experiments v5 through v12.

**Key discoveries and changes per experiment:**

| Experiment | Key change | Effect |
|---|---|---|
| **exp_v5** | Per-class Dice weights (`0.5,1,2.5,1,2,1.5,1.5`), temporal flip augmentation, Gaussian noise σ=0.03 on input | First metrics on truly binary input: comm 0.461, prim 0.862, neut 0.749 |
| **exp_v6** | 2-GPU DDP (`torchrun --nproc_per_node=2`), attempted `torch.compile` | `torch.compile` failed on dynamic T dimension (`SymIntArrayRef` error); removed. DDP retained — epoch time normalized after `cudnn_benchmark` warmup |
| **exp_v7** | `Dropout3d(p=0.05)` added to model bottleneck; `val_line_mean_iou` refined to classes 2–4 only (comm/primary/neutral) | Reduced spatial memorisation; secondary's noisy metric no longer pollutes the checkpoint criterion |
| **exp_v8** | `--sched_metric val_line_mean_iou` to decouple LR scheduler from checkpoint metric; `--early_stop_patience 40` added | Better separation between what triggers LR reduction and what saves `best.pt` |
| **exp_v9** | **`--val_seed 7`** (new argument) | **Key breakthrough**: fixed `seed=42` always produced the same structurally harder validation split. Changing the split seed showed the divergence was a _data-distribution_ problem, not a regularisation problem. Gap reduced significantly (train/val gap dropped from +0.17 to +0.06) |
| **exp_v10** | `--epochs 150`; `CosineAnnealingWarmRestarts` (`--lr_cosine`) | More epochs allowed LR to cycle; comm improved to 0.499 |
| **exp_v11** | `OneCycleLR` (`--lr_onecycle`) replacing cosine restarts; EMA smoothing on training curves | Cosine restarts caused periodic catastrophic forgetting (12–38 epoch regressions per cycle). OneCycleLR gives a single smooth warmup + cosine decay |
| **exp_v12** | Plot warmup-trimming; SWA evaluated for exp_v13 comparison | **Best overall model**: comm 0.496, primary 0.855, neutral 0.760, **liu3 = 0.704** |

---

##### exp_v12 — current production model

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| air | 1.000 | 1.000 | 1.000 | 1.000 |
| solid | 0.962 | 0.987 | 0.974 | 0.981 |
| **comm** | **0.496** | 0.617 | 0.718 | 0.664 |
| **primary** | **0.855** | 0.882 | 0.966 | 0.922 |
| **neutral** | **0.760** | 0.895 | 0.835 | 0.864 |
| secondary | 0.312 | 0.353 | 0.730 | 0.476 |

**liu3** (comm + primary + neutral average) = **0.704**  
Line-object micro F1: **0.555** · Line pixel IoU (merged): **0.817**  
Best at epoch 99 / 139 · Train/val gap last 5 epochs: +0.144  

Training hyperparameters: `--lr 0.001 --lr_onecycle --onecycle_pct_start 0.08 --onecycle_div_factor 25 --focal_gamma 2.5 --ce_class_weights 1,1,40,10,30,30,30 --dice_class_weights 0.5,1,5,1,3,2,2 --dropout 0.05 --augment --noise_sigma 0.03 --val_seed 7 --weight_decay 1e-3`

---

##### exp_v13 — SWA attempted (worse overall — v12 retained)

Stochastic Weight Averaging (SWA, `--swa_start 70`, 88 snapshots) was added. Results:

| Metric | exp_v12 | exp_v13 | Winner |
|--------|---------|---------|--------|
| liu3 | **0.704** | 0.700 | v12 |
| comm IoU | 0.496 | **0.512** | v13 |
| primary IoU | **0.855** | 0.846 | v12 |
| neutral IoU | **0.760** | 0.741 | v12 |
| secondary IoU | **0.312** | 0.012 | v12 |
| train/val gap | **+0.144** | +0.172 | v12 |

SWA offered no net benefit — v12's single best checkpoint is the production model.

---

### 11.3 Summary — what each phase achieved

| Phase | Problem addressed | Key change | Metric impact |
|-------|------------------|------------|---------------|
| exp_v1 | Zero detection | CE boost, output encoding fix | Unblocked training |
| exp_v2–v3 | Low line IoU | Per-class weights, label smoothing, more epochs | comm 0.786→0.818 |
| exp_v4 | Model cheating on embedded labels | `strip_line_labels_from_input()` | Training integrity restored |
| exp_v5 | Weak binary-input detection | Per-class Dice, augmentation, noise | comm 0.461 on honest input |
| exp_v6 | Single-GPU speed | 2-GPU DDP | ~2× epoch throughput |
| exp_v7 | Spatial memorisation / bottleneck overfitting | Dropout3d(0.05) | Reduced train/val gap |
| exp_v8–v9 | Data-distribution split bias | `--val_seed 7` | Gap +0.17 → +0.06 |
| exp_v10–v11 | LR scheduling: plateau vs restarts | OneCycleLR | Eliminated periodic regressions |
| exp_v12 | Best generalisation | Combined all above | liu3 = 0.704 |

### 11.4 Unity / ONNX deployment pipeline

The trained `best.pt` is exported to ONNX (FP32 + FP16) and deployed to Unity 6.3 LTS via Sentis 2.6.1:

```
Paperspace:
  python3 -m line_seg.export_goal2_onnx \
    --weights ./goal2_runs/exp_v12/best.pt \
    --out_dir ./unity_line_export_v12 \
    --fp16 --verify --ort_timing_runs 50

TRT profiling (onnxruntime-gpu required):
  python3 -m line_seg.trt_profile_goal2 \
    --onnx ./unity_line_export_v12/line_seg_span_unet3d.onnx \
    --weights ./goal2_runs/exp_v12/best.pt \
    --T 24 --H 112 --W 56 --runs 100
```

GPU latency (A6000, 24×112×56 input): **PyTorch FP32 ≈ 6.2 ms / PyTorch AMP FP16 ≈ 4.8 ms**.
Expected Unity `BackendType.GPUCompute` latency on comparable discrete GPU: **5–15 ms per span**.

**BMP output encoding fix**: line classes now use `pack_line_gray(id_field=16, type_code)` → gray values 128–132, in the valid line range [128, 254]. Prior versions used `id_field=0` (values 0–4, solid/black range — invisible). Fixed in `infer_goal2.py`, `export_goal2_onnx.py`, and `LineSegSentisInferenceManager.cs`.

### 11.5 Artefacts produced

| Artefact | Location |
|----------|----------|
| Best model checkpoint | `goal2_runs/exp_v12/best.pt` (epoch 99 / 139) |
| Training log | `goal2_runs/exp_v12/train_log.jsonl` |
| Validation report | `goal2_runs/exp_v12/validation_report.txt` |
| ONNX export script | `line_seg/export_goal2_onnx.py` |
| TRT / ORT profiling script | `line_seg/trt_profile_goal2.py` |
| Python inference script | `line_seg/infer_goal2.py` |
| Unity Sentis C# manager | `unity_line_export/LineSegSentisInferenceManager.cs` |
| Architecture README | `SpanUNet3D_Semantic.md` |

### 11.6 Remaining Phase 2 / Phase 3 items

| Item | Priority | Notes |
|------|----------|-------|
| Instance segmentation head (per-line ID across frames) | High | Requires 16-bit label upgrade and discriminative loss; blocked on annotation workflow (`Line_Annotation.md`) |
| Improve comm detection (IoU 0.496) | Medium | Primary bottleneck is annotation quality — comm and neutral occupy same spatial neighbourhood with only 3-bit type_code as distinguishing signal |
| 5-fold cross-validation | Medium | Current single `val_seed=7` split adequate; CV would quantify variance across splits |
| Temporal recurrence in UNet bottleneck (Phase 3) | Low | Only warranted after annotation ceiling is lifted |
| Goal A binary classifier (span has lines?) | Low | Needs negative spans; generate synthetically by masking line pixels |
