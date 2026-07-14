# Line Type Classification — Algorithm Reference

Stage-2 of the power-line pipeline. Receives the binary cross-section BMP produced by Stage-1 (line detection) and assigns each connected wire component one of three semantic classes: **comm**, **primary**, or **neutral**.

---

## 1. Input Representation

Each span is represented as a tiny grayscale BMP (~57 × 31 px). Line pixels are 0, background is 255.

A **7-channel spatial tensor** `[1, 7, H, W]` is built from this mask before being fed to the model:

| Ch | v17–v22 (DUKE only) | v31–v36 (joint DUKE + NYSEG) |
|----|---------------------|-------------------------------|
| 0 | Binary line mask (pixel/255) | same |
| 1 | Global row y ∈ [0,1]  top=0 | same |
| 2 | Component relative y ∈ [0,1]  topmost=0, bottommost=1 | same |
| 3 | Bottom-rank / max(1, n−1) | **n\_lines / 5** clipped to [0,1] |
| 4 | Vertical gap to next wire above / H | same |
| 5 | Vertical gap to next wire below / H | same |
| 6 | n\_lines / 5 clipped | **Interior-ness** = 2·min(rank, n−1−rank) / max(1, n−1) |

The v31+ redesign was motivated by NYSEG 2-wire spans. Replacing bottom-rank with n\_lines/5 (ch3) makes 2-wire and 5-wire edge conductors distinguishable. The interior-ness feature (ch6) uniquely tags any wire sandwiched between two others — neutral is always an interior conductor in ≥3-wire spans, so ch6 > 0 is an unambiguous neutral signal.

---

## 2. Model Architecture — `LineTypeUNet2D`

A **lightweight 2-encoder UNet** (`model_classify.py`). Three pooling stages over a 57×31 grid would collapse the narrowest spatial dimension, so only two max-pool stages are used.

```
Input [B, 7, H, W]
  → enc1: ConvBlock2D(7, base)        GroupNorm + ReLU × 2
  → pool1: MaxPool2d(2)
  → enc2: ConvBlock2D(base, 2·base)
  → pool2: MaxPool2d(2)
  → mid:  ConvBlock2D(2·base, 4·base) + Dropout2D
  → dec2: upsample → cat(enc2) → ConvBlock2D(6·base, 2·base)
  → dec1: upsample → cat(enc1) → ConvBlock2D(3·base, base)
  → 1×1 Conv → class_logits [B, 3, H, W]
```

Default configuration: `base=48`, `dropout=0.17`. All norms are GroupNorm (avoids batch-size dependence at small spans). The model is fine-tuned from a DUKE-only baseline (`hq_only_v22`) for every joint run.

An alternative **ResNet-18 UNet** (`ResNet18UNet2D`) is available for richer feature extraction but was not used in production due to over-parameterisation on tiny cross-sections.

---

## 3. Loss Function

The total per-batch loss combines four terms:

### 3.1 Focal Cross-Entropy with Class Weights and Label Smoothing
```
L_focal = −∑ w_c · (1 − p_c)^γ · log(p̃_c)
```
- **Class weights** (`w_c`): computed by median-frequency balancing across all training spans so rare classes (neutral) are not starved.
- **Focal modulation** (`γ ≈ 0.35–0.45`): down-weights easy-to-classify background pixels so the gradient concentrates on hard confusions.
- **Label smoothing** (`ε ≈ 0.09`): softens one-hot targets to `(1−ε)·1 + ε/C`, preventing overconfident logits.

### 3.2 Symmetric Confusion Penalties
Explicit penalty for producing the wrong class at pixels where the ground-truth class is known:
```
L_conf = α_cn · L(true=comm, pred=neutral)
       + α_nc · L(true=neutral, pred=comm)
       + α_np · L(true=neutral, pred=primary)   + ...
```
Separate weights allow targeted correction — e.g., if neutral→primary confusion dominates, `neutral_to_primary_confusion_weight` is raised without disturbing comm performance.

### 3.3 Component-Pooled Loss (training parity)
Rather than minimising pixel-wise softmax, the loss is computed on the **mean softmax pooled over each 8-connected wire component**, then an argmax is taken once per component. This directly matches the inference post-processing mode and avoids penalising within-component gradient variance.

### 3.4 Optional Feeder-Graph KD (v36)
A KL-divergence consistency term penalises disagreement between the model's predictions and **feeder-graph pseudo-labels** — consensus probabilities aggregated from topologically connected neighboring spans (same wire count, ≤3 hops on the pole graph). This acts as structured regularisation that encodes continuity priors across a feeder run.

### Training Recipe
| Ingredient | Value |
|---|---|
| Optimiser | AdamW |
| LR schedule | Cosine decay with linear warm-up |
| Grad clip | 1.0 |
| SWA | Stochastic Weight Averaging from epoch 7 |
| Distributed | `torchrun` DDP (2 × A100) |
| Checkpoint metric | `val_line_type_min_pr` (minimum per-class P and R) |
| Fairness gate | Checkpoint saved only if `val_loss − train_loss ≤ 0.06` |

---

## 4. Inference Post-Processing — Component-Pooled Prediction

1. Run `[1, 7, H, W]` through the model → raw logits `[1, 3, H, W]`.
2. Apply softmax pixel-wise.
3. Label all 8-connected line-pixel components.
4. For each component: **average** softmax probabilities over all its pixels.
5. Take **argmax** of the averaged vector → single class label for the whole component.
6. Write type grays to output BMP: `comm=128`, `primary=136`, `neutral=144`.

This guarantees that every pixel of a wire gets the same label and prevents fragmented multi-class predictions within a single physical wire.

---

## 5. Data Pools

| Pool | Source | Wire configs |
|---|---|---|
| `1_span` | DUKE Florida 150 kV | Single-circuit (comm + primaries, no neutral) |
| `2_span` | DUKE Florida 150 kV | Double-circuit (comm + primaries) |
| `NYSEG_AUB` | NYSEG Upstate NY | Mixed 2-wire (comm + neutral) and 3-wire (comm + primary + neutral) |

Joint training (v31–v36) merges all three pools with equal weight, using a random 70/20/10 train/val/test split stratified by the presence of each line type.

---

## 6. Audit — Identifying False Positives and False Negatives

Running `line_seg/audit_classify_labels.py` (or `line_seg/infer_classify.py --output_dir`) writes a `line_type_compare.bmp` into every labeled span's output folder.

**Color coding:**
- Red = `comm`
- Green = `primary`
- Blue = `neutral`

**Rendering rule:**

| Prediction outcome | What is drawn |
|---|---|
| Correct | Filled colored circle only |
| Wrong | Ground-truth colored **ring** + predicted-class **asterisk** |

**Example readings:**
- Solid red circle → comm predicted and correct.
- Green ring + blue asterisk → true `primary` predicted as `neutral` (false neutral / false negative for primary).
- Blue ring + red asterisk → true `neutral` predicted as `comm` (false comm / false negative for neutral).

The scalar summary (`evaluation_report.txt`) reports per-class precision and recall alongside a row-normalised confusion matrix, making it straightforward to identify the dominant confusion direction.

---

## 7. Achieved Performance

| Model | Training data | comm minPR | primary minPR | neutral minPR | Overall minPR |
|---|---|---|---|---|---|
| v22 (baseline) | DUKE only | **0.943** | **0.942** | — | **0.942** |
| v36 (joint) | DUKE + NYSEG | 0.774 | 0.917 | 0.637 | 0.637 |

v22 exceeds the 0.93 target on DUKE spans because DUKE geometry is unambiguous (neutral is always the middle wire in 3-wire circuits). The NYSEG pool introduces 2-wire spans where the two conductors are spatially indistinguishable by the current 7-channel feature set — this is the binding bottleneck for v36.

---

## 8. Improvement Steps for Lifting Per-Class Precision and Recall

The steps below are ranked roughly from most to least likely to provide meaningful gains, given the current architecture and data.

### 8.1 Domain-Specific Annotation Corrections *(highest impact)*
The NYSEG 2-wire bottleneck is partly a label ambiguity problem. In roughly 41% of NYSEG spans the two conductors have identical feature vectors; which one is labeled neutral varies across spans with no consistent geometric rule. Systematic re-labeling of these spans (using pole-hardware annotation or field records) would directly remove the information-theoretic ceiling.

### 8.2 New Discriminating Input Features
Features the current 7-channel set cannot provide:
- **Wire attachment type** — neutral in NYSEG is typically on a dead-end insulator or messenger clamp; comm uses a different hardware signature visible in high-res imagery.
- **Span-level wire count trajectory** — neutral disappears at service drops; tracking presence/absence along the feeder axis distinguishes it from comm.
- **Sag ratio** — neutral commonly sags more than primary due to lower tension; extractable from the cross-section height profile.
Adding any of these as additional input channels would not require model architecture changes (only `in_channels` in the ONNX export).

### 8.3 Longer Feeder-Graph Voting Window
The v36 inference uses 3-hop neighborhood voting. Increasing to 5–7 hops on feeders with consistent wire arrangements would improve consensus strength. Requires re-running `infer_classify.py --feeder_hops 5` — no retraining.

### 8.4 Domain-Specific Models
Train a DUKE-only model (v22 already achieves this) and a separate NYSEG-only model. The NYSEG-only model can specialise its confusion penalties and feature priors without being diluted by the different geometry of DUKE spans. Combine at inference using the detected utility (DUKE / NYSEG) as a routing flag.

### 8.5 Feeder KD with a Larger Teacher Window
The v36 feeder KD used hops=3 and triggered from epoch 3 only. A two-stage schedule — CE-only until convergence, then KD from a frozen snapshot — would let the model first develop strong pixel-level features before the consistency signal is introduced, reducing the risk of KD amplifying early errors.

### 8.6 Targeted Post-Processing Rules
The `--two_wire_nyseg_rule` flag implements a geometric override for the close-spaced upper-image 2-wire configuration, recovering a portion of comm→neutral confusions without retraining. Additional hand-coded rules for specific geometric archetypes (e.g. "in a 3-wire NYSEG span, the lowest conductor is primary") can be layered on top of model predictions as a cheap precision booster.

### 8.7 Semi-Supervised Pseudo-Labeling on Unlabeled Spans
If there are unlabeled NYSEG spans available, pseudo-labels generated by the current best model (filtered by high-confidence predictions, e.g. max softmax ≥ 0.90) can expand the effective training set for the less-ambiguous subset, improving generalisation without requiring full manual annotation.
