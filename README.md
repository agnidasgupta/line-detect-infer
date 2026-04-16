# Dataset Summary
The dataset contains 139 pole-to-pole spans with 17,541 total BMP frames — tiny 31x57 grayscale cross-section images, like MRI slices through the corridor between two poles. Each frame has white (air), black (solids: vegetation/structures), and gray (labeled lines encoded with type + ID). Lines are extremely sparse (0.1–0.9% of pixels) and thin (1–3px wide). Across a span, all lines exhibit a smooth catenary sag pattern — their y-position dips in the middle frames and rises at both ends, all moving in unison.

# Two ML Goals
Goal A (Primary): Binary classification — does this span have power lines? Used to filter false-positive poles (e.g., tree trunks).
Goal B (Secondary): Instance segmentation — segment individual lines with consistent IDs across all frames and classify type (comm/primary/neutral/secondary/transmission).

# Recommended Strategy: Three Phases
Phase 1 (1–2 weeks): Build a catenary-fitting baseline using RANSAC on the (frame, y) projection of black pixels. This is a non-ML approach that exploits the physics of power line sag and establishes a performance floor. Also build a visualization toolkit.

Phase 2 (2–4 weeks): Train a custom 3D UNet with instance embedding heads and a semantic classification head. The 3D approach is ideal because the volumes are small (~350K voxels per span) and lines form continuous "tubes" — exactly what 3D medical segmentation excels at. Uses asymmetric kernels (larger along temporal axis) and discriminative loss for instance separation.

Phase 3 (1–2 weeks): Add temporal refinement (BiLSTM/Transformer at bottleneck), train the Goal A binary classifier using synthetic negatives, and ensemble with catenary confidence.

# Critical Risks
The biggest risks are dataset size (only 139 spans — heavy augmentation and cross-validation are essential), no negative examples for Goal A (synthetic negatives must be crafted), and incomplete labels (some lower lines are unlabeled by design). 

# Catenary / Parabola RANSAC Baseline + Visualization

This folder adds a **non-ML baseline** that fits smooth sag curves to **(frame index, row y)** samples extracted from span BMP sequences. It establishes a performance floor and produces plots for inspection.

## What it does

1. **Loads** every `frame_*.bmp` in a span directory (8-bit grayscale, 31×57).
2. **Builds (t, y) points**:
   - Default **`pixel_source=black`**: pixels with value `0` (non-line solids per `Lines_Detect_Goals.txt`).
   - Default **`aggregate=centroid`**: for each frame, one sample `(t, mean_y)` over selected pixels (stable for RANSAC).
3. **Fits** a **parabola** `y = a·t² + b·t + c` with **RANSAC** (robust to outliers).
4. **Optionally refits** inliers with a **catenary-shaped** curve  
   `y = y₀ + a·(cosh((t − t₀)/s) − 1)` using `scipy.optimize.least_squares` (requires SciPy).
5. **Repeats** up to `n_curves` times (sequential RANSAC) to sketch multiple traces when clusters exist.
6. **Writes** JSON metrics, a **(t, y) scatter + fit plot**, an optional **frame montage**, and optional **GIF flipbook**.

### Note on “black” vs “lines”

- In the **labeled** BMPs, **power lines are gray** (128–254), **solids are black** (0).
- The baseline you asked for uses **`black` pixels** → it measures whether **solid clutter** traces a smooth sag. That is **not the same** as fitting **conductors** (gray). For comparison with labels, run again with `--pixel_source line_gray`.

---

## Prerequisites

- Python 3.10+ recommended  
- From `DUKE_FLORIDA_150/`:

```bash
python3 -m pip install -r requirements_baseline.txt
```

---

## Step-by-step usage

### 1. Open a terminal in the dataset root

```bash
cd /path/to/DUKE_FLORIDA_150
```

### 2. Run the baseline on one span

Example span folder: `206_213` (adjust if your tree differs).

```bash
python3 -m tools.run_baseline --span 206_213 --out_dir ./baseline_out/206_213
```

Or pass a full path:

```bash
python3 -m tools.run_baseline --span /path/to/DUKE_FLORIDA_150/206_213 --out_dir ./baseline_out
```

### 3. Inspect outputs

| File | Contents |
|------|----------|
| `*_report.json` | Parabola coefficients, inlier counts, RMSE, catenary parameters (if SciPy refit ran) |
| `*_ty_plot.png` | Scatter of (frame, y) with parabola (solid) and catenary (dashed) overlays |
| `*_montage.png` | Grid of frames with selected pixels highlighted in red |
| `*_flipbook.gif` | (Only if `--gif`) Animated sequence |

### 4. Optional flags

```bash
# Use every black pixel (dense; slower, noisier)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --aggregate none

# Fit labeled line pixels (for label-aligned evaluation)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --pixel_source line_gray

# All non-white pixels (air removed)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --pixel_source not_white

# Tighter / looser RANSAC
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --residual_thresh 2.5

# Skip catenary refit (parabola only)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --no_catenary_refine

# GIF flipbook (needs Pillow)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --gif
```

### 5. Batch many spans (example shell loop)

```bash
out=./baseline_out_batch
mkdir -p "$out"
for d in */; do
  [[ -d "$d" ]] || continue
  python3 -m tools.run_baseline --span "${d%/}" --out_dir "$out/${d%/}" --no_montage || true
done
```

---

## Programmatic API

```python
from pathlib import Path
from tools.catenary_ransac import run_span_baseline
from tools.visualize import plot_span_ty

report = run_span_baseline(Path("206_213"), pixel_source="black", aggregate="centroid")
plot_span_ty(report, "ty.png")
```

---

## Interpretation

- **Strong parabola/catenary fit** on **`line_gray`**: labeled conductors follow a smooth sag — expected on good spans.
- **Fit on `black`**: vegetation and hardware may or may not follow sag; useful mainly as a **rough geometric prior** or for comparing to **`line_gray`**.
- **Goal A (span has lines?)**: a simple rule could combine: presence of **`line_gray`** pixels + low RMSE of a sag fit on **`line_gray`** centroids — this baseline does **not** implement a calibrated classifier; it only provides **fits and residuals** for you to threshold empirically.

---

## File layout

```
DUKE_FLORIDA_150/
  README_CATENARY_BASELINE.md   ← this file
  requirements_baseline.txt
  tools/
    __init__.py
    bmp_io.py              # BMP reader
    catenary_ransac.py     # RANSAC parabola + optional catenary refit
    visualize.py           # Plots, montage, GIF
    run_baseline.py        # CLI
```
