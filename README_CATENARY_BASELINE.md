# Catenary / Parabola RANSAC Baseline + Visualization

This folder adds a **non-ML baseline** that fits smooth sag curves to **(frame index, row y)** samples extracted from span BMP sequences. It establishes a performance floor and produces plots for inspection.

## Line voxel encoding (labeled BMPs)

Per `Lines_Detect_Goals.txt`, each **labeled conductor** pixel uses a grayscale value in **[128, 254]** (never 255; that is empty air):

| Part | Bits | Meaning |
|------|------|---------|
| **Line type** | lower 3 bits: `gray & 0x07` | `0` = comm, `1` = primary, `2` = neutral, `3` = secondary, `4` = transmission |
| **Instance id field** | upper 5 bits: `(gray >> 3) & 0x1F` | Compact id **0–31** packed in the byte |

**Packed value:** `gray = (id_field << 3) | type_code`

Documentation may refer to a “semantic” conductor id starting at 32; that is a naming layer **outside** this 8-bit packing. The tools decode and report **`id_field`** (the 5-bit field) and **`type_code`**.

**Default baseline behavior:** sample **`line_gray`** pixels only — i.e. `128 <= gray < 255`, and by default **only type codes 0–4** (use `--no_strict_line_types` to also include reserved low-3-bit values 5–7). This matches the physical **conductors**, not vegetation (black).

---

## What the pipeline does

1. **Loads** every `frame_*.bmp` in a span directory (8-bit grayscale, 31×57).
2. **Builds (t, y) points** from the chosen mask (default: **labeled line voxels**).
3. **Fits** a **parabola** `y = a·t² + b·t + c` with **RANSAC** (robust to outliers).
4. **Optionally refits** inliers with a **catenary-shaped** curve  
   `y = y₀ + a·(cosh((t − t₀)/s) − 1)` using `scipy.optimize.least_squares` (requires SciPy).
5. **Repeats** up to `n_curves` times (sequential RANSAC) when multiple traces exist.
6. **Writes** JSON (includes **`line_labels_summary`**: unique grays in the span with decoded type/id), a **(t, y) plot**, optional **montage**, optional **GIF**.

### Other pixel modes (optional)

| `--pixel_source` | Mask |
|------------------|------|
| `line_gray` (default) | Labeled lines: `128 <= gray < 255` (+ optional type 0–4 filter) |
| `black` | Non-line solids: `gray == 0` |
| `not_white` | Everything except air: `gray < 255` |

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

### 2. Run the baseline on one span (defaults: line pixels + centroid + strict types)

```bash
python3 -m tools.run_baseline --span 206_213 --out_dir ./baseline_out/206_213
```

Full path:

```bash
python3 -m tools.run_baseline --span /path/to/DUKE_FLORIDA_150/206_213 --out_dir ./baseline_out
```

### 3. Inspect outputs

| File | Contents |
|------|----------|
| `*_report.json` | Curves, RMSE, **`line_labels_summary`** (per unique gray: `type_name`, `type_code`, `id_field`), encoding note |
| `*_ty_plot.png` | Scatter (frame, y) + parabola / catenary overlays |
| `*_montage.png` | Frames with masked pixels highlighted |
| `*_flipbook.gif` | With `--gif` only |

### 4. Per-line sag (one curve per conductor)

Uses **packed line gray** as identity: each distinct `128…254` value is one labeled conductor (same `id_field` can appear twice with different `type_code`, e.g. 130 vs 131 — **not** merged).

```bash
python3 -m tools.run_baseline --span 206_213 --out_dir ./out_per_line --per_line_id
```

Writes `*_report_per_line.json` and `*_ty_plot_per_line.png`. Implies `line_gray`; do not combine with `--pixel_source black`.

### 5. Other useful flags

```bash
# Dense samples: every selected pixel (heavy for line_gray)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --aggregate none

# Solids only (vegetation / hardware — not conductor labels)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --pixel_source black

# Include reserved line-type bits 5–7 in line_gray mask
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --no_strict_line_types

# Tighter / looser RANSAC (pixels)
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --residual_thresh 2.5

python3 -m tools.run_baseline --span 206_213 --out_dir ./out --no_catenary_refine
python3 -m tools.run_baseline --span 206_213 --out_dir ./out --gif
```

### 6. Batch many spans

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
from tools.line_encoding import decode_line_gray, pack_line_gray
from tools.visualize import plot_span_ty

# Example: gray 130 → type + id_field
print(decode_line_gray(130))

report = run_span_baseline(Path("206_213"), pixel_source="line_gray", aggregate="centroid")
plot_span_ty(report, "ty.png")

# Per conductor (packed gray):
from tools.catenary_ransac import run_span_baseline_per_line
from tools.visualize import plot_span_ty_per_line
pl = run_span_baseline_per_line(Path("206_213"))
plot_span_ty_per_line(pl, "ty_per_line.png")
```

---

## Interpretation

- **Strong parabola/catenary fit on `line_gray`** (centroid per frame): bundled conductors share similar sag; one dominant curve often captures most centroids unless lines are vertically separated (then sequential RANSAC helps).
- **`black` / `not_white`**: geometric stress tests only — **not** aligned with conductor label encoding.
- **Goal A (span has lines?)**: combine presence of **`line_gray`** voxels with low RMSE of a sag fit on **`line_gray`** centroids; tune thresholds on your validation spans.

---

## File layout

```
DUKE_FLORIDA_150/
  README_CATENARY_BASELINE.md
  requirements_baseline.txt
  tools/
    __init__.py
    bmp_io.py
    line_encoding.py      # decode / pack / masks per spec
    catenary_ransac.py     # RANSAC + optional catenary refit
    visualize.py
    run_baseline.py
```
