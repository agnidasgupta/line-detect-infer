# Paperspace: What to Upload, How to Run, What Each Output Does

This guide assumes a **Paperspace Gradient notebook** or **Linux machine** with GPU, and that your project root on the remote will be something like `/notebooks/duke_florida` or `/notebooks/DUKE_FLORIDA_150`. Adjust paths to match where you unpack the data.

**Dataset size:** The planning docs used a **sample** subset (e.g. ~139 spans) to size the problem. Your full `DUKE_FLORIDA_150` upload may include **more** span folders. **No code changes are required** — `line_seg/volume.py:list_span_dirs()` and training **enumerate every** immediate subfolder under `--data_root` that contains `frame_*.bmp`. More spans generally **help** training (same label format).

---

## 1. Directory you must upload (dataset + code)

Upload **one folder** that contains **both** the span image data **and** the Python package layout below. The training script expects `--data_root` to point at the **parent** of all span subfolders.

### 1.1 Required layout on Paperspace

```
DUKE_FLORIDA_150/                    ← upload this entire directory (name can differ)
├── Lines_Detect_Goals.txt           # optional but useful reference
├── POWER_LINE_DETECTION_PLAN.md     # optional reference
├── README_CATENARY_BASELINE.md
├── README_GOAL2.md
├── PAPERSPACE_UPLOAD_AND_RUN.md     # this file
├── requirements_baseline.txt
├── requirements_goal2.txt
├── line_seg/                        # Goal 2 training + eval code (upload entire folder)
│   ├── __init__.py
│   ├── volume.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── eval_metrics.py
│   ├── train_goal2.py
│   ├── infer_goal2.py               # inference: weights + input_dir → predicted BMPs + optional eval
│   └── export_goal2_metrics.py
├── tools/                           # RANSAC baseline + BMP I/O (required for baseline; volume.py imports tools.bmp_io)
│   ├── __init__.py
│   ├── bmp_io.py
│   ├── line_encoding.py
│   ├── catenary_ransac.py
│   ├── visualize.py
│   └── run_baseline.py
├── 206_213/                         # example span — each span is a subfolder like this
│   ├── frame_0000_h141.bmp
│   ├── frame_0001_h141.bmp
│   └── ...
├── 2884_2888/                       # another span
│   └── frame_*.bmp
├── ...                              # all other span folders (pole1_pole2 names)
└── (no need to upload empty folders)
```

**Critical rule:** `line_seg/train_goal2.py` uses `--data_root .` meaning “the directory that **directly contains** the span folders” (e.g. `206_213`, `2884_2888`). That directory must also contain `line_seg/` and `tools/` so imports work when you `cd` there.

**What you are uploading in practice**

| Path | Role |
|------|------|
| **`DUKE_FLORIDA_150/` (root)** | Working directory for all commands; must hold `line_seg/`, `tools/`, requirements, and **every span subfolder** with `frame_*.bmp`. |
| **`DUKE_FLORIDA_150/line_seg/`** | PyTorch 3D UNet training: dataset, model, losses, `train_goal2.py`. |
| **`DUKE_FLORIDA_150/tools/`** | BMP reader, line gray encoding, optional RANSAC baseline CLI. `line_seg/volume.py` imports `tools.bmp_io`. |
| **`DUKE_FLORIDA_150/<span_name>/`** | One folder per pole–pole span; only files needed are `frame_*.bmp` (thousands total across spans). |
| **`requirements_baseline.txt`** | `numpy`, `matplotlib`, `scipy`, `pillow` — used for baseline + shared deps. |
| **`requirements_goal2.txt`** | `torch`, `scikit-learn`, `matplotlib`, `scipy` — add on top for training and evaluation plots. |

You do **not** need to upload `goal2_runs/` or `baseline_runs/` before training; those are **created** when you run the scripts.

### 1.2 Upload checklist — **Goal 2 training** (strict minimum)

Upload **one** tree whose root you will `cd` into and pass as `--data_root .`. That root **must** contain all of the following:

| Path / pattern | Required? | Why |
|----------------|-----------|-----|
| **`line_seg/`** (entire folder) | **Yes** | Training, dataset, model, losses, metrics, `train_goal2.py`, `infer_goal2.py`, `export_goal2_metrics.py`. |
| **`line_seg/__init__.py`** | **Yes** | Part of `line_seg/` (Python package). |
| **`tools/`** (entire folder) | **Yes** | `line_seg/volume.py` imports **`tools.bmp_io`** to read BMPs. |
| **`tools/__init__.py`**, **`tools/bmp_io.py`**, and any files `bmp_io` / `volume.py` pull in | **Yes** | Keep the whole `tools/` package as in the repo. |
| **`requirements_baseline.txt`** | **Yes** (install) | Shared deps (`numpy`, `scipy`, `pillow`, …). |
| **`requirements_goal2.txt`** | **Yes** (install) | `torch`, `scikit-learn`, `matplotlib`, `scipy` for train + eval plots. |
| **Every span subdirectory** named like `206_213/`, `2884_2888/`, … each containing **`frame_*.bmp`** | **Yes** | Training and validation sample spans from these folders only. |
| Markdown docs (`README_GOAL2.md`, this file, `POWER_LINE_*.md`, …) | **No** | Helpful for humans; not imported at runtime. |
| **`goal2_runs/`**, **`output/`**, **`baseline_runs/`** | **No** before first run | Created on the machine when you train or run baseline. |

If anything in the **Yes** rows is missing, training will fail (`ModuleNotFoundError`, empty dataset, or missing BMPs).

### 1.3 What you need on Paperspace for **evaluation only** (`export_goal2_metrics`)

Evaluation **reloads validation spans from disk** using `run_meta.json`, so the machine must still have:

| Item | Required? |
|------|-----------|
| Same **`line_seg/`** + **`tools/`** + **`requirements_*`** as training | **Yes** |
| Same **`DUKE_FLORIDA_150` root** with all span folders listed in **`run_meta.json`** (`train_spans` / `val_spans` paths resolve under `--data_root`) | **Yes** (at least all **`val_spans`** folders and BMPs) |
| **`goal2_runs/<exp>/best.pt`** | **Yes** |
| **`goal2_runs/<exp>/run_meta.json`** | **Yes** |
| **`goal2_runs/<exp>/train_log.jsonl`** | **No** (optional; improves history plots and the log section inside `validation_report.txt`) |

---

## 2. Upload methods (Paperspace)

Choose one:

1. **Zip on your Mac, upload via notebook UI, unzip on the machine**  
   - Zip `DUKE_FLORIDA_150/` (including all span folders).  
   - Upload to `/notebooks/` (or your persistent volume).  
   - `unzip duke_florida.zip -d /notebooks/` then `cd /notebooks/DUKE_FLORIDA_150` (or whatever the top folder is named).

2. **Gradient dataset / persistent storage**  
   - If you already sync data to a mounted volume, copy or symlink so the layout in §1.1 holds under one path.

3. **`scp` / `rsync` from your machine to the notebook VM**  
   - Example:  
     `rsync -avz --progress /local/DUKE_FLORIDA_150/ user@host:/notebooks/DUKE_FLORIDA_150/`

After upload, verify:

```bash
cd /notebooks/DUKE_FLORIDA_150   # use your actual path
ls line_seg tools requirements_*.txt
ls -d */ | head   # should list span folders, not only line_seg and tools
```

---

## 3. Step-by-step: environment (once per machine)

Run from **`DUKE_FLORIDA_150`** root (the directory that contains `line_seg` and span folders).

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_baseline.txt
python3 -m pip install -r requirements_goal2.txt
```

**Check GPU:**

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 4. Step-by-step: optional RANSAC baseline (no PyTorch training)

**Purpose:** geometric sag fits and plots; does **not** train the neural network.

**When:** optional sanity check before or after Goal 2 training.

```bash
cd /notebooks/DUKE_FLORIDA_150

# Merged bundle (all line pixels in one curve family)
python3 -m tools.run_baseline --span 206_213 --out_dir ./baseline_runs/206_213_merged

# One curve per packed line gray (per conductor label)
python3 -m tools.run_baseline --span 206_213 --out_dir ./baseline_runs/206_213_per_line --per_line_id
```

### 4.1 Files produced under `baseline_runs/...` and their roles

| File pattern | Produced when | Role |
|--------------|-----------------|------|
| **`*_report.json`** | Default (merged) run | JSON: parabola coefficients, inlier counts, RMSE, catenary params, `line_labels_summary` (decoded grays). |
| **`*_ty_plot.png`** | Default run | Scatter of (frame, y) + fitted curves — quick visual of sag. |
| **`*_montage.png`** | Unless `--no_montage` | Grid of frames with masked pixels highlighted. |
| **`*_flipbook.gif`** | If `--gif` | Animated span sequence (needs Pillow). |
| **`*_report_per_line.json`** | `--per_line_id` | One RANSAC result **per** packed `line_gray` conductor. |
| **`*_ty_plot_per_line.png`** | `--per_line_id` | Multi-color plot: one trace + fit per conductor. |

You can zip `baseline_runs/` and download it for offline review.

---

## 5. Goal 2: training and evaluation commands

**Prerequisites:** `cd` to the **`DUKE_FLORIDA_150` root** (see §1.2). Install once (§3).

### 5.1 Training — **single GPU**

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m line_seg.train_goal2 \
  --data_root . \
  --out_dir ./goal2_runs/exp1 \
  --epochs 30 \
  --lr 1e-3 \
  --device cuda
```

- **`--data_root .`** = directory that contains **`line_seg/`**, **`tools/`**, and every **`*/frame_*.bmp`** span folder.  
- **`--out_dir ./goal2_runs/exp1`** = new run directory (change `exp1` per experiment). Writes **`best.pt`**, **`last.pt`**, logs, plots, **`validation_report.txt`**, etc.

### 5.2 Training — **multi-GPU** (NCCL + DDP; logs and checkpoints on rank 0 only)

**Rule:** `--nproc_per_node` must equal the number of **visible** CUDA devices (`echo $CUDA_VISIBLE_DEVICES` may reduce that count). Example: **two A6000s** → `--nproc_per_node=2` (not 4).

```bash
cd /notebooks/DUKE_FLORIDA_150

torchrun --standalone --nproc_per_node=2 -m line_seg.train_goal2 \
  --data_root . \
  --out_dir ./goal2_runs/exp_mgpu \
  --epochs 30 \
  --lr 1e-3
```

- **`--device`** is ignored under `torchrun`; each process uses **`cuda:LOCAL_RANK`**.  
- You need **at least as many training spans as `--nproc_per_node`** or training exits with an error.

### 5.2b Detect **how many GPUs** the job will see (before training)

PyTorch only uses GPUs that are **visible** to the process (`CUDA_VISIBLE_DEVICES` can hide some):

```bash
nvidia-smi -L
python3 << 'EOF'
import torch
print("cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
EOF
```

Use **`device_count`** as the number passed to **`torchrun --nproc_per_node=...`**, or rely on **`--multi_gpu`** below (same count).

### 5.2c Training — **use every visible GPU automatically** (`--multi_gpu`)

`train_goal2` reads **`torch.cuda.device_count()`** and **re-execs** via `torch.distributed.run` with **`--nproc_per_node=<that count>`**. You do **not** need to hard-code 2 vs 4 vs 8.

**Minimal example (any machine: 1 GPU → single process; 2+ GPUs → DDP):**

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m line_seg.train_goal2 \
  --multi_gpu \
  --data_root . \
  --out_dir ./goal2_runs/exp_mgpu \
  --epochs 30 \
  --lr 1e-3
```

**Same idea with optional per-class CE + focal** (still auto-detects GPU count):

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m line_seg.train_goal2 \
  --multi_gpu \
  --data_root . \
  --out_dir ./goal2_runs/exp_weighted \
  --epochs 30 \
  --lr 1e-3 \
  --ce_class_weights 1,1,1.5,1.5,1.5,10,3 \
  --focal_gamma 2.0
```

- If **`device_count == 1`**, `--multi_gpu` does **not** launch DDP (you stay on a single process; a short message is printed).  
- To **force a single GPU** on a multi-GPU box: `CUDA_VISIBLE_DEVICES=0` then run **without** `--multi_gpu`, or keep only one index visible.  
- **`--multi_gpu`** cannot be combined with a manual **`torchrun`** on the same command (already under distributed); launch one or the other.

### 5.3 Evaluation — **regenerate metrics / `validation_report.txt`** (after training)

Use this to refresh confusion matrix, plots, and the combined text report from an existing **`best.pt`** (same or new machine, as long as §1.3 paths exist):

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m line_seg.export_goal2_metrics \
  --data_root . \
  --out_dir ./goal2_runs/exp1 \
  --device cuda
```

- **`--data_root .`** = same dataset root used for training (span folders must exist; **`val_spans`** from `run_meta.json` are resolved relative to this root when paths are not absolute).  
- **`--out_dir`** = the run folder that contains **`best.pt`** and **`run_meta.json`** (e.g. `./goal2_runs/exp1`).  
- Optional: **`--num_workers 4`**, **`--no_amp`** if you hit AMP issues on a specific GPU stack.

**Note:** A full training run already writes **`validation_report.txt`** at the end; **`export_goal2_metrics`** is for re-runs, new machines, or changed code without retraining.

### 5.4 Inference — **run `best.pt` on a folder of spans** (`infer_goal2`)

**Input:** `--input_dir` may be the **repo root** (spans + `line_seg` + `tools`). Only immediate child directories that contain **at least one `.bmp` anywhere** are considered; others are skipped. Each candidate must have **`frame_*.bmp`** in that folder or in **one immediate subdirectory** (same as training, optional one-level nesting). BMPs may be **unlabeled** (only `0` and `255`) or **labeled** (line grays `128..254`). Discovery details are in **`inference_manifest.json`** (`span_discovery`, `span_frame_dirs`).

**Output:** `--output_dir` receives one subfolder per span with **`predicted_bmp/`** (same filenames as input), **`predicted_semantic_classes.npy`**, **`meta.json`**.  
**`inference_timings.txt`** (root) lists **wall-clock seconds per span**.  
If **at least one** input span contains line labels, **`evaluation_report.txt`** is written in the **same format** as training **`validation_report.txt`** (ASCII CM + sklearn + JSON + object metrics), aggregated over **only** spans that had labels; plus **`evaluation_confusion_matrix.npy`**. If **no** span has labels, evaluation files are skipped.

```bash
cd /notebooks/DUKE_FLORIDA_150

python3 -m line_seg.infer_goal2 \
  --weights ./goal2_runs/exp1/best.pt \
  --input_dir ./path/to/spans_or_holdout \
  --output_dir ./inference_runs/exp1_eval \
  --device cuda
```

Optional: **`--max_frames N`**, **`--no_amp`**.

---

## 6. Files produced under `goal2_runs/<experiment>/` and their roles

| File | Role |
|------|------|
| **`best.pt`** | PyTorch checkpoint when the tracked metric improved (default **highest `val_mean_iou`**; see `run_meta.json` → `checkpoint_metric`). Contains `model_state`, `num_classes` (7), `base_channels`. **Use this for inference** when deploying the semantic model. |
| **`last.pt`** | Weights after the **final** epoch (may overfit). Fallback if you prefer last-epoch weights. |
| **`train_log.jsonl`** | One JSON line per epoch: losses, `val_mean_iou`, macro precision/recall/F1, per-class IoU list, **line object** micro P/R/F1 (merged line classes, 3D CC + IoU match), `seconds`. |
| **`metrics_history.json`** | Same fields as an array (convenient for plotting). |
| **`history_loss.png`** / **`history_scores.png`** | Train vs val loss and validation score curves. |
| **`confusion_matrix.npy`** / **`confusion_matrix.png`** | Pixel confusion matrix on the **full val set** with **`best.pt`** weights (after training finishes). |
| **`validation_report.txt`** | **Single text file**: ASCII confusion matrix, sklearn classification report, JSON scalar summary (including object-line metrics), and training-log diagnosis. |
| **`run_meta.json`** | Lists **which span folder names** went to train vs validation (`train_spans`, `val_spans`) and class name list. **Reproducibility and debugging splits.** |

Nothing else is required for a minimal train; there is no separate ONNX export in this stack yet.

---

## 7. What to download back from Paperspace

| Download | Why |
|----------|-----|
| **`goal2_runs/<exp>/best.pt`** | Trained model for inference or further fine-tuning. |
| **`goal2_runs/<exp>/train_log.jsonl`**, **`metrics_history.json`**, plots, **`validation_report.txt`**, confusion matrix files, **`run_meta.json`** | Metrics, curves, and split documentation. |
| **`baseline_runs/`** (if you ran baseline) | Reports and plots for documentation. |

Optional: entire `goal2_runs/` zip for backup.

---

## 8. Quick troubleshooting

| Issue | What to check |
|-------|----------------|
| `ModuleNotFoundError: tools` or `line_seg` | Your shell **`cd`** must be the **`DUKE_FLORIDA_150` root** (parent of `tools` and `line_seg`), and `data_root` for training must be that same root. |
| `No frame_*.bmp` / empty dataset | Span folders missing from upload or wrong `--data_root`. |
| CUDA OOM | Use a larger GPU or `--max_frames` (caps frames per span) in code if you add that flag; default uses full spans. |
| Very slow on CPU | Training expects **CUDA**; `--device cuda` is default when available. |
| `RuntimeError: invalid device ordinal` under `torchrun` | **`--nproc_per_node` is larger than visible GPUs** (e.g. 4 processes on a 2-GPU box). Use `--nproc_per_node=2` or **`python3 -m line_seg.train_goal2 --multi_gpu ...`**. |

---

## 9. One-line summary

**Upload (training):** **`line_seg/`**, **`tools/`**, **`requirements_baseline.txt`**, **`requirements_goal2.txt`**, and **every span folder** with **`frame_*.bmp`**, all under one **`DUKE_FLORIDA_150`** root.  
**Train:** `cd` that root → `pip install -r` both requirements → **`python3 -m line_seg.train_goal2 --data_root . --out_dir ./goal2_runs/exp1`**.  
**Evaluate (optional):** **`python3 -m line_seg.export_goal2_metrics --data_root . --out_dir ./goal2_runs/exp1`**.  
**Download:** **`best.pt`**, logs, plots, **`validation_report.txt`**, confusion matrix files, **`run_meta.json`**.
