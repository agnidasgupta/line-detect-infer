"""
Visualization for catenary / parabola baseline: (frame, y) plots and frame montages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .bmp_io import read_bmp_gray
from .catenary_ransac import (
    PerLineSpanReport,
    PixelSource,
    SpanBaselineReport,
    collect_ty_points,
    collect_ty_points_per_unique_line_gray,
    mask_from_source,
    parabola_predict,
)


def catenary_eval(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """y = y0 + a * (cosh((t - t0) / s) - 1)"""
    y0, a, t0, s = params
    s = max(float(s), 1e-9)
    return y0 + a * (np.cosh((t - t0) / s) - 1.0)


def plot_span_ty(
    report: SpanBaselineReport,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 120,
) -> None:
    """Scatter (t, y) with fitted parabola / catenary overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Recompute t, y for plotting (same pipeline as baseline)
    span_dir = Path(report.span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    frames = [read_bmp_gray(p) for p in paths]
    t, y = collect_ty_points(
        frames,
        source=report.pixel_source,  # type: ignore[arg-type]
        aggregate=report.aggregate,  # type: ignore[arg-type]
        strict_line_types=report.strict_line_types,
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.scatter(t, y, s=8, c="0.35", alpha=0.6, label="samples")

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    t_dense = np.linspace(float(np.min(t)), float(np.max(t)), 256)

    for i, curve in enumerate(report.curves):
        c = colors[i % len(colors)]
        pred = parabola_predict(curve.coef_parabola, t_dense)
        ax.plot(t_dense, pred, "-", color=c, lw=2, label=f"parabola #{i+1} (inliers={int(np.sum(curve.inlier_mask))})")
        if curve.catenary_params is not None:
            yc = catenary_eval(curve.catenary_params, t_dense)
            ax.plot(t_dense, yc, "--", color=c, lw=1.5, alpha=0.85, label=f"catenary refit #{i+1}")

    ax.set_xlabel("frame index (along span)")
    ax.set_ylabel("y (row, top = 0)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title or f"{span_dir.name}  ({report.n_frames} frames, {report.n_points} points)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_span_ty_per_line(
    report: PerLineSpanReport,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 120,
) -> None:
    """Scatter + fits for ``--per_line_id``: one color per packed ``line_gray``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    span_dir = Path(report.span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    frames = [read_bmp_gray(p) for p in paths]
    by_gray = collect_ty_points_per_unique_line_gray(
        frames,
        aggregate=report.aggregate,  # type: ignore[arg-type]
        strict_line_types=report.strict_line_types,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=dpi)

    for i, pl in enumerate(report.per_line):
        g = pl.line_gray
        if g not in by_gray:
            continue
        t, y = by_gray[g]
        color = cm.tab10(i % 10)
        lbl = f"g={g} {pl.decode.get('type_name', '')}"
        ax.scatter(t, y, s=10, color=color, alpha=0.65, label=lbl)

        t_min, t_max = float(np.min(t)), float(np.max(t))
        t_dense = np.linspace(t_min, t_max, 128)
        pred = parabola_predict(pl.curve.coef_parabola, t_dense)
        ax.plot(t_dense, pred, "-", color=color, lw=2)
        if pl.curve.catenary_params is not None:
            yc = catenary_eval(pl.curve.catenary_params, t_dense)
            ax.plot(t_dense, yc, "--", color=color, lw=1.2, alpha=0.9)

    ax.set_xlabel("frame index (along span)")
    ax.set_ylabel("y (row, top = 0)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_title(title or f"{span_dir.name}  per-line sag (packed gray)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_frame_montage(
    span_dir: str | Path,
    out_path: str | Path,
    *,
    pixel_source: PixelSource = "line_gray",
    strict_line_types: bool = True,
    indices: Sequence[int] | None = None,
    max_frames: int = 12,
    dpi: int = 120,
) -> None:
    """Grid of frames with selected pixels overlaid in red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span_dir = Path(span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.bmp in {span_dir}")

    if indices is None:
        n = len(paths)
        if n <= max_frames:
            pick = list(range(n))
        else:
            pick = [int(round(i)) for i in np.linspace(0, n - 1, max_frames)]
    else:
        pick = list(indices)

    cols = min(4, len(pick))
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows), dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for k, idx in enumerate(pick):
        r, cc = divmod(k, cols)
        ax = axes[r, cc]
        im = read_bmp_gray(paths[idx])
        rgb = np.stack([im, im, im], axis=-1)
        mask = mask_from_source(im, pixel_source, strict_line_types=strict_line_types)
        rgb[mask, 0] = 255
        rgb[mask, 1] = 0
        rgb[mask, 2] = 0
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(paths[idx].name, fontsize=7)
        ax.axis("off")

    # Hide unused axes
    for j in range(len(pick), rows * cols):
        r, cc = divmod(j, cols)
        axes[r, cc].axis("off")

    fig.suptitle(f"{span_dir.name} — {pixel_source} pixels highlighted", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_flipbook_gif(
    span_dir: str | Path,
    out_path: str | Path,
    *,
    pixel_source: PixelSource = "line_gray",
    strict_line_types: bool = True,
    fps: float = 8.0,
) -> None:
    """Animated GIF cycling through frames (requires pillow)."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("flipbook GIF requires pillow: pip install pillow") from e

    span_dir = Path(span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    frames_rgb: list[Image.Image] = []
    for p in paths:
        g = read_bmp_gray(p)
        rgb = np.stack([g, g, g], axis=-1).astype(np.uint8)
        mask = mask_from_source(g, pixel_source, strict_line_types=strict_line_types)
        rgb[mask, 0] = 255
        rgb[mask, 1] = 0
        rgb[mask, 2] = 0
        frames_rgb.append(Image.fromarray(rgb))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000.0 / max(fps, 0.1))
    if frames_rgb:
        frames_rgb[0].save(
            out_path,
            save_all=True,
            append_images=frames_rgb[1:],
            duration=duration_ms,
            loop=0,
        )
