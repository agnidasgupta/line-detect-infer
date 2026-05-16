#!/usr/bin/env python3
"""
CLI: RANSAC parabola (+ optional catenary refit) baseline on DUKE_FLORIDA_150 spans.

Run from the DUKE_FLORIDA_150 directory (defaults: labeled **line** pixels per gray encoding):

    python3 -m tools.run_baseline --span 206_213 --out_dir ./baseline_out

See README_CATENARY_BASELINE.md for full steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catenary_ransac import (
    PerLineSpanReport,
    PixelSource,
    run_span_baseline,
    run_span_baseline_per_line,
)
from .visualize import plot_span_ty, plot_span_ty_per_line, save_frame_montage, write_flipbook_gif


def _curve_dict(c):
    d = {
        "coef_parabola": c.coef_parabola.tolist(),
        "n_inliers": int(c.inlier_mask.sum()),
        "rmse_parabola": c.rmse_parabola,
        "rmse_catenary": c.rmse_catenary,
    }
    if c.catenary_params is not None:
        d["catenary_params_y0_a_t0_s"] = c.catenary_params.tolist()
    return d


def _report_to_dict(report):
    return {
        "span_dir": report.span_dir,
        "n_frames": report.n_frames,
        "n_points": report.n_points,
        "pixel_source": report.pixel_source,
        "aggregate": report.aggregate,
        "strict_line_types": report.strict_line_types,
        "line_gray_encoding": "gray = (id_field << 3) | type_code; type in low 3 bits (0=comm..4=transmission); id_field in bits 3-7",
        "line_labels_summary": report.line_labels_summary,
        "curves": [_curve_dict(c) for c in report.curves],
    }


def _per_line_report_to_dict(report: PerLineSpanReport):
    return {
        "mode": "per_line_id",
        "span_dir": report.span_dir,
        "n_frames": report.n_frames,
        "pixel_source": report.pixel_source,
        "aggregate": report.aggregate,
        "strict_line_types": report.strict_line_types,
        "line_gray_encoding": "one curve per packed line gray (distinct conductors; same id_field + different type => different gray)",
        "line_labels_summary": report.line_labels_summary,
        "per_line": [
            {
                "line_gray": pl.line_gray,
                "decode": pl.decode,
                "n_points": pl.n_points,
                "curve": _curve_dict(pl.curve),
            }
            for pl in report.per_line
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Catenary/parabola RANSAC baseline for span BMPs")
    p.add_argument(
        "--span",
        type=str,
        required=True,
        help="Span folder name (under cwd) or full path to a span directory containing frame_*.bmp",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="If --span is a relative name, join with this root (default: current directory)",
    )
    p.add_argument("--out_dir", type=str, default="./baseline_runs", help="Output directory for artifacts")
    p.add_argument(
        "--per_line_id",
        action="store_true",
        help=(
            "Fit one sag curve per packed line gray (each distinct conductor label). "
            "Requires labeled BMPs; implies line_gray sampling. Incompatible with --pixel_source black/not_white."
        ),
    )
    p.add_argument(
        "--pixel_source",
        type=str,
        default="line_gray",
        choices=("black", "not_white", "line_gray"),
        help=(
            "line_gray (default): labeled conductors 128<=gray<255, type=gray&7, id_field=gray>>3; "
            "black=0 solids; not_white=gray<255"
        ),
    )
    p.add_argument(
        "--no_strict_line_types",
        action="store_true",
        help="For line_gray, include low-3-bit types 5-7 (normally excluded; spec types are 0-4 only)",
    )
    p.add_argument(
        "--aggregate",
        type=str,
        default="centroid",
        choices=("none", "centroid"),
        help="centroid: one (t, mean_y) per frame per line (per_line_id) or merged (default); none: every pixel",
    )
    p.add_argument("--n_curves", type=int, default=3, help="Sequential RANSAC rounds (ignored with --per_line_id)")
    p.add_argument("--residual_thresh", type=float, default=4.0, help="Parabola RANSAC residual threshold in pixels")
    p.add_argument("--no_catenary_refine", action="store_true", help="Skip scipy catenary refit on inliers")
    p.add_argument("--max_frames", type=int, default=None, help="Load at most this many frames (debug)")
    p.add_argument("--gif", action="store_true", help="Write flipbook.gif (needs pillow)")
    p.add_argument("--no_montage", action="store_true", help="Skip frame montage PNG")
    args = p.parse_args()

    span_path = Path(args.span)
    if not span_path.is_dir():
        span_path = Path(args.data_root) / args.span
    if not span_path.is_dir():
        raise SystemExit(f"Span directory not found: {span_path}")

    if args.per_line_id and args.pixel_source != "line_gray":
        raise SystemExit("--per_line_id only applies to labeled line pixels; use --pixel_source line_gray (default).")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = span_path.name
    st = not args.no_strict_line_types

    if args.per_line_id:
        pl_report = run_span_baseline_per_line(
            span_path,
            aggregate=args.aggregate,  # type: ignore[arg-type]
            strict_line_types=st,
            residual_thresh=args.residual_thresh,
            refine_catenary=not args.no_catenary_refine,
            max_frames=args.max_frames,
        )
        rep_path = out / f"{tag}_report_per_line.json"
        with rep_path.open("w", encoding="utf-8") as f:
            json.dump(_per_line_report_to_dict(pl_report), f, indent=2)

        plot_path = out / f"{tag}_ty_plot_per_line.png"
        plot_span_ty_per_line(pl_report, plot_path, title=f"{tag}  per-line sag")

        if not args.no_montage:
            save_frame_montage(
                span_path,
                out / f"{tag}_montage.png",
                pixel_source="line_gray",
                strict_line_types=st,
            )
        if args.gif:
            write_flipbook_gif(span_path, out / f"{tag}_flipbook.gif", pixel_source="line_gray", strict_line_types=st)

        n_pts = sum(pl.n_points for pl in pl_report.per_line)
        print(f"[ok] span={span_path}  mode=per_line_id")
        print(f"     frames={pl_report.n_frames}  lines={len(pl_report.per_line)}  total_samples={n_pts}")
        print(f"     aggregate={pl_report.aggregate}  strict_line_types={pl_report.strict_line_types}")
        print(f"     wrote {rep_path}")
        print(f"     wrote {plot_path}")
        if not args.no_montage:
            print(f"     wrote {out / f'{tag}_montage.png'}")
        if args.gif:
            print(f"     wrote {out / f'{tag}_flipbook.gif'}")
        return

    report = run_span_baseline(
        span_path,
        pixel_source=args.pixel_source,  # type: ignore[arg-type]
        aggregate=args.aggregate,  # type: ignore[arg-type]
        strict_line_types=st,
        n_curves=args.n_curves,
        residual_thresh=args.residual_thresh,
        refine_catenary=not args.no_catenary_refine,
        max_frames=args.max_frames,
    )

    rep_path = out / f"{tag}_report.json"
    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(_report_to_dict(report), f, indent=2)

    plot_path = out / f"{tag}_ty_plot.png"
    plot_span_ty(report, plot_path, title=f"{tag}  RANSAC sag fit")

    if not args.no_montage:
        save_frame_montage(
            span_path,
            out / f"{tag}_montage.png",
            pixel_source=args.pixel_source,  # type: ignore[arg-type]
            strict_line_types=st,
        )

    if args.gif:
        write_flipbook_gif(
            span_path,
            out / f"{tag}_flipbook.gif",
            pixel_source=args.pixel_source,  # type: ignore[arg-type]
            strict_line_types=st,
        )

    best = report.best_curve
    print(f"[ok] span={span_path}")
    print(
        f"     frames={report.n_frames}  points={report.n_points}  source={report.pixel_source}  "
        f"aggregate={report.aggregate}  strict_line_types={report.strict_line_types}"
    )
    print(f"     wrote {rep_path}")
    print(f"     wrote {plot_path}")
    if not args.no_montage:
        print(f"     wrote {out / f'{tag}_montage.png'}")
    if args.gif:
        print(f"     wrote {out / f'{tag}_flipbook.gif'}")
    if best is not None:
        print(f"     best curve inliers={int(best.inlier_mask.sum())}  rmse_parabola={best.rmse_parabola:.4f}", end="")
        if best.rmse_catenary is not None:
            print(f"  rmse_catenary={best.rmse_catenary:.4f}")
        else:
            print()


if __name__ == "__main__":
    main()
