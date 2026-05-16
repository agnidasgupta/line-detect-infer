"""
RANSAC fitting of smooth sag curves in (frame_index, y) space.

Default samples are **labeled conductor pixels** (``128 <= gray < 255``) with
type/id decoded per ``line_encoding.py`` (low 3 bits = line type, upper 5 bits =
id field). Optional modes use black (0) or all non-white voxels.

We fit a parabola y = a*t^2 + b*t + c inside RANSAC (fast, stable hypothesis).
Optionally refine inliers with a symmetric catenary-style curve:

    y = y0 + a * (cosh((t - t0) / s) - 1)

which has a minimum at t = t0 when a > 0 (U-shaped sag). Parameters are estimated
with bounded nonlinear least squares when scipy is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .bmp_io import read_bmp_gray
from .line_encoding import decode_line_gray, line_label_mask, unique_line_grays_in_span

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


PixelSource = Literal["black", "not_white", "line_gray"]


def mask_from_source(
    img: np.ndarray,
    source: PixelSource,
    *,
    strict_line_types: bool = True,
) -> np.ndarray:
    """
    Boolean mask of pixels to use for (t, y) samples.

    ``line_gray``: labeled conductor voxels per dataset spec: 128 <= gray < 255,
    optionally restricted to type codes 0..4 in the low 3 bits (comm..transmission).
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if source == "black":
        return img == 0
    if source == "not_white":
        return img < 255
    if source == "line_gray":
        return line_label_mask(img, strict_type=strict_line_types)
    raise ValueError(f"Unknown pixel source: {source}")


def collect_ty_points(
    frames: list[np.ndarray],
    source: PixelSource = "line_gray",
    aggregate: Literal["none", "centroid"] = "centroid",
    *,
    strict_line_types: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build arrays t (frame index), y (row index, top=0).

    Default ``source='line_gray'``: uses labeled line voxels (encoding per
    ``line_encoding.py``). ``aggregate='centroid'`` yields one (t, mean_y) per
    frame that has any selected pixel (stable for RANSAC over merged conductors).

    aggregate='none': one sample per selected pixel (can be large).
    """
    ts: list[float] = []
    ys: list[float] = []

    for fi, im in enumerate(frames):
        m = mask_from_source(im, source, strict_line_types=strict_line_types)
        if not np.any(m):
            continue
        yy, xx = np.where(m)
        if aggregate == "centroid":
            ts.append(float(fi))
            ys.append(float(np.mean(yy)))
        elif aggregate == "none":
            ts.extend([float(fi)] * yy.size)
            ys.extend(yy.astype(np.float64).tolist())
        else:
            raise ValueError(aggregate)

    return np.asarray(ts, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def collect_ty_points_per_unique_line_gray(
    frames: list[np.ndarray],
    *,
    aggregate: Literal["none", "centroid"] = "centroid",
    strict_line_types: bool = True,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    One (t, y) trace per **packed line gray** (unique conductor label in the span).

    Each physical line uses a distinct gray value ``128 <= g < 255``; grouping by
    ``g`` separates conductors that share the same ``id_field`` but differ in
    ``type_code`` (e.g. 130 vs 131).
    """
    present: set[int] = set()
    for im in frames:
        m = line_label_mask(im, strict_type=strict_line_types)
        if np.any(m):
            present.update(int(v) for v in np.unique(im[m]).tolist())

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for g in sorted(present):
        ts: list[float] = []
        ys: list[float] = []
        for fi, im in enumerate(frames):
            sel = im == np.uint8(g)
            if not np.any(sel):
                continue
            yy, _ = np.where(sel)
            if aggregate == "centroid":
                ts.append(float(fi))
                ys.append(float(np.mean(yy)))
            elif aggregate == "none":
                ts.extend([float(fi)] * yy.size)
                ys.extend(yy.astype(np.float64).tolist())
            else:
                raise ValueError(aggregate)
        if ts:
            out[g] = (np.asarray(ts, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return out


def fit_parabola_lstsq(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fit [a, b, c] for y = a*t^2 + b*t + c."""
    A = np.column_stack([t * t, t, np.ones_like(t)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # a, b, c


def parabola_predict(coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, c = coef
    return a * t * t + b * t + c


def ransac_parabola(
    t: np.ndarray,
    y: np.ndarray,
    *,
    n_iter: int = 800,
    min_samples: int = 3,
    residual_thresh: float = 4.0,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC for y = a*t^2 + b*t + c.

    Returns:
        coef: (3,) array [a, b, c]
        inlier_mask: (N,) bool
    """
    rng = np.random.default_rng(random_state)
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = t.size
    if n < min_samples:
        coef = fit_parabola_lstsq(t, y)
        return coef, np.ones(n, dtype=bool)

    best_inliers = None
    best_count = -1
    best_coef = None

    for _ in range(n_iter):
        idx = rng.choice(n, size=min_samples, replace=False)
        try:
            coef = fit_parabola_lstsq(t[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        pred = parabola_predict(coef, t)
        resid = np.abs(y - pred)
        inliers = resid < residual_thresh
        c = int(np.sum(inliers))
        if c > best_count:
            best_count = c
            best_inliers = inliers
            best_coef = coef

    if best_inliers is None:
        coef = fit_parabola_lstsq(t, y)
        return coef, np.ones(n, dtype=bool)

    # Refit on all inliers from best model
    mask = best_inliers
    coef = fit_parabola_lstsq(t[mask], y[mask])
    pred = parabola_predict(coef, t)
    inliers = np.abs(y - pred) < residual_thresh
    return coef, inliers


def refine_catenary_residual(
    t: np.ndarray,
    y: np.ndarray,
    *,
    max_nfev: int = 200,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Fit y ≈ y0 + a * (cosh((t - t0) / s) - 1) on points (t, y).

    Returns (params, pred) where params = [y0, a, t0, s] or (None, None) if scipy missing.
    """
    if least_squares is None or t.size < 4:
        return None, None

    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # Initial guess from parabola: minimum near middle of t
    t0 = float(np.median(t))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    y0 = ymin
    amp = max(ymax - ymin, 1e-3)
    s = max((float(t.max() - t.min()) / 6.0), 1e-3)
    a = amp

    p0 = np.array([y0, a, t0, s], dtype=np.float64)

    def fun(p):
        y0_, a_, t0_, s_ = p
        s_ = max(s_, 1e-6)
        pred = y0_ + a_ * (np.cosh((t - t0_) / s_) - 1.0)
        return pred - y

    bounds_lower = [-np.inf, 1e-6, -np.inf, 1e-6]
    bounds_upper = [np.inf, np.inf, np.inf, np.inf]
    res = least_squares(
        fun,
        p0,
        bounds=(bounds_lower, bounds_upper),
        max_nfev=max_nfev,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    p = res.x
    pred = p[0] + p[1] * (np.cosh((t - p[2]) / p[3]) - 1.0)
    return p, pred


@dataclass
class RansacCurveResult:
    """One fitted sag curve (parabola + optional catenary refit)."""
    coef_parabola: np.ndarray  # [a, b, c]
    inlier_mask: np.ndarray
    t: np.ndarray
    y: np.ndarray
    pred_parabola: np.ndarray
    catenary_params: np.ndarray | None = None
    pred_catenary: np.ndarray | None = None
    rmse_parabola: float = 0.0
    rmse_catenary: float | None = None


def sequential_ransac_curves(
    t: np.ndarray,
    y: np.ndarray,
    *,
    n_curves: int = 3,
    residual_thresh: float = 4.0,
    n_iter: int = 800,
    refine_catenary: bool = True,
    random_state: int = 42,
) -> list[RansacCurveResult]:
    """
    Fit multiple parabolas sequentially: remove inliers, repeat.

    Useful when several spatially separated sag traces exist in (t,y).
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    remain = np.ones_like(t, dtype=bool)
    out: list[RansacCurveResult] = []

    for _ in range(n_curves):
        idx = np.where(remain)[0]
        if idx.size < 3:
            break
        coef, inl = ransac_parabola(
            t[idx],
            y[idx],
            residual_thresh=residual_thresh,
            n_iter=n_iter,
            random_state=random_state,
        )
        # Map local inliers back to full array
        full_inl = np.zeros_like(t, dtype=bool)
        full_inl[idx[inl]] = True
        pred_all = parabola_predict(coef, t)
        rmse = float(np.sqrt(np.mean((y[full_inl] - pred_all[full_inl]) ** 2)))

        cat_p = cat_pred = None
        rmse_cat = None
        if refine_catenary and np.sum(full_inl) >= 4:
            tt = t[full_inl]
            yy = y[full_inl]
            cat_p, cat_pred = refine_catenary_residual(tt, yy)
            if cat_p is not None and cat_pred is not None:
                rmse_cat = float(np.sqrt(np.mean((yy - cat_pred) ** 2)))

        out.append(
            RansacCurveResult(
                coef_parabola=coef,
                inlier_mask=full_inl,
                t=t.copy(),
                y=y.copy(),
                pred_parabola=pred_all,
                catenary_params=cat_p,
                pred_catenary=cat_pred,
                rmse_parabola=rmse,
                rmse_catenary=rmse_cat,
            )
        )
        remain &= ~full_inl
        if np.sum(remain) < 3:
            break

    return out


@dataclass
class SpanBaselineReport:
    span_dir: str
    n_frames: int
    n_points: int
    pixel_source: str
    aggregate: str
    strict_line_types: bool
    line_labels_summary: dict[str, dict] = field(default_factory=dict)
    curves: list[RansacCurveResult] = field(default_factory=list)

    @property
    def best_curve(self) -> RansacCurveResult | None:
        if not self.curves:
            return None
        return max(self.curves, key=lambda c: int(np.sum(c.inlier_mask)))


@dataclass
class PerLineSagResult:
    """Single conductor (packed gray) sag fit."""
    line_gray: int
    decode: dict
    n_points: int
    curve: RansacCurveResult


@dataclass
class PerLineSpanReport:
    """Baseline when ``--per_line_id`` is used (one curve per packed line gray)."""
    span_dir: str
    n_frames: int
    pixel_source: str
    aggregate: str
    strict_line_types: bool
    line_labels_summary: dict[str, dict]
    per_line: list[PerLineSagResult]

    @property
    def best_curve(self) -> RansacCurveResult | None:
        if not self.per_line:
            return None
        return max(
            (pl.curve for pl in self.per_line),
            key=lambda c: int(np.sum(c.inlier_mask)),
        )


def _single_curve_from_ransac(
    t: np.ndarray,
    y: np.ndarray,
    *,
    residual_thresh: float,
    refine_catenary: bool,
    n_iter: int = 800,
    random_state: int = 42,
) -> RansacCurveResult:
    coef, inl = ransac_parabola(
        t,
        y,
        residual_thresh=residual_thresh,
        n_iter=n_iter,
        random_state=random_state,
    )
    pred_all = parabola_predict(coef, t)
    rmse = float(np.sqrt(np.mean((y[inl] - pred_all[inl]) ** 2)))
    cat_p = cat_pred = None
    rmse_cat = None
    if refine_catenary and int(np.sum(inl)) >= 4:
        tt = t[inl]
        yy = y[inl]
        cat_p, cat_pred = refine_catenary_residual(tt, yy)
        if cat_p is not None and cat_pred is not None:
            rmse_cat = float(np.sqrt(np.mean((yy - cat_pred) ** 2)))
    return RansacCurveResult(
        coef_parabola=coef,
        inlier_mask=inl,
        t=t.copy(),
        y=y.copy(),
        pred_parabola=pred_all,
        catenary_params=cat_p,
        pred_catenary=cat_pred,
        rmse_parabola=rmse,
        rmse_catenary=rmse_cat,
    )


def run_span_baseline_per_line(
    span_dir: str | Path,
    *,
    aggregate: Literal["none", "centroid"] = "centroid",
    strict_line_types: bool = True,
    residual_thresh: float = 4.0,
    refine_catenary: bool = True,
    max_frames: int | None = None,
) -> PerLineSpanReport:
    """
    One RANSAC sag curve per **packed line gray** (distinct conductor label).

    Only meaningful for labeled BMPs (line voxels). Ignores merged-bundle mode.
    """
    span_dir = Path(span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    if max_frames is not None:
        paths = paths[:max_frames]
    frames = [read_bmp_gray(p) for p in paths]
    by_gray = collect_ty_points_per_unique_line_gray(
        frames,
        aggregate=aggregate,
        strict_line_types=strict_line_types,
    )
    uniq = unique_line_grays_in_span(frames)
    summary = {
        str(g): {
            "type_name": d.type_name,
            "type_code": d.type_code,
            "id_field": d.id_field,
            "valid_type_per_spec": d.valid_line_byte,
        }
        for g, d in sorted(uniq.items(), key=lambda kv: kv[0])
    }

    per_line: list[PerLineSagResult] = []
    for g, (t, y) in sorted(by_gray.items(), key=lambda kv: kv[0]):
        dec = decode_line_gray(g)
        dct = {
            "type_name": dec.type_name,
            "type_code": dec.type_code,
            "id_field": dec.id_field,
            "valid_type_per_spec": dec.valid_line_byte,
        }
        if t.size < 3:
            continue
        curve = _single_curve_from_ransac(
            t,
            y,
            residual_thresh=residual_thresh,
            refine_catenary=refine_catenary,
        )
        per_line.append(
            PerLineSagResult(
                line_gray=g,
                decode=dct,
                n_points=int(t.size),
                curve=curve,
            )
        )

    return PerLineSpanReport(
        span_dir=str(span_dir),
        n_frames=len(frames),
        pixel_source="line_gray",
        aggregate=aggregate,
        strict_line_types=strict_line_types,
        line_labels_summary=summary,
        per_line=per_line,
    )


def run_span_baseline(
    span_dir: str | Path,
    *,
    pixel_source: PixelSource = "line_gray",
    aggregate: Literal["none", "centroid"] = "centroid",
    strict_line_types: bool = True,
    n_curves: int = 3,
    residual_thresh: float = 4.0,
    refine_catenary: bool = True,
    max_frames: int | None = None,
) -> SpanBaselineReport:
    """
    Load all frame_*.bmp under span_dir, extract (t,y), run sequential RANSAC.

    By default uses **labeled line pixels** (``line_gray``): ``128 <= gray < 255``,
    with optional restriction to type codes 0..4 in the low three bits, matching
    the comm/primary/neutral/secondary/transmission encoding.
    """
    span_dir = Path(span_dir)
    paths = sorted(span_dir.glob("frame_*.bmp"))
    if max_frames is not None:
        paths = paths[:max_frames]
    frames = [read_bmp_gray(p) for p in paths]
    t, y = collect_ty_points(
        frames,
        source=pixel_source,
        aggregate=aggregate,
        strict_line_types=strict_line_types,
    )
    curves = sequential_ransac_curves(
        t,
        y,
        n_curves=n_curves,
        residual_thresh=residual_thresh,
        refine_catenary=refine_catenary,
    )
    uniq = unique_line_grays_in_span(frames)
    summary = {
        str(g): {
            "type_name": d.type_name,
            "type_code": d.type_code,
            "id_field": d.id_field,
            "valid_type_per_spec": d.valid_line_byte,
        }
        for g, d in sorted(uniq.items(), key=lambda kv: kv[0])
    }
    return SpanBaselineReport(
        span_dir=str(span_dir),
        n_frames=len(frames),
        n_points=int(t.size),
        pixel_source=pixel_source,
        aggregate=aggregate,
        strict_line_types=strict_line_types,
        line_labels_summary=summary,
        curves=curves,
    )
