#!/usr/bin/env python
"""
Matplotlib-based plotting utilities for Hi-C noise distribution visualization.

These functions produce density histograms of the noise metric values from bedgraph
files, useful for visually assessing data quality and comparing samples.
"""
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-ready figure dimensions (width x height in inches).
# 6×4 inches at 200 dpi produces a 1200×800 pixel PNG — suitable for manuscripts
# and presentations without being excessively large.
_FIGURE_SIZE = (6, 4)
_FIGURE_DPI = 200

# Default histogram colour: Vega/Altair's standard blue (#4C78A8).
# This is a perceptually uniform, accessible blue commonly used in scientific figures.
_DEFAULT_COLOR = "#4C78A8"


def _transform_noise(values: np.ndarray) -> np.ndarray:
    """
    Apply a pseudo-log transform to noise values for histogram display.

    The noise metric (1 / |lag-1 autocovariance|) spans many orders of magnitude
    (typically 0.001 to 1000+), making a linear histogram uninformative. A log10
    transform compresses the range, but raw noise values can be slightly negative
    (due to floating-point artefacts near zero) or exactly zero (for degenerate bins).

    The transform used is:  log10(clip(x, -0.999, None) + 1.0)

    Breaking this down:
    - ``clip(x, -0.999, None)``:
        Clips the minimum value to -0.999. Values below -1.0 would produce
        log10(x + 1.0) < -Inf (since log10(0) = -Inf and log10(negative) = NaN).
        The clip at -0.999 ensures x + 1.0 >= 0.001, so log10 is always finite.
        Why -0.999 specifically: this preserves values that are just slightly
        negative (e.g. -0.5) in their approximate log position while safely
        handling pathologically negative values.
    - ``+ 1.0``:
        Shifts the range so that x=0 maps to log10(1) = 0 (the histogram origin).
        Without this shift, x=0 would produce log10(0) = -Inf.
        This is the standard "log(x + 1)" or "log1p" pseudo-log transform used
        when data includes zeros.
    - ``log10(...)``:
        Base-10 logarithm. Using base 10 (rather than natural log) makes the
        axis intuitive: each unit corresponds to a 10× change in noise.

    The resulting x-axis label "log10(noise + 1)" accurately describes this transform.

    Parameters
    ----------
    values : np.ndarray
        Raw noise values from a bedgraph file (may include NaN, Inf, negative values).

    Returns
    -------
    np.ndarray
        Transformed values suitable for histogram display.
    """
    return np.log10(np.clip(values, a_min=-0.999, a_max=None) + 1.0)


def plot_noise_density_from_bed(noise_bed_path: str, out_png: str) -> None:
    """
    Render a density histogram of noise values from a single bedgraph file.

    Reads the bedgraph, applies the pseudo-log transform, and saves a density
    histogram to a PNG file. The y-axis shows probability density (area = 1),
    making it easy to compare the shape of distributions across samples even
    when they have different numbers of bins.

    Parameters
    ----------
    noise_bed_path : str
        Path to a noise bedgraph file (4-column: chrom start end noise_value).
        Typically the merged ``{res}.bedgraph`` from ``noise_fullmap``, or a
        per-chromosome ``{chrom}_{res}.bedgraph`` from ``noise_sampling``.
    out_png : str
        Output PNG file path (overwritten if it exists).
    """
    df = pd.read_csv(
        noise_bed_path,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    values = df["noise"].astype(float).to_numpy()
    transformed = _transform_noise(values)
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
    ax.hist(transformed, bins=20, density=True, alpha=0.6, color=_DEFAULT_COLOR)
    ax.set_xlabel("log10(noise + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Noise Distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_FIGURE_DPI)


def plot_two_noise_beds(
    noise_bed_a: str,
    noise_bed_b: str,
    labels: Tuple[str, str] = ("A", "B"),
    out_png: str = "noise_compare.png",
) -> None:
    """
    Compare two bedgraph files by plotting overlaid noise density histograms.

    Useful for visually comparing noise distributions between two Hi-C samples
    (e.g. before and after additional sequencing, or two different cell lines).
    The two histograms are overlaid with partial transparency (alpha=0.5) so
    both are visible where they overlap.

    Parameters
    ----------
    noise_bed_a : str
        Path to the first bedgraph file.
    noise_bed_b : str
        Path to the second bedgraph file.
    labels : Tuple[str, str]
        Legend labels for the two datasets (default: ("A", "B")).
    out_png : str
        Output PNG file path (default: ``"noise_compare.png"``).
    """
    df_a = pd.read_csv(
        noise_bed_a,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    df_b = pd.read_csv(
        noise_bed_b,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    vals_a = df_a["noise"].astype(float).to_numpy()
    vals_b = df_b["noise"].astype(float).to_numpy()
    trans_a = _transform_noise(vals_a)
    trans_b = _transform_noise(vals_b)

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
    # alpha=0.5 makes both histograms visible where they overlap
    ax.hist(trans_a, bins=20, density=True, alpha=0.5, label=labels[0])
    ax.hist(trans_b, bins=20, density=True, alpha=0.5, label=labels[1])
    ax.legend()
    ax.set_xlabel("log10(noise + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Noise Distributions")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_FIGURE_DPI)


# ---------------------------------------------------------------------------
# Elbow diagnostic figure
# ---------------------------------------------------------------------------

_ELBOW_UPPER_COLOR = "#d62728"
_ELBOW_LOWER_COLOR = "#1f77b4"
_ELBOW_BAND_ALPHA  = 0.15


def _elbow_apply_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Variance-stabilising transform for elbow display (mirrors filters.py)."""
    if transform == "sqrt":
        return np.sqrt(np.clip(y, 0.0, None))
    if transform == "cbrt":
        return np.cbrt(y)
    if transform == "log1p":
        return np.log1p(np.clip(y, 0.0, None))
    return y.astype(float)


def _elbow_overlay_panel(
    ax,
    per_chrom_curves: dict,
    sorted_key: int,
    hi_key: int,
    lo_key: int,
    transform: str,
    title: str,
) -> None:
    """Overlay normalised sorted curves for all chromosomes on one axes."""
    cmap = matplotlib.colormaps["tab20"]
    chrom_list = list(per_chrom_curves.keys())
    hi_pcts, lo_pcts = [], []

    for i, chrom in enumerate(chrom_list):
        vals   = per_chrom_curves[chrom][sorted_key]
        hi_idx = per_chrom_curves[chrom][hi_key]
        lo_idx = per_chrom_curves[chrom][lo_key]

        x_n    = np.linspace(0.0, 100.0, len(vals))
        y_disp = _elbow_apply_transform(vals, transform)
        span   = float(y_disp[-1]) - float(y_disp[0])
        if span < 1e-12:
            continue
        y_plot = (y_disp - y_disp[0]) / span
        ax.plot(x_n, y_plot, lw=0.5, alpha=0.4, color=cmap(i % 20))
        hi_pcts.append(100.0 * hi_idx / len(vals))
        lo_pcts.append(100.0 * lo_idx / len(vals))

    if hi_pcts:
        med_hi = float(np.median(hi_pcts))
        med_lo = float(np.median(lo_pcts))
        ax.axvspan(med_hi - 1, med_hi + 1, color=_ELBOW_UPPER_COLOR, alpha=_ELBOW_BAND_ALPHA)
        ax.axvline(med_hi, color=_ELBOW_UPPER_COLOR, lw=1.5, ls="--",
                   label=f"Median upper elbow: {med_hi:.1f}th pct")
        ax.axvspan(med_lo - 1, med_lo + 1, color=_ELBOW_LOWER_COLOR, alpha=_ELBOW_BAND_ALPHA)
        ax.axvline(med_lo, color=_ELBOW_LOWER_COLOR, lw=1.5, ls="--",
                   label=f"Median lower elbow: {med_lo:.1f}th pct")

    ylabel = f"Normalised value ({transform})" if transform != "none" else "Normalised value"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Bin percentile (%)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)


def _elbow_dotplot_panel(
    ax,
    results: List[dict],
    hi_col: str,
    lo_col: str,
    title: str,
) -> None:
    """Per-chromosome scatter of elbow percentiles (upper=red, lower=blue)."""
    chroms  = [r["chrom"] for r in results]
    hi_pcts = [r[hi_col] for r in results]
    lo_pcts = [r[lo_col] for r in results]
    x = np.arange(len(chroms))

    ax.scatter(x, hi_pcts, color=_ELBOW_UPPER_COLOR, s=30, zorder=3, label="Upper elbow")
    ax.scatter(x, lo_pcts, color=_ELBOW_LOWER_COLOR, s=30, zorder=3, label="Lower elbow")
    ax.axhline(float(np.median(hi_pcts)), color=_ELBOW_UPPER_COLOR, lw=1.0, ls=":", alpha=0.6)
    ax.axhline(float(np.median(lo_pcts)), color=_ELBOW_LOWER_COLOR, lw=1.0, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(chroms, rotation=90, fontsize=6)
    ax.set_ylabel("Elbow percentile (%)", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, lw=0.3, alpha=0.4, axis="y")


def plot_elbow_figure(
    per_chrom_curves: dict,
    results: List[dict],
    out_prefix: str,
    density_transform: str = "log1p",
    noise_transform: str = "log1p",
) -> None:
    """
    Save the 2×2 elbow diagnostic figure as both PNG and PDF.

    ``out_prefix`` is the output path without extension
    (e.g. ``/out/sample_10000_elbow``).  Writes ``{out_prefix}.png`` (150 dpi)
    and ``{out_prefix}.pdf`` (vector).

    Parameters
    ----------
    per_chrom_curves : dict
        ``chrom → (d_sorted, n_sorted, d_lo_idx, d_hi_idx, n_lo_idx, n_hi_idx)``
        as returned by ``filters.detect_elbow_thresholds()``.
    results : list of dict
        Per-chromosome threshold rows (from ``thresholds_df.to_dict("records")``).
    out_prefix : str
        Output path without file extension.
    density_transform, noise_transform : str
        The same transforms used during detection, applied here for display.
    """
    # Tuple layout: (d_sorted=0, n_sorted=1, d_lo_idx=2, d_hi_idx=3, n_lo_idx=4, n_hi_idx=5)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Per-chromosome elbow detection — density & noise metrics",
        fontsize=11,
    )

    _elbow_overlay_panel(
        axes[0, 0], per_chrom_curves,
        sorted_key=0, hi_key=3, lo_key=2,
        transform=density_transform,
        title=f"Density — sorted curves (normalised, transform={density_transform})",
    )
    _elbow_overlay_panel(
        axes[0, 1], per_chrom_curves,
        sorted_key=1, hi_key=5, lo_key=4,
        transform=noise_transform,
        title=f"Noise — sorted curves (normalised, transform={noise_transform})",
    )
    _elbow_dotplot_panel(
        axes[1, 0], results,
        hi_col="density_upper_pct", lo_col="density_lower_pct",
        title="Density elbow percentile per chromosome",
    )
    _elbow_dotplot_panel(
        axes[1, 1], results,
        hi_col="noise_upper_pct", lo_col="noise_lower_pct",
        title="Noise elbow percentile per chromosome",
    )

    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)
