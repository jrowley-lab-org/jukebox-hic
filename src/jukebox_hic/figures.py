#!/usr/bin/env python
"""
Matplotlib-based plotting utilities for Hi-C noise distribution visualization.

These functions produce density histograms of the noise metric values from bedgraph
files, useful for visually assessing data quality and comparing samples.
"""
from typing import Tuple

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
