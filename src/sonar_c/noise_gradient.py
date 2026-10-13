#!/usr/bin/env python
"""
Per-bin noise gradient: a graded alternative to the binary noise blacklist.

The blacklist answers "is this bin usable?" with yes or no. This module answers
the question that is actually being asked of the data — *how confident can I be
that a feature here is real, given the local noise background?* — with a
continuous score per bin.

Two things distinguish the gradient from the noise track it is derived from:

**It is conditioned on density.** The raw noise metric is higher in sparsely
sequenced regions for purely statistical reasons, so thresholding it directly
penalises real-but-sparse loci. The gradient instead scores how far a bin's
noise sits above what the density→noise trend predicts *for its own density*,
reusing ``filters._density_conditioned_residuals()``. A sparse region whose
noise is merely typical for that sparsity scores near zero.

**It includes the neighbouring rows.** A feature's credibility depends on its
surroundings, not just on the single bin it lands in, so each bin's score also
carries the residual z of the bins at +1 and -1. All three values are kept as
columns in the TSV output alongside the summary, so a different aggregation can
be tried later without recomputing anything.

Relationship to the blacklist
-----------------------------
The gradient and the ``density-residual`` blacklist rule are two views of one
quantity: selecting bins with ``z_self >= k`` reproduces exactly the bins that
rule flags at ``density_residual_k = k`` (leaving aside the unmappable bins the
blacklist always flags). Both go through ``filters._robust_centre_scale()`` so
they cannot drift apart. Run the two with the same ``fit_window`` for the
correspondence to hold.

Inputs are the two bedgraphs the ``noise-bedgraph`` command already produces, so
the gradient is pure post-processing — it never needs that expensive step re-run.

Main entry point: ``build_noise_gradient()``.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from . import filters
from .noise_to_weights import (
    _is_decoy_chrom,
    _load_bedgraph,
    _load_chrom_sizes,
    _reindex_to_full_grid,
)

# A chromosome needs at least this many usable bins before a density→noise trend
# and a robust spread mean anything. Matches the guard in
# filters.detect_elbow_thresholds().
_MIN_FINITE_BINS = 4

# Columns of the detailed TSV output, in order.
_TSV_COLUMNS = [
    "chrom", "start", "end",
    "density", "noise", "residual",
    "z_minus1", "z_self", "z_plus1",
    "z_max", "z_mean", "n_finite", "unmappable",
]

# Which column each --summary choice writes into the 4-column bedgraph.
_SUMMARY_COLUMNS = {"max": "z_max", "mean": "z_mean", "self": "z_self"}


def _infer_res(df: pd.DataFrame) -> int:
    """
    Infer bin size from a bedgraph as the most common interval width.

    The modal width rather than the first one, because the final bin of each
    chromosome is usually short (the chromosome length is rarely a multiple of
    the bin size) and a sorted file can start on such a bin after a merge.
    """
    widths = (df["end"] - df["start"]).to_numpy(dtype=np.int64)
    return int(pd.Series(widths).mode().iloc[0])


def compute_chrom_gradient(
    density: np.ndarray,
    noise: np.ndarray,
    fit_window: int = 0,
) -> pd.DataFrame:
    """
    Compute the gradient columns for one chromosome.

    Both inputs must already be projected onto the **full bin grid**, with NaN
    where a bin is absent or unmeasurable — index i must be the bin covering
    ``[i*res, (i+1)*res)``. That is what makes "+1 row" mean "+1 bin"; on a
    gapped or unsorted array the neighbour columns would silently refer to the
    wrong loci.

    The residual is fitted on the finite subset only (matching what the
    ``density-residual`` blacklist rule does), then scattered back to full-grid
    positions before the neighbour shift.

    Parameters
    ----------
    density, noise : np.ndarray
        Full-grid per-bin values for one chromosome, same length.
    fit_window : int
        Running-median window for the density→noise trend; 0 selects a width
        from the bin count. Must match the blacklist's value for the two to
        agree.

    Returns
    -------
    pd.DataFrame
        One row per bin, with the per-bin columns of ``_TSV_COLUMNS`` (the
        caller adds chrom/start/end).
    """
    n = len(density)
    finite = np.isfinite(density) & np.isfinite(noise)

    residual = np.full(n, np.nan)
    z_self = np.full(n, np.nan)
    if int(finite.sum()) >= _MIN_FINITE_BINS:
        fitted = filters._density_conditioned_residuals(
            density[finite], noise[finite], fit_window,
        )
        residual[finite] = fitted
        z_self[finite] = filters._residual_robust_z(fitted)

    # Neighbours by bin index. The first bin has no -1 and the last no +1, so
    # those stay NaN and are simply skipped by the summary below.
    z_minus1 = np.full(n, np.nan)
    z_minus1[1:] = z_self[:-1]
    z_plus1 = np.full(n, np.nan)
    z_plus1[:-1] = z_self[1:]

    # A NaN neighbour is skipped rather than poisoning the summary: an
    # unmappable bin next door says nothing about how noisy this locus is, and
    # n_finite records how many of the three actually contributed.
    stack = np.vstack([z_minus1, z_self, z_plus1])
    n_finite = np.isfinite(stack).sum(axis=0)
    z_max = np.full(n, np.nan)
    z_mean = np.full(n, np.nan)
    have_any = n_finite > 0
    if have_any.any():
        z_max[have_any] = np.nanmax(stack[:, have_any], axis=0)
        z_mean[have_any] = np.nanmean(stack[:, have_any], axis=0)

    return pd.DataFrame({
        "density": density,
        "noise": noise,
        "residual": residual,
        "z_minus1": z_minus1,
        "z_self": z_self,
        "z_plus1": z_plus1,
        "z_max": z_max,
        "z_mean": z_mean,
        "n_finite": n_finite.astype(int),
        # The centre bin having no measurable noise is worth recording
        # separately: z_max may still be finite from its neighbours, and a
        # consumer needs to be able to tell that apart from a measured bin.
        "unmappable": (~finite).astype(int),
    })


def build_noise_gradient(
    density_bedgraph: str,
    noise_bedgraph: str,
    out_dir: str,
    res: Optional[int] = None,
    chrom_sizes_path: Optional[str] = None,
    fit_window: int = 0,
    summary: str = "max",
    skip_decoys: bool = True,
) -> pd.DataFrame:
    """
    Build a genome-wide noise gradient from the noise and density bedgraphs.

    Writes two files into ``out_dir``:

    ``gradient.bedgraph``
        Four space-separated columns (chrom, start, end, value) with no header,
        carrying the chosen ``summary`` column. Bins whose summary is not finite
        are omitted so that ``bedtools map`` returns "." for them rather than
        tripping over a NaN.

    ``gradient.tsv``
        Every bin, including unmappable ones, with all the component columns —
        so the neighbourhood aggregation can be revisited without recomputing.

    Parameters
    ----------
    density_bedgraph, noise_bedgraph : str
        The ``{res}_density.bedgraph`` and ``{res}.bedgraph`` outputs of
        ``noise-bedgraph``.
    out_dir : str
        Output directory, created if needed.
    res : int, optional
        Bin size. Inferred from the bedgraph's modal interval width if omitted.
    chrom_sizes_path : str, optional
        Chrom sizes TSV. Without it the grid is sized from the largest bin
        present, which is only correct when the track reaches the chromosome end.
    fit_window : int
        Passed to the trend fit; see ``compute_chrom_gradient()``.
    summary : {"max", "mean", "self"}
        Which column the bedgraph carries. "max" is the conservative choice: a
        locus is only as trustworthy as the noisiest bin in its neighbourhood.
    skip_decoys : bool
        Drop unplaced/alt/decoy sequences via ``_is_decoy_chrom()``.

    Returns
    -------
    pd.DataFrame
        The full per-bin table that was written to ``gradient.tsv``.
    """
    if summary not in _SUMMARY_COLUMNS:
        raise ValueError(
            f"unknown summary {summary!r}; expected one of {sorted(_SUMMARY_COLUMNS)}"
        )

    d_df = _load_bedgraph(density_bedgraph)
    n_df = _load_bedgraph(noise_bedgraph)
    d_df["value"] = pd.to_numeric(d_df["value"], errors="coerce")
    n_df["value"] = pd.to_numeric(n_df["value"], errors="coerce")

    if res is None:
        res = _infer_res(n_df)

    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)

    chroms = [
        c for c in sorted(
            set(d_df["chrom"].unique()) & set(n_df["chrom"].unique()),
            key=filters._canonical_chrom_key,
        )
        if not (skip_decoys and _is_decoy_chrom(c))
    ]

    frames = []
    for chrom in chroms:
        chrom_len = chrom_sizes.get(chrom)
        # Project both tracks onto the same full grid, so index i means bin i in
        # both and the neighbour shift is meaningful.
        density = _reindex_to_full_grid(d_df[d_df["chrom"] == chrom], res, chrom_len)
        noise = _reindex_to_full_grid(n_df[n_df["chrom"] == chrom], res, chrom_len)
        if len(density) != len(noise):
            # Only possible when chrom_len is unknown and the two tracks end on
            # different bins; pad the shorter one so the grids line up.
            n_bins = max(len(density), len(noise))
            density = np.pad(density, (0, n_bins - len(density)), constant_values=np.nan)
            noise = np.pad(noise, (0, n_bins - len(noise)), constant_values=np.nan)

        usable = int((np.isfinite(density) & np.isfinite(noise)).sum())
        if usable < _MIN_FINITE_BINS:
            print(f"[WARN] {chrom}: only {usable} usable bins — skipping")
            continue

        frame = compute_chrom_gradient(density, noise, fit_window)
        starts = np.arange(len(frame), dtype=np.int64) * res
        ends = starts + res
        if chrom_len is not None:
            ends = np.minimum(ends, int(chrom_len))
        frame.insert(0, "end", ends)
        frame.insert(0, "start", starts)
        frame.insert(0, "chrom", chrom)
        frames.append(frame)

    if not frames:
        raise ValueError("no chromosomes produced a gradient — check the input bedgraphs")

    out = pd.concat(frames, ignore_index=True)[_TSV_COLUMNS]

    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "gradient.tsv"), sep="\t", index=False,
               float_format="%.6g")

    # The bedgraph carries only finite summary values, in bedtools-friendly form.
    summary_col = _SUMMARY_COLUMNS[summary]
    bed = out.loc[np.isfinite(out[summary_col]), ["chrom", "start", "end", summary_col]]
    bed.to_csv(os.path.join(out_dir, "gradient.bedgraph"), sep=" ",
               header=False, index=False, float_format="%.6g")

    return out
