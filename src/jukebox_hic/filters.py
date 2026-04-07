#!/usr/bin/env python
"""
Blacklist generation from per-chromosome Hi-C noise bedgraphs.

A "blacklist" in Hi-C analysis is a set of genomic regions that should be excluded
from downstream analyses (loop calling, compartment detection, TAD analysis) because
they produce unreliable or artifactual signals. Common sources of blacklisted regions:
- Repetitive sequences with multi-mapping reads (high stochastic contact noise).
- Low-mappability regions (telomeres, centromeres, satellite repeats).
- Extreme GC bias regions.
- Bins with essentially no contacts (NaN noise values).

This module flags bins based on their noise values: bins above a user-specified noise
threshold (either a z-score cutoff or a quantile threshold) and bins with non-finite
noise (NaN/Inf) are merged into a BED-format blacklist.

Main entry point: ``build_blacklist_from_bedgraphs()``.
"""
import math
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _gather_paths(patterns: Sequence[str]) -> list[str]:
    """
    Expand a list of file paths and/or glob patterns into a sorted list of absolute paths.

    Accepts a mix of:
    - Literal file paths (e.g. ``"/data/chr1_10000.bedgraph"``): included if the file
      exists, silently ignored if it does not.
    - Glob patterns (containing ``*``, ``?``, or ``[...]``): expanded using
      ``glob.glob()``; matched files are included, non-matching patterns produce no
      error (the empty match is silently ignored).

    Duplicate paths (from overlapping globs or explicit + glob matches) are deduplicated.
    The returned list is sorted alphabetically for deterministic processing order.

    Parameters
    ----------
    patterns : Sequence[str]
        Mix of literal file paths and glob patterns to expand.

    Returns
    -------
    list of str
        Sorted, deduplicated list of absolute paths to existing files.
    """
    unique: set[str] = set()
    for pat in patterns:
        if "*" in pat or "?" in pat or "[" in pat:
            import glob
            # Expand glob pattern; only include files (not directories)
            matches = glob.glob(pat)
            unique.update(os.path.abspath(p) for p in matches if os.path.isfile(p))
        else:
            if os.path.isfile(pat):
                unique.add(os.path.abspath(pat))
    return sorted(unique)


def _merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping and adjacent genomic intervals within each chromosome.

    Takes a DataFrame of flagged intervals (which may overlap or touch each other)
    and returns a new DataFrame where all contiguous regions are collapsed into a
    single non-overlapping interval. This is the standard "bedtools merge" operation.

    Algorithm (per chromosome)
    --------------------------
    Intervals are processed in sorted order by start position using a single pass:

    - ``current`` — the interval currently being extended. Starts as the first interval.
    - ``ordered`` — the sorted array of all intervals for this chromosome.

    For each subsequent interval ``row``:

    1. **Overlapping**: ``row.start <= current.end AND row.start >= current.start``
       → The new interval begins before or at the end of ``current``, meaning they
       overlap. Extend ``current.end`` to ``max(current.end, row.end)``.
       (The ``>= current.start`` condition protects against pathological cases where
       a very short interval is fully contained within ``current``.)

    2. **Adjacent**: ``row.start == current.end``
       → The new interval begins exactly where ``current`` ends (zero gap).
       This case is technically handled by condition 1 but the original check is
       kept for clarity: adjacent intervals are merged.

    3. **Neither**: ``row.start > current.end``
       → The new interval is disjoint from ``current``. Emit ``current`` as a
       completed merged interval and reset ``current = row``.

    After the loop, the final ``current`` interval is always emitted.

    Variables
    ---------
    ``ordered`` : 2D numpy array, shape (N, 3), columns = [chrom, start, end]
        The sorted intervals for one chromosome, as numpy rows for fast iteration.
    ``current`` : 1D numpy array, shape (3,) = [chrom, start, end]
        The interval currently being extended. Mutated in-place for merging.
    ``row`` : 1D numpy array, shape (3,) = [chrom, start, end]
        The next interval being examined. Accessed by positional index:
        - ``row[1]`` = start coordinate
        - ``row[2]`` = end coordinate

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``chrom``, ``start``, ``end``.
        Values may have any dtype; coordinates are cast to int during comparison.

    Returns
    -------
    pd.DataFrame
        Merged intervals with columns ``chrom``, ``start``, ``end``.
        Empty DataFrame (with correct columns) if input is empty.
    """
    merged_rows = []
    for chrom, group in df.groupby("chrom", sort=True):
        ordered = group.sort_values("start").to_numpy()
        if ordered.size == 0:
            continue
        # Start with the first interval as the "current" interval being extended
        current = ordered[0].copy()
        for row in ordered[1:]:
            # row[1] = start, row[2] = end (positional access into [chrom, start, end])
            if int(row[1]) <= int(current[2]) and int(row[1]) >= int(current[1]):
                # Overlapping: extend current's end to cover this interval
                current[2] = max(int(current[2]), int(row[2]))
            elif int(row[1]) == int(current[2]):
                # Adjacent (abutting): treat as overlap and extend
                # Note: this branch is logically subsumed by the condition above
                # (row[1] == current[2] satisfies row[1] <= current[2]), but is
                # kept for clarity.
                current[2] = int(row[2])
            else:
                # Disjoint: emit the completed interval and start a new one
                merged_rows.append(current.copy())
                current = row.copy()
        # Emit the final interval
        merged_rows.append(current.copy())
    if not merged_rows:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    out = pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    return out


def build_blacklist_from_bedgraphs(
    inputs: Sequence[str],
    output_path: str,
    zscore_cutoff: float | None = None,
    top_quantile: float = 0.95,
) -> None:
    """
    Load one or more per-chromosome bedgraphs, flag rows with NaN noise or high noise
    signal, and emit a blacklist BED file with merged intervals.

    Two flagging modes are available:

    **Z-score mode** (``zscore_cutoff`` is not None):
        Compute z-scores of the noise values across all loaded bedgraphs:
            z = (noise - mean) / std
        Flag bins with z ≥ zscore_cutoff. This mode is scale-invariant: the same
        cutoff applies regardless of the absolute noise level.
        If std == 0 (all values identical), bins at or above the mean are flagged.

    **Quantile mode** (``zscore_cutoff`` is None, default):
        Flag bins whose noise value is at or above the ``top_quantile`` percentile
        across all loaded bedgraphs. The default threshold of 0.95 flags the top 5%
        of bins. Use a higher value (e.g. 0.99) for a more conservative blacklist.

    In both modes, bins with non-finite noise (NaN or Inf) are also always flagged —
    these correspond to bins that were too sparse for any noise estimate.

    Flagged intervals are merged per chromosome using ``_merge_intervals()`` so that
    adjacent flagged bins appear as a single BED entry rather than many individual lines.

    Parameters
    ----------
    inputs : Sequence[str]
        One or more file paths or glob patterns pointing to per-chromosome bedgraph
        files (e.g. ``["chr1_10000.bedgraph", "chr*.bedgraph"]``).
    output_path : str
        Path for the output BED file (created; parent directories are created if absent).
    zscore_cutoff : float or None
        Z-score threshold for flagging. If None, ``top_quantile`` is used instead.
    top_quantile : float
        Quantile threshold (0–1) for flagging (default 0.95 = top 5%).
        Only used when ``zscore_cutoff`` is None.

    Raises
    ------
    ValueError
        If no bedgraph files are found or all loaded files are empty.
    """
    paths = _gather_paths(inputs)
    if not paths:
        raise ValueError("No bedgraph files found for provided inputs")

    frames = []
    for path in paths:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["chrom", "start", "end", "noise"],
            dtype={"chrom": str},
            engine="c",
        )
        df["source"] = os.path.basename(path)
        # Coerce noise column to float; non-numeric values (e.g. "nan", "NaN") become NaN
        df["noise"] = pd.to_numeric(df["noise"], errors="coerce")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("No rows found in bedgraph inputs")

    # Always flag non-finite bins (NaN/Inf = no reliable noise estimate)
    flagged = ~np.isfinite(data["noise"])
    valid_mask = np.isfinite(data["noise"])
    valid_values = data.loc[valid_mask, "noise"]

    if not valid_values.empty:
        if zscore_cutoff is not None:
            # Z-score mode: flag bins that deviate far above the mean
            mean = float(valid_values.mean())
            std = float(valid_values.std(ddof=0))
            if std <= 0:
                # Degenerate: all values are identical. Flag anything at or above mean.
                thresh_mask = valid_values >= mean
            else:
                zscores = (valid_values - mean) / std
                thresh_mask = zscores >= float(zscore_cutoff)
        else:
            # Quantile mode: flag bins at or above the Nth percentile
            quantile_value = float(np.nanquantile(valid_values, float(top_quantile)))
            thresh_mask = valid_values >= quantile_value
        flagged.loc[valid_mask] |= thresh_mask

    flagged_df = data.loc[flagged, ["chrom", "start", "end"]].copy()
    if flagged_df.empty:
        merged = pd.DataFrame(columns=["chrom", "start", "end"])
    else:
        merged = _merge_intervals(flagged_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    # Write as tab-separated BED (no header, no index) — compatible with bedtools and IGV
    merged.to_csv(output_path, sep="\t", header=False, index=False)
