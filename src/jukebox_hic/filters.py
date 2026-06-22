#!/usr/bin/env python
"""
Blacklist generation for Hi-C contact matrices using per-chromosome elbow detection.

Bins are flagged using the Kneedle algorithm applied independently to two metrics:

- **Contact density** (per-bin row sums from ``noise_fullmap``)
- **Noise values** (lag-1 autocovariance metric from ``noise_fullmap``)

The union rule flags a bin when it falls outside the data-driven threshold in
either metric.  Flagged intervals are merged into a BED-format blacklist.

Main entry points:

- ``detect_elbow_thresholds()``            — compute per-chromosome thresholds
- ``build_blacklist_from_elbow_thresholds()`` — run detection and write BED
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d



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



# ---------------------------------------------------------------------------
# Elbow detection helpers (Kneedle method)
# ---------------------------------------------------------------------------

def _apply_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Apply a variance-stabilising transform before Kneedle normalisation."""
    if transform == "sqrt":
        return np.sqrt(np.clip(y, 0.0, None))
    if transform == "cbrt":
        return np.cbrt(y)
    if transform == "log1p":
        return np.log1p(np.clip(y, 0.0, None))
    return y.astype(float)  # "none"


def _kneedle(values: np.ndarray, smooth_sigma: float, transform: str = "none") -> int:
    """
    Core Kneedle on a sorted ascending 1-D array.

    Returns the local index at which the perpendicular distance from the
    normalised curve to the unit diagonal y = x is maximised.  ``transform``
    is applied after Gaussian smoothing so right-skewed distributions are
    spread before normalisation; the returned index is into the original array.
    """
    y = gaussian_filter1d(values.astype(float), sigma=max(1.0, float(smooth_sigma)))
    y = _apply_transform(y, transform)
    span = float(y[-1]) - float(y[0])
    if span < 1e-12:
        return len(values) - 1
    y_n = (y - y[0]) / span
    x_n = np.linspace(0.0, 1.0, len(y))
    return int(np.argmax(np.abs(y_n - x_n) / np.sqrt(2.0)))


def detect_upper_elbow(
    values: np.ndarray,
    search_frac: float = 0.40,
    smooth_sigma: float = 10.0,
    transform: str = "none",
) -> int:
    """
    Return the global index of the upper (high-value) elbow in a sorted array.

    Restricts the Kneedle search to the top ``search_frac`` fraction so that
    normalisation spans the exponential spike region rather than the gradual bulk.
    """
    n = len(values)
    start = max(0, int(n * (1.0 - search_frac)))
    return start + _kneedle(values[start:], smooth_sigma, transform)


def detect_lower_elbow(
    values: np.ndarray,
    search_frac: float = 0.30,
    smooth_sigma: float = 10.0,
    transform: str = "none",
) -> int:
    """
    Return the global index of the lower (near-zero) elbow in a sorted array.

    Restricts the Kneedle search to the bottom ``search_frac`` fraction to find
    where near-zero / empty bins transition into the normal population.
    """
    end = max(3, int(len(values) * search_frac))
    return _kneedle(values[:end], smooth_sigma, transform)


def _analyse_chrom_elbow(
    d_sorted: np.ndarray,
    n_sorted: np.ndarray,
    d_upper_frac: float,
    d_lower_frac: float,
    n_upper_frac: float,
    n_lower_frac: float,
    sigma: float,
    d_transform: str = "none",
    n_transform: str = "sqrt",
) -> Dict:
    """Compute all four elbow thresholds for one chromosome."""

    def _threshold(arr: np.ndarray, idx: int) -> Dict:
        return {"value": float(arr[idx]), "pct": 100.0 * idx / len(arr)}

    d_lo = detect_lower_elbow(d_sorted, d_lower_frac, sigma, d_transform)
    d_hi = detect_upper_elbow(d_sorted, d_upper_frac, sigma, d_transform)
    n_lo = detect_lower_elbow(n_sorted, n_lower_frac, sigma, n_transform)
    n_hi = detect_upper_elbow(n_sorted, n_upper_frac, sigma, n_transform)

    t: Dict = {}
    for k, v in _threshold(d_sorted, d_lo).items():
        t[f"density_lower_{k}"] = v
    for k, v in _threshold(d_sorted, d_hi).items():
        t[f"density_upper_{k}"] = v
    for k, v in _threshold(n_sorted, n_lo).items():
        t[f"noise_lower_{k}"] = v
    for k, v in _threshold(n_sorted, n_hi).items():
        t[f"noise_upper_{k}"] = v
    t["d_lo_idx"] = d_lo
    t["d_hi_idx"] = d_hi
    t["n_lo_idx"] = n_lo
    t["n_hi_idx"] = n_hi
    return t


def _canonical_chrom_key(c: str):
    """Sort key that places chr1–22 before chrX/Y/M, decoys last."""
    num_str = (
        c.replace("chr", "")
         .replace("X", "23")
         .replace("Y", "24")
         .replace("M", "25")
         .split("_")[0]
    )
    return (not c.startswith("chr"), int(num_str) if num_str.isdigit() else 99, c)


# ---------------------------------------------------------------------------
# Elbow-based blacklist — public API
# ---------------------------------------------------------------------------

_ELBOW_TSV_COLS = [
    "chrom", "n_bins",
    "density_lower_value", "density_lower_pct",
    "density_upper_value", "density_upper_pct",
    "noise_lower_value",   "noise_lower_pct",
    "noise_upper_value",   "noise_upper_pct",
]


def detect_elbow_thresholds(
    density_bedgraph: str,
    noise_bedgraph: str,
    smooth_sigma: float = 10.0,
    density_upper_frac: float = 0.01,
    density_lower_frac: float = 0.01,
    noise_upper_frac: float = 0.10,
    noise_lower_frac: float = 0.01,
    density_transform: str = "none",
    noise_transform: str = "sqrt",
) -> Tuple[pd.DataFrame, dict]:
    """
    Run per-chromosome dual-metric elbow detection on density and noise bedgraphs.

    Returns
    -------
    thresholds_df : pd.DataFrame
        One row per chromosome with columns defined by ``_ELBOW_TSV_COLS``.
    per_chrom_curves : dict
        ``chrom → (d_sorted, n_sorted, d_lo_idx, d_hi_idx, n_lo_idx, n_hi_idx)``
        passed to ``figures.plot_elbow_figure()`` for the diagnostic figure.
    """
    from .noise_to_weights import _load_bedgraph, _is_decoy_chrom

    d_df = _load_bedgraph(density_bedgraph)
    n_df = _load_bedgraph(noise_bedgraph)
    d_df["value"] = pd.to_numeric(d_df["value"], errors="coerce")
    n_df["value"] = pd.to_numeric(n_df["value"], errors="coerce")

    chroms = [
        c for c in sorted(
            set(d_df["chrom"].unique()) & set(n_df["chrom"].unique()),
            key=_canonical_chrom_key,
        )
        if not _is_decoy_chrom(c)
    ]

    results = []
    per_chrom_curves: dict = {}

    for chrom in chroms:
        d_raw = d_df.loc[d_df["chrom"] == chrom, "value"].to_numpy(float)
        n_raw = n_df.loc[n_df["chrom"] == chrom, "value"].to_numpy(float)
        d_sorted = np.sort(d_raw[np.isfinite(d_raw)])
        n_sorted = np.sort(n_raw[np.isfinite(n_raw)])

        if len(d_sorted) < 4 or len(n_sorted) < 4:
            continue

        t = _analyse_chrom_elbow(
            d_sorted, n_sorted,
            d_upper_frac=density_upper_frac,
            d_lower_frac=density_lower_frac,
            n_upper_frac=noise_upper_frac,
            n_lower_frac=noise_lower_frac,
            sigma=smooth_sigma,
            d_transform=density_transform,
            n_transform=noise_transform,
        )
        results.append({"chrom": chrom, "n_bins": len(d_sorted), **t})
        per_chrom_curves[chrom] = (
            d_sorted, n_sorted,
            t["d_lo_idx"], t["d_hi_idx"],
            t["n_lo_idx"], t["n_hi_idx"],
        )

    tsv_rows = [{k: r[k] for k in _ELBOW_TSV_COLS} for r in results]
    return pd.DataFrame(tsv_rows, columns=_ELBOW_TSV_COLS), per_chrom_curves


def build_blacklist_from_elbow_thresholds(
    density_bedgraph: str,
    noise_bedgraph: str,
    output_path: str,
    smooth_sigma: float = 10.0,
    density_upper_frac: float = 0.01,
    density_lower_frac: float = 0.01,
    noise_upper_frac: float = 0.10,
    noise_lower_frac: float = 0.01,
    density_transform: str = "none",
    noise_transform: str = "sqrt",
    thresholds_df: Optional[pd.DataFrame] = None,
    require_both_metrics: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a BED blacklist using per-chromosome elbow thresholds.

    If ``thresholds_df`` is None, ``detect_elbow_thresholds()`` is called first.

    Two flagging rules are available via ``require_both_metrics``:

    Union rule (default, ``require_both_metrics=False``)::

        flagged = non-finite(density) OR non-finite(noise)
               OR density <= density_lower_value
               OR density >= density_upper_value
               OR noise   <= noise_lower_value
               OR noise   >= noise_upper_value

    Intersection rule (``require_both_metrics=True``)::

        density_oob = density <= density_lower_value OR density >= density_upper_value
        noise_oob   = noise   <= noise_lower_value   OR noise   >= noise_upper_value
        flagged     = non-finite(density) OR non-finite(noise)
                   OR (density_oob AND noise_oob)

    The intersection rule is less conservative: a bin must be out-of-bounds for
    both metrics simultaneously to be blacklisted.  Non-finite bins are always
    flagged regardless of rule.

    Returns
    -------
    (thresholds_df, per_chrom_curves)
        The caller can save ``thresholds_df`` as a TSV and pass both to
        ``figures.plot_elbow_figure()`` without re-running detection.
    """
    from .noise_to_weights import _load_bedgraph

    per_chrom_curves: dict = {}
    if thresholds_df is None:
        thresholds_df, per_chrom_curves = detect_elbow_thresholds(
            density_bedgraph=density_bedgraph,
            noise_bedgraph=noise_bedgraph,
            smooth_sigma=smooth_sigma,
            density_upper_frac=density_upper_frac,
            density_lower_frac=density_lower_frac,
            noise_upper_frac=noise_upper_frac,
            noise_lower_frac=noise_lower_frac,
            density_transform=density_transform,
            noise_transform=noise_transform,
        )

    d_df = _load_bedgraph(density_bedgraph)
    n_df = _load_bedgraph(noise_bedgraph)
    d_df["value"] = pd.to_numeric(d_df["value"], errors="coerce")
    n_df["value"] = pd.to_numeric(n_df["value"], errors="coerce")

    # Join density and noise on genomic coordinates
    combined = d_df.rename(columns={"value": "density"}).merge(
        n_df[["chrom", "start", "end", "value"]].rename(columns={"value": "noise"}),
        on=["chrom", "start", "end"],
        how="outer",
    )

    # Always flag non-finite bins
    flagged = ~np.isfinite(combined["density"]) | ~np.isfinite(combined["noise"])

    # Per-chromosome threshold application
    thresh_by_chrom = thresholds_df.set_index("chrom")
    for chrom in thresh_by_chrom.index:
        row = thresh_by_chrom.loc[chrom]
        mask = combined["chrom"] == chrom
        d_lo = float(row["density_lower_value"])
        d_hi = float(row["density_upper_value"])
        n_lo = float(row["noise_lower_value"])
        n_hi = float(row["noise_upper_value"])
        density_oob = (
            (combined.loc[mask, "density"] <= d_lo) |
            (combined.loc[mask, "density"] >= d_hi)
        )
        noise_oob = (
            (combined.loc[mask, "noise"] <= n_lo) |
            (combined.loc[mask, "noise"] >= n_hi)
        )
        if require_both_metrics:
            flagged.loc[mask] |= density_oob & noise_oob
        else:
            flagged.loc[mask] |= density_oob | noise_oob

    flagged_df = combined.loc[flagged, ["chrom", "start", "end"]].copy()
    merged = _merge_intervals(flagged_df) if not flagged_df.empty else pd.DataFrame(
        columns=["chrom", "start", "end"]
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    merged.to_csv(output_path, sep="\t", header=False, index=False)

    return thresholds_df, per_chrom_curves
