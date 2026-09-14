#!/usr/bin/env python
"""
Blacklist generation for Hi-C contact matrices using per-chromosome elbow detection.

Bins are flagged using the Kneedle algorithm applied independently to two metrics:

- **Contact density** (per-bin row sums from the ``noise-bedgraph`` command)
- **Noise values** (lag-1 autocovariance metric from the ``noise-bedgraph`` command)

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

    2. **Neither**: ``row.start > current.end``
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


def _stratified_noise_thresholds(
    density: np.ndarray,
    noise: np.ndarray,
    n_strata: int,
    noise_upper_frac: float,
    smooth_sigma: float,
    noise_transform: str,
    min_stratum_size: int = 30,
) -> np.ndarray:
    """
    Per-bin noise upper-threshold, conditioned on local density.

    Bins are grouped into ``n_strata`` equal-count groups by sorted density, then
    ``detect_upper_elbow()`` is run independently on each group's noise values —
    the same Kneedle logic the ``noise-high`` rule uses, just localized to bins of
    similar density instead of the whole chromosome. This lets a bin whose density
    is low (and whose noise is merely typical for that density, e.g. a real but
    rare loop in a sparsely-sequenced region) avoid being judged against noisier,
    higher-density bins it isn't comparable to.

    A stratum with fewer than ``min_stratum_size`` bins (short chromosome, coarse
    resolution) falls back to the chromosome-wide (unconditioned) noise-high
    threshold, since Kneedle is unstable on very small arrays.

    Parameters
    ----------
    density, noise : np.ndarray
        Finite per-bin values for one chromosome, same order and length.
    n_strata : int
        Number of equal-count density groups to split bins into.
    noise_upper_frac, smooth_sigma, noise_transform
        Passed through to ``detect_upper_elbow()`` for both the fallback
        threshold and each stratum's local threshold.
    min_stratum_size : int
        Minimum bins a stratum needs before it gets its own local threshold.

    Returns
    -------
    np.ndarray
        Same length/order as the input: threshold[i] is the noise value bin i
        must meet or exceed to be flagged.
    """
    n = len(density)
    thresholds = np.empty(n, dtype=float)

    # Chromosome-wide fallback — identical to what the "noise-high" rule computes.
    n_sorted_global = np.sort(noise)
    fallback = float(n_sorted_global[
        detect_upper_elbow(n_sorted_global, noise_upper_frac, smooth_sigma, noise_transform)
    ])

    order = np.argsort(density)
    for group in np.array_split(order, max(1, n_strata)):
        if len(group) < min_stratum_size:
            thresholds[group] = fallback
            continue
        stratum_sorted = np.sort(noise[group])
        local_idx = detect_upper_elbow(stratum_sorted, noise_upper_frac, smooth_sigma, noise_transform)
        thresholds[group] = float(stratum_sorted[local_idx])

    return thresholds


def _density_conditioned_residuals(
    density: np.ndarray,
    noise: np.ndarray,
    fit_window: int = 0,
) -> np.ndarray:
    """
    Residual of each bin's noise against the noise expected at its density.

    Instead of thresholding noise directly, the density→noise trend is first
    estimated from the data itself: bins are ordered by density and a centred
    running median of log10(noise) gives the typical noise for bins of similar
    density. The returned residual, log10(noise) − trend, is what the caller
    thresholds, so "noisy" means "noisier than other bins sequenced this
    deeply" rather than "noisy in absolute terms".

    Working in log space matters because the density→noise relationship is
    roughly power-law; a running median (rather than a mean) keeps the trend
    from being dragged upward by the very outliers the blacklist is looking for.

    This is the property the per-stratum elbow approach (``"density-stratified"``)
    lacks. An elbow computed inside each density stratum always flags that
    stratum's top fraction, whether or not anything is wrong with it, so
    well-sampled strata donate bins to the blacklist purely as an artefact of
    the method. Here a group of bins that all sit on the trend produces small
    residuals and contributes no flags at all.

    Parameters
    ----------
    density, noise : np.ndarray
        Finite per-bin values for one chromosome, same order and length. Values
        are floored at a small positive constant before the log, so zero-contact
        or zero-noise bins are handled without producing -inf.
    fit_window : int
        Width, in bins, of the running-median window used to estimate the trend.
        0 selects a width from the chromosome's bin count. A wider window gives
        a smoother, stiffer trend; a narrower one tracks local structure more
        closely and therefore leaves smaller residuals.

    Returns
    -------
    np.ndarray
        Residual per bin, same length and order as the input.
    """
    # Floor before the log so empty bins do not become -inf. 1e-12 matches the
    # floor used in reference._preprocess_noise_track.
    x = np.log10(np.clip(density.astype(float), 1e-12, None))
    y = np.log10(np.clip(noise.astype(float), 1e-12, None))

    if fit_window > 0:
        window = int(fit_window)
    else:
        # ~2% of the chromosome's bins, bounded so the trend is neither noisy
        # on short chromosomes nor over-smoothed on long ones.
        window = int(np.clip(len(y) // 50, 51, 2001))
    window = max(3, window | 1)  # force odd so the window can be centred

    # Estimate the trend in density order, then map it back to bin order.
    order = np.argsort(x)
    trend_sorted = (
        pd.Series(y[order])
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    trend = np.empty_like(trend_sorted)
    trend[order] = trend_sorted

    return y - trend


def _robust_centre_scale(residual: np.ndarray) -> Tuple[float, float]:
    """
    Median and robust standard deviation of a residual vector.

    Spread is the median absolute deviation scaled by 1.4826, which makes it
    match the standard deviation for normally distributed data. Both the centre
    and the scale are medians, so the outliers being searched for do not inflate
    the statistics that are supposed to catch them.

    Factored out so that ``_robust_residual_threshold()`` (used by the
    density-residual blacklist rule) and ``_residual_robust_z()`` (used by the
    noise gradient) are guaranteed to agree: thresholding the gradient at k must
    select exactly the bins the blacklist flags at the same k, and that contract
    would silently break if the two computed their centre or scale separately.

    The MAD breaks down when more than half the residuals are *exactly* equal,
    which happens on near-empty chromosomes: K562 chrY has 698 measurable bins
    out of 5723, their contact counts take only a handful of discrete values, and
    over half the residuals land on precisely 0.0. The MAD is then floating-point
    dust (~1e-15) rather than zero, so it slips past a ``<= 0`` guard and
    z-scores explode to ~1e15 — and the blacklist, dividing the same dust into
    its threshold, flags a quarter of the measurable bins for no real reason.

    So the scale falls back through progressively less robust estimators, each
    used only when the previous one is degenerate: MAD, then the IQR (which
    survives a smaller atom at the median), then the standard deviation. The
    fallbacks cannot fire on well-behaved data, where the MAD is orders of
    magnitude above the floor, so normal chromosomes are untouched and the
    calibration of k is unchanged. Falling back to the non-robust standard
    deviation is the conservative direction: outliers inflate it, so a
    degenerate chromosome under-flags rather than over-flags.
    """
    centre = float(np.median(residual))

    # Below this a "spread" is numerical dust, not signal: residuals are log10
    # differences, so 1e-9 is a few parts per billion of the noise value.
    negligible = 1e-9

    sigma = 1.4826 * float(np.median(np.abs(residual - centre)))
    if sigma > negligible:
        return centre, sigma

    q75, q25 = np.percentile(residual, [75, 25])
    sigma = float(q75 - q25) / 1.349          # IQR → sigma for a normal
    if sigma > negligible:
        return centre, sigma

    sigma = float(np.std(residual))
    return centre, sigma if sigma > negligible else 0.0


def _robust_residual_threshold(residual: np.ndarray, k: float) -> float:
    """
    Cutoff at *k* robust standard deviations above the median residual.

    Unlike an elbow, this returns a cutoff that nothing need exceed: on a
    chromosome whose bins all sit on the density→noise trend, no residual
    reaches k robust sigma and no bins are flagged.

    A degenerate all-identical residual vector gives MAD = 0; the threshold then
    falls back to just above the median so that only strictly larger residuals
    are flagged, rather than the whole chromosome tying with the cutoff.
    """
    centre, sigma = _robust_centre_scale(residual)
    if sigma <= 0.0:
        return np.nextafter(centre, np.inf)
    return centre + float(k) * sigma


def _residual_robust_z(residual: np.ndarray) -> np.ndarray:
    """
    Per-bin robust z-score: how many robust sigma above the median each residual sits.

    This is the continuous form of ``_robust_residual_threshold()``. Comparing
    ``z >= k`` is equivalent to comparing ``residual >= _robust_residual_threshold(residual, k)``,
    which is what lets the noise gradient and the density-residual blacklist be
    two views of one quantity rather than two independent calculations.

    The degenerate MAD = 0 case mirrors the threshold's fallback exactly: only
    strictly-greater residuals clear the bar, so they get +inf and everything
    else gets 0, making ``z >= k`` true for the same bins at any finite k > 0.
    """
    centre, sigma = _robust_centre_scale(residual)
    if sigma <= 0.0:
        z = np.zeros(len(residual), dtype=float)
        z[residual > centre] = np.inf
        return z
    return (residual - centre) / sigma


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
    rule: Optional[str] = None,
    density_strata: int = 5,
    density_fit_window: int = 0,
    density_residual_k: float = 4.0,
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

    Non-finite bins are always flagged regardless of rule: a bin with no computable
    noise estimate is unusable whatever the thresholds say.

    ``rule`` selects the flagging logic explicitly and, when given, overrides
    ``require_both_metrics``:

    ``"noise-high"`` (recommended)
        non-finite(density) OR non-finite(noise) OR noise >= noise_upper_value.

        Only the extreme-disorder tail, and no density term. On the corrected
        tracks this is the only rule whose flagged bins are reliably noisier than
        background: median log2(NME) is +7.7 across five cell lines, against -5.2
        for the intersection, which is negative in every one of them. It also adds
        under 1% of the genome on top of the unmappability mask instead of 8-10%.

        The two tails are worth separating because they are opposite populations.
        The low tail flags bins that are *unusually smooth*, and mixing them into
        one median makes the statistic unstable -- in HEPG2 the low tail
        outnumbers the high tail and flips the sign of the whole comparison.

    ``"union"``        density OR noise out-of-bounds (either tail). Legacy default.
    ``"intersection"`` density AND noise out-of-bounds. Legacy option.
    ``"mask"``         non-finite bins only: a pure mappability mask, no thresholds.

    ``"density-residual"``
        non-finite(density) OR non-finite(noise) OR residual >= k robust sigma,
        where residual is log10(noise) minus the noise expected at that bin's
        density (see ``_density_conditioned_residuals()``) and the cutoff is
        ``density_residual_k`` median-absolute-deviations above the median
        residual (see ``_robust_residual_threshold()``).

        This is the one rule here that does not use an elbow, and the departure
        is the point: Kneedle always returns an index inside its search window,
        so it flags a share of the data whether or not anything is wrong. A
        robust-sigma cutoff can come back empty on a clean chromosome and flag
        heavily on a bad one, which is what "is this bin noisy for its density?"
        actually requires. ``noise_upper_frac``/``noise_transform`` are unused
        by this rule; ``density_residual_k`` is its sensitivity knob.

        This is the density-conditioned rule to prefer over ``"density-stratified"``.
        Both ask whether a bin is noisy *for its density*, but this one thresholds
        one distribution, so a population of bins that all sit on the density→noise
        trend contributes nothing to the blacklist. ``"density-stratified"`` instead
        runs an elbow inside each stratum and therefore always flags each stratum's
        top fraction — on a six-cell-line benchmark that put it below the plain
        unmappability mask on noise enrichment while masking more of the genome.

    ``"density-stratified"``
        non-finite(density) OR non-finite(noise) OR noise >= local_noise_upper_value,
        where the noise upper threshold is computed independently within each of
        ``density_strata`` equal-count density groups (see
        ``_stratified_noise_thresholds()``) instead of once for the whole
        chromosome. Same upper-tail-only philosophy as ``"noise-high"`` — and the
        same reason for leaving the low tail alone (it's a different, unusually-smooth
        population, not a disorder tail) — but a bin is only flagged when its noise
        is anomalous *relative to bins of similar density*, so density and noise are
        no longer independent axes: a low-density bin with noise typical for that
        density (e.g. a real but rare loop in a sparsely-sequenced region) is not
        penalized just for being low-density.

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
        effective = rule if rule is not None else (
            "intersection" if require_both_metrics else "union"
        )
        if effective == "noise-high":
            flagged.loc[mask] |= combined.loc[mask, "noise"] >= n_hi
        elif effective == "mask":
            pass          # non-finite bins were flagged above; nothing more to add
        elif effective == "intersection":
            flagged.loc[mask] |= density_oob & noise_oob
        elif effective == "union":
            flagged.loc[mask] |= density_oob | noise_oob
        elif effective == "density-residual":
            d_vals = combined.loc[mask, "density"].to_numpy(float)
            n_vals = combined.loc[mask, "noise"].to_numpy(float)
            finite = np.isfinite(d_vals) & np.isfinite(n_vals)
            local_flag = np.zeros(len(d_vals), dtype=bool)
            if finite.any():
                residual = _density_conditioned_residuals(
                    d_vals[finite], n_vals[finite], density_fit_window,
                )
                # Deliberately NOT an elbow. Kneedle always returns an index
                # inside its search window, so it flags a share of the data
                # whether or not anything is actually wrong — that is what made
                # "density-stratified" flag well-behaved bins. A fixed multiple
                # of the residual spread can express "nothing here deviates",
                # which is the whole point of conditioning on density.
                local_flag[finite] = residual >= _robust_residual_threshold(
                    residual, density_residual_k
                )
            flagged.loc[mask] |= local_flag
        elif effective == "density-stratified":
            d_vals = combined.loc[mask, "density"].to_numpy(float)
            n_vals = combined.loc[mask, "noise"].to_numpy(float)
            finite = np.isfinite(d_vals) & np.isfinite(n_vals)
            local_flag = np.zeros(len(d_vals), dtype=bool)
            if finite.any():
                local_thresholds = _stratified_noise_thresholds(
                    d_vals[finite], n_vals[finite], density_strata,
                    noise_upper_frac, smooth_sigma, noise_transform,
                )
                local_flag[finite] = n_vals[finite] >= local_thresholds
            flagged.loc[mask] |= local_flag
        else:
            raise ValueError(
                f"unknown blacklist rule {effective!r}; expected one of "
                "'noise-high', 'density-residual', 'union', 'intersection', "
                "'mask', 'density-stratified'"
            )

    flagged_df = combined.loc[flagged, ["chrom", "start", "end"]].copy()
    merged = _merge_intervals(flagged_df) if not flagged_df.empty else pd.DataFrame(
        columns=["chrom", "start", "end"]
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    merged.to_csv(output_path, sep="\t", header=False, index=False)

    return thresholds_df, per_chrom_curves
