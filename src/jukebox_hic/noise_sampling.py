#!/usr/bin/env python
"""
Sampled row-based Hi-C noise estimation.

Instead of computing noise for every genomic bin (which ``noise_fullmap`` does),
this module *samples* a fraction of rows from each chromosome's contact matrix,
computes per-row noise metrics, and aggregates them into per-chromosome summaries.

Why sampling?
-------------
Full-genome noise calculation at multiple resolutions is computationally intensive.
Sampling a representative subset of rows (default: all rows, configurable via
``sample_fraction``) gives a fast, robust estimate of the median noise level and
autocorrelation structure, which is then used to:
- Compute the z-map quality score (via ``reference.compute_zmap``).
- Power the sequencing advisor (via ``reference.sequencing_advisor``).
- Optionally simulate what noise would look like at lower sequencing depths
  (via ``subsample_ratios``) to model sequencing depth effects.

Main entry point: ``compute_sampled_noise()``.
"""
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.tsa.stattools import acf, acovf

from .backends import ChromMatrix, read_chrom_sizes, select_provider, total_contacts
from .metrics import collect_memory_usage_mb
from . import reference


def _default_window_bp(resolution: int) -> int:
    """
    Choose a sensible default genomic window size (in bp) for noise estimation.

    The window defines how far from the diagonal the noise calculation looks —
    i.e. the maximum genomic distance between contact pairs included in the noise
    metric for a given bin. Larger windows capture more long-range signal but are
    slower; smaller windows focus on local structure.

    The defaults are chosen empirically to capture enough contacts for a stable
    autocorrelation estimate while staying close enough to the diagonal that the
    expected contact model (distance-decay) is reliable:

    - ≥100 kb resolution → 10 Mbp window  (~100 bins)
    - ≥50 kb resolution  → 6 Mbp window   (~120 bins)
    - ≥25 kb resolution  → 4 Mbp window   (~160 bins)
    - <25 kb resolution  → 1 Mbp window   (still ~40 bins at 25 kb)

    These correspond roughly to 100–160 bins on each side of the diagonal, which
    is the range where the distance-decay signal is strong and autocorrelation can
    be estimated reliably.
    """
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _precompute_expectation(coo: sparse.coo_matrix) -> Tuple[np.ndarray, float]:
    """
    Build a lookup table of expected (average) contact counts by genomic distance.

    In Hi-C, contacts between bins that are close on the linear genome are far more
    frequent than contacts between distant bins — this is the well-known distance-decay
    (or "expected") curve. Dividing observed counts by this expected value normalises
    for the distance effect, making contact enrichment comparable across different
    genomic distances.

    This function computes the expected curve from the data itself:
    - For every non-zero contact (i, j), the lag = |i - j| (bin offset = genomic distance
      in bins).
    - All contacts at the same lag are averaged to get ``expected[lag]``.

    Parameters
    ----------
    coo : scipy.sparse.coo_matrix
        Sparse contact matrix in COO format. The ``row``, ``col``, and ``data`` arrays
        are iterated to compute per-lag averages.

    Returns
    -------
    means : np.ndarray of shape (max_lag + 1,)
        ``means[k]`` is the average contact count across all pairs separated by k bins.
        Lags with zero observed contacts have ``means[k] == 0``.
    default : float
        Fallback expected value used when a lag has no observed data (i.e., ``means[k]``
        would be zero). Computed as the mean of all non-zero lag means, so it represents
        a "typical" contact density for the chromosome.
    """
    if coo.data.size == 0:
        return np.zeros(1, dtype=float), 0.0
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    data = coo.data.astype(float)
    # lag = bin distance between the two interacting loci
    lags = np.abs(cols - rows)
    max_lag = int(lags.max())
    # np.bincount counts occurrences; with weights= it sums values instead.
    sums = np.bincount(lags, weights=data, minlength=max_lag + 1)
    counts = np.bincount(lags, minlength=max_lag + 1)
    # Divide sums by counts only where counts > 0 to avoid divide-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    default = float(means[counts > 0].mean()) if np.count_nonzero(counts) else 0.0
    return means, default


def _expected_for_lag(expected_lookup: np.ndarray, default: float, lag: int) -> float:
    """
    Retrieve the expected contact count for a specific genomic distance (lag).

    Parameters
    ----------
    expected_lookup : np.ndarray
        The means array returned by ``_precompute_expectation()``.
        ``expected_lookup[k]`` is the average contact count at lag k bins.
    default : float
        Fallback value used when the lag is beyond the array or has no data.
    lag : int
        Genomic distance in bins (|bin_i - bin_j|).

    Returns
    -------
    float
        Expected contact count at this lag. Falls back to ``default`` when the lag
        exceeds the precomputed range or when the stored value is zero (which would
        cause division-by-zero downstream).
    """
    if lag < len(expected_lookup):
        val = expected_lookup[lag]
        return float(val) if val > 0 else default
    return default


def _row_window_vectors(
    matrices: ChromMatrix,
    row_idx: int,
    bindist_bins: int,
    expected_lookup: np.ndarray,
    default_expected: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the observed and expected contact vectors for a single genomic bin (row).

    For a given bin ``row_idx``, this function collects all contacts within
    ``bindist_bins`` bins of the diagonal — i.e. contacts to all bins j where
    |row_idx - j| ≤ bindist_bins. These are assembled into fixed-length vectors
    centred on the diagonal, where position ``bindist_bins`` (the middle) corresponds
    to the bin itself and positions further from centre correspond to larger genomic
    distances.

    The window vector has length ``bindist_bins * 2``:
    - Positions 0 … bindist_bins−1  : contacts *upstream* (j < row_idx), i.e.
      bins that come before row_idx on the chromosome. Fetched from the CSC matrix
      (column slicing is efficient in CSC format).
    - Positions bindist_bins … 2*bindist_bins−1 : contacts *downstream* (j > row_idx),
      i.e. bins that come after row_idx. Fetched from the CSR matrix (row slicing).

    The upstream and downstream halves share an asymmetric indexing scheme to keep
    the diagonal at the boundary between them rather than at the centre:
    - Upstream position: ``pos = bindist_bins - lag``   (lag 1 → pos bindist_bins-1)
    - Downstream position: ``pos = bindist_bins - 1 + lag``  (lag 1 → pos bindist_bins)

    Parameters
    ----------
    matrices : ChromMatrix
        Contact matrix for the chromosome (CSR and CSC formats used).
    row_idx : int
        The bin index whose noise is being computed.
    bindist_bins : int
        Maximum number of bins from the diagonal to include (= window_bp // resolution).
    expected_lookup : np.ndarray
        Distance-decay expected values from ``_precompute_expectation()``.
    default_expected : float
        Fallback expected value for lags with no data.

    Returns
    -------
    observed : np.ndarray of shape (bindist_bins * 2,)
        Raw contact counts from the Hi-C matrix. Positions with no observed contact
        remain 0.0.
    expected : np.ndarray of shape (bindist_bins * 2,)
        Expected (distance-decay) contact counts for each position. Initialised to
        ``default_expected`` everywhere; updated with the actual expected value where
        contacts exist.
    """
    window = bindist_bins * 2
    observed = np.zeros(window, dtype=float)
    # Initialise expected to the global default so positions with no contact still
    # have a meaningful denominator when computing observed/expected ratios.
    expected = np.full(window, default_expected, dtype=float)

    # --- Upstream contacts (j < row_idx): use CSC column slicing ---
    # getcol(row_idx) returns all rows i that contact column row_idx.
    # We only want entries where i < row_idx (upstream bins).
    col_vec = matrices.csc.getcol(row_idx)
    for src_idx, value in zip(col_vec.indices, col_vec.data):
        if src_idx >= row_idx:
            # Skip downstream or diagonal entries (handled by CSR below).
            continue
        lag = row_idx - int(src_idx)
        if lag > bindist_bins:
            continue
        pos = bindist_bins - lag  # map lag to upstream window position
        observed[pos] = float(value)
        expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

    # --- Downstream contacts (j > row_idx): use CSR row slicing ---
    # getrow(row_idx) returns all columns j that row_idx contacts.
    # We only want entries where j > row_idx (downstream bins).
    row_vec = matrices.csr.getrow(row_idx)
    for dst_idx, value in zip(row_vec.indices, row_vec.data):
        if dst_idx <= row_idx:
            # Skip upstream or diagonal entries (already handled above).
            continue
        lag = int(dst_idx) - row_idx
        if lag > bindist_bins:
            continue
        pos = bindist_bins - 1 + lag  # map lag to downstream window position
        observed[pos] = float(value)
        expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

    return observed, expected


def _compute_row_metrics(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> Dict[str, float]:
    """
    Compute noise and signal quality metrics for a single bin's contact window.

    The core noise measure is based on the **lag-1 autocovariance** of the
    observed/expected ratio series. Here is the reasoning:

    - We compute the ratio ``score[k] = (observed[k] + 1) / (expected[k] + 1)``
      (pseudo-count +1 prevents division by zero and stabilises sparse windows).
    - In a "noisy" bin, these scores fluctuate erratically — alternating between
      high and low values with no smooth spatial trend. This erratic fluctuation
      is captured by **low or negative lag-1 autocovariance**: adjacent positions
      along the window have uncorrelated (or anti-correlated) scores.
    - In a "clean" bin, scores vary smoothly with distance (following the expected
      decay), producing **high positive lag-1 autocovariance**.
    - The noise metric is therefore ``1 / |autocovariance_lag1|``:
      high autocovariance (smooth signal) → small noise value;
      low autocovariance (erratic signal) → large noise value.

    This is **not** classical signal autocorrelation; it is the raw autocovariance
    (``acovf``, not normalised), so the magnitude is in units of (score)².

    Additionally, the normalised autocorrelation coefficient at lag 1 (``acf``) is
    recorded for reporting, as well as basic statistics (mean, std, empty_ratio).

    Degenerate cases
    ----------------
    - ``finite.size < 2`` — fewer than two finite ratio values; autocorrelation is
      undefined. Returns NaN for noise_raw.
    - ``np.allclose(finite, finite[0])`` — all ratios are identical (constant series);
      variance is zero, so autocovariance is zero and the noise formula would divide
      by zero. Returns NaN for noise_raw (the bin is too uniform to measure noise).

    Parameters
    ----------
    observed_counts : np.ndarray
        Raw contact counts in the window (from ``_row_window_vectors()``).
    expected_counts : np.ndarray
        Distance-decay expected values for each window position.

    Returns
    -------
    dict with keys:
        noise_raw   : float — the primary noise metric (1 / |acovf_lag1|); NaN if
                      the window is too sparse or uniform to compute reliably.
        row_mean    : float — mean of the finite observed/expected ratios.
        row_std     : float — standard deviation of the finite ratios.
        acf         : float — normalised autocorrelation coefficient at lag 1.
        empty_ratio : float — fraction of window positions with zero observed count.
    """
    # --- Empty bin ratio: fraction of the window with no observed contacts ---
    empty_ratio = 1.0
    if observed_counts.size:
        non_empty = int(np.count_nonzero(observed_counts))
        empty_ratio = 1.0 - (non_empty / float(observed_counts.size))

    # --- Compute observed/expected ratio (pseudo-count added to both sides) ---
    scores = (observed_counts + 1.0) / (expected_counts + 1.0)
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]  # drop any Inf/NaN ratios

    # --- Handle degenerate cases where autocovariance is undefined ---
    if finite.size < 2 or np.allclose(finite, finite[0]):
        mean_val = float(finite.mean()) if finite.size else 0.0
        std_val = 0.0
        ac_val = 0.0
        noise_raw = float("nan")
    else:
        mean_val = float(finite.mean())
        try:
            std_val = float(statistics.stdev(finite.tolist()))
        except statistics.StatisticsError:
            std_val = 0.0

        # --- Core noise metric: 1 / |lag-1 autocovariance| ---
        # acovf returns the full autocovariance sequence: [var, cov_lag1, cov_lag2, ...]
        # We only request nlag=1 so we get [variance, lag-1 autocovariance].
        try:
            ac_vals = acovf(finite, nlag=1, fft=True)
            ac1 = float(ac_vals[1]) if len(ac_vals) > 1 else 0.0
            # ac1 == 0 means a perfectly alternating series; the noise formula would
            # divide by zero, so we return NaN (treated as unmeasurable noise).
            noise_raw = float("nan") if ac1 == 0 else 1.0 / abs(ac1)
        except Exception:
            noise_raw = float("nan")

        # --- Normalised ACF at lag 1 (for reporting, not used in noise_raw) ---
        try:
            ac_series = acf(finite, nlags=1, fft=True)
            ac_val = float(ac_series[1]) if len(ac_series) > 1 else 0.0
        except Exception:
            ac_val = 0.0

    return {
        "noise_raw": noise_raw,
        "row_mean": mean_val,
        "row_std": std_val,
        "acf": ac_val,
        "empty_ratio": empty_ratio,
    }


def compute_sampled_noise(
    hic_path: str,
    res: int,
    sample_fraction: float = 1.0,
    window_bp: Optional[int] = None,
    chrom_sizes_path: Optional[str] = None,
    out_dir: str = ".",
    norm: str = "none",
    cooler_selection: Optional[str] = None,
    min_mean_score: float = 0.2,
    min_nonzero_frac: float = 0.01,
    subsample_ratios: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    profile_path: Optional[str] = None,
    chroms: Optional[List[str]] = None,
    chunk_rows: Optional[int] = None,
    precomputed_contacts: Optional[Dict[str, float]] = None,
) -> None:
    """
    Sample a subset of rows, compute noise metrics, and optionally profile usage.

    For each chromosome, this function:
    1. Loads the contact matrix via the appropriate provider (hic or cooler).
    2. Precomputes the distance-decay expected curve (``_precompute_expectation``).
    3. Optionally samples a fraction of row bins (``sample_fraction < 1.0`` enables
       random row sampling; default is all rows).
    4. For each sampled row, extracts the contact window (``_row_window_vectors``)
       and filters rows that are too sparse or too low-signal to be informative:
       - ``min_mean_score``: minimum mean observed/expected ratio in the window.
         Rows below this are considered "empty" and skipped (default 0.2).
         This prevents noise calculation on bins that have essentially no contacts.
       - ``min_nonzero_frac``: minimum fraction of window positions with any contact.
         Rows below this threshold are also skipped (default 0.01 = 1% of positions).
    5. Computes row metrics (``_compute_row_metrics``) for each valid row.
    6. Optionally simulates lower sequencing depths by applying binomial downsampling:
       For each ``ratio`` in ``subsample_ratios``, the observed integer counts are
       subsampled using a binomial draw. Ratios are applied *incrementally* (from
       highest to lowest) to avoid re-drawing from the original data each time:
       ``counts_at_ratio = binomial(counts_at_prev_ratio, ratio / prev_ratio)``.
    7. Aggregates per-row metrics into per-chromosome summaries, computes z-map and
       sequencing advisor results, and writes outputs.

    Outputs written to ``out_dir``:
    - ``{chrom}_{res}.bedgraph`` — per-bin noise values for each chromosome
      (NaN for rows that failed quality filters).
    - ``subsample_summary.tsv`` — per-chromosome summary statistics (median noise,
      z-map score, sequencing advisor output, etc.), one row per (chrom, ratio) pair.

    Parameters
    ----------
    hic_path : str
        Path to the input .hic or cooler file.
    res : int
        Bin size in base pairs.
    sample_fraction : float
        Fraction of eligible row bins to evaluate (0 < fraction ≤ 1.0). A value of
        1.0 processes all rows; 0.5 randomly samples half.
    window_bp : int or None
        Genomic window size in bp (distance from diagonal). Defaults to
        ``_default_window_bp(res)``.
    chrom_sizes_path : str or None
        Optional path to chrom sizes TSV to restrict or override chromosomes.
    out_dir : str
        Directory for output files (created if it does not exist).
    norm : str
        Normalisation: ``"none"``, ``"balance"``, or path to a custom weight file.
    cooler_selection : str or None
        HDF5 group path for .scool or specific .mcool URI selection.
    min_mean_score : float
        Minimum mean observed/expected ratio for a row to be kept (default 0.2).
    min_nonzero_frac : float
        Minimum fraction of non-zero bins in the window (default 0.01).
    subsample_ratios : list of float or None
        Sequencing depth ratios to simulate (e.g. [1.0, 0.5, 0.25]). 1.0 is always
        included. Useful for modelling how noise changes with sequencing depth.
    seed : int or None
        Random seed for reproducible row sampling and subsampling.
    profile_path : str or None
        If provided, per-chromosome timing and memory usage is appended to a CSV
        at this path (columns: chrom, resolution, elapsed_s, peak_rss_mb, ...).
    chroms : list of str or None
        If provided, restrict analysis to these chromosome names.
    chunk_rows : int or None
        When set, each chromosome is processed in genomic bands of this many rows
        rather than loading the entire chromosome matrix at once.  Each band fetches
        a square submatrix ``[band_start − bindist_bins, band_end + bindist_bins]``
        so that every row in the band has its full contact window available.
        The distance-decay expected curve is recomputed from the local band matrix,
        which introduces a small per-band approximation but avoids ever holding the
        full chromosome matrix in memory.  Recommended value: 2000–5000 rows.
        ``None`` (default) preserves the original full-chromosome load behaviour.
    precomputed_contacts : dict or None
        Pre-computed total contact counts per chromosome (chromosome → float).
        When provided, the full-matrix load that was previously needed solely to call
        ``total_contacts()`` is skipped entirely, even in full-load mode.  The CLI
        automatically passes the contacts dict produced by ``_dump_contacts_once()``.
    """

    provider, backend = select_provider(hic_path, int(res), norm, cooler_selection)
    chrom_sizes = read_chrom_sizes(provider, chrom_sizes_path)
    if not chrom_sizes:
        raise ValueError("No chromosomes available for analysis")

    if chroms is not None:
        requested = set(chroms)
        missing = requested - set(chrom_sizes.keys())
        if missing:
            print(f"[WARN] Requested chromosomes not found: {sorted(missing)}")
        chrom_sizes = {c: chrom_sizes[c] for c in chroms if c in chrom_sizes}
        if not chrom_sizes:
            raise ValueError("None of the requested chromosomes are available")

    rng = np.random.default_rng(seed)

    # Always include ratio 1.0 (full depth). Sort descending so we downsample
    # incrementally from highest to lowest ratio (avoids resampling from full data).
    ratios = list(subsample_ratios) if subsample_ratios else [1.0]
    if 1.0 not in ratios:
        ratios.append(1.0)
    ratios = sorted(set(float(r) for r in ratios), reverse=True)

    window = window_bp or _default_window_bp(int(res))
    # Convert window size from bp to number of bins on each side of the diagonal.
    bindist_bins = max(1, window // int(res))

    summary_records: List[Dict[str, Any]] = []
    contacts_by_chrom: Dict[str, float] = {}
    chrom_bp_sizes: Dict[str, int] = {}
    profile_rows: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    psutil_peak_so_far, ru_peak_so_far = collect_memory_usage_mb()

    for chrom, chrom_len in chrom_sizes.items():
        current_rss_mb, current_ru_peak = collect_memory_usage_mb()
        psutil_peak_so_far = max(psutil_peak_so_far, current_rss_mb)
        ru_peak_so_far = max(ru_peak_so_far, current_ru_peak)
        start_psutil_peak = psutil_peak_so_far
        start_ru_peak = ru_peak_so_far
        chrom_start_time = time.perf_counter()

        n_bins = (chrom_len + int(res) - 1) // int(res)

        # --- Load matrix / get contact count ---
        # precomputed_contacts avoids a redundant full-matrix load when the caller
        # (CLI) has already counted contacts at a coarser resolution.
        matrices: Optional[ChromMatrix] = None
        if precomputed_contacts is not None and chrom in precomputed_contacts:
            chrom_contacts = float(precomputed_contacts[chrom])
            if chrom_contacts <= 0:
                continue
            chrom_bp_sizes[chrom] = chrom_len
            if chunk_rows is None:
                # Full-matrix mode needs the matrix for row iteration even when
                # contact count was precomputed at a coarser resolution.
                matrices = provider.fetch_chrom(chrom)
                if matrices is None or matrices.coo.nnz == 0:
                    continue
        else:
            # Full matrix needed: either for row iteration (chunk_rows is None)
            # or just for contact counting (chunk_rows set, no precomputed value).
            matrices = provider.fetch_chrom(chrom)
            if matrices is None or matrices.coo.nnz == 0:
                continue
            chrom_bp_sizes[chrom] = chrom_len
            chrom_contacts = total_contacts(matrices)
            if chunk_rows is not None:
                # Discard immediately after counting; bands will be fetched below.
                del matrices
                matrices = None

        contacts_by_chrom[chrom] = chrom_contacts

        # --- Set up expected lookup ---
        if matrices is not None:
            # Full matrix path: compute expected curve once from the whole chromosome.
            expected_lookup, default_expected = _precompute_expectation(matrices.coo)
            rowbins = matrices.coo.shape[0]
        else:
            # Band path: expected_lookup computed per-band inside the row loop.
            expected_lookup = np.zeros(1, dtype=float)
            default_expected = 0.0
            rowbins = n_bins
        # Skip the first and last ``bindist_bins`` rows: these edge bins have fewer
        # than ``bindist_bins`` neighbours on one side, making their window incomplete
        # and the noise estimate unreliable.
        start_idx = bindist_bins
        stop_idx = max(bindist_bins + 1, rowbins - bindist_bins)
        if stop_idx <= start_idx:
            continue

        candidate_rows = np.arange(start_idx, stop_idx)
        requested_n = len(candidate_rows)
        sample_all = True
        if float(sample_fraction) < 1.0:
            requested_n = max(1, int(len(candidate_rows) * float(sample_fraction)))
            sample_all = False

        # Shuffle row order for random sampling; use sequential order for full scan.
        if sample_all:
            row_order = candidate_rows
        else:
            row_order = rng.permutation(candidate_rows)

        selected_rows: List[int] = []   # rows that passed quality filters
        failed_rows = 0                  # rows rejected by quality filters
        exhausted = False               # True if we ran out of rows before reaching requested_n
        # chrom_bed stores (bin_index, noise_value) pairs for writing the per-chromosome bedgraph.
        chrom_bed: List[Tuple[int, float]] = []
        # ratio_stats accumulates per-row metrics for each downsampling ratio,
        # to be aggregated into per-chromosome summary statistics.
        ratio_stats = {
            ratio: {
                "noise_raw": [],
                "row_mean": [],
                "row_std": [],
                "acf": [],
                "empty_ratio": [],
            }
            for ratio in ratios
        }

        # In band mode, sort rows ascending so each band is fetched exactly once.
        # This changes iteration order but not the statistical properties of the sample.
        if chunk_rows is not None:
            row_order = np.sort(row_order)

        # Band state — only active when chunk_rows is not None.
        _band_matrix: Optional[ChromMatrix] = None
        _band_start_idx: int = -1        # core start row of the currently-loaded band
        _band_fetch_start_bin: int = 0   # first bin of the padded fetch window
        _band_expected_lookup: np.ndarray = expected_lookup
        _band_default_expected: float = default_expected

        for row_idx in row_order:
            # --- Band fetch dispatch (chunk_rows mode) ---
            if chunk_rows is not None:
                current_band_start = start_idx + (
                    (int(row_idx) - start_idx) // chunk_rows
                ) * chunk_rows
                if current_band_start != _band_start_idx:
                    band_end = min(stop_idx, current_band_start + chunk_rows)
                    fetch_start_bin = max(0, current_band_start - bindist_bins)
                    fetch_end_bin = min(n_bins, band_end + bindist_bins)
                    _band_matrix = provider.fetch_region(
                        chrom, fetch_start_bin * int(res), fetch_end_bin * int(res)
                    )
                    if _band_matrix is not None and _band_matrix.coo.nnz > 0:
                        _band_expected_lookup, _band_default_expected = (
                            _precompute_expectation(_band_matrix.coo)
                        )
                    else:
                        _band_expected_lookup = np.zeros(1, dtype=float)
                        _band_default_expected = 0.0
                    _band_fetch_start_bin = fetch_start_bin
                    _band_start_idx = current_band_start

                if _band_matrix is None or _band_matrix.coo.nnz == 0:
                    if sample_all:
                        chrom_bed.append((row_idx, float("nan")))
                    failed_rows += 1
                    continue

                observed_vec, expected_vec = _row_window_vectors(
                    _band_matrix,
                    row_idx=int(row_idx) - _band_fetch_start_bin,
                    bindist_bins=bindist_bins,
                    expected_lookup=_band_expected_lookup,
                    default_expected=_band_default_expected,
                )
            else:
                observed_vec, expected_vec = _row_window_vectors(
                    matrices,
                    row_idx=row_idx,
                    bindist_bins=bindist_bins,
                    expected_lookup=expected_lookup,
                    default_expected=default_expected,
                )

            # --- Quality filter: skip rows that are too sparse or too weak ---
            # mean_score < min_mean_score: the row's average observed/expected ratio
            # is too low, meaning the bin has almost no contacts relative to expectation.
            # nonzero_frac < min_nonzero_frac: fewer than 1% of the window positions
            # have any contact at all — not enough data for a stable autocorrelation.
            scores = (observed_vec + 1.0) / (expected_vec + 1.0)
            finite_scores = scores[np.isfinite(scores)]
            mean_score = float(np.nanmean(finite_scores)) if finite_scores.size else 0.0
            nonzero_frac = float(np.count_nonzero(observed_vec) / observed_vec.size) if observed_vec.size else 0.0

            if mean_score < float(min_mean_score) or nonzero_frac < float(min_nonzero_frac):
                failed_rows += 1
                if sample_all:
                    # In full-scan mode, record the failed row as NaN in the bedgraph
                    # to preserve the full-chromosome coordinate structure.
                    chrom_bed.append((row_idx, float("nan")))
                continue

            selected_rows.append(row_idx)

            # --- Subsampling simulation ---
            # Round observed counts to nearest integer (contact counts should be integers,
            # but floating-point normalisation may introduce fractional values).
            observed_int = np.clip(np.rint(observed_vec), 0, None).astype(int)
            # current_counts tracks the downsampled state as we move from ratio to ratio.
            current_counts = observed_int.copy()
            prev_ratio = 1.0

            for ratio in ratios:
                if ratio == 1.0:
                    # Full-depth data — use the original counts unchanged.
                    counts = observed_int.copy()
                    prev_ratio = ratio
                else:
                    if prev_ratio == 0:
                        counts = np.zeros_like(current_counts)
                    else:
                        # Compute the *incremental* fraction relative to the previous step.
                        # Example: going from ratio=0.5 to ratio=0.25 means keeping
                        # 0.25/0.5 = 50% of the already-downsampled counts.
                        # This avoids drawing new binomial samples from the full data
                        # at each ratio, which would be inconsistent (different samples
                        # at ratio 0.5 vs. using the ratio-0.5 sample as input for 0.25).
                        incremental = ratio / prev_ratio
                        incremental = float(np.clip(incremental, 0.0, 1.0))
                        counts = rng.binomial(current_counts, incremental)
                        prev_ratio = ratio
                        current_counts = counts

                # Scale expected values proportionally to match the downsampled depth.
                expected_scaled = expected_vec * ratio
                metrics = _compute_row_metrics(counts.astype(float), expected_scaled.astype(float))

                ratio_stats[ratio]["noise_raw"].append(metrics["noise_raw"])
                ratio_stats[ratio]["row_mean"].append(metrics["row_mean"])
                ratio_stats[ratio]["row_std"].append(metrics["row_std"])
                ratio_stats[ratio]["acf"].append(metrics["acf"])
                ratio_stats[ratio]["empty_ratio"].append(metrics["empty_ratio"])

                if ratio == 1.0:
                    # Record the full-depth noise value for this row in the bedgraph.
                    chrom_bed.append((row_idx, metrics["noise_raw"]))

            if len(selected_rows) >= requested_n:
                break

        if len(selected_rows) < requested_n:
            exhausted = True

        # --- Write per-chromosome bedgraph ---
        if chrom_bed:
            bed_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
            with open(bed_path, "w") as handle:
                for row_idx, value in sorted(chrom_bed):
                    start_bp = row_idx * int(res)
                    stop_bp = min((row_idx + 1) * int(res), chrom_len)
                    handle.write(f"{chrom} {start_bp} {stop_bp} {value}\n")

        # --- Aggregate per-chromosome summary statistics ---
        for ratio, stats_dict in ratio_stats.items():
            noise_values = stats_dict["noise_raw"]
            if not noise_values:
                continue
            chrom_contacts = contacts_by_chrom.get(chrom, float("nan"))
            median_noise = float(np.nanmedian(noise_values))
            mean_ebr = float(np.nanmean(stats_dict["empty_ratio"]))
            record = {
                "chrom": chrom,
                "ratio": ratio,
                "rows_evaluated": len(noise_values),
                "mean_noise": float(np.nanmean(noise_values)),
                "median_noise": median_noise,
                "std_noise": float(np.nanstd(noise_values, ddof=0)),
                "mean_row_mean": float(np.nanmean(stats_dict["row_mean"])),
                "mean_row_std": float(np.nanmean(stats_dict["row_std"])),
                "mean_acf": float(np.nanmean(stats_dict["acf"])),
                "mean_empty_bin_ratio": mean_ebr,
                # estimated_contacts at this ratio = actual contacts * downsampling ratio
                "estimated_contacts": float(chrom_contacts * float(ratio)),
                # Placeholder columns filled in below by compute_noise_model / sequencing_advisor
                "z_map": float("nan"),
                "gamma": float("nan"),
                "pred_log_N": float("nan"),
                "pred_log_N_upper": float("nan"),
                "pred_log_N_lower": float("nan"),
                "epsilon": float("nan"),
                "quality_status": "",
                "quality_percentile": -1,
                "advisor_target_density": float("nan"),
                "advisor_fold_increase": float("nan"),
                "advisor_recommendation": "",
            }
            # Only compute model metrics and advisor for the full-depth (ratio == 1.0)
            # run, since the reference model is calibrated against unsubsampled data.
            if (
                ratio == 1.0
                and np.isfinite(median_noise)
                and median_noise > 0
                and np.isfinite(chrom_contacts)
                and chrom_contacts > 0
            ):
                c_len = chrom_bp_sizes.get(chrom, 0)
                if c_len > 0:
                    try:
                        model_result = reference.compute_noise_model(
                            contacts=chrom_contacts,
                            chrom_len=c_len,
                            resolution=int(res),
                            window_bp=window,
                            median_noise=median_noise,
                            ebr=mean_ebr,
                        )
                        record["z_map"]            = model_result["z_map"]
                        record["gamma"]            = model_result["gamma"]
                        record["pred_log_N"]       = model_result["pred_log_N"]
                        record["pred_log_N_upper"] = model_result["pred_log_N_upper"]
                        record["pred_log_N_lower"] = model_result["pred_log_N_lower"]
                        record["epsilon"]          = model_result["epsilon"]
                        record["quality_status"]   = model_result["quality_status"]

                        # Decile Diagnostic — classify noise relative to 4DN percentile landscape
                        _rho_win  = model_result["rho_win"]
                        _L_rho    = float(np.log10(max(_rho_win, 1e-300)))
                        _obs_log_N = float(np.log10(max(median_noise, 1e-300)))
                        record["quality_percentile"] = reference.classify_quality_percentile(
                            L_rho=_L_rho,
                            ebr=mean_ebr,
                            obs_noise_log10=_obs_log_N,
                        )

                        # current_rho_win = effective matrix coverage for this chrom
                        current_rho_win = chrom_contacts / (
                            (c_len / int(res)) * (window / int(res))
                        )
                        advisor_result = reference.sequencing_advisor(
                            obs_noise=median_noise,
                            current_rho_win=current_rho_win,
                            ebr=mean_ebr,
                            window_bp=window,
                        )
                        record["advisor_target_density"] = advisor_result["target_density"]
                        record["advisor_fold_increase"]  = advisor_result["fold_increase"]
                        record["advisor_recommendation"] = advisor_result["recommendation"]
                    except Exception:
                        # Expected failure modes: chromosome too sparse for a meaningful
                        # model evaluation (e.g. very short or unplaced chromosomes with
                        # near-zero contacts). Leave the fields as NaN silently.
                        pass
            summary_records.append(record)

        total_considered = len(selected_rows) + failed_rows
        if total_considered:
            frac_kept = len(selected_rows) / float(total_considered)
            if exhausted:
                print(
                    f"[WARN] {chrom}: only {frac_kept:.2%} of sampled rows met minimum thresholds "
                    f"(requested {requested_n}, kept {len(selected_rows)})."
                )

        if profile_path:
            elapsed = time.perf_counter() - chrom_start_time
            current_rss_mb, ru_peak_curr = collect_memory_usage_mb()
            psutil_peak_so_far = max(psutil_peak_so_far, current_rss_mb)
            ru_peak_so_far = max(ru_peak_so_far, ru_peak_curr)
            psutil_delta = max(0.0, psutil_peak_so_far - start_psutil_peak)
            ru_delta = max(0.0, ru_peak_so_far - start_ru_peak)
            profile_rows.append(
                {
                    "input": os.path.abspath(hic_path),
                    "backend": backend,
                    "chrom": chrom,
                    "resolution": int(res),
                    "sample_fraction": float(sample_fraction),
                    "rows_requested": int(requested_n),
                    "rows_attempted": int(total_considered),
                    "rows_kept": int(len(selected_rows)),
                    "rows_failed": int(failed_rows),
                    "elapsed_s": float(elapsed),
                    "current_rss_mb": float(current_rss_mb),
                    "ru_maxrss_mb": float(ru_peak_curr),
                    "peak_rss_mb": float(max(psutil_peak_so_far, ru_peak_so_far)),
                    "peak_increase_mb": float(max(psutil_delta, ru_delta)),
                    "min_mean_score": float(min_mean_score),
                    "min_nonzero_frac": float(min_nonzero_frac),
                }
            )

    # --- Write genome-wide summary TSV ---
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.sort_values(["chrom", "ratio"], inplace=True)
        summary_path = os.path.join(out_dir, "subsample_summary.tsv")
        with open(summary_path, "w") as handle:
            # Header comment lines encode metadata parsed by downstream tools:
            # - "# resolution=N" is read by the sequencing-advisor CLI to determine
            #   which resolution the summary applies to.
            # - "# window_bp=N" is read by the sequencing-advisor CLI to compute
            #   ρ_win (effective matrix coverage), which requires window size.
            # - "# chrom_info" lines provide chromosome sizes (not stored in the TSV
            #   body) so the advisor can compute contacts-per-bin (rho).
            handle.write(f"# resolution={int(res)}\n")
            handle.write(f"# window_bp={int(window)}\n")
            for chrom in sorted(contacts_by_chrom):
                handle.write(
                    f"# chrom_info\t{chrom}\tcontacts={contacts_by_chrom[chrom]:.6f}\t"
                    f"size_bp={chrom_bp_sizes.get(chrom, 'NA')}\n"
                )
            summary_df.to_csv(handle, sep="\t", index=False)

    # --- Append profiling data ---
    if profile_path and profile_rows:
        profile_abs = os.path.abspath(profile_path)
        os.makedirs(os.path.dirname(profile_abs) or ".", exist_ok=True)
        exists = os.path.exists(profile_abs)
        mode = "a" if exists else "w"
        header = not exists
        pd.DataFrame(profile_rows).to_csv(profile_abs, index=False, mode=mode, header=header)
