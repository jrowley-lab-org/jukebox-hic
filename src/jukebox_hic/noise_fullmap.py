#!/usr/bin/env python
"""
Genome-wide (full-map) Hi-C noise computation — backing module for the ``noise-bedgraph`` CLI command.

Unlike ``noise_sampling``, which samples a fraction of rows for a fast global
estimate, this module computes a noise value for **every** genomic bin on every
chromosome. The output is a per-bin bedgraph file suitable for downstream bias
vector construction (``noise_to_weights``) and blacklisting (``filters``).

Why per-bin noise?
------------------
The per-row noise metric captures how "erratic" the contact pattern is around
each genomic bin. Bins in heterochromatic, repetitive, or low-mappability regions
tend to have high stochastic noise (random spiky contacts) while euchromatic
regions in well-sequenced samples have smooth, predictable decay patterns.

By computing noise for every bin, this module produces a genome-wide noise
track (like a coverage bedgraph, but measuring signal quality rather than depth)
that can be transformed into a normalization bias vector to down-weight noisy bins
during Hi-C matrix normalization.

Architecture
------------
Each (chromosome, resolution) pair is processed as an independent task
(``_FullNoiseTask``). Tasks can be distributed across multiple worker processes
using Python's ``multiprocessing.Pool``. The main process collects results and
concatenates the per-chromosome output files into a single resolution-level bedgraph.

Main entry point: ``compute_full_noise()``.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from statsmodels.tsa.stattools import acovf

from .backends import ChromMatrix, select_provider
from . import metrics


def _default_bindist_bp(resolution: int) -> int:
    """
    Choose a sensible default maximum genomic distance (in bp) for the noise window.

    "Bindist" is short for "bin distance" — the maximum number of bins away from the
    diagonal that the noise calculation considers. Only contacts within this distance
    are included in the noise metric for a given bin.

    The defaults are chosen to capture enough contacts for a stable lag-1 autocovariance
    estimate without extending into the long-range regime where the distance-decay model
    is less reliable. The window is approximately 100–160 bins wide at each resolution:

    - ≥100 kb resolution → 10 Mbp  (100 bins on each side of the diagonal)
    - ≥50 kb resolution  → 6 Mbp   (120 bins)
    - ≥25 kb resolution  → 4 Mbp   (160 bins)
    - <25 kb resolution  → 1 Mbp   (40+ bins at fine resolutions)

    These are the same defaults as ``_default_window_bp()`` in ``noise_sampling``;
    both functions use the same empirical thresholds.
    """
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _compute_dump_bins(n_bins: int, noise_bins: int, dump_factor: float = 10.0) -> int:
    """
    Number of core bins to process per sliding-window chunk.

    The chunk is at least ``2 * noise_bins`` wide (avoids pathologically small chunks
    that would thrash the fetch API) but never larger than the chromosome (so small
    chromosomes are always processed in a single pass).  ``dump_factor`` scales
    linearly with ``noise_bins``: larger values mean fewer, larger fetches and higher
    peak RAM; smaller values mean more fetches and lower peak RAM.
    """
    return int(min(n_bins, max(2 * noise_bins, int(dump_factor * noise_bins))))


def _precompute_expectation(coo) -> Tuple[np.ndarray, float]:
    """
    Build a lookup table of expected (average) contact counts by genomic distance.

    In Hi-C, contact frequency decreases with increasing genomic distance — the
    "distance-decay" or "expected" curve. This function computes that curve from
    the data itself by averaging all contact counts that share the same bin offset
    (lag = |i - j|).

    Dividing observed counts by this expected value when building the noise window
    normalises away the distance effect, so the noise metric captures erratic
    deviations from the expected pattern rather than the expected pattern itself.

    Parameters
    ----------
    coo : scipy.sparse.coo_matrix
        Sparse contact matrix in COO format (no type annotation here to accept any
        sparse COO-like object). The ``row``, ``col``, and ``data`` arrays are used.

    Returns
    -------
    means : np.ndarray of shape (max_lag + 1,)
        ``means[k]`` = average contact count at lag k bins. Zero where no contacts
        exist at that lag (or when the bin has no data).
    default : float
        Global fallback expected value — the mean of all non-zero lag averages.
        Used when a specific lag has no observed contacts in ``_expected_for_lag()``.
    """
    if coo.data.size == 0:
        return np.zeros(1, dtype=float), 0.0
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    data = coo.data.astype(float)
    # lag = genomic distance between the two interacting bins, in bin units
    lags = np.abs(cols - rows)
    max_lag = int(lags.max())
    # np.bincount with weights= sums data values grouped by lag index
    sums = np.bincount(lags, weights=data, minlength=max_lag + 1)
    counts = np.bincount(lags, minlength=max_lag + 1)
    # Safely divide: leave lags with zero count as 0.0 in the output
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    default = float(means[counts > 0].mean()) if np.count_nonzero(counts) else 0.0
    return means, default


def _expected_for_lag(expected_lookup: np.ndarray, default: float, lag: int) -> float:
    """
    Retrieve the expected contact count for a specific genomic distance (lag in bins).

    Parameters
    ----------
    expected_lookup : np.ndarray
        The means array from ``_precompute_expectation()``.
    default : float
        Fallback used when the lag is beyond the precomputed range or when the stored
        value is zero (which would cause a division-by-zero downstream).
    lag : int
        Genomic distance in bins (|bin_i - bin_j|).

    Returns
    -------
    float
        Expected contact count at this lag; falls back to ``default`` as needed.
    """
    if lag < len(expected_lookup):
        val = expected_lookup[lag]
        return float(val) if val > 0 else default
    return default


def _row_window_noise(
    matrix: ChromMatrix,
    rowbins: int,
    bindist_bins: int,
    expected_lookup: np.ndarray,
    default_expected: float,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the noise value for a range of bins in a contact matrix.

    Processes rows ``row_start`` to ``row_end - 1`` (default: all rows).  The returned
    array has length ``row_end - row_start``, with index 0 corresponding to
    ``row_start`` in the matrix.  Passing ``row_start`` / ``row_end`` restricts
    processing to core rows of a sliding-window chunk without reallocating the matrix.

    For each bin ``idx``, two contact windows are built:
    - **Upstream** (CSC column slice): contacts with bins j < idx, positions ``bindist_bins - lag``.
    - **Downstream** (CSR row slice): contacts with bins j > idx, positions ``bindist_bins - 1 + lag``.

    Noise = ``1 / |lag-1 autocovariance|`` of the observed window.  NaN when the
    autocovariance is zero or the computation fails.

    Parameters
    ----------
    matrix : ChromMatrix
        Sparse contact matrix (CSR and CSC formats used).
    rowbins : int
        Total number of bins in the matrix dimension.
    bindist_bins : int
        Half-window size in bins.
    expected_lookup : np.ndarray
        Distance-decay lookup from ``_precompute_expectation()``.
    default_expected : float
        Fallback expected value for lags with no data.
    row_start : int
        First row index to process (default 0).
    row_end : int or None
        One-past-last row index to process (default ``rowbins``).

    Returns
    -------
    noise_vals : np.ndarray of shape (row_end - row_start,)
        Per-bin noise values for the requested row range.
    window_sums : np.ndarray of shape (row_end - row_start,)
        Per-bin total observed contacts inside the SAME 2W diagonal window used
        for the noise estimate.  Returned here rather than computed separately so
        the density track and the noise track are guaranteed to describe the same
        span of the matrix (the Methods define both over the 2W window).
    """
    if row_end is None:
        row_end = rowbins
    out_len = row_end - row_start
    noise_vals = np.zeros(out_len, dtype=float)
    window_sums = np.zeros(out_len, dtype=float)
    for out_i, idx in enumerate(range(row_start, row_end)):
        window = bindist_bins * 2
        observed = np.zeros(window, dtype=float)
        # Initialise expected to the global default so un-contacted positions still
        # have a meaningful denominator in the observed/expected ratio.
        expected = np.full(window, default_expected, dtype=float)

        # --- Upstream contacts: CSC column slice (bins before idx) ---
        col_vec = matrix.csc.getcol(idx)
        for src_idx, value in zip(col_vec.indices, col_vec.data):
            if src_idx >= idx:
                # Skip diagonal and downstream entries — handled by CSR below.
                continue
            lag = idx - int(src_idx)
            if lag > bindist_bins:
                continue  # beyond the window — exclude this contact
            pos = bindist_bins - lag  # upstream position in the window vector
            observed[pos] = float(value)
            expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

        # --- Downstream contacts: CSR row slice (bins after idx) ---
        row_vec = matrix.csr.getrow(idx)
        for dst_idx, value in zip(row_vec.indices, row_vec.data):
            if dst_idx <= idx:
                # Skip diagonal and upstream entries — already handled above.
                continue
            lag = int(dst_idx) - idx
            if lag > bindist_bins:
                continue  # beyond the window — exclude this contact
            pos = bindist_bins - 1 + lag  # downstream position in the window vector
            observed[pos] = float(value)
            expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

        # --- Compute noise = 1 / |lag-1 autocovariance of observed/expected ratios| ---
        # The distance-decay expectation MUST be divided out before the
        # autocovariance is taken.  Over a window half-width of W bins the expected
        # contact frequency falls by one to two orders of magnitude, so the
        # autocovariance of the raw counts is dominated by the distance-decay
        # envelope and by the row's sequencing depth rather than by the local
        # roughness this metric is meant to capture.  A pseudo-count of 1 is added
        # to both numerator and denominator, matching the Methods section and
        # noise_sampling._compute_row_metrics().
        window_sums[out_i] = float(observed.sum())

        scores = (observed + 1.0) / (expected + 1.0)
        finite = scores[np.isfinite(scores)]
        if finite.size < 2 or np.allclose(finite, finite[0]):
            # Fewer than two finite ratios, or a perfectly constant series: the
            # lag-1 autocovariance is zero/undefined and 1/|ac| would divide by
            # zero.  Report as unmeasurable, as the sampling path does.
            noise_vals[out_i] = float("nan")
            continue
        try:
            ac = acovf(finite, nlag=1, fft=True)[1]
            noise_vals[out_i] = float("nan") if ac == 0 else 1.0 / abs(ac)
        except Exception:
            noise_vals[out_i] = float("nan")

    return noise_vals, window_sums


def _write_bedgraph(
    chrom: str,
    res: int,
    chrom_len: int,
    noise_vals: np.ndarray,
    out_path: str,
    include_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Write a per-bin noise track as a UCSC bedgraph file.

    BedGraph format: four space-separated columns per line:
        chromosome  start_bp  end_bp  value

    Coordinates are 0-based half-open intervals [start, end), consistent with
    the BED specification. The last bin may be shorter than ``res`` if the
    chromosome length is not a multiple of the bin size.

    Parameters
    ----------
    chrom : str
        Chromosome name (written verbatim as column 1).
    res : int
        Bin size in base pairs.
    chrom_len : int
        Chromosome length in bp (used to clip the last bin's end coordinate).
    noise_vals : np.ndarray
        Per-bin noise values, one per row in the matrix (index 0 = first bin).
    out_path : str
        Output file path (overwritten if it exists).
    include_mask : np.ndarray of bool, optional
        When provided, only bins where ``include_mask[i]`` is True are written.
        None (default) writes all bins.
    """
    with open(out_path, "w") as handle:
        for i, val in enumerate(noise_vals):
            if include_mask is not None and not include_mask[i]:
                continue
            start = i * res
            stop = min((i + 1) * res, chrom_len)
            handle.write(f"{chrom} {start} {stop} {val}\n")


def _worker_init(per_worker_limit_bytes: Optional[int]) -> None:
    """Pool initializer: set a per-worker address space limit if requested."""
    if per_worker_limit_bytes is not None:
        metrics.set_address_space_limit(per_worker_limit_bytes)


@dataclass(frozen=True)
class _FullNoiseTask:
    """
    Immutable task descriptor for one (chromosome, resolution) noise computation.

    Each instance is passed to a worker process (or run serially) by
    ``compute_full_noise()``. Using a frozen dataclass ensures the task cannot be
    accidentally mutated between the main process building it and the worker consuming
    it, which is important for correctness in multiprocessing.

    Attributes
    ----------
    hic_path : str
        Path to the input Hi-C or cooler file. Each worker opens its own provider
        from this path because provider objects are not safe to share across processes.
    chrom : str
        Chromosome name to process (e.g. ``"chr1"``).
    chrom_len : int
        Chromosome length in bp (needed to clip the last bin in the bedgraph).
    res : int
        Bin size in base pairs.
    norm : str
        Normalisation mode (passed to ``select_provider()``).
    cooler_selection : str or None
        HDF5 group path for cooler files that require an explicit selection.
    bindist_bp : int or None
        Maximum genomic distance in bp for the noise window. If None, the default
        computed by ``_default_bindist_bp(res)`` is used.
    out_dir : str
        Directory where the per-chromosome bedgraph will be written (unused by the
        worker itself; kept for reference and error reporting by the main process).
    out_path : str
        Full path to the per-chromosome output bedgraph file (pre-computed by the
        main process to avoid per-worker path construction).
    """
    hic_path: str
    chrom: str
    chrom_len: int
    res: int
    norm: str
    cooler_selection: Optional[str]
    bindist_bp: Optional[int]
    out_dir: str
    out_path: str
    density_out_path: str
    dump_factor: float = 10.0
    regions: Optional[Tuple[Tuple[int, int], ...]] = None


def _full_noise_worker(task: _FullNoiseTask) -> Tuple[str, int, str, bool]:
    """
    Worker function that computes full-map noise for one (chromosome, resolution) task.

    This function is designed to run either in the main process (serial mode) or in a
    subprocess (parallel mode via ``multiprocessing.Pool``). Each call opens a fresh
    provider connection to the Hi-C file because file handles cannot be safely shared
    between processes.

    Steps:
    1. Open the provider and load the contact matrix for the chromosome.
    2. If the matrix is empty or missing, clean up any stale output file and return
       ``(chrom, res, out_path, False)`` — the ``False`` signals failure to the caller.
    3. Precompute the distance-decay expected curve.
    4. Compute per-bin noise values for all bins in the matrix.
    5. Write the noise bedgraph to ``task.out_path``.
    6. Return ``(chrom, res, out_path, True)`` — the ``True`` signals success.

    Parameters
    ----------
    task : _FullNoiseTask
        Immutable task descriptor built by ``compute_full_noise()``.

    Returns
    -------
    Tuple[str, int, str, bool]
        ``(chrom_name, resolution_bp, output_path, success_flag)``
        The success flag is ``True`` if a bedgraph was written, ``False`` if the
        chromosome had no contacts or an error occurred.
    """
    provider, _ = select_provider(task.hic_path, task.res, task.norm, task.cooler_selection)

    res = int(task.res)
    window_bp = task.bindist_bp or _default_bindist_bp(res)
    noise_bins = max(1, window_bp // res)
    n_bins = (task.chrom_len + res - 1) // res

    if n_bins == 0:
        for stale in (task.out_path, task.density_out_path):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        return task.chrom, task.res, task.out_path, False

    dump_bins = _compute_dump_bins(n_bins, noise_bins, task.dump_factor)

    noise_vals = np.full(n_bins, float("nan"), dtype=float)
    density_vals = np.zeros(n_bins, dtype=float)
    any_data = False

    # --- Region filtering: determine which bins to output and the iteration span ---
    target_mask: Optional[np.ndarray] = None
    iter_start = 0
    iter_end = n_bins

    if task.regions is not None and len(task.regions) > 0:
        target_mask = np.zeros(n_bins, dtype=bool)
        for r_start, r_end in task.regions:
            b_start = r_start // res
            b_end = (r_end + res - 1) // res  # ceiling: include partial bins
            b_end = min(b_end, n_bins)
            if b_start < b_end:
                target_mask[b_start:b_end] = True
        if not target_mask.any():
            for stale in (task.out_path, task.density_out_path):
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            return task.chrom, task.res, task.out_path, False
        target_indices = np.where(target_mask)[0]
        iter_start = int(target_indices[0])
        iter_end = int(target_indices[-1]) + 1

    # --- Pass 1: accumulate the distance-decay expectation over the WHOLE chromosome.
    #
    # The expectation must be a property of the chromosome, not of the chunk that
    # happens to contain a bin: E(k) is defined as the mean contact count at lag k
    # across the chromosome.  Computing it per chunk made every reported value
    # depend on --dump_factor, a pure memory-tuning knob, so the same map analysed
    # on two machines with different memory budgets produced different noise and
    # different blacklists.  Only lags inside the noise window are needed.
    lag_sums = np.zeros(noise_bins + 1, dtype=float)
    lag_counts = np.zeros(noise_bins + 1, dtype=np.int64)

    _scan_start = iter_start
    while _scan_start < iter_end:
        _scan_end = min(_scan_start + dump_bins, iter_end)
        _fs = max(0, _scan_start - noise_bins)
        _fe = min(n_bins, _scan_end + noise_bins)
        _m = provider.fetch_region(task.chrom, _fs * res, _fe * res)
        if _m is not None and _m.coo.nnz > 0:
            _coo = _m.coo
            _lags = np.abs(_coo.col.astype(np.int64) - _coo.row.astype(np.int64))
            # Chunks are fetched with `noise_bins` of padding on each side so that
            # every core row sees its full window.  That padding makes consecutive
            # fetches overlap, so a contact in an overlap would be accumulated
            # twice -- and how often that happens depends on the chunk size, i.e.
            # on --dump_factor.  Attribute each entry to exactly one chunk by
            # keeping only those whose global ROW index falls in this chunk's core
            # span, which makes the pooled curve independent of the chunking.
            _grow = _coo.row.astype(np.int64) + _fs
            _keep = (_lags <= noise_bins) & (_grow >= _scan_start) & (_grow < _scan_end)
            if _keep.any():
                lag_sums += np.bincount(
                    _lags[_keep], weights=_coo.data.astype(float)[_keep],
                    minlength=noise_bins + 1,
                )
                lag_counts += np.bincount(_lags[_keep], minlength=noise_bins + 1)
        del _m
        _scan_start = _scan_end

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_lookup = np.divide(
            lag_sums, lag_counts,
            out=np.zeros_like(lag_sums), where=lag_counts > 0,
        )
    default_expected = (
        float(expected_lookup[lag_counts > 0].mean()) if np.any(lag_counts > 0) else 0.0
    )

    # --- Pass 2: per-bin noise and window density against that fixed expectation.
    chunk_start = iter_start
    while chunk_start < iter_end:
        chunk_end = min(chunk_start + dump_bins, iter_end)

        fetch_start_bin = max(0, chunk_start - noise_bins)
        fetch_end_bin = min(n_bins, chunk_end + noise_bins)
        fetch_start_bp = fetch_start_bin * res
        fetch_end_bp = fetch_end_bin * res

        matrix = provider.fetch_region(task.chrom, fetch_start_bp, fetch_end_bp)

        if matrix is not None and matrix.coo.nnz > 0:
            any_data = True

            local_chunk_start = chunk_start - fetch_start_bin
            local_chunk_end = chunk_end - fetch_start_bin
            fetch_bins = fetch_end_bin - fetch_start_bin

            chunk_noise, chunk_density = _row_window_noise(
                matrix,
                rowbins=fetch_bins,
                bindist_bins=noise_bins,
                expected_lookup=expected_lookup,
                default_expected=default_expected,
                row_start=local_chunk_start,
                row_end=local_chunk_end,
            )
            noise_vals[chunk_start:chunk_end] = chunk_noise
            # Density is the row sum over the same 2W window the noise used, so the
            # two blacklist metrics describe the same span.  Previously this summed
            # the whole fetched submatrix, whose width is set by --dump_factor.
            density_vals[chunk_start:chunk_end] = chunk_density

        chunk_start = chunk_end

    if not any_data:
        for stale in (task.out_path, task.density_out_path):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        return task.chrom, task.res, task.out_path, False

    _write_bedgraph(task.chrom, res, task.chrom_len, noise_vals, task.out_path, target_mask)
    _write_bedgraph(task.chrom, res, task.chrom_len, density_vals, task.density_out_path, target_mask)

    return task.chrom, task.res, task.out_path, True


# Type alias for a parsed region specification.
# On-diagonal: Tuple[int, int] = (start_bp, end_bp)
# Off-diagonal: Tuple[Tuple[int,int], Tuple[int,int]] = ((l_start,l_end),(r_start,r_end))
_RegionSpec = Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]


def _parse_single_region(
    s: str,
    size: int,
    chrom: str,
    col_i: int,
    lineno: int,
    path: str,
    side: str = "",
) -> Optional[Tuple[int, int]]:
    """
    Parse a ``start-end`` region string and validate against chromosome size.

    ``-`` is the only accepted separator within a single region (0-based, half-open).
    Returns ``(start, end)`` on success or ``None`` (with a ``[WARN]`` line printed)
    on any validation failure.  ``side`` is an optional label (``"left"``/``"right"``)
    for clearer error messages in off-diagonal context.
    """
    label = f"{side} " if side else ""
    if "-" not in s:
        print(
            f"[WARN] {path} line {lineno} col {col_i}: "
            f"{label}region {s!r} has no '-' separator, skipping"
        )
        return None
    halves = s.split("-", 1)
    if not halves[0] or not halves[1]:
        print(
            f"[WARN] {path} line {lineno} col {col_i}: "
            f"{label}region {s!r} has empty coordinate, skipping"
        )
        return None
    try:
        r_start = int(halves[0])
        r_end = int(halves[1])
    except ValueError:
        print(
            f"[WARN] {path} line {lineno} col {col_i}: "
            f"cannot parse {label}region {s!r} as integers, skipping"
        )
        return None
    if r_start < 0 or r_end < 0:
        print(f"[WARN] {chrom} {label}region {s!r}: negative coordinate, skipping")
        return None
    if r_end > size:
        print(
            f"[WARN] {chrom} {label}region {s!r}: end ({r_end}) exceeds "
            f"chromosome size ({size}), skipping"
        )
        return None
    if r_start >= r_end:
        print(
            f"[WARN] {chrom} {label}region {s!r}: start ({r_start}) >= end ({r_end}), skipping"
        )
        return None
    return (r_start, r_end)


def _parse_extended_chrom_sizes(
    path: str,
) -> Dict[str, Tuple[int, Optional[List[_RegionSpec]]]]:
    """
    Parse a chromosome sizes file with optional region columns.

    File format (tab-separated):
        chrom_name  size_bp  [col3  col4  ...]

    Region column formats (0-based, half-open BED coordinates):

    - ``start-end``                         — on-diagonal region
    - ``l_start-l_end:r_start-r_end``       — off-diagonal pair (left × right)

    The ``-`` character is the exclusive within-region separator.
    The ``:`` character exclusively joins two ``start-end`` halves into an off-diagonal
    pair.  The old ``start:end`` notation is no longer accepted as an on-diagonal region.

    Returns
    -------
    Dict mapping chrom_name → (size_bp, region_specs_or_None).
    Each element of region_specs is either:

    - ``(start, end)``                              — on-diagonal
    - ``((l_start, l_end), (r_start, r_end))``     — off-diagonal

    ``None`` when no region columns were given (whole chromosome is processed).
    Duplicate chromosome entries: last row wins.
    """
    result: Dict[str, Tuple[int, Optional[List[_RegionSpec]]]] = {}
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(
                    f"[WARN] {path} line {lineno}: expected ≥2 columns, got {len(parts)}, skipping"
                )
                continue
            chrom = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                print(
                    f"[WARN] {path} line {lineno}: cannot parse size {parts[1]!r} "
                    f"for {chrom}, skipping"
                )
                continue

            if len(parts) == 2:
                result[chrom] = (size, None)
                continue

            region_specs: List[_RegionSpec] = []
            for col_i, raw_col in enumerate(parts[2:], start=3):
                col = raw_col.strip()
                if not col:
                    continue
                colon_count = col.count(":")
                if colon_count == 0:
                    # On-diagonal: parse as start-end
                    spec = _parse_single_region(col, size, chrom, col_i, lineno, path)
                    if spec is not None:
                        region_specs.append(spec)
                elif colon_count == 1:
                    # Off-diagonal: two halves joined by ':'
                    left_str, right_str = col.split(":", 1)
                    left = _parse_single_region(
                        left_str.strip(), size, chrom, col_i, lineno, path, side="left"
                    )
                    right = _parse_single_region(
                        right_str.strip(), size, chrom, col_i, lineno, path, side="right"
                    )
                    if left is not None and right is not None:
                        if left[0] < right[1] and right[0] < left[1]:
                            print(
                                f"[WARN] {chrom} col {col_i}: off-diagonal regions "
                                f"overlap ({col!r}); continuing anyway"
                            )
                        region_specs.append((left, right))
                else:
                    print(
                        f"[WARN] {path} line {lineno} col {col_i}: "
                        f"region {col!r} contains more than one ':', skipping"
                    )

            if not region_specs:
                print(
                    f"[WARN] {chrom}: no valid regions found in {path}, skipping chromosome"
                )
                continue

            result[chrom] = (size, region_specs)

    return result


@dataclass(frozen=True)
class _OffDiagNoiseTask:
    """
    Immutable task descriptor for one off-diagonal (region_A × region_B) noise computation.

    Unlike ``_FullNoiseTask``, which operates on a symmetric diagonal block, this task
    fetches the rectangular contact block where rows come from ``left_region`` and columns
    come from ``right_region``.  For each row bin in the left region the lag-1
    autocovariance of its contact profile across the entire right region is computed.

    Attributes
    ----------
    left_start, left_end : int
        0-based half-open bp coordinates for the row (left) region.
    right_start, right_end : int
        0-based half-open bp coordinates for the column (right) region.
    out_path : str
        Output bedgraph path (5-column format; see ``_write_off_diag_bedgraph``).
    """
    hic_path: str
    chrom: str
    chrom_len: int
    res: int
    norm: str
    cooler_selection: Optional[str]
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    out_dir: str
    out_path: str
    dump_factor: float = 10.0


def _write_off_diag_bedgraph(task: "_OffDiagNoiseTask", noise_vals: np.ndarray) -> None:
    """
    Write off-diagonal noise values as a 5-column bedgraph.

    Format per line:
        chrom  left_bin_start  left_bin_end  chrom:right_start-right_end  noise_value

    Columns 1–3 give the standard BED coordinate of each bin in the left region.
    Column 4 encodes the right region as ``chrom:start-end`` (using ``:`` to join
    the chromosome name to the coordinate range).  Column 5 is the noise value.
    """
    res = int(task.res)
    right_label = f"{task.chrom}:{task.right_start}-{task.right_end}"
    with open(task.out_path, "w") as fh:
        for i, val in enumerate(noise_vals):
            start = task.left_start + i * res
            end = min(task.left_start + (i + 1) * res, task.left_end)
            fh.write(f"{task.chrom}\t{start}\t{end}\t{right_label}\t{val}\n")


def _off_diag_noise_worker(task: "_OffDiagNoiseTask") -> Tuple[str, str, bool]:
    """
    Worker function for one off-diagonal (region_A × region_B) noise task.

    Designed to run either in the main process (serial mode) or in a subprocess
    (parallel mode).  Steps:

    1. Open the provider and compute bin counts for the left and right regions.
    2. Iterate over left-region row chunks (sized by ``dump_factor × n_right_bins``).
       For each chunk, fetch the rectangular contact block via ``fetch_region_pair``.
    3. For each left bin, compute lag-1 autocovariance of the contact profile across
       the right region.  Noise = ``1 / |autocovariance|`` (NaN when undefined).
    4. Write the per-bin noise track with ``_write_off_diag_bedgraph``.

    Returns
    -------
    Tuple[str, str, bool]
        ``(chrom_name, output_path, success_flag)``
    """
    provider, _ = select_provider(task.hic_path, task.res, task.norm, task.cooler_selection)

    res = int(task.res)
    n_left_bins = (task.left_end - task.left_start + res - 1) // res
    n_right_bins = (task.right_end - task.right_start + res - 1) // res

    if n_left_bins == 0 or n_right_bins == 0:
        if os.path.exists(task.out_path):
            try:
                os.remove(task.out_path)
            except OSError:
                pass
        return task.chrom, task.out_path, False

    # Chunk the left region; use n_right_bins as the "window size" analogue so that
    # dump_factor has the same RAM-scaling meaning as in the on-diagonal case.
    dump_bins = _compute_dump_bins(n_left_bins, n_right_bins, task.dump_factor)

    noise_vals = np.full(n_left_bins, float("nan"), dtype=float)
    any_data = False

    chunk_start = 0
    while chunk_start < n_left_bins:
        chunk_end = min(chunk_start + dump_bins, n_left_bins)

        abs_left_start = task.left_start + chunk_start * res
        abs_left_end = task.left_start + chunk_end * res

        mat = provider.fetch_region_pair(
            task.chrom,
            abs_left_start, abs_left_end,
            task.right_start, task.right_end,
        )

        if mat is not None and mat.nnz > 0:
            any_data = True
            for local_i in range(chunk_end - chunk_start):
                row_vec = mat.getrow(local_i).toarray().ravel()
                try:
                    ac = acovf(row_vec, nlag=1, fft=True)[1]
                    noise_vals[chunk_start + local_i] = (
                        float("nan") if ac == 0 else 1.0 / abs(ac)
                    )
                except Exception:
                    noise_vals[chunk_start + local_i] = float("nan")

        chunk_start = chunk_end

    if not any_data:
        if os.path.exists(task.out_path):
            try:
                os.remove(task.out_path)
            except OSError:
                pass
        return task.chrom, task.out_path, False

    _write_off_diag_bedgraph(task, noise_vals)
    return task.chrom, task.out_path, True


def compute_full_noise(
    hic_path: str,
    res_list: List[int],
    chrom_sizes_path: Optional[str] = None,
    norm: str = "none",
    bindist_bp: Optional[int] = None,
    out_dir: str = ".",
    cooler_selection: Optional[str] = None,
    cpu: int = 1,
    chroms: Optional[List[str]] = None,
    dump_factor: float = 10.0,
    max_memory_gb: Optional[float] = None,
) -> None:
    """
    Compute noise values for every row/bin across the genome at the requested resolutions.
    All calculations rely on the selected backend for the input format.

    For each resolution in ``res_list``, this function:
    1. Builds a list of ``_FullNoiseTask`` objects — one per (chromosome, resolution).
    2. Cleans up any stale output files from previous runs.
    3. Dispatches tasks either serially (cpu=1) or via a multiprocessing pool (cpu>1).
       Each task is run with a single retry on failure (see below).
    4. Concatenates per-chromosome bedgraphs into a single ``{res}.bedgraph`` file
       per resolution in ``out_dir``.
    5. If any tasks failed after retry, writes a ``noise_bedgraph_errors.tsv`` log and
       raises a RuntimeError.

    Retry logic
    -----------
    Each task gets one automatic retry if it fails. This handles transient errors
    that can occur under memory pressure or in multiprocessing contexts (e.g. a
    worker process killed by the OS, a transient I/O error on a network filesystem).
    If the retry also fails, the error is logged and (after all tasks complete)
    a RuntimeError is raised summarising all failures.

    In parallel mode, the retry is run in the **main process** (not a worker) to
    avoid cascading failures if the worker pool is in a bad state.

    Multiprocessing
    ---------------
    ``maxtasksperchild=10`` limits each worker to 10 tasks before being replaced.
    This prevents memory accumulation in long-running workers (Hi-C matrices are
    large; loading many chromosomes in one worker can exhaust RAM over time).

    Output files
    ------------
    Per-chromosome files (intermediate): ``{out_dir}/{chrom}_{res}.bedgraph``
    Per-resolution merged file: ``{out_dir}/{res}.bedgraph``
    Error log (if failures): ``{out_dir}/noise_bedgraph_errors.tsv``

    Parameters
    ----------
    hic_path : str
        Path to the input .hic or cooler file.
    res_list : list of int
        One or more bin sizes in bp to process.
    chrom_sizes_path : str or None
        Optional TSV to restrict analysis to specific chromosomes and/or regions.
        Accepted formats:

        - Two columns: ``chrom_name<tab>size_bp`` — whole chromosome is processed.
        - Three or more columns: additional columns are regions as ``start:end``
          or ``start-end`` (0-based, half-open).  Only bins overlapping those
          regions are emitted; the noise window may extend beyond the region but
          is clamped at chromosome ends as usual.  Multiple regions per chromosome
          are supported (one per extra column, same row).  Regions that exceed the
          chromosome size or have start >= end are skipped with a warning.  A
          region smaller than one bin outputs the single bin that overlaps it.
    norm : str
        Normalisation mode: ``"none"``, ``"balance"``, or a custom weight path.
    bindist_bp : int or None
        Maximum genomic distance in bp for noise windows. None = use per-resolution
        default from ``_default_bindist_bp()``.
    out_dir : str
        Output directory (created if absent).
    cooler_selection : str or None
        HDF5 group path for cooler files.
    cpu : int
        Number of parallel worker processes (default 1 = serial).
    chroms : list of str or None
        If provided, restrict to these chromosome names.
    dump_factor : float
        Chunk size multiplier: each sliding-window chunk covers
        ``max(2, dump_factor) * noise_bins`` core bins.  Larger values increase
        peak RAM per worker; smaller values reduce RAM at the cost of more fetches.
        Default 10.0 is a good balance for resolutions down to ~1 kb.
    max_memory_gb : float or None
        Optional hard memory budget in gigabytes.  When set:
        - The worker count is capped so that total estimated peak RAM stays within
          this budget (using ``metrics.estimate_worker_memory_mb()``).
        - Each worker process receives a ``RLIMIT_AS`` address-space limit equal to
          ``max_memory_gb / workers`` GB, so allocations beyond the cap raise
          ``MemoryError`` (Linux/macOS only; silently ignored on Windows).
        The existing single-retry logic catches ``MemoryError`` from workers and
        re-runs the failed task in the main process.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Clamp requested workers to the number of available CPU cores.
    cpu_count = os.cpu_count() or 1
    requested_cpu = int(cpu) if isinstance(cpu, int) else 1
    if requested_cpu < 1:
        requested_cpu = 1
    workers = min(requested_cpu, cpu_count)

    # Memory-aware worker count: estimate per-worker peak RAM from dump_factor and
    # the noise window of the coarsest (most memory-intensive) resolution requested.
    per_worker_bytes: Optional[int] = None
    if max_memory_gb is not None and max_memory_gb > 0:
        coarsest_res = max(int(r) for r in res_list) if res_list else 10000
        noise_bins_est = max(1, _default_bindist_bp(coarsest_res) // coarsest_res)
        per_worker_mb = metrics.estimate_worker_memory_mb(dump_factor, noise_bins_est)
        if per_worker_mb > 0:
            safe_workers = max(1, int(max_memory_gb * 1024 / per_worker_mb))
            workers = min(workers, safe_workers)
        per_worker_bytes = max(
            int(1.5 * 1024**3),  # floor: 1.5 GB for Python + shared libs overhead
            int(max_memory_gb * 1024**3 / workers),
        )

    # --- Build task lists ---
    # A provider is opened briefly to enumerate chromosomes, then closed (``del provider``)
    # before tasks are dispatched to workers. Worker processes open their own providers.

    # Parse the extended chrom_sizes file once (resolution-independent).
    ext_sizes: Optional[Dict[str, Tuple[int, Optional[List[_RegionSpec]]]]] = None
    if chrom_sizes_path and os.path.exists(chrom_sizes_path):
        ext_sizes = _parse_extended_chrom_sizes(chrom_sizes_path)

    tasks: List[_FullNoiseTask] = []
    off_tasks: List[_OffDiagNoiseTask] = []
    for res_raw in res_list:
        res = int(res_raw)
        provider, _ = select_provider(hic_path, res, norm, cooler_selection)
        provider_sizes = provider.chrom_sizes()

        # Build chrom_info: chrom -> (chrom_len, region_specs_or_None)
        # chrom_len always comes from the provider (authoritative); specs from the file.
        if ext_sizes is not None:
            missing_in_hic = set(ext_sizes.keys()) - set(provider_sizes.keys())
            if missing_in_hic:
                print(
                    f"[WARN] Chromosomes in {chrom_sizes_path} not found in file at "
                    f"{res} bp: {sorted(missing_in_hic)}"
                )
            chrom_info: Dict[str, Tuple[int, Optional[List[_RegionSpec]]]] = {
                chrom: (provider_sizes[chrom], region_specs)
                for chrom, (_, region_specs) in ext_sizes.items()
                if chrom in provider_sizes
            }
        else:
            chrom_info = {c: (s, None) for c, s in provider_sizes.items()}

        if chroms is not None:
            requested = set(chroms)
            missing = requested - set(chrom_info.keys())
            if missing:
                print(f"[WARN] Requested chromosomes not found at {res} bp: {sorted(missing)}")
            chrom_info = {c: chrom_info[c] for c in chroms if c in chrom_info}

        # A chromosome shorter than the noise window cannot support a 2W diagonal
        # window, so its noise estimate would be meaningless.  These are also the
        # chromosomes that crash the native .hic reader (chrM at fine resolutions
        # makes hicstraw print "Error finding block data" and die with SIGFPE,
        # which used to take the whole run down).  Drop them up front.
        _window_bp = bindist_bp or _default_bindist_bp(res)
        _too_short = {c: L for c, (L, _) in chrom_info.items() if L < _window_bp}
        if _too_short:
            print(
                f"[INFO] {res} bp: skipping {len(_too_short)} chromosome(s) shorter than the "
                f"{_window_bp / 1e6:.3g} Mb noise window: "
                + ", ".join(f"{c} ({L / 1e3:.4g} kb)" for c, L in sorted(_too_short.items()))
            )
            chrom_info = {c: v for c, v in chrom_info.items() if c not in _too_short}

        for chrom, (chrom_len, region_specs) in chrom_info.items():
            # Partition region specs into on-diagonal and off-diagonal.
            # On-diagonal:  Tuple[int, int]                       — isinstance(spec[0], int) is True
            # Off-diagonal: Tuple[Tuple[int,int], Tuple[int,int]] — spec[0] is a tuple
            if region_specs is None:
                on_diag: Optional[List[Tuple[int, int]]] = None
                off_diag: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
            else:
                on_diag = [
                    spec for spec in region_specs  # type: ignore[misc]
                    if isinstance(spec[0], int)
                ]
                off_diag = [
                    spec for spec in region_specs  # type: ignore[misc]
                    if not isinstance(spec[0], int)
                ]

            # On-diagonal task: create when whole-chrom (None) or when explicit on-diag specs exist.
            if on_diag is not None or region_specs is None:
                out_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
                density_out_path = os.path.join(out_dir, f"{chrom}_{res}_density.bedgraph")
                tasks.append(
                    _FullNoiseTask(
                        hic_path=hic_path,
                        chrom=chrom,
                        chrom_len=chrom_len,
                        res=res,
                        norm=norm,
                        cooler_selection=cooler_selection,
                        bindist_bp=bindist_bp,
                        out_dir=out_dir,
                        out_path=out_path,
                        density_out_path=density_out_path,
                        dump_factor=dump_factor,
                        regions=tuple(tuple(r) for r in on_diag) if on_diag else None,
                    )
                )

            # Off-diagonal tasks: one per region pair.
            for (l_start, l_end), (r_start, r_end) in off_diag:
                safe_name = f"{l_start}-{l_end}_{r_start}-{r_end}"
                od_path = os.path.join(
                    out_dir, f"{chrom}_{safe_name}_{res}_offdiag.bedgraph"
                )
                off_tasks.append(
                    _OffDiagNoiseTask(
                        hic_path=hic_path,
                        chrom=chrom,
                        chrom_len=chrom_len,
                        res=res,
                        norm=norm,
                        cooler_selection=cooler_selection,
                        left_start=l_start,
                        left_end=l_end,
                        right_start=r_start,
                        right_end=r_end,
                        out_dir=out_dir,
                        out_path=od_path,
                        dump_factor=dump_factor,
                    )
                )

        del provider  # release file handle before forking worker processes

    # --- Remove stale output files from previous runs ---
    # This ensures the concatenation step only sees freshly-generated files.
    for task in tasks:
        for stale in (task.out_path, task.density_out_path):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

    results: List[Tuple[str, int, str, bool]] = []
    errors: List[Tuple[_FullNoiseTask, Exception, Exception]] = []

    def _run_with_retry(task: _FullNoiseTask) -> Tuple[str, int, str, bool]:
        """
        Run a noise task and retry once on failure.

        On the first failure: remove any partial output file, then retry.
        On the second failure: record the error in ``errors`` and re-raise so
        the caller can decide whether to continue or abort.

        This inner function is used in serial mode; parallel mode has its own
        equivalent retry logic in the main process after collecting pool results.
        """
        try:
            return _full_noise_worker(task)
        except Exception as first_exc:
            # Clean up any partial output before retrying.
            if os.path.exists(task.out_path):
                try:
                    os.remove(task.out_path)
                except OSError:
                    pass
            try:
                return _full_noise_worker(task)
            except Exception as second_exc:
                errors.append((task, first_exc, second_exc))
                raise

    if not tasks and not off_tasks:
        return

    # --- On-diagonal dispatch ---
    if tasks:
        if workers == 1:
            # --- Serial execution ---
            for task in tasks:
                try:
                    results.append(_run_with_retry(task))
                except Exception:
                    # Error already recorded in ``errors`` by ``_run_with_retry``.
                    continue
        else:
            # --- Parallel execution ---
            # ``apply_async`` submits all tasks immediately and returns AsyncResult handles.
            # Results are collected in submission order (``async_res.get()`` blocks until
            # the specific task completes). This preserves ordering without limiting
            # parallelism: the pool runs up to ``workers`` tasks simultaneously.
            # ProcessPoolExecutor, not multiprocessing.Pool: a worker killed by a
            # signal (hicstraw raises SIGFPE on some degenerate chromosomes) leaves
            # Pool.apply_async().get() blocking forever with no message, so a single
            # bad chromosome silently hangs a genome-wide run.  The executor raises
            # BrokenProcessPool instead, which is recoverable.
            ctx = mp.get_context()
            futures = {}
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(per_worker_bytes,),
                ) as pool:
                    futures = {pool.submit(_full_noise_worker, task): task for task in tasks}
                    pending = list(futures)
                    for fut in pending:
                        task = futures[fut]
                        try:
                            results.append(fut.result())
                        except BrokenProcessPool:
                            # The pool is dead; every remaining task must be retried
                            # in-process below rather than waited on.
                            raise
                        except Exception as first_exc:
                            if os.path.exists(task.out_path):
                                try:
                                    os.remove(task.out_path)
                                except OSError:
                                    pass
                            try:
                                results.append(_full_noise_worker(task))
                            except Exception as second_exc:
                                errors.append((task, first_exc, second_exc))
            except BrokenProcessPool:
                done_paths = {r[2] for r in results}
                stragglers = [t for t in tasks if t.out_path not in done_paths]
                print(
                    f"[WARN] a worker process died (likely a native crash in the .hic reader); "
                    f"retrying {len(stragglers)} chromosome(s) serially",
                    flush=True,
                )
                for task in stragglers:
                    # Retry in a fresh single-worker subprocess rather than in this
                    # process: whatever killed the pool is a native crash, and running
                    # it here would take the parent down with it.
                    try:
                        with ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=ctx,
                            initializer=_worker_init,
                            initargs=(per_worker_bytes,),
                        ) as solo:
                            results.append(solo.submit(_full_noise_worker, task).result())
                    except BrokenProcessPool:
                        print(
                            f"[WARN] {task.chrom} @ {task.res} bp crashed the .hic reader; "
                            f"skipping this chromosome",
                            flush=True,
                        )
                        for stale in (task.out_path, task.density_out_path):
                            if os.path.exists(stale):
                                try:
                                    os.remove(stale)
                                except OSError:
                                    pass
                        continue
                    except Exception:
                        continue

    # --- Off-diagonal dispatch ---
    od_results: List[Tuple[str, str, bool]] = []
    od_errors: List[Tuple[_OffDiagNoiseTask, Exception, Exception]] = []

    def _run_off_diag_with_retry(task: _OffDiagNoiseTask) -> Tuple[str, str, bool]:
        """Same retry logic as ``_run_with_retry`` but for off-diagonal tasks."""
        try:
            return _off_diag_noise_worker(task)
        except Exception as first_exc:
            if os.path.exists(task.out_path):
                try:
                    os.remove(task.out_path)
                except OSError:
                    pass
            try:
                return _off_diag_noise_worker(task)
            except Exception as second_exc:
                od_errors.append((task, first_exc, second_exc))
                raise

    if off_tasks:
        if workers == 1:
            for task in off_tasks:
                try:
                    od_results.append(_run_off_diag_with_retry(task))
                except Exception:
                    continue
        else:
            ctx = mp.get_context()
            with ctx.Pool(
                processes=workers,
                maxtasksperchild=10,
                initializer=_worker_init,
                initargs=(per_worker_bytes,),
            ) as pool:
                async_od = [
                    (task, pool.apply_async(_off_diag_noise_worker, (task,)))
                    for task in off_tasks
                ]
                for task, async_res in async_od:
                    try:
                        od_results.append(async_res.get())
                    except Exception as first_exc:
                        if os.path.exists(task.out_path):
                            try:
                                os.remove(task.out_path)
                            except OSError:
                                pass
                        try:
                            od_results.append(_off_diag_noise_worker(task))
                        except Exception as second_exc:
                            od_errors.append((task, first_exc, second_exc))

    # --- Collect successfully-generated per-chromosome bedgraph paths by resolution ---
    per_res_outputs: Dict[int, List[str]] = {}
    for chrom, res, path, generated in results:
        if not generated:
            continue
        per_res_outputs.setdefault(res, []).append(path)

    # --- Concatenate per-chromosome files into one per-resolution bedgraph ---
    # Chromosomes are sorted alphabetically within each resolution file.
    for res_raw in res_list:
        res = int(res_raw)
        merged = os.path.join(out_dir, f"{res}.bedgraph")
        if os.path.exists(merged):
            try:
                os.remove(merged)
            except OSError:
                pass
        paths = sorted(per_res_outputs.get(res, []))
        if not paths:
            continue
        with open(merged, "wb") as merged_handle:
            for path in paths:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as part:
                    merged_handle.write(part.read())

    # --- Concatenate per-chromosome density files into one per-resolution density bedgraph ---
    per_res_density: Dict[int, List[str]] = {}
    for chrom, res_val, _, generated in results:
        if not generated:
            continue
        dp = os.path.join(out_dir, f"{chrom}_{res_val}_density.bedgraph")
        if os.path.exists(dp):
            per_res_density.setdefault(res_val, []).append(dp)

    for res_raw in res_list:
        res_val = int(res_raw)
        merged_density = os.path.join(out_dir, f"{res_val}_density.bedgraph")
        if os.path.exists(merged_density):
            try:
                os.remove(merged_density)
            except OSError:
                pass
        dpaths = sorted(per_res_density.get(res_val, []))
        if not dpaths:
            continue
        with open(merged_density, "wb") as mh:
            for dp in dpaths:
                if os.path.exists(dp):
                    with open(dp, "rb") as part:
                        mh.write(part.read())

    # --- Report failures (on-diagonal and off-diagonal combined) ---
    all_failed = len(errors) + len(od_errors)
    if all_failed:
        error_path = os.path.join(out_dir, "noise_bedgraph_errors.tsv")
        with open(error_path, "w") as handle:
            handle.write("chrom\tres\ttask_type\tout_path\tmessage\n")
            for task, first_exc, second_exc in errors:
                handle.write(
                    f"{task.chrom}\t{task.res}\ton_diagonal\t{task.out_path}\t"
                    f"initial: {repr(first_exc)}; retry: {repr(second_exc)}\n"
                )
            for task, first_exc, second_exc in od_errors:
                handle.write(
                    f"{task.chrom}\t{task.res}\toff_diagonal\t{task.out_path}\t"
                    f"initial: {repr(first_exc)}; retry: {repr(second_exc)}\n"
                )
        raise RuntimeError(
            f"{all_failed} noise-bedgraph task(s) failed. See {error_path} for details."
        )
