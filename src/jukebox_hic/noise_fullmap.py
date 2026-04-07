#!/usr/bin/env python
"""
Genome-wide (full-map) Hi-C noise computation.

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

By computing noise for every bin, ``noise_fullmap`` produces a genome-wide noise
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
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.stattools import acovf

from .backends import ChromMatrix, read_chrom_sizes, select_provider


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
) -> np.ndarray:
    """
    Compute the noise value for every bin in a chromosome's contact matrix.

    For each genomic bin ``idx`` (from 0 to rowbins-1), this function:
    1. Constructs an observed-count window vector of length ``bindist_bins * 2``
       covering contacts within ``bindist_bins`` bins of the diagonal.
    2. Constructs a matching expected-count vector using the distance-decay model.
    3. Computes the lag-1 autocovariance of the ratio series and returns
       ``1 / |autocovariance_lag1|`` as the noise value.

    The window is split into two halves:
    - **Upstream half** (positions 0 … bindist_bins−1): contacts with bins j < idx.
      Fetched from the CSC matrix (column slicing is efficient in CSC format).
      Position mapping: ``pos = bindist_bins - lag`` where ``lag = idx - j``.
      So lag=1 → position bindist_bins−1 (closest upstream neighbour).
    - **Downstream half** (positions bindist_bins … 2*bindist_bins−1): contacts with
      bins j > idx. Fetched from the CSR matrix (row slicing is efficient in CSR).
      Position mapping: ``pos = bindist_bins - 1 + lag`` where ``lag = j - idx``.
      So lag=1 → position bindist_bins (closest downstream neighbour).

    The asymmetric indexing keeps the diagonal at the boundary between the two halves
    (between positions bindist_bins−1 and bindist_bins) rather than at the centre.
    This simplifies the position arithmetic and makes it clear which half is upstream
    vs. downstream.

    Noise metric
    ------------
    For each bin, the ratio series ``score[k] = observed[k] / expected[k]`` (with
    pseudo-counts) is computed. The lag-1 autocovariance of this series (``acovf``)
    captures how smoothly the ratios vary with genomic distance:
    - High autocovariance → smooth, predictable variation → LOW noise (small value).
    - Low autocovariance → erratic, random variation → HIGH noise (large value).

    The noise value is ``1 / |acovf_lag1|``, so noisier bins get larger values.
    NaN is returned if the autocovariance is zero (perfectly alternating series)
    or if an exception occurs during computation (e.g. too few data points).

    Parameters
    ----------
    matrix : ChromMatrix
        Sparse contact matrix for this chromosome (CSR and CSC formats used).
    rowbins : int
        Total number of bins in the chromosome (= matrix dimension).
    bindist_bins : int
        Half-window size in bins (= bindist_bp // resolution).
    expected_lookup : np.ndarray
        Distance-decay lookup table from ``_precompute_expectation()``.
    default_expected : float
        Fallback expected value for lags with no data.

    Returns
    -------
    noise_vals : np.ndarray of shape (rowbins,)
        Per-bin noise values. NaN for bins where the metric could not be computed.
    """
    noise_vals = np.zeros(rowbins, dtype=float)
    for idx in range(rowbins):
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
        try:
            # ``ac`` is the lag-1 autocovariance of the raw observed vector.
            # Using observed (not the ratio) here for speed — the expected curve is
            # roughly constant over the short window, so the ratio and observed series
            # have nearly the same autocorrelation structure.
            ac = acovf(observed, nlag=1, fft=True)[1]
            noise_vals[idx] = float("nan") if ac == 0 else 1.0 / abs(ac)
        except Exception:
            noise_vals[idx] = float("nan")

    return noise_vals


def _write_bedgraph(
    chrom: str,
    res: int,
    chrom_len: int,
    noise_vals: np.ndarray,
    out_path: str,
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
    """
    with open(out_path, "w") as handle:
        for i, val in enumerate(noise_vals):
            start = i * res
            stop = min((i + 1) * res, chrom_len)
            handle.write(f"{chrom} {start} {stop} {val}\n")


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
    matrix = provider.fetch_chrom(task.chrom)
    if matrix is None or matrix.coo.nnz == 0:
        # No contacts for this chromosome — remove any stale output and signal failure.
        if os.path.exists(task.out_path):
            try:
                os.remove(task.out_path)
            except OSError:
                pass
        return task.chrom, task.res, task.out_path, False

    window_bp = task.bindist_bp or _default_bindist_bp(int(task.res))
    bindist_bins = max(1, window_bp // int(task.res))
    expected_lookup, default_expected = _precompute_expectation(matrix.coo)
    rowbins = matrix.coo.shape[0]
    noise_vals = _row_window_noise(
        matrix,
        rowbins=rowbins,
        bindist_bins=bindist_bins,
        expected_lookup=expected_lookup,
        default_expected=default_expected,
    )

    _write_bedgraph(task.chrom, int(task.res), task.chrom_len, noise_vals, task.out_path)
    return task.chrom, task.res, task.out_path, True


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
    5. If any tasks failed after retry, writes a ``full_noise_errors.tsv`` log and
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
    Error log (if failures): ``{out_dir}/full_noise_errors.tsv``

    Parameters
    ----------
    hic_path : str
        Path to the input .hic or cooler file.
    res_list : list of int
        One or more bin sizes in bp to process.
    chrom_sizes_path : str or None
        Optional TSV to restrict or filter chromosomes.
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
    """
    os.makedirs(out_dir, exist_ok=True)

    # Clamp requested workers to the number of available CPU cores.
    cpu_count = os.cpu_count() or 1
    requested_cpu = int(cpu) if isinstance(cpu, int) else 1
    if requested_cpu < 1:
        requested_cpu = 1
    workers = min(requested_cpu, cpu_count)

    # --- Build task list ---
    # A provider is opened briefly to enumerate chromosomes, then closed (``del provider``)
    # before tasks are dispatched to workers. Worker processes open their own providers.
    tasks: List[_FullNoiseTask] = []
    for res_raw in res_list:
        res = int(res_raw)
        provider, _ = select_provider(hic_path, res, norm, cooler_selection)
        chrom_sizes = read_chrom_sizes(provider, chrom_sizes_path)
        if chroms is not None:
            requested = set(chroms)
            missing = requested - set(chrom_sizes.keys())
            if missing:
                print(f"[WARN] Requested chromosomes not found at {res} bp: {sorted(missing)}")
            chrom_sizes = {c: chrom_sizes[c] for c in chroms if c in chrom_sizes}
        for chrom, chrom_len in chrom_sizes.items():
            out_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
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
                )
            )
        del provider  # release file handle before forking worker processes

    # --- Remove stale output files from previous runs ---
    # This ensures the concatenation step only sees freshly-generated files.
    for task in tasks:
        if os.path.exists(task.out_path):
            try:
                os.remove(task.out_path)
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

    if not tasks:
        return

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
        ctx = mp.get_context()
        with ctx.Pool(processes=workers, maxtasksperchild=10) as pool:
            async_results = [(task, pool.apply_async(_full_noise_worker, (task,))) for task in tasks]
            for task, async_res in async_results:
                try:
                    results.append(async_res.get())
                except Exception as first_exc:
                    # Worker failed — clean up partial output and retry in main process.
                    if os.path.exists(task.out_path):
                        try:
                            os.remove(task.out_path)
                        except OSError:
                            pass
                    try:
                        results.append(_full_noise_worker(task))
                    except Exception as second_exc:
                        errors.append((task, first_exc, second_exc))

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

    # --- Report failures ---
    if errors:
        error_path = os.path.join(out_dir, "full_noise_errors.tsv")
        with open(error_path, "w") as handle:
            handle.write("chrom\tres\tmessage\n")
            for task, first_exc, second_exc in errors:
                handle.write(
                    f"{task.chrom}\t{task.res}\tinitial: {repr(first_exc)}; retry: {repr(second_exc)}\n"
                )
        raise RuntimeError(
            f"{len(errors)} full-noise task(s) failed. See {error_path} for details."
        )
