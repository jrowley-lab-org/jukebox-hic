#!/usr/bin/env python
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.stattools import acovf

from .backends import ChromMatrix, read_chrom_sizes, select_provider


def _default_bindist_bp(resolution: int) -> int:
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _precompute_expectation(coo) -> Tuple[np.ndarray, float]:
    if coo.data.size == 0:
        return np.zeros(1, dtype=float), 0.0
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    data = coo.data.astype(float)
    lags = np.abs(cols - rows)
    max_lag = int(lags.max())
    sums = np.bincount(lags, weights=data, minlength=max_lag + 1)
    counts = np.bincount(lags, minlength=max_lag + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    default = float(means[counts > 0].mean()) if np.count_nonzero(counts) else 0.0
    return means, default


def _expected_for_lag(expected_lookup: np.ndarray, default: float, lag: int) -> float:
    if lag < len(expected_lookup):
        val = expected_lookup[lag]
        return float(val) if val > 0 else default
    return default


def _row_window_noise(matrix: ChromMatrix, rowbins: int, bindist_bins: int, expected_lookup: np.ndarray, default_expected: float) -> np.ndarray:
    noise_vals = np.zeros(rowbins, dtype=float)
    for idx in range(rowbins):
        window = bindist_bins * 2
        observed = np.zeros(window, dtype=float)
        expected = np.full(window, default_expected, dtype=float)

        col_vec = matrix.csc.getcol(idx)
        for src_idx, value in zip(col_vec.indices, col_vec.data):
            if src_idx >= idx:
                continue
            lag = idx - int(src_idx)
            if lag > bindist_bins:
                continue
            pos = bindist_bins - lag
            observed[pos] = float(value)
            expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

        row_vec = matrix.csr.getrow(idx)
        for dst_idx, value in zip(row_vec.indices, row_vec.data):
            if dst_idx <= idx:
                continue
            lag = int(dst_idx) - idx
            if lag > bindist_bins:
                continue
            pos = bindist_bins - 1 + lag
            observed[pos] = float(value)
            expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

        try:
            ac = acovf(observed, nlag=1, fft=True)[1]
            noise_vals[idx] = float("nan") if ac == 0 else 1.0 / abs(ac)
        except Exception:
            noise_vals[idx] = float("nan")

    return noise_vals


def _write_bedgraph(chrom: str, res: int, chrom_len: int, noise_vals: np.ndarray, out_path: str) -> None:
    with open(out_path, "w") as handle:
        for i, val in enumerate(noise_vals):
            start = i * res
            stop = min((i + 1) * res, chrom_len)
            handle.write(f"{chrom} {start} {stop} {val}\n")


@dataclass(frozen=True)
class _FullNoiseTask:
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
    provider, _ = select_provider(task.hic_path, task.res, task.norm, task.cooler_selection)
    matrix = provider.fetch_chrom(task.chrom)
    if matrix is None or matrix.coo.nnz == 0:
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
    """
    os.makedirs(out_dir, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    requested_cpu = int(cpu) if isinstance(cpu, int) else 1
    if requested_cpu < 1:
        requested_cpu = 1
    workers = min(requested_cpu, cpu_count)

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
        del provider

    for task in tasks:
        if os.path.exists(task.out_path):
            try:
                os.remove(task.out_path)
            except OSError:
                pass

    results: List[Tuple[str, int, str, bool]] = []
    errors: List[Tuple[_FullNoiseTask, Exception, Exception]] = []

    def _run_with_retry(task: _FullNoiseTask) -> Tuple[str, int, str, bool]:
        try:
            return _full_noise_worker(task)
        except Exception as first_exc:
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
        for task in tasks:
            try:
                results.append(_run_with_retry(task))
            except Exception:
                continue
    else:
        ctx = mp.get_context()
        with ctx.Pool(processes=workers, maxtasksperchild=10) as pool:
            async_results = [(task, pool.apply_async(_full_noise_worker, (task,))) for task in tasks]
            for task, async_res in async_results:
                try:
                    results.append(async_res.get())
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

    per_res_outputs: Dict[int, List[str]] = {}
    for chrom, res, path, generated in results:
        if not generated:
            continue
        per_res_outputs.setdefault(res, []).append(path)

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
