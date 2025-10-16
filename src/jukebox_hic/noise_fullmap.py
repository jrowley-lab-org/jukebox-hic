#!/usr/bin/env python
import os
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


def compute_full_noise(
    hic_path: str,
    res_list: List[int],
    chrom_sizes_path: Optional[str] = None,
    norm: str = "none",
    bindist_bp: Optional[int] = None,
    out_dir: str = ".",
    cooler_selection: Optional[str] = None,
) -> None:
    """
    Compute noise values for every row/bin across the genome at the requested resolutions.
    All calculations rely solely on hicstraw outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    for res in res_list:
        provider, _ = select_provider(hic_path, int(res), norm, cooler_selection)
        chrom_sizes = read_chrom_sizes(provider, chrom_sizes_path)
        window_bp = bindist_bp or _default_bindist_bp(int(res))
        bindist_bins = max(1, window_bp // int(res))
        per_chrom_paths: List[str] = []

        for chrom, chrom_len in chrom_sizes.items():
            matrix = provider.fetch_chrom(chrom)
            if matrix is None or matrix.coo.nnz == 0:
                continue

            expected_lookup, default_expected = _precompute_expectation(matrix.coo)
            rowbins = matrix.coo.shape[0]
            noise_vals = _row_window_noise(
                matrix,
                rowbins=rowbins,
                bindist_bins=bindist_bins,
                expected_lookup=expected_lookup,
                default_expected=default_expected,
            )

            out_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
            _write_bedgraph(chrom, int(res), chrom_len, noise_vals, out_path)
            per_chrom_paths.append(out_path)

        merged = os.path.join(out_dir, f"{res}.bedgraph")
        with open(merged, "wb") as merged_handle:
            for path in per_chrom_paths:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as part:
                    merged_handle.write(part.read())
