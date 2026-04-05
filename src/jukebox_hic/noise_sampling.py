#!/usr/bin/env python
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
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _precompute_expectation(coo: sparse.coo_matrix) -> Tuple[np.ndarray, float]:
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


def _row_window_vectors(
    matrices: ChromMatrix,
    row_idx: int,
    bindist_bins: int,
    expected_lookup: np.ndarray,
    default_expected: float,
) -> Tuple[np.ndarray, np.ndarray]:
    window = bindist_bins * 2
    observed = np.zeros(window, dtype=float)
    expected = np.full(window, default_expected, dtype=float)

    col_vec = matrices.csc.getcol(row_idx)
    for src_idx, value in zip(col_vec.indices, col_vec.data):
        if src_idx >= row_idx:
            continue
        lag = row_idx - int(src_idx)
        if lag > bindist_bins:
            continue
        pos = bindist_bins - lag
        observed[pos] = float(value)
        expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

    row_vec = matrices.csr.getrow(row_idx)
    for dst_idx, value in zip(row_vec.indices, row_vec.data):
        if dst_idx <= row_idx:
            continue
        lag = int(dst_idx) - row_idx
        if lag > bindist_bins:
            continue
        pos = bindist_bins - 1 + lag
        observed[pos] = float(value)
        expected[pos] = _expected_for_lag(expected_lookup, default_expected, lag)

    return observed, expected


def _compute_row_metrics(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> Dict[str, float]:
    empty_ratio = 1.0
    if observed_counts.size:
        non_empty = int(np.count_nonzero(observed_counts))
        empty_ratio = 1.0 - (non_empty / float(observed_counts.size))

    scores = (observed_counts + 1.0) / (expected_counts + 1.0)
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]

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

        try:
            ac_vals = acovf(finite, nlag=1, fft=True)
            ac1 = float(ac_vals[1]) if len(ac_vals) > 1 else 0.0
            noise_raw = float("nan") if ac1 == 0 else 1.0 / abs(ac1)
        except Exception:
            noise_raw = float("nan")

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
    min_mean_score: float = 0.9,
    min_nonzero_frac: float = 0.05,
    subsample_ratios: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    profile_path: Optional[str] = None,
    chroms: Optional[List[str]] = None,
) -> None:
    """Sample a subset of rows, compute noise metrics, and optionally profile usage."""

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

    ratios = list(subsample_ratios) if subsample_ratios else [1.0]
    if 1.0 not in ratios:
        ratios.append(1.0)
    ratios = sorted(set(float(r) for r in ratios), reverse=True)

    window = window_bp or _default_window_bp(int(res))
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

        matrices = provider.fetch_chrom(chrom)
        if matrices is None or matrices.coo.nnz == 0:
            continue
        chrom_bp_sizes[chrom] = chrom_len
        chrom_contacts = total_contacts(matrices)
        contacts_by_chrom[chrom] = chrom_contacts

        expected_lookup, default_expected = _precompute_expectation(matrices.coo)

        rowbins = matrices.coo.shape[0]
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

        if sample_all:
            row_order = candidate_rows
        else:
            row_order = rng.permutation(candidate_rows)

        selected_rows: List[int] = []
        failed_rows = 0
        exhausted = False
        chrom_bed: List[Tuple[int, float]] = []
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

        for row_idx in row_order:
            observed_vec, expected_vec = _row_window_vectors(
                matrices,
                row_idx=row_idx,
                bindist_bins=bindist_bins,
                expected_lookup=expected_lookup,
                default_expected=default_expected,
            )

            scores = (observed_vec + 1.0) / (expected_vec + 1.0)
            finite_scores = scores[np.isfinite(scores)]
            mean_score = float(np.nanmean(finite_scores)) if finite_scores.size else 0.0
            nonzero_frac = float(np.count_nonzero(observed_vec) / observed_vec.size) if observed_vec.size else 0.0

            if mean_score < float(min_mean_score) or nonzero_frac < float(min_nonzero_frac):
                failed_rows += 1
                if sample_all:
                    chrom_bed.append((row_idx, float("nan")))
                continue

            selected_rows.append(row_idx)

            observed_int = np.clip(np.rint(observed_vec), 0, None).astype(int)
            current_counts = observed_int.copy()
            prev_ratio = 1.0

            for ratio in ratios:
                if ratio == 1.0:
                    counts = observed_int.copy()
                    prev_ratio = ratio
                else:
                    if prev_ratio == 0:
                        counts = np.zeros_like(current_counts)
                    else:
                        incremental = ratio / prev_ratio
                        incremental = float(np.clip(incremental, 0.0, 1.0))
                        counts = rng.binomial(current_counts, incremental)
                        prev_ratio = ratio
                        current_counts = counts

                expected_scaled = expected_vec * ratio
                metrics = _compute_row_metrics(counts.astype(float), expected_scaled.astype(float))

                ratio_stats[ratio]["noise_raw"].append(metrics["noise_raw"])
                ratio_stats[ratio]["row_mean"].append(metrics["row_mean"])
                ratio_stats[ratio]["row_std"].append(metrics["row_std"])
                ratio_stats[ratio]["acf"].append(metrics["acf"])
                ratio_stats[ratio]["empty_ratio"].append(metrics["empty_ratio"])

                if ratio == 1.0:
                    chrom_bed.append((row_idx, metrics["noise_raw"]))

            if len(selected_rows) >= requested_n:
                break

        if len(selected_rows) < requested_n:
            exhausted = True

        if chrom_bed:
            bed_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
            with open(bed_path, "w") as handle:
                for row_idx, value in sorted(chrom_bed):
                    start_bp = row_idx * int(res)
                    stop_bp = min((row_idx + 1) * int(res), chrom_len)
                    handle.write(f"{chrom} {start_bp} {stop_bp} {value}\n")

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
                "estimated_contacts": float(chrom_contacts * float(ratio)),
                "z_map": float("nan"),
                "gamma": float("nan"),
                "pred_log_N": float("nan"),
                "residual": float("nan"),
                "advisor_target_density": float("nan"),
                "advisor_fold_increase": float("nan"),
                "advisor_efficiency_index": float("nan"),
                "advisor_recommendation": "",
            }
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
                        zmap_result = reference.compute_zmap(
                            contacts=chrom_contacts,
                            chrom_len=c_len,
                            resolution=int(res),
                            median_noise=median_noise,
                            ebr=mean_ebr,
                        )
                        record["z_map"] = zmap_result["z_map"]
                        record["gamma"] = zmap_result["gamma"]
                        record["pred_log_N"] = zmap_result["pred_log_N"]
                        record["residual"] = zmap_result["residual"]

                        current_rho = chrom_contacts / (c_len / int(res))
                        advisor_result = reference.sequencing_advisor(
                            obs_noise=median_noise,
                            current_rho=current_rho,
                            ebr=mean_ebr,
                        )
                        record["advisor_target_density"] = advisor_result["target_density"]
                        record["advisor_fold_increase"] = advisor_result["fold_increase"]
                        record["advisor_efficiency_index"] = advisor_result["efficiency_index"]
                        record["advisor_recommendation"] = advisor_result["recommendation"]
                    except Exception:
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

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.sort_values(["chrom", "ratio"], inplace=True)
        summary_path = os.path.join(out_dir, "subsample_summary.tsv")
        with open(summary_path, "w") as handle:
            handle.write(f"# resolution={int(res)}\n")
            for chrom in sorted(contacts_by_chrom):
                handle.write(
                    f"# chrom_info\t{chrom}\tcontacts={contacts_by_chrom[chrom]:.6f}\t"
                    f"size_bp={chrom_bp_sizes.get(chrom, 'NA')}\n"
                )
            summary_df.to_csv(handle, sep="\t", index=False)

    if profile_path and profile_rows:
        profile_abs = os.path.abspath(profile_path)
        os.makedirs(os.path.dirname(profile_abs) or ".", exist_ok=True)
        exists = os.path.exists(profile_abs)
        mode = "a" if exists else "w"
        header = not exists
        pd.DataFrame(profile_rows).to_csv(profile_abs, index=False, mode=mode, header=header)
