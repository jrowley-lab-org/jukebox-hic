#!/usr/bin/env python
import math
import os
import statistics
from typing import Dict, List, Optional, Sequence

import hicstraw
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, acovf


def _read_chrom_sizes(hic_path: str, chrom_sizes_path: Optional[str]) -> Dict[str, int]:
    if chrom_sizes_path and os.path.exists(chrom_sizes_path):
        df = pd.read_csv(
            chrom_sizes_path,
            sep="\t",
            header=None,
            names=["chr", "size"],
            dtype={"chr": str, "size": int},
        )
        return dict(zip(df["chr"], df["size"]))

    hic = hicstraw.HiCFile(hic_path)
    sizes: Dict[str, int] = {}
    for chrom in hic.getChromosomes():
        if chrom.name == "ALL":
            continue
        sizes[str(chrom.name)] = int(chrom.length)
    return sizes


def _default_window_bp(resolution: int) -> int:
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _precompute_expectations(df_obs: pd.DataFrame) -> tuple[Dict[int, float], float]:
    df_obs = df_obs.copy()
    df_obs["lag"] = (df_obs["binY"] - df_obs["binX"]).abs()
    grouped = df_obs.groupby("lag")["count"].mean()
    lookup = grouped.to_dict()
    default = float(grouped.mean()) if len(grouped) else 0.0
    return lookup, default


def _row_observed_expected(
    df_obs: pd.DataFrame,
    row_idx: int,
    bindist_bins: int,
    expected_lookup: Dict[int, float],
    default_expected: float,
) -> tuple[np.ndarray, np.ndarray]:
    by_x = df_obs.groupby("binX")
    by_y = df_obs.groupby("binY")

    observed = np.zeros(bindist_bins * 2, dtype=float)
    expected = np.full(bindist_bins * 2, default_expected, dtype=float)

    left = by_y.get_group(row_idx) if row_idx in by_y.groups else pd.DataFrame(columns=df_obs.columns)
    left = left[(left["binX"] >= row_idx - bindist_bins) & (left["binX"] < row_idx)]

    for _, rec in left.iterrows():
        lag = row_idx - int(rec["binX"])
        idx = int(bindist_bins - lag)
        if 0 <= idx < len(observed):
            observed[idx] = float(rec["count"])
            expected[idx] = expected_lookup.get(lag, default_expected)

    right = by_x.get_group(row_idx) if row_idx in by_x.groups else pd.DataFrame(columns=df_obs.columns)
    right = right[(right["binY"] > row_idx) & (right["binY"] <= row_idx + bindist_bins)]

    for _, rec in right.iterrows():
        lag = int(rec["binY"]) - row_idx
        idx = int(bindist_bins + lag)
        if 0 <= idx < len(observed):
            observed[idx] = float(rec["count"])
            expected[idx] = expected_lookup.get(lag, default_expected)

    return observed, expected


def _compute_row_metrics(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> dict[str, float]:
    empty_ratio = 1.0
    if observed_counts.size:
        non_empty = int(np.count_nonzero(observed_counts))
        empty_ratio = 1.0 - (non_empty / float(observed_counts.size))

    scores = (observed_counts + 1.0) / (expected_counts + 1.0)
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size < 2 or np.allclose(finite, finite[0]):
        mean_val = float(finite.mean()) if finite.size else 0.0
        std_val = 0.0
        ac_val = 0.0
        noise_raw = 10000.0
    else:
        mean_val = float(finite.mean())
        try:
            std_val = float(statistics.stdev(finite.tolist()))
        except statistics.StatisticsError:
            std_val = 0.0

        try:
            ac_vals = acovf(finite, nlag=1, fft=True)
            ac1 = float(ac_vals[1]) if len(ac_vals) > 1 else 0.0
            noise_raw = 10000.0 if ac1 == 0 else 1.0 / abs(ac1)
        except Exception:
            noise_raw = 10000.0

        try:
            ac_series = ac(finite, nlags=1, fft=True)
            ac_val = float(ac_series[1]) if len(ac_series) > 1 else 0.0
        except Exception:
            ac_val = 0.0

    noise_log = math.log10(noise_raw + 1.0)

    return {
        "noise_raw": noise_raw,
        "noise_log": noise_log,
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
    subsample_ratios: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Compute per-row noise on the original map and derive subsampling summaries.

    Outputs
    -------
    - <chrom>_<res>.bedgraph : noise from the non-subsampled rows.
    - subsample_summary.tsv  : per-chromosome summary statistics for each subsample ratio.
    """
    rng = np.random.default_rng(seed)

    ratios = list(subsample_ratios) if subsample_ratios else [1.0]
    if 1.0 not in ratios:
        ratios.append(1.0)
    # Sort in descending order so that we can progressively subsample.
    ratios = sorted(set(ratios), reverse=True)

    hic = hicstraw.HiCFile(hic_path)
    chrom_sizes = _read_chrom_sizes(hic_path, chrom_sizes_path)

    window = window_bp or _default_window_bp(int(res))
    bindist_bins = max(1, window // int(res))

    summary_records: List[dict[str, float]] = []

    os.makedirs(out_dir, exist_ok=True)

    for chrom, chrom_len in chrom_sizes.items():
        records = hicstraw.straw("observed", "NONE", hic_path, chrom, chrom, "BP", int(res))
        if not records:
            continue
        df_obs = pd.DataFrame(
            {
                "binX": [(r.binX // res) for r in records],
                "binY": [(r.binY // res) for r in records],
                "count": [r.counts for r in records],
            }
        )

        expected_lookup, default_expected = _precompute_expectations(df_obs)

        total_rows = chrom_len // int(res) + 1
        start_idx = bindist_bins
        stop_idx = max(bindist_bins + 1, total_rows - bindist_bins)
        if stop_idx <= start_idx:
            continue

        candidate_rows = np.arange(start_idx, stop_idx)
        if float(sample_fraction) < 1.0:
            sample_n = max(1, int(len(candidate_rows) * float(sample_fraction)))
            row_indices = np.sort(rng.choice(candidate_rows, size=sample_n, replace=False))
        else:
            row_indices = candidate_rows

        chrom_bed: List[tuple[int, float]] = []
        ratio_stats = {
            ratio: {
                "noise_log": [],
                "row_mean": [],
                "row_std": [],
                "acf": [],
                "empty_ratio": [],
            }
            for ratio in ratios
        }

        for row_idx in row_indices:
            observed_vec, expected_vec = _row_observed_expected(
                df_obs,
                row_idx=row_idx,
                bindist_bins=bindist_bins,
                expected_lookup=expected_lookup,
                default_expected=default_expected,
            )

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
                        # guard for numerical issues
                        incremental = min(max(incremental, 0.0), 1.0)
                        counts = rng.binomial(current_counts, incremental)
                        prev_ratio = ratio
                        current_counts = counts

                expected_scaled = expected_vec * ratio
                metrics = _compute_row_metrics(counts.astype(float), expected_scaled.astype(float))

                ratio_stats[ratio]["noise_log"].append(metrics["noise_log"])
                ratio_stats[ratio]["row_mean"].append(metrics["row_mean"])
                ratio_stats[ratio]["row_std"].append(metrics["row_std"])
                ratio_stats[ratio]["acf"].append(metrics["acf"])
                ratio_stats[ratio]["empty_ratio"].append(metrics["empty_ratio"])

                if ratio == 1.0:
                    chrom_bed.append((row_idx, metrics["noise_raw"]))

        if chrom_bed:
            bed_path = os.path.join(out_dir, f"{chrom}_{res}.bedgraph")
            with open(bed_path, "w") as handle:
                for row_idx, value in sorted(chrom_bed):
                    start_bp = row_idx * int(res)
                    stop_bp = min((row_idx + 1) * int(res), chrom_len)
                    handle.write(f"{chrom} {start_bp} {stop_bp} {value}\n")

        for ratio, stats_dict in ratio_stats.items():
            if not stats_dict["noise_log"]:
                continue
            summary_records.append(
                {
                    "chrom": chrom,
                    "ratio": ratio,
                    "rows_evaluated": len(stats_dict["noise_log"]),
                    "mean_noise_log10": float(np.nanmean(stats_dict["noise_log"])),
                    "median_noise_log10": float(np.nanmedian(stats_dict["noise_log"])),
                    "std_noise_log10": float(np.nanstd(stats_dict["noise_log"], ddof=0)),
                    "mean_row_mean": float(np.nanmean(stats_dict["row_mean"])),
                    "mean_row_std": float(np.nanmean(stats_dict["row_std"])),
                    "mean_acf": float(np.nanmean(stats_dict["acf"])),
                    "mean_empty_bin_ratio": float(np.nanmean(stats_dict["empty_ratio"])),
                }
            )

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.sort_values(["chrom", "ratio"], inplace=True)
        summary_path = os.path.join(out_dir, "subsample_summary.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
