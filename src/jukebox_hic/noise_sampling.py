#!/usr/bin/env python
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.tsa.stattools import acf, acovf

try:
    import hicstraw  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise RuntimeError("hicstraw is required for .hic inputs") from exc

try:
    import cooler  # type: ignore
    from cooler.fileops import list_coolers  # type: ignore
except ImportError:
    cooler = None  # type: ignore
    list_coolers = None  # type: ignore


COOLER_SUFFIXES = (".cool", ".mcool", ".scool")


def _read_chrom_sizes_tsv(path: str) -> Dict[str, int]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "size"],
        dtype={"chr": str, "size": int},
    )
    return dict(zip(df["chr"], df["size"]))


def _default_window_bp(resolution: int) -> int:
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _normalize_hic_norm(norm: str) -> str:
    lower = norm.lower()
    if lower == "none":
        return "NONE"
    if lower == "balance":
        return "KR"
    if os.path.exists(norm):
        raise ValueError("Custom normalization vectors are not supported for .hic inputs")
    return norm.upper()


def _is_cooler_like(path: str) -> bool:
    return "::" in path or any(path.lower().endswith(sfx) for sfx in COOLER_SUFFIXES)


def _resolve_cooler_uri(path: str, res: int, selection: Optional[str]) -> str:
    if "::" in path:
        return path
    lower = path.lower()
    if lower.endswith(".cool"):
        return path
    if lower.endswith(".mcool"):
        if list_coolers is None:
            raise RuntimeError("cooler is required for .mcool handling")
        # Build the target group and accept either a full URI or a bare group path
        target_group = f"/resolutions/{res}"
        candidate_full = f"{path}::{target_group}"
        available = list(list_coolers(path))
        # Normalize to both full-URI and group-only sets
        avail_full = set(available)
        avail_groups = set(u.split("::", 1)[1] if "::" in u else u for u in available)
        if candidate_full in avail_full or target_group in avail_groups:
            return candidate_full
        # Provide a helpful error listing what's present
        pretty = sorted(avail_groups)
        raise ValueError(
            f"Resolution {res} bp not found in {path}. Available groups: {pretty}"
        )
    if lower.endswith(".scool"):
        if not selection:
            raise ValueError("Provide --cooler_path=<group> for .scool files (e.g. cell name)")
        return f"{path}::{selection}"
    raise ValueError(f"Unrecognized cooler-like suffix in {path}")


def _load_custom_weights(path: str, bins_df: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 1:
        if len(df) != len(bins_df):
            raise ValueError(f"Custom vector length {len(df)} does not match number of bins {len(bins_df)}")
        weights = df.iloc[:, 0].to_numpy(dtype=float)
    elif df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["chrom", "start", "end", "value"]
        merge = bins_df[["chrom", "start"]].merge(df[["chrom", "start", "value"]], on=["chrom", "start"], how="left")
        weights = merge["value"].fillna(0.0).to_numpy(dtype=float)
        if np.count_nonzero(np.isnan(weights)):
            raise ValueError("Custom normalization has NaNs after alignment; please verify formatting")
    else:
        raise ValueError("Custom normalization file must have either 1 column or chrom/start/end/value columns")
    return weights


@dataclass
class ChromMatrix:
    chrom: str
    chrom_len: int
    coo: sparse.coo_matrix
    csr: sparse.csr_matrix
    csc: sparse.csc_matrix


class BaseProvider:
    def chrom_sizes(self) -> Dict[str, int]:
        raise NotImplementedError

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        raise NotImplementedError


class HiCProvider(BaseProvider):
    def __init__(self, path: str, res: int, norm: str) -> None:
        self.path = path
        self.res = int(res)
        self.norm = _normalize_hic_norm(norm)
        self._hic = hicstraw.HiCFile(path)
        self._chrom_sizes = {
            str(chrom.name): int(chrom.length)
            for chrom in self._hic.getChromosomes()
            if chrom.name != "ALL"
        }

    def chrom_sizes(self) -> Dict[str, int]:
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        chrom_len = self._chrom_sizes.get(chrom)
        if chrom_len is None:
            return None
        records = hicstraw.straw(
            "observed",
            self.norm,
            self.path,
            chrom,
            chrom,
            "BP",
            self.res,
        )
        if not records:
            return None
        rows = []
        cols = []
        data = []
        for rec in records:
            rows.append(rec.binX // self.res)
            cols.append(rec.binY // self.res)
            data.append(float(rec.counts))
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        data_arr = np.asarray(data, dtype=float)

        mask = row_arr != col_arr
        row_sym = np.concatenate([row_arr, col_arr[mask]])
        col_sym = np.concatenate([col_arr, row_arr[mask]])
        data_sym = np.concatenate([data_arr, data_arr[mask]])

        rowbins = chrom_len // self.res + 1
        coo = sparse.coo_matrix((data_sym, (row_sym, col_sym)), shape=(rowbins, rowbins))
        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)


class CoolerProvider(BaseProvider):
    def __init__(self, path: str, res: int, norm: str, selection: Optional[str]) -> None:
        if cooler is None:
            raise RuntimeError("cooler is required for .cool/.mcool/.scool inputs")
        self.res = int(res)
        uri = _resolve_cooler_uri(path, self.res, selection)
        self._cooler = cooler.Cooler(uri)

        bin_size = int(self._cooler.info.get("bin-size", self.res))
        if bin_size != self.res:
            raise ValueError(f"Input resolution {self.res} bp does not match cooler bin size {bin_size} bp")

        self._chrom_sizes = dict(self._cooler.chromsizes)
        self._bins = self._cooler.bins()[:]
        self._bins["chrom"] = self._bins["chrom"].astype(str)
        self._norm_mode = "none"
        self._custom_weights: Optional[np.ndarray] = None

        norm_lower = norm.lower()
        if norm_lower == "none":
            self._norm_mode = "none"
        elif norm_lower == "balance":
            if "weight" not in self._bins.columns or self._bins["weight"].isna().all():
                raise ValueError("Cooler file does not contain balance weights")
            self._norm_mode = "balance"
        elif os.path.exists(norm):
            self._norm_mode = "custom"
            self._custom_weights = _load_custom_weights(norm, self._bins)
        else:
            raise ValueError("Unsupported cooler normalization; choose none, balance, or provide vector path")

        self._raw_matrix = self._cooler.matrix(balance=False, sparse=True)
        self._balanced_matrix = (
            self._cooler.matrix(balance=True, sparse=True) if self._norm_mode == "balance" else None
        )

    def chrom_sizes(self) -> Dict[str, int]:
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        if chrom not in self._chrom_sizes:
            return None
        start, end = self._cooler.extent(chrom)
        if end <= start:
            return None
        chrom_len = int(self._chrom_sizes[chrom])
        if self._norm_mode == "balance":
            sub = self._balanced_matrix.fetch(chrom)  # type: ignore[union-attr]
        else:
            sub = self._raw_matrix.fetch(chrom)
        coo = sub.tocoo()

        if self._norm_mode == "custom":
            weights = self._custom_weights[start:end]
            denom = weights[coo.row] * weights[coo.col]
            with np.errstate(divide="ignore", invalid="ignore"):
                data = np.divide(coo.data, denom, out=np.zeros_like(coo.data), where=denom != 0)
            coo = sparse.coo_matrix((data, (coo.row, coo.col)), shape=coo.shape)

        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)


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

    # Left of diagonal via CSC
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

    # Right of diagonal via CSR
    row_vec = matrices.csr.getrow(row_idx)
    for dst_idx, value in zip(row_vec.indices, row_vec.data):
        if dst_idx <= row_idx:
            continue
        lag = int(dst_idx) - row_idx
        if lag > bindist_bins:
            continue
        # Indices [bindist_bins .. 2*bindist_bins-1] map to lags 1..bindist_bins
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
            ac_series = acf(finite, nlags=1, fft=True)
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


def _select_provider(path: str, res: int, norm: str, cooler_selection: Optional[str]) -> BaseProvider:
    if _is_cooler_like(path):
        return CoolerProvider(path, res, norm, cooler_selection)
    return HiCProvider(path, res, norm)


def compute_sampled_noise(
    hic_path: str,
    res: int,
    sample_fraction: float = 1.0,
    window_bp: Optional[int] = None,
    chrom_sizes_path: Optional[str] = None,
    out_dir: str = ".",
    norm: str = "none",
    cooler_selection: Optional[str] = None,
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
    provider = _select_provider(hic_path, int(res), norm, cooler_selection)

    chrom_sizes = provider.chrom_sizes()
    if chrom_sizes_path:
        user_sizes = _read_chrom_sizes_tsv(chrom_sizes_path)
        chrom_sizes = {chrom: user_sizes[chrom] for chrom in chrom_sizes if chrom in user_sizes}

    if not chrom_sizes:
        raise ValueError("No chromosomes available for analysis")

    rng = np.random.default_rng(seed)

    ratios = list(subsample_ratios) if subsample_ratios else [1.0]
    if 1.0 not in ratios:
        ratios.append(1.0)
    ratios = sorted(set(float(r) for r in ratios), reverse=True)

    window = window_bp or _default_window_bp(int(res))
    bindist_bins = max(1, window // int(res))

    summary_records = []
    os.makedirs(out_dir, exist_ok=True)

    for chrom, chrom_len in chrom_sizes.items():
        matrices = provider.fetch_chrom(chrom)
        if matrices is None or matrices.coo.nnz == 0:
            continue

        expected_lookup, default_expected = _precompute_expectation(matrices.coo)

        rowbins = matrices.coo.shape[0]
        start_idx = bindist_bins
        stop_idx = max(bindist_bins + 1, rowbins - bindist_bins)
        if stop_idx <= start_idx:
            continue

        candidate_rows = np.arange(start_idx, stop_idx)
        if float(sample_fraction) < 1.0:
            sample_n = max(1, int(len(candidate_rows) * float(sample_fraction)))
            row_indices = np.sort(rng.choice(candidate_rows, size=sample_n, replace=False))
        else:
            row_indices = candidate_rows

        chrom_bed = []
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
            observed_vec, expected_vec = _row_window_vectors(
                matrices,
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
                        incremental = float(np.clip(incremental, 0.0, 1.0))
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
