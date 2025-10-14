#!/usr/bin/env python
import os
from typing import Dict, List, Optional

import hicstraw
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf


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


def _default_bindist_bp(resolution: int) -> int:
    if resolution >= 100_000:
        return 10_000_000
    if resolution >= 50_000:
        return 6_000_000
    if resolution >= 25_000:
        return 4_000_000
    return 1_000_000


def _distance_normalize(df_obs: pd.DataFrame, res: int) -> pd.DataFrame:
    """
    Normalize observed counts using distance-based expected values derived from hicstraw counts.
    """
    lags = ((df_obs["binY"] - df_obs["binX"]).abs()).astype(int)
    with_lag = df_obs.copy()
    with_lag["lag"] = lags
    expected = with_lag.groupby("lag")["count"].mean()

    # Use the overall mean if a lag is missing from expected (rare for sparse data)
    overall = float(expected.mean())
    expected_vec = expected.to_dict()
    exp_vals = np.array([expected_vec.get(int(lag), overall) for lag in with_lag["lag"]], dtype=float)
    norm = (with_lag["count"].to_numpy() + 1.0) / (exp_vals + 1.0)

    out = with_lag.copy()
    out["distnorm"] = norm
    return out[["binX", "binY", "distnorm"]]


def _row_window_noise(df_dn: pd.DataFrame, rowbins: int, bindist_bins: int) -> np.ndarray:
    by_x = df_dn.groupby("binX")
    by_y = df_dn.groupby("binY")
    noise_vals = np.zeros(rowbins, dtype=float)

    for idx in range(rowbins):
        left = by_y.get_group(idx) if idx in by_y.groups else pd.DataFrame(columns=df_dn.columns)
        left = left[(left["binX"] >= idx - bindist_bins) & (left["binX"] < idx)]

        right = by_x.get_group(idx) if idx in by_x.groups else pd.DataFrame(columns=df_dn.columns)
        right = right[(right["binY"] > idx) & (right["binY"] <= idx + bindist_bins)]

        filled = np.zeros(bindist_bins * 2, dtype=float)
        for _, rec in left.iterrows():
            pos = int(bindist_bins - (idx - int(rec["binX"])))
            if 0 <= pos < len(filled):
                filled[pos] = float(rec["distnorm"])
        for _, rec in right.iterrows():
            pos = int(bindist_bins + (int(rec["binY"]) - idx))
            if 0 <= pos < len(filled):
                filled[pos] = float(rec["distnorm"])

        try:
            ac = acovf(filled, nlag=1, fft=True)[1]
            noise_vals[idx] = 10000.0 if ac == 0 else 1.0 / abs(ac)
        except Exception:
            noise_vals[idx] = 10000.0

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
    norm: str = "NONE",
    bindist_bp: Optional[int] = None,
    out_dir: str = ".",
) -> None:
    """
    Compute noise values for every row/bin across the genome at the requested resolutions.
    All calculations rely solely on hicstraw outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    chrom_sizes = _read_chrom_sizes(hic_path, chrom_sizes_path)

    for res in res_list:
        window_bp = bindist_bp or _default_bindist_bp(int(res))
        bindist_bins = max(1, window_bp // int(res))
        per_chrom_paths: List[str] = []

        for chrom, chrom_len in chrom_sizes.items():
            records = hicstraw.straw("observed", norm, hic_path, chrom, chrom, "BP", int(res))
            if not records:
                continue
            df_obs = pd.DataFrame(
                {
                    "binX": [(r.binX // res) for r in records],
                    "binY": [(r.binY // res) for r in records],
                    "count": [r.counts for r in records],
                }
            )

            df_dn = _distance_normalize(df_obs, int(res))
            rowbins = chrom_len // int(res) + 1
            noise_vals = _row_window_noise(df_dn, rowbins=rowbins, bindist_bins=bindist_bins)

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
