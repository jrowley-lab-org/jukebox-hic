#!/usr/bin/env python
import math
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _gather_paths(patterns: Sequence[str]) -> list[str]:
    unique: set[str] = set()
    for pat in patterns:
        if "*" in pat or "?" in pat or "[" in pat:
            import glob

            matches = glob.glob(pat)
            unique.update(os.path.abspath(p) for p in matches if os.path.isfile(p))
        else:
            if os.path.isfile(pat):
                unique.add(os.path.abspath(pat))
    return sorted(unique)


def _merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []
    for chrom, group in df.groupby("chrom", sort=True):
        ordered = group.sort_values("start").to_numpy()
        if ordered.size == 0:
            continue
        current = ordered[0].copy()
        for row in ordered[1:]:
            if int(row[1]) <= int(current[2]) and int(row[1]) >= int(current[1]):
                current[2] = max(int(current[2]), int(row[2]))
            elif int(row[1]) == int(current[2]):
                current[2] = int(row[2])
            else:
                merged_rows.append(current.copy())
                current = row.copy()
        merged_rows.append(current.copy())
    if not merged_rows:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    out = pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    return out


def build_blacklist_from_bedgraphs(
    inputs: Sequence[str],
    output_path: str,
    zscore_cutoff: float | None = None,
    top_quantile: float = 0.95,
) -> None:
    """
    Load one or more per-chromosome bedgraphs, flag rows with NaN noise or high noise signal,
    and emit a blacklist BED file with merged intervals.
    """
    paths = _gather_paths(inputs)
    if not paths:
        raise ValueError("No bedgraph files found for provided inputs")

    frames = []
    for path in paths:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["chrom", "start", "end", "noise"],
            dtype={"chrom": str},
            engine="c",
        )
        df["source"] = os.path.basename(path)
        df["noise"] = pd.to_numeric(df["noise"], errors="coerce")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("No rows found in bedgraph inputs")

    flagged = ~np.isfinite(data["noise"])
    valid_mask = np.isfinite(data["noise"])
    valid_values = data.loc[valid_mask, "noise"]

    if not valid_values.empty:
        if zscore_cutoff is not None:
            mean = float(valid_values.mean())
            std = float(valid_values.std(ddof=0))
            if std <= 0:
                thresh_mask = valid_values >= mean  # degenerate distribution; keep only above-mean rows
            else:
                zscores = (valid_values - mean) / std
                thresh_mask = zscores >= float(zscore_cutoff)
        else:
            quantile_value = float(np.nanquantile(valid_values, float(top_quantile)))
            thresh_mask = valid_values >= quantile_value
        flagged.loc[valid_mask] |= thresh_mask

    flagged_df = data.loc[flagged, ["chrom", "start", "end"]].copy()
    if flagged_df.empty:
        merged = pd.DataFrame(columns=["chrom", "start", "end"])
    else:
        merged = _merge_intervals(flagged_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    merged.to_csv(output_path, sep="\t", header=False, index=False)
