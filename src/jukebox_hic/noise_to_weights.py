#!/usr/bin/env python
"""Transform full-noise bedGraphs into adaptive normalization bias vectors."""

from __future__ import annotations

import io
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import reference


def _resolve_bedgraph_path(noise_dir: str, res: int) -> str:
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"Not a directory: {noise_dir}")
    candidates = [
        os.path.join(noise_dir, f"{res}.bedgraph"),
        os.path.join(noise_dir, f"{res}.bedGraph"),
        os.path.join(noise_dir, f"{res}.bg"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate bedGraph for resolution {res} in directory {noise_dir}"
    )


def _load_bedgraph(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"chrom": str},
        engine="c",
    )
    if df.empty:
        raise ValueError(f"BedGraph {path} is empty")
    return df


def _load_chrom_sizes(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chrom sizes file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["chrom", "size"],
        dtype={"chrom": str, "size": int},
    )
    return dict(zip(df["chrom"], df["size"]))


def _is_decoy_chrom(chrom: str) -> bool:
    lowered = chrom.lower()
    if lowered.startswith("chrun") or lowered.startswith("un_"):
        return True
    if "_" in chrom:
        return True
    decoy_tokens = ("random", "alt", "fix", "decoy", "hap", "patch")
    if any(token in lowered for token in decoy_tokens):
        return True
    if lowered.startswith(("gl", "ki", "jh", "hs", "chrz", "chrw")):
        return True
    return False


def _validate_grid(df: pd.DataFrame, res: int, chrom_len: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    if (df["start"] < 0).any() or (df["end"] < 0).any():
        raise ValueError("BedGraph contains negative coordinates")
    if (df["end"] <= df["start"]).any():
        raise ValueError("BedGraph contains zero or negative length intervals")
    if (df["start"].diff().fillna(res) < 0).any():
        df = df.sort_values("start").reset_index(drop=True)
    if df["start"].duplicated().any():
        dup_starts = df.loc[df["start"].duplicated(), "start"].tolist()
        raise ValueError(f"Duplicate bins detected at starts: {dup_starts[:5]}")
    if (df["start"] % res != 0).any():
        bad = df.loc[(df["start"] % res) != 0, "start"].iloc[0]
        raise ValueError(f"Start {bad} not aligned to resolution {res}")
    lengths = (df["end"] - df["start"]).astype(int)
    approx_res = np.isclose(lengths, res, atol=1)
    final_idx = len(df) - 1
    shorter_final = np.zeros(len(df), dtype=bool)
    if final_idx >= 0:
        final_end = int(df.iloc[final_idx]["end"])
        within_terminal = True
        if chrom_len is not None:
            within_terminal = abs(final_end - chrom_len) <= res
        shorter_final[final_idx] = lengths.iloc[final_idx] <= res and within_terminal
    valid_lengths = approx_res | shorter_final
    valid_lengths &= lengths > 0
    if not valid_lengths.all():
        bad_row = df.iloc[np.where(~valid_lengths)[0][0]]
        raise ValueError(
            f"Intervals do not match expected resolution grid (start={int(bad_row['start'])} "
            f"end={int(bad_row['end'])} length={int(bad_row['end']) - int(bad_row['start'])}, expected {res})"
        )
    if chrom_len is not None:
        max_end = int(df["end"].max())
        if max_end > chrom_len:
            raise ValueError(
                f"Intervals exceed declared length for chromosome (end={max_end} > {chrom_len})"
            )
    return df.reset_index(drop=True)


def _load_zmap_summary(path: str) -> Dict[str, Dict[str, float]]:
    """Load per-chromosome gamma, EBR, and pred_log_N from a subsample_summary.tsv."""
    data_lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                data_lines.append(line)
    if not data_lines:
        return {}
    df = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t")
    if "ratio" in df.columns:
        df = df[df["ratio"] == 1.0]
    result: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        chrom = str(row["chrom"])
        entry: Dict[str, float] = {}
        if "gamma" in row.index and pd.notna(row["gamma"]):
            entry["gamma"] = float(row["gamma"])
        if "mean_empty_bin_ratio" in row.index and pd.notna(row["mean_empty_bin_ratio"]):
            entry["ebr"] = float(row["mean_empty_bin_ratio"])
        if "pred_log_N" in row.index and pd.notna(row["pred_log_N"]):
            entry["pred_log_N"] = float(row["pred_log_N"])
        if entry:
            result[chrom] = entry
    return result


def _write_juicer_vector_block(
    handle,
    chrom: str,
    res: int,
    bias_vector: np.ndarray,
    vector_name: str,
    nan_fill_value: float,
) -> None:
    handle.write(f"vector {vector_name} {chrom} {res} BP\n")
    for val in bias_vector:
        if np.isnan(val):
            if np.isnan(nan_fill_value):
                handle.write("NaN\n")
            else:
                handle.write(f"{nan_fill_value}\n")
        else:
            handle.write(f"{val}\n")


_VALID_MODES = ("baseline", "adaptive")


def build_bias_vectors_from_bedgraphs(
    noise_dir: str,
    res: int,
    out_dir: str,
    zmap_summary_path: Optional[str] = None,
    chrom_sizes_path: Optional[str] = None,
    nan_fill_value: float = float("nan"),
    skip_decoys: bool = True,
    mode: str = "baseline",
    alpha: float = 0.7,
) -> None:
    """
    Transform a full-noise bedGraph into a Juicer-format normalization vector file.

    Reads ``{noise_dir}/{res}.bedgraph`` and writes ``{out_dir}/{res}.juicervector``.
    The vector name in the output encodes the mode:
      - ``mode="baseline"``  → ``vector JUKEBOX-BASELINE``
      - ``mode="adaptive"``  → ``vector JUKEBOX-ADAPTIVE``

    Both modes require ``zmap_summary_path`` to provide the per-chromosome
    predicted log10-noise (``pred_log_N``).

    Parameters
    ----------
    noise_dir         : directory containing ``{res}.bedgraph`` (from full-noise)
    res               : resolution in bp
    out_dir           : output directory; file will be written as ``{res}.juicervector``
    zmap_summary_path : path to ``subsample_summary.tsv`` from the sampling phase;
                        required for baseline and adaptive modes
    chrom_sizes_path  : chrom sizes TSV for grid validation (optional)
    nan_fill_value    : value written for NaN bins (default: NaN; use 0.0
                        for matrix balancing engines that do not accept NaN)
    skip_decoys       : skip unplaced/decoy chromosomes (default: True)
    mode              : normalization mode — ``"baseline"`` (default) or ``"adaptive"``
    alpha             : damping factor for adaptive mode (default: 0.7; range 0.5–0.8)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    if not zmap_summary_path or not os.path.isfile(zmap_summary_path):
        raise ValueError(
            f"zmap_summary_path is required for mode={mode!r} but was not provided or "
            f"does not exist: {zmap_summary_path!r}. "
            "Run sample-noise first and pass its subsample_summary.tsv."
        )

    noise_bed_path = _resolve_bedgraph_path(noise_dir, res)
    noise_df = _load_bedgraph(noise_bed_path)
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)
    zmap_data = _load_zmap_summary(zmap_summary_path)

    vector_name = "JUKEBOX-BASELINE" if mode == "baseline" else "JUKEBOX-ADAPTIVE"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{res}.juicervector")

    with open(out_path, "w") as out_handle:
        for chrom, group in noise_df.groupby("chrom", sort=True):
            if skip_decoys and _is_decoy_chrom(chrom):
                continue
            chrom_len = chrom_sizes.get(chrom)
            try:
                group = _validate_grid(group.copy(), res, chrom_len)
            except ValueError as exc:
                print(f"[WARN] {chrom}: skipping — {exc}")
                continue

            noise_track = group["value"].to_numpy(dtype=float)
            chrom_info = zmap_data.get(chrom, {})
            ebr = chrom_info.get("ebr", 0.0)
            pred_log_N = chrom_info.get("pred_log_N")

            if pred_log_N is None:
                print(f"[WARN] {chrom}: pred_log_N missing from zmap_summary — skipping")
                continue

            if mode == "baseline":
                bias_vector = reference.process_normalization_vectors_baseline(
                    noise_track, pred_log_N, ebr
                )
            else:
                bias_vector = reference.process_normalization_vectors_adaptive(
                    noise_track, pred_log_N, ebr, alpha
                )

            _write_juicer_vector_block(
                out_handle, chrom, res, bias_vector, vector_name, nan_fill_value
            )
