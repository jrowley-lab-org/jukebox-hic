#!/usr/bin/env python
"""
Transform full-noise bedGraphs into JUKEBOX normalization bias vectors.

This module is the final step in the jukebox-hic pipeline. It reads the genome-wide
per-bin noise track produced by ``noise_fullmap`` (as a bedgraph file) and converts
it into a normalization bias vector in Juicer's custom vector format.

Pipeline context
----------------
1. ``noise_sampling`` → ``subsample_summary.tsv`` (per-chrom z-map, pred_log_N, EBR)
2. ``noise_fullmap``  → ``{res}.bedgraph`` (per-bin noise values for every bin)
3. **This module**    → ``{res}.juicervector`` (per-bin bias weights for Juicer)

The bias vector tells Juicer how to weight each genomic bin when normalising the
Hi-C matrix. Bins with high noise get a larger weight (down-weighted contacts);
bins with low noise get a smaller weight (contacts kept closer to observed values).

Juicer vector format
--------------------
The output file uses Juicer's custom normalization vector format:

    vector JUKEBOX chr1 10000 BP
    1.234
    0.987
    NaN
    ...
    vector JUKEBOX chr2 10000 BP
    ...

Each ``vector`` header line introduces a block for one chromosome. The header
specifies: vector name, chromosome, resolution (bp), coordinate system (always BP).
One value per line follows, one per bin in order. NaN means "mask this bin".

Main entry point: ``build_bias_vectors_from_bedgraphs()``.
"""

from __future__ import annotations

import io
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import reference


def _resolve_bedgraph_path(noise_dir: str, res: int) -> str:
    """
    Find the full-noise bedgraph file for a given resolution in a directory.

    ``noise_fullmap`` writes output files named ``{res}.bedgraph``, ``{res}.bedGraph``,
    or ``{res}.bg`` depending on the platform or user convention. This function tries
    all three variants and returns the first one found.

    Parameters
    ----------
    noise_dir : str
        Directory to search (must exist as a directory, not a file).
    res : int
        Resolution in base pairs (the integer file stem to look for).

    Returns
    -------
    str
        Absolute path to the bedgraph file.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist, or no matching file is found.
    """
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
    """
    Load a 4-column bedgraph file into a DataFrame.

    BedGraph format: whitespace-separated columns:
        chromosome  start_bp  end_bp  value

    The value column contains the per-bin noise metric (or NaN for unmeasurable bins).

    Parameters
    ----------
    path : str
        Path to the bedgraph file.

    Returns
    -------
    pd.DataFrame
        Columns: ``chrom`` (str), ``start`` (int), ``end`` (int), ``value`` (float).

    Raises
    ------
    ValueError
        If the file is empty.
    """
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
    """
    Load a chromosome sizes file into a dictionary.

    Chromosome sizes files are 2-column TSV files:
        chromosome_name<tab>length_in_bp

    These are commonly produced by ``samtools faidx`` (the ``.fai`` file's first two
    columns) or downloaded from UCSC (e.g. ``hg38.chrom.sizes``).

    Parameters
    ----------
    path : str or None
        Path to the chrom sizes file. If None or empty string, returns an empty dict,
        which downstream code interprets as "no size validation".

    Returns
    -------
    dict
        Chromosome name → length in bp. Empty dict if ``path`` is None.

    Raises
    ------
    FileNotFoundError
        If ``path`` is provided but the file does not exist.
    """
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
    """
    Return True if *chrom* appears to be an unplaced, alternate, or decoy sequence.

    In reference genome assemblies, the primary assembled chromosomes (chr1–chr22,
    chrX, chrY, chrM for human) are supplemented by additional sequences that are
    not part of the canonical genome:

    - **Unlocalized sequences**: contigs that are assigned to a chromosome but cannot
      be precisely placed (e.g. ``chr1_KI270762v1_alt``, ``chr1_gl000191v1_random``).
    - **Unplaced sequences**: contigs not assigned to any chromosome
      (e.g. ``chrUn_gl000220v1``).
    - **Alternate loci**: alternate representations of genomic regions
      (e.g. ``chr6_ssto_hap7``).
    - **Fix patches**: corrections to the reference (e.g. ``chr1_KN196472v1_fix``).
    - **Decoy sequences**: sequences to capture reads that would otherwise map
      poorly (e.g. ``chrEBV``, ``hs37d5``).

    These sequences are noisy in Hi-C because they are short, repetitive, or present
    in multiple copies, producing spurious contacts. They should generally be excluded
    from normalization vector generation.

    Detection heuristics (applied in order):
    1. Starts with ``chrun`` or ``un_``: classic UCSC notation for unplaced contigs
       (e.g. ``chrUn_gl000220``).
    2. Contains ``_``: almost all unlocalized/alternate sequences contain an underscore
       (e.g. ``chr1_KI270762v1_alt``). Note: ``chrX`` and ``chrY`` do not contain ``_``.
    3. Contains one of the keyword substrings: ``"random"``, ``"alt"``, ``"fix"``,
       ``"decoy"``, ``"hap"``, ``"patch"`` — these appear in sequence names from
       Ensembl, UCSC, and NCBI assemblies for non-canonical sequences.
    4. Starts with one of the NCBI GenBank accession prefixes: ``gl``, ``ki``, ``jh``,
       ``hs`` — these are the first two letters of GenBank accession numbers for
       unlocalized scaffolds in human assemblies (e.g. ``GL000191.1``, ``KI270762.1``).
    5. Starts with ``chrz`` or ``chrw``: sex chromosome aliases used in some non-human
       vertebrate assemblies (e.g. bird Z/W chromosomes sometimes named chrZ/chrW).

    Parameters
    ----------
    chrom : str
        Chromosome or sequence name to test.

    Returns
    -------
    bool
        True if the sequence should be treated as a decoy and excluded.
    """
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
    """
    Validate that a per-chromosome bedgraph slice conforms to a regular bin grid.

    The noise-to-weights pipeline requires a strictly regular grid: every bin must
    start at a multiple of ``res``, all bins must have size ``res`` (except the final
    bin which may be shorter), and there must be no gaps or duplicates.

    Validation checks (in order):
    1. **No negative coordinates**: start and end must be ≥ 0.
    2. **No zero-length or inverted intervals**: end > start for every row.
    3. **Sorted order**: if starts are not monotonically increasing, the DataFrame
       is sorted in place (a warning is not raised; sorting is silently applied).
    4. **No duplicate bin starts**: two bins with the same start position would
       create an ambiguous grid — raises ValueError with the duplicate positions listed.
    5. **Start alignment**: every bin start must be a multiple of ``res``.
       Mis-aligned bins indicate a resolution mismatch (e.g. a 10 kb bedgraph
       loaded with a 5 kb resolution setting).
    6. **Bin length uniformity**: all bins must have length ``res`` (within 1 bp
       tolerance for floating-point rounding). The **final bin only** is allowed to
       be shorter — this is expected when the chromosome length is not a multiple of
       ``res``. The ``shorter_final`` flag tracks this exception:
       - ``shorter_final[i]`` is True only for the last bin (index ``final_idx``),
         and only if its end coordinate is within ``res`` of the chromosome boundary
         (``within_terminal``). This check uses ``chrom_len`` if available; if not,
         the last bin is assumed to be legitimately short.
    7. **No overflow**: if ``chrom_len`` is known, no bin may extend beyond it.

    Parameters
    ----------
    df : pd.DataFrame
        Per-chromosome subset of the bedgraph (columns: chrom, start, end, value).
        Should already be filtered to a single chromosome.
    res : int
        Expected bin size in bp.
    chrom_len : int or None
        Chromosome length in bp for boundary checks. If None, overflow is not checked.

    Returns
    -------
    pd.DataFrame
        The validated (and potentially sorted) DataFrame, with index reset.

    Raises
    ------
    ValueError
        On any validation failure, with a descriptive message including the
        offending coordinate values.
    """
    if df.empty:
        return df
    if (df["start"] < 0).any() or (df["end"] < 0).any():
        raise ValueError("BedGraph contains negative coordinates")
    if (df["end"] <= df["start"]).any():
        raise ValueError("BedGraph contains zero or negative length intervals")
    # Check if starts are sorted; sort silently if not
    if (df["start"].diff().fillna(res) < 0).any():
        df = df.sort_values("start").reset_index(drop=True)
    if df["start"].duplicated().any():
        dup_starts = df.loc[df["start"].duplicated(), "start"].tolist()
        raise ValueError(f"Duplicate bins detected at starts: {dup_starts[:5]}")
    if (df["start"] % res != 0).any():
        bad = df.loc[(df["start"] % res) != 0, "start"].iloc[0]
        raise ValueError(f"Start {bad} not aligned to resolution {res}")
    lengths = (df["end"] - df["start"]).astype(int)
    # approx_res: bins whose length is within 1 bp of the expected resolution
    approx_res = np.isclose(lengths, res, atol=1)
    final_idx = len(df) - 1
    # shorter_final: the last bin is allowed to be shorter than res if it reaches
    # the chromosome boundary (i.e. chromosome length is not a multiple of res)
    shorter_final = np.zeros(len(df), dtype=bool)
    if final_idx >= 0:
        final_end = int(df.iloc[final_idx]["end"])
        within_terminal = True
        if chrom_len is not None:
            # The final bin is "within terminal" if its end is within one bin-width
            # of the chromosome boundary — allows for rounding differences.
            within_terminal = abs(final_end - chrom_len) <= res
        shorter_final[final_idx] = lengths.iloc[final_idx] <= res and within_terminal
    valid_lengths = approx_res | shorter_final
    valid_lengths &= lengths > 0  # exclude any zero-length bins
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
    """
    Load per-chromosome EBR and pred_log_N from a subsample_summary.tsv.

    The ``subsample_summary.tsv`` produced by ``noise_sampling.compute_sampled_noise()``
    contains comment lines (starting with ``#``) encoding metadata, followed by a
    TSV body with one row per (chromosome, subsample_ratio) combination.

    This function:
    1. Skips comment lines (the ``# resolution=`` and ``# chrom_info`` headers).
    2. Loads the TSV body and filters to rows where ``ratio == 1.0`` (full-depth data).
    3. Extracts per-chromosome ``mean_empty_bin_ratio`` (EBR) and ``pred_log_N``.
       These are needed to compute the bias vectors.

    Fields extracted:
    - ``ebr`` (from ``mean_empty_bin_ratio``): mean empty bin ratio; used as the
      Gaussian smoothing sigma offset in ``_preprocess_noise_track()``.
    - ``pred_log_N``: 4DN-predicted log10 noise for this chromosome; used as the
      anchor point in the JUKEBOX bias-vector formula.

    Parameters
    ----------
    path : str
        Path to the ``subsample_summary.tsv`` file.

    Returns
    -------
    dict
        Chromosome name → dict of available fields (``ebr``, ``pred_log_N``,
        ``pred_log_N_upper``, ``pred_log_N_lower``). Chromosomes not in the
        file, or with all-NaN fields, are omitted.
    """
    data_lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                data_lines.append(line)
    if not data_lines:
        return {}
    df = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t")
    # Keep only full-depth rows (ratio=1.0) — subsampled rows are not used for
    # bias vector construction.
    if "ratio" in df.columns:
        df = df[df["ratio"] == 1.0]
    result: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        chrom = str(row["chrom"])
        entry: Dict[str, float] = {}
        if "mean_empty_bin_ratio" in row.index and pd.notna(row["mean_empty_bin_ratio"]):
            entry["ebr"] = float(row["mean_empty_bin_ratio"])
        if "pred_log_N" in row.index and pd.notna(row["pred_log_N"]):
            entry["pred_log_N"] = float(row["pred_log_N"])
        if "pred_log_N_upper" in row.index and pd.notna(row["pred_log_N_upper"]):
            entry["pred_log_N_upper"] = float(row["pred_log_N_upper"])
        if "pred_log_N_lower" in row.index and pd.notna(row["pred_log_N_lower"]):
            entry["pred_log_N_lower"] = float(row["pred_log_N_lower"])
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
    """
    Write one chromosome's bias vector in Juicer custom normalization format.

    Juicer's custom normalization vector format consists of blocks, one per chromosome:

        vector <vector_name> <chrom> <resolution> BP
        <value_bin_0>
        <value_bin_1>
        ...
        <value_last_bin>

    The ``vector`` header line is required by Juicer to identify the vector type,
    chromosome, and resolution. The values follow immediately, one per line, in
    ascending bin order (bin 0 = start of chromosome, bin N = end).

    NaN values in the bias vector represent bins that should be masked (too sparse
    or otherwise invalid). Juicer can accept ``NaN`` literally, or some downstream
    tools require a numeric fill (e.g. ``0.0`` or ``1.0``).

    Parameters
    ----------
    handle
        Open file handle for writing (text mode).
    chrom : str
        Chromosome name (written verbatim in the header line).
    res : int
        Resolution in bp (written verbatim in the header line).
    bias_vector : np.ndarray
        Per-bin bias weights. NaN entries are written as ``NaN`` or
        ``nan_fill_value`` depending on the configuration.
    vector_name : str
        Name to use in the ``vector`` header line (e.g. ``"JUKEBOX"``).
    nan_fill_value : float
        Value to write for NaN bins. If ``nan_fill_value`` is itself NaN, the literal
        string ``"NaN"`` is written (compatible with Juicer). Use ``0.0`` for tools
        that reject non-numeric values in normalization vectors.
    """
    handle.write(f"vector {vector_name} {chrom} {res} BP\n")
    for val in bias_vector:
        if np.isnan(val):
            if np.isnan(nan_fill_value):
                handle.write("NaN\n")
            else:
                handle.write(f"{nan_fill_value}\n")
        else:
            handle.write(f"{val}\n")


def build_bias_vectors_from_bedgraphs(
    noise_dir: str,
    res: int,
    out_dir: str,
    zmap_summary_path: Optional[str] = None,
    chrom_sizes_path: Optional[str] = None,
    nan_fill_value: float = float("nan"),
    skip_decoys: bool = True,
    alpha: float = 0.5,
    p_factor: float = 2.0,
) -> None:
    """
    Transform a full-noise bedGraph into a Juicer-format normalization vector file.

    Reads ``{noise_dir}/{res}.bedgraph`` and writes ``{out_dir}/{res}.juicervector``
    with vector name ``JUKEBOX`` in the Juicer header line.

    Requires ``zmap_summary_path`` to provide the per-chromosome predicted
    log10-noise (``pred_log_N``).

    Processing pipeline per chromosome
    -----------------------------------
    1. Filter decoy chromosomes (if ``skip_decoys=True``) using ``_is_decoy_chrom()``.
    2. Validate the bin grid with ``_validate_grid()`` — skip the chromosome with a
       warning if the grid is malformed (rather than aborting the whole run).
    3. Look up per-chromosome ``pred_log_N`` and ``ebr`` from the zmap summary.
       Skip chromosomes missing from the summary (they were not sampled, probably
       because they had no contacts at the sampling resolution).
    4. Compute the bias vector using ``reference.process_normalization_vectors_jukebox()``.
    5. Write the bias vector block to the output file using ``_write_juicer_vector_block()``.

    Parameters
    ----------
    noise_dir         : directory containing ``{res}.bedgraph`` (from full-noise)
    res               : resolution in bp
    out_dir           : output directory; file will be written as ``{res}.juicervector``
    zmap_summary_path : path to ``subsample_summary.tsv`` from the sampling phase;
                        required to anchor the JUKEBOX bias-vector formula
    chrom_sizes_path  : chrom sizes TSV for grid validation (optional)
    nan_fill_value    : value written for NaN bins (default: NaN; use 0.0
                        for matrix balancing engines that do not accept NaN)
    skip_decoys       : skip unplaced/decoy chromosomes (default: True)
    alpha             : scaling constant for the JUKEBOX bias formula (default: 0.5)
    p_factor          : compression factor for the JUKEBOX bias formula
                        (default: 2.0, square root; use 3.0 for cube root)
    """
    if not zmap_summary_path or not os.path.isfile(zmap_summary_path):
        raise ValueError(
            "zmap_summary_path is required for JUKEBOX normalization but was not "
            f"provided or does not exist: {zmap_summary_path!r}. "
            "Run sample-noise first and pass its subsample_summary.tsv."
        )

    noise_bed_path = _resolve_bedgraph_path(noise_dir, res)
    noise_df = _load_bedgraph(noise_bed_path)
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)
    zmap_data = _load_zmap_summary(zmap_summary_path)

    vector_name = "JUKEBOX"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{res}.juicervector")

    with open(out_path, "w") as out_handle:
        for chrom, group in noise_df.groupby("chrom", sort=True):
            # Skip decoy/unplaced sequences that would produce unreliable bias vectors
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
                # This chromosome was not in the sampling summary — either it was
                # excluded from sampling (no contacts) or the summary is from a
                # different run. Skip rather than producing a zero-anchored vector.
                print(f"[WARN] {chrom}: pred_log_N missing from zmap_summary — skipping")
                continue

            bias_vector = reference.process_normalization_vectors_jukebox(
                noise_track, pred_log_N, ebr, p_factor, alpha
            )

            _write_juicer_vector_block(
                out_handle, chrom, res, bias_vector, vector_name, nan_fill_value
            )
