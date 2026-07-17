#!/usr/bin/env python
"""
Command-line interface for jukebox-hic.

This module defines the ``main()`` entry point, which is registered as the
``jukebox-hic`` console script in ``pyproject.toml``. It uses Python's
``argparse`` library to expose eight subcommands:

Subcommands
-----------
sample-noise
    Sample a fraction of genomic rows, compute per-row noise metrics, and produce
    per-chromosome bedgraphs and a genome-wide summary TSV. This is the fast quality
    assessment step used to compute z-map scores and sequencing advisor results.

full-noise
    Compute noise for every genomic bin genome-wide (no sampling). Produces a per-bin
    bedgraph at each requested resolution, suitable for bias vector construction.

bias-vectors
    Transform a full-noise bedgraph into a JUKEBOX normalization vector (Juicer
    custom-vector format) for a single resolution. Requires the subsample_summary.tsv
    from sample-noise at the same resolution to obtain per-chromosome z-map
    parameters. Run once per resolution.

plot
    Render a density histogram of noise values from a bedgraph file.

blacklist
    Build a per-chromosome BED blacklist from noise and density bedgraphs using
    data-driven elbow detection.  Requires --density_bedgraph and --noise_bedgraph
    (outputs of full-noise).

sequencing-advisor
    Whole-map ENCODE/4DN sequencing-sufficiency check: reads one
    subsample_summary.tsv per resolution under --summary_dir, picks a
    resolution tier from total genome-wide contacts, and produces a single
    genome-wide "Sequence Further" / "Sufficient" recommendation.

default-run
    Run the full jukebox-hic pipeline in sequence:
    Phase 1: sample-noise → Phase 1.5: sequencing-advisor →
    Phase 2: full-noise → Phase 3: bias-vectors → Phase 4: blacklist.
    Use this for a complete first-pass analysis.
"""
import argparse
import glob
import io
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import figures, noise_fullmap, noise_sampling, filters, noise_to_weights, reference
from .backends import (
    cooler_resolutions,
    hic_resolutions,
    is_cooler_path,
    read_chrom_sizes,
    select_provider,
    total_contacts,
)
from .metrics import collect_memory_usage_mb


def _comma_ints(value: str) -> List[int]:
    """
    Argparse type converter: parse a comma-separated string of integers.

    Used for the ``--res`` argument, which accepts one or more resolution values
    separated by commas (e.g. ``"5000,10000,25000"``).

    Parameters
    ----------
    value : str
        Comma-separated integer string from the command line (e.g. ``"10000,25000"``).

    Returns
    -------
    List[int]
        List of parsed integers in the order they were provided.
    """
    return list(map(int, value.split(",")))


def _comma_strs(value: str) -> List[str]:
    """
    Argparse type converter: parse a comma-separated string of non-empty tokens.

    Used for the ``--chroms`` argument, which accepts chromosome names separated by
    commas (e.g. ``"chr1,chr2,chrX"``). Empty tokens from trailing commas or
    double-commas are silently dropped.

    Parameters
    ----------
    value : str
        Comma-separated string from the command line.

    Returns
    -------
    List[str]
        List of stripped, non-empty string tokens.
    """
    return [s.strip() for s in value.split(",") if s.strip()]


def _detect_resolutions_in_dir(noise_dir: str) -> List[int]:
    """Return sorted list of resolutions found as {integer}.bedgraph in noise_dir."""
    resolutions = []
    if not os.path.isdir(noise_dir):
        return resolutions
    for path in glob.glob(os.path.join(noise_dir, "*.bedgraph")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.isdigit():
            resolutions.append(int(name))
    return sorted(resolutions)


def _profile_command(name: str, func: Callable[[], Optional[int]]) -> Optional[int]:
    """
    Execute *func*, then print a one-line profiling summary to stdout.

    Wraps any callable (typically a closure that runs one CLI subcommand) with
    timing and memory measurement. The summary line format is:

        [PROFILE] cmd=<name> elapsed=<s>s rss_now=<MB>MB ru_max=<MB>MB
                  delta_rss=<MB>MB delta_ru=<MB>MB

    Fields explained:
    - ``elapsed``: wall-clock time in seconds for the entire command.
    - ``rss_now``: current process RSS in MB after the command (from psutil).
    - ``ru_max``: peak RSS ever reached by the process in MB (from getrusage).
    - ``delta_rss``: increase in RSS during the command (end − start, clamped at 0).
    - ``delta_ru``: increase in peak RSS during the command (end − start, clamped at 0).

    The ``finally`` block ensures the profile line is always printed, even if
    ``func()`` raises an exception (allowing the caller to see timing before the
    traceback).

    Parameters
    ----------
    name : str
        Human-readable command name printed in the profile line.
    func : Callable[[], Optional[int]]
        Zero-argument callable to execute and profile.

    Returns
    -------
    Optional[int]
        The return value of ``func()`` (typically None for successful subcommands,
        or an exit code integer for error paths).
    """
    start_time = time.perf_counter()
    start_rss, start_ru = collect_memory_usage_mb()
    try:
        return func()
    finally:
        end_rss, end_ru = collect_memory_usage_mb()
        elapsed = time.perf_counter() - start_time
        delta_rss = max(0.0, end_rss - start_rss)
        delta_ru = max(0.0, end_ru - start_ru)
        print(
            f"[PROFILE] cmd={name} elapsed={elapsed:.2f}s rss_now={end_rss:.1f}MB "
            f"ru_max={end_ru:.1f}MB delta_rss={delta_rss:.1f}MB delta_ru={delta_ru:.1f}MB"
        )


def _available_resolutions(path: str, cooler_selection: Optional[str]) -> List[int]:
    """
    Return all available bin resolutions for a Hi-C or cooler file.

    Dispatches to the appropriate resolution-discovery function based on the file type:
    - Cooler files (.cool, .mcool, .scool, or URI with ``::``): use ``cooler_resolutions()``.
    - Juicer .hic files: use ``hic_resolutions()``.

    The ``cooler_selection`` argument is only relevant for .scool files, where a
    specific cell group must be selected to determine the bin size.

    Parameters
    ----------
    path : str
        File path or URI.
    cooler_selection : str or None
        Internal group path for .scool files (e.g. a cell name).

    Returns
    -------
    List[int]
        Sorted list of available resolutions in bp. Empty list on error.
    """
    if is_cooler_path(path):
        return cooler_resolutions(path, cooler_selection)
    return hic_resolutions(path)


def _coarsest_resolution(path: str, cooler_selection: Optional[str]) -> Optional[int]:
    """
    Return the coarsest (largest bin size) resolution available in the file.

    "Coarsest" means the largest bin size in base pairs — the resolution that
    captures the most data per bin with the fewest total bins. Used by
    ``_dump_contacts_once()`` to choose a resolution for counting total contacts
    across chromosomes before per-resolution analysis begins.

    Choosing the coarsest resolution for contact counting is an efficiency choice:
    - Loading a coarse-resolution matrix is faster (fewer bins, smaller matrix).
    - Total contact counts are resolution-independent (contacts are summed the same
      way regardless of bin size, because the deduplication in ``total_contacts()``
      works at the contact-pair level).

    Parameters
    ----------
    path : str
        File path or URI.
    cooler_selection : str or None
        Internal group path for .scool files.

    Returns
    -------
    int or None
        The largest available bin size in bp, or None if no resolutions are available.
    """
    resolutions = _available_resolutions(path, cooler_selection)
    if not resolutions:
        return None
    return max(resolutions)


def _dump_contacts_once(
    hic_path: str,
    norm: str,
    cooler_selection: Optional[str],
    chrom_sizes_path: Optional[str],
) -> Dict[str, float]:
    """
    Count total contacts per chromosome, using the coarsest available resolution.

    This function is called once at the start of ``sample-noise``, ``full-noise``,
    and ``default-run`` to pre-compute a contacts overview. The results are:
    - Written to ``contacts_overview.tsv`` for user reference.
    - Passed to ``compute_sampled_noise()`` for computing sequencing density (ρ),
      which feeds into the z-map and sequencing advisor calculations.

    Using the coarsest resolution is an efficiency optimisation — total contact
    counts are the same regardless of resolution (the same read pairs are present
    at all resolutions; only their bin assignments change). Loading at coarse
    resolution minimises memory usage.

    Parameters
    ----------
    hic_path : str
        Path to the Hi-C or cooler file.
    norm : str
        Normalisation mode passed to ``select_provider()`` (typically ``"none"``
        for contact counting, since we want raw contact numbers).
    cooler_selection : str or None
        Internal group path for .scool files.
    chrom_sizes_path : str or None
        Optional path to filter chromosomes to a specific set.

    Returns
    -------
    Dict[str, float]
        Chromosome name → total contact count (float because contact values may
        be normalised and fractional in some files). Empty dict if no resolutions
        are available or the file cannot be opened.
    """
    res = _coarsest_resolution(hic_path, cooler_selection)
    if res is None:
        return {}
    provider, _ = select_provider(hic_path, res, norm, cooler_selection)
    chrom_sizes = read_chrom_sizes(provider, chrom_sizes_path)
    contacts: Dict[str, float] = {}
    for chrom in chrom_sizes:
        matrices = provider.fetch_chrom(chrom)
        if matrices is None or matrices.coo.nnz == 0:
            continue
        contacts[chrom] = total_contacts(matrices)
    return contacts


def _parse_subsample_summary(
    path: str,
) -> Tuple[Optional[int], Optional[int], Dict[str, int], "pd.DataFrame"]:
    """
    Parse one ``subsample_summary.tsv`` produced by ``sample-noise``.

    The file format uses comment lines:
        "# resolution=N"    — the resolution this summary was computed at
        "# window_bp=N"     — the analysis window size used during sampling
        "# chrom_info<tab>chrom<tab>contacts=N<tab>size_bp=N" — chromosome metadata
    These are needed because chromosome sizes are not stored in the TSV body.

    Parameters
    ----------
    path : str
        Path to a subsample_summary.tsv file.

    Returns
    -------
    (resolution, window_bp, chrom_sizes, df)
        ``df`` is filtered to ``ratio == 1.0`` rows (full-depth) only.
    """
    resolution: Optional[int] = None
    window_bp: Optional[int] = None
    chrom_sizes: Dict[str, int] = {}
    data_lines: List[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("# resolution="):
                try:
                    resolution = int(line.strip().split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("# window_bp="):
                try:
                    window_bp = int(line.strip().split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("# chrom_info"):
                parts = line.strip().split("\t")
                chrom_name = parts[1] if len(parts) > 1 else None
                for part in parts[2:]:
                    if part.startswith("size_bp="):
                        try:
                            size = int(part.split("=", 1)[1])
                            if chrom_name:
                                chrom_sizes[chrom_name] = size
                        except ValueError:
                            pass
            elif not line.startswith("#"):
                data_lines.append(line)

    df = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t")
    if "ratio" in df.columns:
        df = df[df["ratio"] == 1.0].copy()
    return resolution, window_bp, chrom_sizes, df


def _read_total_from_contacts_overview(path: str) -> float:
    """Read a contacts_overview.tsv and return the TOTAL row's contacts value."""
    df = pd.read_csv(path, sep="\t")
    return float(df.loc[df["chrom"] == "TOTAL", "contacts"].iloc[0])


def _run_sequencing_advisor(summary_dir: str, out_path: str) -> Dict[str, object]:
    """
    Whole-map sequencing-sufficiency check (ENCODE/4DN rule).

    Reads one ``{resolution}/subsample_summary.tsv`` per resolution
    subdirectory of ``summary_dir`` (the layout produced by
    ``sample-noise --out_dir <summary_dir>``, or ``default-run``'s
    ``<out_dir>/sample`` directory), selects a resolution tier from the total
    genome-wide contact count, aggregates per-chromosome noise/EBR at that
    resolution, and evaluates ``reference.map_sequencing_advisor()``.

    Writes a single-row TSV to ``out_path`` and prints a one-line summary.

    Parameters
    ----------
    summary_dir : str
        Directory containing one ``{resolution}/subsample_summary.tsv`` per
        resolution.
    out_path : str
        Output TSV path for the advisor report.

    Returns
    -------
    Dict[str, object]
        The advisor result (see ``reference.map_sequencing_advisor()``), plus
        ``resolution_selected`` (tier target) and ``resolution_used`` (actual,
        possibly a fallback).
    """
    available: Dict[int, str] = {}
    for name in sorted(os.listdir(summary_dir)):
        sub_path = os.path.join(summary_dir, name, "subsample_summary.tsv")
        if name.isdigit() and os.path.isfile(sub_path):
            available[int(name)] = sub_path
    if not available:
        raise SystemExit(
            f"No <resolution>/subsample_summary.tsv found under {summary_dir}. "
            "Run sample-noise (or default-run) first."
        )

    # total_contacts is resolution-independent — read from any one available resolution.
    any_res = sorted(available)[0]
    _, _, _, any_df = _parse_subsample_summary(available[any_res])
    overview_path = os.path.join(summary_dir, str(any_res), "contacts_overview.tsv")
    if os.path.isfile(overview_path):
        total_contacts_val = _read_total_from_contacts_overview(overview_path)
    else:
        total_contacts_val = float(any_df["estimated_contacts"].sum())

    resolution_selected = reference.select_advisor_resolution(total_contacts_val, list(available.keys()))

    if resolution_selected is None:
        result: Dict[str, object] = {
            "total_contacts":  total_contacts_val,
            "genome_len":      None,
            "resolution":      None,
            "window_bp":       None,
            "epsilon":         float("nan"),
            "quality_status":  "",
            "recommendation":  "Sufficient",
            "resolution_selected": None,
            "resolution_used":    None,
        }
    else:
        if resolution_selected in available:
            resolution_used = resolution_selected
        else:
            resolution_used = min(available, key=lambda r: abs(r - resolution_selected))
            print(
                f"[WARN] sequencing-advisor: tier requires {resolution_selected} bp, "
                f"but only {sorted(available)} bp were run. Falling back to nearest "
                f"available: {resolution_used} bp. For an accurate check, re-run "
                f"sample-noise with --res including {resolution_selected}."
            )

        _, window_bp, chrom_sizes, df = _parse_subsample_summary(available[resolution_used])
        chrom_noise    = dict(zip(df["chrom"], df["median_noise"].astype(float)))
        chrom_ebr      = dict(zip(df["chrom"], df["mean_empty_bin_ratio"].astype(float)))
        chrom_contacts = dict(zip(df["chrom"], df["estimated_contacts"].astype(float)))
        genome_len = int(sum(chrom_sizes.values()))

        agg_noise, agg_ebr = reference.aggregate_genome_wide_noise_ebr(chrom_noise, chrom_ebr, chrom_contacts)
        if window_bp is None:
            window_bp = noise_sampling._default_window_bp(resolution_used)
        result = reference.map_sequencing_advisor(
            total_contacts=total_contacts_val,
            genome_len=genome_len,
            resolution=resolution_used,
            window_bp=window_bp,
            median_noise=agg_noise,
            ebr=agg_ebr,
        )
        result["resolution_selected"] = resolution_selected
        result["resolution_used"] = resolution_used

    out_df = pd.DataFrame([result])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(
        f"[sequencing-advisor] total_contacts={result['total_contacts']:.0f} "
        f"resolution_used={result.get('resolution_used')} epsilon={result.get('epsilon')} "
        f"-> {result['recommendation']}  (report: {out_path})"
    )
    return result


def main() -> None:
    """
    Entry point for the ``jukebox-hic`` command-line tool.

    Parses arguments and dispatches to the appropriate module function based on
    the subcommand selected (``args.cmd``). All subcommands are wrapped in
    ``_profile_command()`` to emit a timing/memory summary line after completion.

    See the module docstring for a description of each subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="jukebox-hic",
        description="Lightweight Hi-C noise analysis toolkit for .hic/.cool matrices",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    parser.epilog = (
        'Note: The default install (pip install jukebox-hic) includes both cooler and hicstraw backends. '
        'Single-backend profiles: pip install "jukebox-hic[cooler]" or pip install "jukebox-hic[straw]". '
        'Windows users experiencing hicstraw build errors: see the README for a cooler-only workaround.'
    )

    # ------------------------------------------------------------------ #
    # sample-noise                                                         #
    # ------------------------------------------------------------------ #
    # Subparser for the "sample a fraction of rows, estimate noise" command.
    sp_sample = sub.add_parser("sample-noise", help="Compute noise on a sampled set of rows")
    sp_sample.add_argument("--hic", required=True, help="Input .hic file path")
    sp_sample.add_argument("--res", required=True, help="Comma-separated resolutions in bp")
    sp_sample.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of rows to evaluate (0-1)")
    sp_sample.add_argument("--window_bp", type=int, default=None, help="Half-window size around diagonal in bp")
    sp_sample.add_argument("--chrom_sizes", help="Chrom sizes TSV (chr\\tsize); default: read from input")
    sp_sample.add_argument("--out_dir", required=True, help="Directory for outputs")
    sp_sample.add_argument(
        "--norm",
        default="none",
        help="Normalization to apply: none | balance | path/to/bedgraph",
    )
    sp_sample.add_argument(
        "--cooler_path",
        help="Internal group path for .cool/.mcool/.scool URIs",
    )
    sp_sample.add_argument(
        "--min_mean_score",
        type=float,
        default=0.2,
        # 0.2 is the empirically chosen minimum mean observed/expected ratio:
        # rows below this have essentially no contacts relative to the distance-decay
        # expectation and produce meaningless noise estimates.
        help="Minimum distance-normalized mean score required to keep a row (default: 0.2)",
    )
    sp_sample.add_argument(
        "--min_nonzero_frac",
        type=float,
        default=0.01,
        # 0.01 = 1%: at least 1% of window positions must have a non-zero contact.
        # Below this, the window is too sparse for a stable autocorrelation estimate.
        help="Minimum fraction of non-zero bins in the window required to keep a row (default: 0.01)",
    )
    sp_sample.add_argument(
        "--subsample_ratios",
        # Example: "1.0,0.5,0.25" simulates the noise at full, half, and quarter depth.
        help="Comma-separated ratios (<=1.0) to simulate (e.g. 1.0,0.5,0.25); defaults to 1.0 only",
    )
    sp_sample.add_argument("--seed", type=int, help="Random seed for subsampling reproducibility")
    sp_sample.add_argument(
        "--profile",
        help="Write per-chromosome profiling metrics (runtime, memory) to the specified CSV",
    )
    sp_sample.add_argument(
        "--chroms",
        help="Comma-separated list of chromosomes to process (default: all)",
    )
    sp_sample.add_argument(
        "--chunk_rows",
        type=int,
        default=None,
        help=(
            "Process each chromosome in bands of this many rows instead of loading "
            "the full chromosome matrix at once.  Reduces peak RAM at the cost of "
            "re-computing the distance-decay curve per band.  Recommended: 2000–5000. "
            "Default: load full chromosome (original behaviour)."
        ),
    )

    # ------------------------------------------------------------------ #
    # full-noise                                                           #
    # ------------------------------------------------------------------ #
    # Subparser for the "compute noise for every bin genome-wide" command.
    sp_full = sub.add_parser("full-noise", help="Compute noise genome-wide using Hi-C expected vectors")
    sp_full.add_argument("--hic", required=True, help="Input .hic file path")
    sp_full.add_argument("--res", required=True, help="Comma-separated resolutions in bp")
    sp_full.add_argument("--chrom_sizes", help="Chrom sizes TSV (chr\\tsize); default: read from .hic")
    sp_full.add_argument(
        "--norm",
        default="none",
        help="Normalization to apply: none | balance | path/to/bedgraph (or hic label)",
    )
    sp_full.add_argument("--bindist_bp", type=int, help="Half-window distance from diagonal in bp")
    sp_full.add_argument("--out_dir", required=True, help="Directory for outputs")
    sp_full.add_argument(
        "--cooler_path",
        help="Internal group path for .cool/.mcool/.scool URIs",
    )
    sp_full.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of worker processes to use for full-noise (default: 1)",
    )
    sp_full.add_argument(
        "--chroms",
        help="Comma-separated list of chromosomes to process (default: all)",
    )
    sp_full.add_argument(
        "--max_memory_gb",
        type=float,
        default=None,
        help=(
            "Hard memory budget in GB.  Caps the number of parallel workers so that "
            "estimated peak RAM stays within this limit.  Each worker also receives a "
            "RLIMIT_AS address-space cap of budget/workers GB (Linux/macOS only). "
            "Default: no cap."
        ),
    )
    sp_full.add_argument(
        "--dump_factor",
        type=float,
        default=10.0,
        help=(
            "Sliding-window chunk size multiplier for RAM control. "
            "Each chunk covers max(2, dump_factor) × noise_bins core bins. "
            "Lower values reduce peak RAM at the cost of more region fetches. "
            "Default 10.0 works well down to ~1 kb resolution. "
            "Try 4.0–6.0 for sub-1 kb or very large chromosomes."
        ),
    )

    # ------------------------------------------------------------------ #
    # bias-vectors                                                         #
    # ------------------------------------------------------------------ #
    # Subparser for transforming full-noise bedgraphs into Juicer normalization vectors.
    sp_bias = sub.add_parser(
        "bias-vectors",
        help="Transform full-noise bedGraphs into JUKEBOX normalization vectors (Juicer format)",
    )
    sp_bias.add_argument(
        "--noise_dir",
        required=True,
        help="Directory containing {res}.bedgraph files (from full-noise)",
    )
    sp_bias.add_argument(
        "--out_dir",
        required=True,
        help="Output directory; vectors written as {res}.juicervector",
    )
    sp_bias.add_argument(
        "--res",
        type=int,
        default=None,
        help=(
            "Resolution in bp to process (single integer, e.g. 10000). "
            "If omitted, auto-detected from --noise_dir only when exactly one "
            "{res}.bedgraph file is present. Run bias-vectors once per resolution."
        ),
    )
    sp_bias.add_argument(
        "--zmap_summary",
        help="Path to subsample_summary.tsv from sample-noise; provides per-chrom pred_log_N and EBR",
    )
    sp_bias.add_argument(
        "--chrom_sizes",
        help="Chrom sizes TSV for grid validation (optional)",
    )
    sp_bias.add_argument(
        "--nan_fill",
        default="nan",
        choices=["nan", "0"],
        help="Value written for NaN/masked bins (default: nan; use 0 for engines that reject NaN)",
    )
    sp_bias.add_argument(
        "--include_decoys",
        action="store_true",
        help="Include decoy/unplaced chromosomes (default: skip)",
    )
    sp_bias.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Scaling constant for JUKEBOX normalization (default: 0.5).",
    )
    sp_bias.add_argument(
        "--p_factor",
        type=float,
        default=2.0,
        help="Compression factor for JUKEBOX normalization (default: 2.0; square root). Use 3.0 for cube root.",
    )

    # ------------------------------------------------------------------ #
    # plot                                                                 #
    # ------------------------------------------------------------------ #
    # Subparser for noise distribution visualization.
    sp_plot = sub.add_parser("plot", help="Plot density of noise values from a bedgraph")
    sp_plot.add_argument("--noise_bed", required=True, help="Noise bedgraph path")
    sp_plot.add_argument("--out_png", required=True, help="Output PNG path")

    # ------------------------------------------------------------------ #
    # blacklist                                                            #
    # ------------------------------------------------------------------ #
    # Subparser for elbow-based blacklist generation.
    sp_mask = sub.add_parser(
        "blacklist",
        help="Build a per-chromosome BED blacklist using elbow detection on density + noise",
    )
    sp_mask.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Output directory.  Writes: elbow_blacklist.bed, elbow_thresholds.tsv, "
            "elbow.png, elbow.pdf"
        ),
    )
    sp_mask.add_argument(
        "--density_bedgraph",
        required=True,
        help="Density bedgraph ({res}_density.bedgraph) produced by full-noise",
    )
    sp_mask.add_argument(
        "--noise_bedgraph",
        required=True,
        help="Noise bedgraph ({res}.bedgraph) produced by full-noise",
    )
    # Elbow tuning
    sp_mask.add_argument(
        "--density_transform", default="none",
        choices=["none", "sqrt", "cbrt", "log1p"],
        help="Pre-transform for density values in elbow detection (default: none)",
    )
    sp_mask.add_argument(
        "--noise_transform", default="sqrt",
        choices=["none", "sqrt", "cbrt", "log1p"],
        help="Pre-transform for noise values in elbow detection (default: sqrt)",
    )
    sp_mask.add_argument(
        "--density_upper_frac", type=float, default=0.01,
        help="Top fraction of density bins searched for the upper elbow (default: 0.01 = top 1%%)",
    )
    sp_mask.add_argument(
        "--density_lower_frac", type=float, default=0.01,
        help="Bottom fraction of density bins searched for the lower elbow (default: 0.01 = bottom 1%%)",
    )
    sp_mask.add_argument(
        "--noise_upper_frac", type=float, default=0.10,
        help="Top fraction of noise bins searched for the upper elbow (default: 0.10 = top 10%%)",
    )
    sp_mask.add_argument(
        "--noise_lower_frac", type=float, default=0.01,
        help="Bottom fraction of noise bins searched for the lower elbow (default: 0.01 = bottom 1%%)",
    )
    sp_mask.add_argument(
        "--elbow_smooth_sigma", type=float, default=10.0,
        help="Gaussian smoothing sigma for elbow detection (default: 10.0)",
    )
    sp_mask.add_argument(
        "--require_both_metrics", action="store_true", default=False,
        help=(
            "Use the intersection rule: flag a bin only when it is out-of-bounds "
            "for BOTH density AND noise.  Default (off) uses the union rule: flag "
            "when out-of-bounds for either metric."
        ),
    )

    # ------------------------------------------------------------------ #
    # sequencing-advisor                                                   #
    # ------------------------------------------------------------------ #
    # Subparser for the standalone whole-map sequencing-sufficiency advisor.
    sp_adv = sub.add_parser(
        "sequencing-advisor",
        help="Whole-map ENCODE/4DN sequencing-sufficiency check (single genome-wide recommendation)",
    )
    sp_adv.add_argument(
        "--summary_dir",
        required=True,
        help=(
            "Directory containing one <resolution>/subsample_summary.tsv per resolution, "
            "as produced by 'sample-noise --out_dir <this>' (or default-run's "
            "<out_dir>/sample directory)."
        ),
    )
    sp_adv.add_argument("--out", required=True, help="Output TSV path for the whole-map advisor report")

    # ------------------------------------------------------------------ #
    # default-run                                                          #
    # ------------------------------------------------------------------ #
    # Subparser for the full automated pipeline (sample → full → vectors → blacklist).
    sp_run = sub.add_parser(
        "default-run",
        help=(
            "Full JUKEBOX pipeline: sample noise → full noise → bias vectors → blacklist. "
            "Runs at no normalization using 4DN reference benchmarking."
        ),
    )
    sp_run.add_argument("--hic", required=True, help="Input .hic or cooler file path")
    sp_run.add_argument("--res", required=True, help="Comma-separated resolutions in bp")
    sp_run.add_argument("--out_dir", required=True, help="Root output directory")
    sp_run.add_argument(
        "--sample_name",
        required=True,
        help="Sample label used in vector headers and output file names",
    )
    sp_run.add_argument("--chrom_sizes", help="Chrom sizes TSV (chr\\tsize)")
    sp_run.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Worker processes for full-noise (default: 1)",
    )
    sp_run.add_argument(
        "--cooler_path",
        help="Internal group path for .cool/.mcool/.scool URIs",
    )
    sp_run.add_argument(
        "--nan_fill",
        default="nan",
        choices=["nan", "0"],
        help="Value for masked NaN bins in output vectors (default: nan)",
    )
    sp_run.add_argument(
        "--chroms",
        help="Comma-separated list of chromosomes to process (default: all)",
    )
    sp_run.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Scaling constant for JUKEBOX normalization (default: 0.5).",
    )
    sp_run.add_argument(
        "--p_factor",
        type=float,
        default=2.0,
        help="Compression factor for JUKEBOX normalization (default: 2.0; square root). Use 3.0 for cube root.",
    )
    sp_run.add_argument(
        "--dump_factor",
        type=float,
        default=10.0,
        help=(
            "Sliding-window chunk size multiplier passed to full-noise. "
            "Lower values reduce peak RAM. Default: 10.0."
        ),
    )
    sp_run.add_argument(
        "--max_memory_gb",
        type=float,
        default=None,
        help=(
            "Hard memory budget in GB for the full-noise phase.  Caps worker count "
            "and sets a per-worker address-space limit (Linux/macOS).  Default: no cap."
        ),
    )
    sp_run.add_argument(
        "--chunk_rows",
        type=int,
        default=None,
        help=(
            "Band size (rows) for sample-noise phase.  Prevents loading the full "
            "chromosome matrix at once.  Recommended: 2000–5000.  Default: full load."
        ),
    )
    # Elbow blacklist tuning (Phase 4)
    sp_run.add_argument(
        "--density_transform", default="none",
        choices=["none", "sqrt", "cbrt", "log1p"],
        help="Pre-transform for density values in elbow detection (default: none)",
    )
    sp_run.add_argument(
        "--noise_transform", default="sqrt",
        choices=["none", "sqrt", "cbrt", "log1p"],
        help="Pre-transform for noise values in elbow detection (default: sqrt)",
    )
    sp_run.add_argument("--density_upper_frac", type=float, default=0.01,
                        help="Top fraction of density bins searched for the upper elbow (default: 0.01 = top 1%%)")
    sp_run.add_argument("--density_lower_frac", type=float, default=0.01,
                        help="Bottom fraction of density bins searched for the lower elbow (default: 0.01 = bottom 1%%)")
    sp_run.add_argument("--noise_upper_frac", type=float, default=0.10,
                        help="Top fraction of noise bins searched for the upper elbow (default: 0.10 = top 10%%)")
    sp_run.add_argument("--noise_lower_frac", type=float, default=0.01,
                        help="Bottom fraction of noise bins searched for the lower elbow (default: 0.01 = bottom 1%%)")
    sp_run.add_argument("--elbow_smooth_sigma", type=float, default=10.0,
                        help="Gaussian smoothing sigma for elbow detection (default: 10.0)")
    sp_run.add_argument(
        "--require_both_metrics", action="store_true", default=False,
        help=(
            "Use the intersection rule for blacklisting: flag a bin only when it is "
            "out-of-bounds for BOTH density AND noise.  Default (off) uses the union rule."
        ),
    )

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # Argument parsing is complete; dispatch to the appropriate subcommand.
    # ------------------------------------------------------------------ #
    args = parser.parse_args()

    if args.cmd == "sample-noise":
        resolutions = _comma_ints(args.res)
        chroms = _comma_strs(args.chroms) if args.chroms else None
        # Count contacts once at the coarsest resolution, shared across all resolutions.
        # This avoids reopening the file repeatedly for the contacts_overview.tsv output.
        contacts = _dump_contacts_once(args.hic, args.norm, args.cooler_path, args.chrom_sizes)

        def run() -> None:
            for res in resolutions:
                # Each resolution gets its own subdirectory inside out_dir
                res_dir = os.path.join(args.out_dir, str(res))
                os.makedirs(res_dir, exist_ok=True)

                if contacts:
                    contacts_path = os.path.join(res_dir, "contacts_overview.tsv")
                    total_value = sum(contacts.values())
                    with open(contacts_path, "w") as handle:
                        handle.write("chrom\tcontacts\n")
                        for chrom in sorted(contacts):
                            handle.write(f"{chrom}\t{contacts[chrom]:.6f}\n")
                        handle.write(f"TOTAL\t{total_value:.6f}\n")

                noise_sampling.compute_sampled_noise(
                    hic_path=args.hic,
                    res=res,
                    sample_fraction=args.sample_fraction,
                    window_bp=args.window_bp,
                    chrom_sizes_path=args.chrom_sizes,
                    out_dir=res_dir,
                    norm=args.norm,
                    cooler_selection=args.cooler_path,
                    min_mean_score=args.min_mean_score,
                    min_nonzero_frac=args.min_nonzero_frac,
                    subsample_ratios=[float(x) for x in args.subsample_ratios.split(",")] if args.subsample_ratios else None,
                    seed=args.seed,
                    profile_path=args.profile,
                    chroms=chroms,
                    chunk_rows=args.chunk_rows,
                    precomputed_contacts=contacts or None,
                )

        _profile_command("sample-noise", run)

    elif args.cmd == "full-noise":
        os.makedirs(args.out_dir, exist_ok=True)
        contacts = _dump_contacts_once(args.hic, args.norm, args.cooler_path, args.chrom_sizes)
        if contacts:
            out_path = os.path.join(args.out_dir, "contacts_overview.tsv")
            total_value = sum(contacts.values())
            with open(out_path, "w") as handle:
                handle.write("chrom\tcontacts\n")
                for chrom in sorted(contacts):
                    handle.write(f"{chrom}\t{contacts[chrom]:.6f}\n")
                handle.write(f"TOTAL\t{total_value:.6f}\n")

        def run() -> None:
            noise_fullmap.compute_full_noise(
                hic_path=args.hic,
                res_list=_comma_ints(args.res),
                chrom_sizes_path=args.chrom_sizes,
                norm=args.norm,
                bindist_bp=args.bindist_bp,
                out_dir=args.out_dir,
                cooler_selection=args.cooler_path,
                cpu=args.cpu,
                chroms=_comma_strs(args.chroms) if args.chroms else None,
                dump_factor=args.dump_factor,
                max_memory_gb=args.max_memory_gb,
            )

        _profile_command("full-noise", run)

    elif args.cmd == "bias-vectors":
        if args.res:
            res = args.res
        else:
            # Auto-detect only when exactly one {integer}.bedgraph exists in noise_dir;
            # error when multiple are found so the user picks one explicitly.
            detected = _detect_resolutions_in_dir(args.noise_dir)
            if not detected:
                parser.error(
                    f"No {{res}}.bedgraph files found in {args.noise_dir}. "
                    "Use --res to specify the resolution explicitly."
                )
            if len(detected) > 1:
                parser.error(
                    f"Multiple resolutions found in {args.noise_dir}: {detected}. "
                    "bias-vectors processes one resolution at a time — "
                    "use --res to specify which resolution to process."
                )
            res = detected[0]
        if not args.zmap_summary:
            parser.error(
                "--zmap_summary is required. "
                "Run sample-noise first and pass the resulting subsample_summary.tsv."
            )
        if not os.path.isfile(args.zmap_summary):
            parser.error(f"--zmap_summary file not found: {args.zmap_summary}")
        os.makedirs(args.out_dir, exist_ok=True)
        # Convert the "nan"/"0" CLI choice to an actual float value
        nan_fill = float("nan") if args.nan_fill == "nan" else 0.0

        def run() -> None:
            noise_to_weights.build_bias_vectors_from_bedgraphs(
                noise_dir=args.noise_dir,
                res=res,
                out_dir=args.out_dir,
                zmap_summary_path=args.zmap_summary,
                chrom_sizes_path=args.chrom_sizes,
                nan_fill_value=nan_fill,
                skip_decoys=not bool(args.include_decoys),
                alpha=args.alpha,
                p_factor=args.p_factor,
            )
            print(f"  → {os.path.join(args.out_dir, f'{res}.juicervector')}")

        _profile_command("bias-vectors", run)

    elif args.cmd == "plot":

        def run() -> None:
            figures.plot_noise_density_from_bed(args.noise_bed, args.out_png)

        _profile_command("plot", run)

    elif args.cmd == "blacklist":
        os.makedirs(args.out_dir, exist_ok=True)
        out_prefix    = os.path.join(args.out_dir, "elbow")
        out_blacklist = out_prefix + "_blacklist.bed"
        out_tsv       = out_prefix + "_thresholds.tsv"

        def run() -> None:
            thresholds_df, per_chrom_curves = filters.build_blacklist_from_elbow_thresholds(
                density_bedgraph=args.density_bedgraph,
                noise_bedgraph=args.noise_bedgraph,
                output_path=out_blacklist,
                smooth_sigma=args.elbow_smooth_sigma,
                density_upper_frac=args.density_upper_frac,
                density_lower_frac=args.density_lower_frac,
                noise_upper_frac=args.noise_upper_frac,
                noise_lower_frac=args.noise_lower_frac,
                density_transform=args.density_transform,
                noise_transform=args.noise_transform,
                require_both_metrics=args.require_both_metrics,
            )
            thresholds_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.6g")
            figures.plot_elbow_figure(
                per_chrom_curves=per_chrom_curves,
                results=thresholds_df.to_dict("records"),
                out_prefix=out_prefix,
                density_transform=args.density_transform,
                noise_transform=args.noise_transform,
            )
            print(f"  → {out_blacklist}")
            print(f"  → {out_tsv}")
            print(f"  → {out_prefix}.png  /  {out_prefix}.pdf")

        _profile_command("blacklist", run)

    elif args.cmd == "sequencing-advisor":
        if not os.path.isdir(args.summary_dir):
            parser.error(f"--summary_dir not found or not a directory: {args.summary_dir}")

        def run() -> None:
            _run_sequencing_advisor(args.summary_dir, args.out)

        _profile_command("sequencing-advisor", run)

    elif args.cmd == "default-run":
        # Full pipeline: all four phases run in sequence.
        resolutions = _comma_ints(args.res)
        cooler_sel = getattr(args, "cooler_path", None)
        nan_fill = float("nan") if args.nan_fill == "nan" else 0.0
        chroms = _comma_strs(args.chroms) if args.chroms else None

        # Output subdirectory layout:
        #   {out_dir}/sample/{res}/   — per-resolution sampling outputs
        #   {out_dir}/full/           — full-noise bedgraphs
        #   {out_dir}/vectors/        — Juicer normalization vectors
        sample_root = os.path.join(args.out_dir, "sample")
        full_dir = os.path.join(args.out_dir, "full")
        vectors_dir = os.path.join(args.out_dir, "vectors")
        os.makedirs(sample_root, exist_ok=True)
        os.makedirs(full_dir, exist_ok=True)
        os.makedirs(vectors_dir, exist_ok=True)

        # Count contacts once (resolution-independent) for the contacts overview.
        # Use "none" normalisation for counting raw contact pairs.
        contacts = _dump_contacts_once(args.hic, "none", cooler_sel, args.chrom_sizes)

        # Phase 1: Sample noise per resolution
        for res in resolutions:
            print(f"\n[Phase 1] Sampling noise at {res} bp ...")
            res_sample_dir = os.path.join(sample_root, str(res))
            os.makedirs(res_sample_dir, exist_ok=True)

            if contacts:
                contacts_path = os.path.join(res_sample_dir, "contacts_overview.tsv")
                total_value = sum(contacts.values())
                with open(contacts_path, "w") as handle:
                    handle.write("chrom\tcontacts\n")
                    for chrom in sorted(contacts):
                        handle.write(f"{chrom}\t{contacts[chrom]:.6f}\n")
                    handle.write(f"TOTAL\t{total_value:.6f}\n")

            noise_sampling.compute_sampled_noise(
                hic_path=args.hic,
                res=res,
                out_dir=res_sample_dir,
                norm="none",  # always use raw counts in default-run
                chrom_sizes_path=args.chrom_sizes,
                cooler_selection=cooler_sel,
                chroms=chroms,
                chunk_rows=args.chunk_rows,
                precomputed_contacts=contacts or None,
            )

        # Phase 1.5: Whole-map sequencing sufficiency check (genome-wide, ENCODE/4DN rule)
        print("\n[Phase 1.5] Sequencing-advisor whole-map check ...")
        adv_out = os.path.join(args.out_dir, "sequencing_advisor_summary.tsv")
        adv_result = _run_sequencing_advisor(sample_root, adv_out)

        # Phase 2: Full noise for all resolutions at once (benefits from shared I/O)
        print(f"\n[Phase 2] Computing full noise at {resolutions} bp ...")
        noise_fullmap.compute_full_noise(
            hic_path=args.hic,
            res_list=resolutions,
            norm="none",
            out_dir=full_dir,
            chrom_sizes_path=args.chrom_sizes,
            cooler_selection=cooler_sel,
            cpu=args.cpu,
            chroms=chroms,
            dump_factor=args.dump_factor,
            max_memory_gb=args.max_memory_gb,
        )

        # Phases 3 & 4: Per resolution
        for res in resolutions:
            res_sample_dir = os.path.join(sample_root, str(res))
            zmap_summary = os.path.join(res_sample_dir, "subsample_summary.tsv")

            # Phase 3: Bias vectors (JUKEBOX normalization)
            print(f"\n[Phase 3] Building bias vectors at {res} bp ...")
            if not os.path.isfile(zmap_summary):
                print(f"  [WARN] subsample_summary.tsv not found for res={res} — skipping bias vectors")
            else:
                noise_to_weights.build_bias_vectors_from_bedgraphs(
                    noise_dir=full_dir,
                    res=res,
                    out_dir=vectors_dir,
                    zmap_summary_path=zmap_summary,
                    chrom_sizes_path=args.chrom_sizes,
                    nan_fill_value=nan_fill,
                    alpha=args.alpha,
                    p_factor=args.p_factor,
                )
                print(f"  → {os.path.join(vectors_dir, f'{res}.juicervector')}")

            # Phase 4: Elbow-based blacklist + diagnostic figures
            print(f"\n[Phase 4] Building elbow blacklist at {res} bp ...")
            density_bed = os.path.join(full_dir, f"{res}_density.bedgraph")
            noise_bed   = os.path.join(full_dir, f"{res}.bedgraph")

            if os.path.isfile(noise_bed) and os.path.isfile(density_bed):
                out_prefix    = os.path.join(args.out_dir, f"{args.sample_name}_{res}_elbow")
                out_blacklist = out_prefix + "_blacklist.bed"
                out_tsv       = out_prefix + "_thresholds.tsv"

                thresholds_df, per_chrom_curves = filters.build_blacklist_from_elbow_thresholds(
                    density_bedgraph=density_bed,
                    noise_bedgraph=noise_bed,
                    output_path=out_blacklist,
                    smooth_sigma=args.elbow_smooth_sigma,
                    density_upper_frac=args.density_upper_frac,
                    density_lower_frac=args.density_lower_frac,
                    noise_upper_frac=args.noise_upper_frac,
                    noise_lower_frac=args.noise_lower_frac,
                    density_transform=args.density_transform,
                    noise_transform=args.noise_transform,
                    require_both_metrics=args.require_both_metrics,
                )
                thresholds_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.6g")
                figures.plot_elbow_figure(
                    per_chrom_curves=per_chrom_curves,
                    results=thresholds_df.to_dict("records"),
                    out_prefix=out_prefix,
                    density_transform=args.density_transform,
                    noise_transform=args.noise_transform,
                )
                print(f"  → {out_blacklist}")
                print(f"  → {out_tsv}")
                print(f"  → {out_prefix}.png  /  {out_prefix}.pdf")
            else:
                print(f"  [WARN] Bedgraph(s) not found for res={res} — skipping blacklist")

        print(
            f"\n[default-run] Complete. Sequencing advisor: {adv_result['recommendation']} "
            f"(epsilon={adv_result.get('epsilon')}, resolution_used={adv_result.get('resolution_used')})"
        )

    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
