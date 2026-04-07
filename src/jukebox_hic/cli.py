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
    Transform full-noise bedgraphs into Juicer-format custom normalization vectors.
    Requires the subsample_summary.tsv from sample-noise to obtain per-chromosome
    z-map parameters.

plot
    Render a density histogram of noise values from a bedgraph file.

blacklist
    Build a BED-format blacklist from per-chromosome noise bedgraphs by flagging
    bins with high noise or NaN values.

sequencing-advisor
    Re-run the sequencing depth advisor from a saved subsample_summary.tsv, optionally
    with different target z-score or fold-increase threshold.

default-run
    Run the full jukebox-hic pipeline in sequence:
    Phase 1: sample-noise → Phase 2: full-noise → Phase 3: bias-vectors → Phase 4: blacklist.
    Use this for a complete first-pass analysis.
"""
import argparse
import glob
import os
import time
from typing import Callable, Dict, List, Optional

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
        'Note: Base install includes "cooler" only. For .hic support, install extras: '
        'pip install "jukebox-hic[straw]" or "jukebox-hic[all]".'
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

    # ------------------------------------------------------------------ #
    # bias-vectors                                                         #
    # ------------------------------------------------------------------ #
    # Subparser for transforming full-noise bedgraphs into Juicer normalization vectors.
    sp_bias = sub.add_parser(
        "bias-vectors",
        help="Transform full-noise bedGraphs into Juicer normalization vectors",
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
        default=None,
        help=(
            "Comma-separated resolutions in bp to process "
            "(default: auto-detect from {res}.bedgraph files in --noise_dir)"
        ),
    )
    sp_bias.add_argument(
        "--zmap_summary",
        help="Path to subsample_summary.tsv from sample-noise; provides per-chrom gamma and EBR",
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
        "--mode",
        default="baseline",
        choices=["baseline", "adaptive"],
        help=(
            "Normalization mode: "
            "baseline = JUKEBOX-BASELINE (precise per-bin alignment to 4DN reference); "
            "adaptive = JUKEBOX-ADAPTIVE (global z-shift + dampened local variance, "
            "recommended for sparse/palaeogenomic data). Default: baseline"
        ),
    )
    sp_bias.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        # Alpha controls how much of the per-bin local noise deviation is retained
        # in the adaptive bias vector. 0.5 = aggressive smoothing, 0.8 = conservative.
        # 0.7 is a balanced default. Values outside 0.5–0.8 are not recommended.
        help="Damping factor for adaptive mode (default: 0.7; valid range 0.5–0.8)",
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
    # Subparser for flagging noisy/sparse bins and producing a BED blacklist.
    sp_mask = sub.add_parser("blacklist", help="Build a blacklist from per-chrom bedgraphs")
    sp_mask.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more bedgraph paths or glob patterns (e.g. chr*.bedgraph)",
    )
    sp_mask.add_argument("--out", required=True, help="Output BED path for blacklisted intervals")
    sp_mask.add_argument(
        "--zscore_cutoff",
        type=float,
        help="Z-score cutoff; rows with noise z-score >= value are blacklisted",
    )
    sp_mask.add_argument(
        "--top_quantile",
        type=float,
        default=0.95,
        # 0.95 = flag the top 5% noisiest bins. Use 0.99 for a more conservative list
        # (only the absolute worst 1%). The default-run pipeline uses 0.99.
        help="Quantile threshold for blacklisting when z-score cutoff is not provided (default: 0.95)",
    )

    # ------------------------------------------------------------------ #
    # sequencing-advisor                                                   #
    # ------------------------------------------------------------------ #
    # Subparser for the standalone sequencing depth advisor (reads a saved summary).
    sp_adv = sub.add_parser(
        "sequencing-advisor",
        help="Predict sequencing depth needed to reach a target noise z-score",
    )
    sp_adv.add_argument(
        "--summary",
        required=True,
        help="Path to subsample_summary.tsv produced by sample-noise",
    )
    sp_adv.add_argument("--out", required=True, help="Output TSV path for the advisor report")
    sp_adv.add_argument(
        "--z_target",
        type=float,
        default=0.0,
        # z_target=0.0 means "reach the 4DN reference quality level".
        # Negative values target better-than-reference quality (more sequencing required).
        # Positive values accept worse-than-reference quality (less sequencing needed).
        help="Target z-score to reach (default: 0.0 = 4DN reference level)",
    )
    sp_adv.add_argument(
        "--fold_threshold",
        type=float,
        default=10.0,
        # If more than 10× additional sequencing is needed, the advisor recommends "Stop"
        # because the return on investment is too low (diminishing returns in Hi-C).
        help='Fold-increase above which recommendation is "Stop" (default: 10.0)',
    )

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
        "--mode",
        default="baseline",
        choices=["baseline", "adaptive"],
        help=(
            "Normalization mode for bias vectors: baseline (default) or adaptive. "
            "See bias-vectors --help for details."
        ),
    )
    sp_run.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Damping factor for adaptive mode (default: 0.7; valid range 0.5–0.8)",
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
            )

        _profile_command("full-noise", run)

    elif args.cmd == "bias-vectors":
        if args.res:
            resolutions = _comma_ints(args.res)
        else:
            # Auto-detect resolutions from files named {integer}.bedgraph in noise_dir
            resolutions = _detect_resolutions_in_dir(args.noise_dir)
            if not resolutions:
                parser.error(
                    f"No {{res}}.bedgraph files found in {args.noise_dir}. "
                    "Use --res to specify resolutions explicitly."
                )
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
            for res in resolutions:
                noise_to_weights.build_bias_vectors_from_bedgraphs(
                    noise_dir=args.noise_dir,
                    res=res,
                    out_dir=args.out_dir,
                    zmap_summary_path=args.zmap_summary,
                    chrom_sizes_path=args.chrom_sizes,
                    nan_fill_value=nan_fill,
                    skip_decoys=not bool(args.include_decoys),
                    mode=args.mode,
                    alpha=args.alpha,
                )
                print(f"  → {os.path.join(args.out_dir, f'{res}.juicervector')}")

        _profile_command("bias-vectors", run)

    elif args.cmd == "plot":

        def run() -> None:
            figures.plot_noise_density_from_bed(args.noise_bed, args.out_png)

        _profile_command("plot", run)

    elif args.cmd == "blacklist":

        def run() -> None:
            filters.build_blacklist_from_bedgraphs(
                inputs=args.input,
                output_path=args.out,
                zscore_cutoff=args.zscore_cutoff,
                top_quantile=args.top_quantile,
            )

        _profile_command("blacklist", run)

    elif args.cmd == "sequencing-advisor":
        import io
        import pandas as pd

        if not os.path.isfile(args.summary):
            parser.error(f"Summary file not found: {args.summary}")

        # Parse the subsample_summary.tsv header comments to extract metadata.
        # The file format uses two types of comment lines:
        #   "# resolution=N" — the resolution this summary was computed at
        #   "# chrom_info<tab>chrom<tab>contacts=N<tab>size_bp=N" — chromosome metadata
        # These are needed because chromosome sizes are not stored in the TSV body.
        summary_res: Optional[int] = None
        chrom_sizes_adv: Dict[str, int] = {}
        data_lines: List[str] = []
        with open(args.summary) as fh:
            for line in fh:
                if line.startswith("# resolution="):
                    try:
                        summary_res = int(line.strip().split("=", 1)[1])
                    except ValueError:
                        pass
                elif line.startswith("# chrom_info"):
                    # Parse: "# chrom_info\tchromname\tcontacts=N\tsize_bp=N"
                    parts = line.strip().split("\t")
                    chrom_name = parts[1] if len(parts) > 1 else None
                    for part in parts[2:]:
                        if part.startswith("size_bp="):
                            try:
                                size = int(part.split("=", 1)[1])
                                if chrom_name:
                                    chrom_sizes_adv[chrom_name] = size
                            except ValueError:
                                pass
                elif not line.startswith("#"):
                    data_lines.append(line)

        if not data_lines:
            parser.error(f"No data rows found in {args.summary}")
        if summary_res is None:
            parser.error(
                "Could not read resolution from the summary file. "
                "Ensure the file was produced by jukebox-hic sample-noise."
            )
        if not chrom_sizes_adv:
            parser.error(
                "Could not recover chromosome sizes from the summary file. "
                "Ensure the file was produced by jukebox-hic sample-noise."
            )

        df = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t")
        if "ratio" in df.columns:
            # Only use full-depth (ratio=1.0) rows for advisor analysis
            df = df[df["ratio"] == 1.0].copy()
        if df.empty:
            parser.error("No ratio=1.0 rows found in the summary file")

        required_cols = {"chrom", "median_noise", "mean_empty_bin_ratio", "estimated_contacts"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            parser.error(f"Summary file is missing columns: {sorted(missing_cols)}")

        def run() -> None:
            records = []
            for _, row in df.iterrows():
                chrom = str(row["chrom"])
                median_noise = float(row["median_noise"])
                ebr = float(row["mean_empty_bin_ratio"])
                est_contacts = float(row["estimated_contacts"])
                z_map_val = float(row["z_map"]) if "z_map" in df.columns else float("nan")
                c_len = chrom_sizes_adv.get(chrom, 0)

                if c_len > 0:
                    # current_rho = contacts per bin = sequencing density for this chrom
                    current_rho = est_contacts / (c_len / summary_res)
                    adv = reference.sequencing_advisor(
                        obs_noise=median_noise,
                        current_rho=current_rho,
                        ebr=ebr,
                        z_target=args.z_target,
                        fold_threshold=args.fold_threshold,
                    )
                else:
                    # Chromosome size not available — cannot compute sequencing density
                    adv = {
                        "target_density": float("nan"),
                        "fold_increase": float("nan"),
                        "efficiency_index": float("nan"),
                        "recommendation": "Insufficient data",
                    }

                records.append({
                    "chrom": chrom,
                    "median_noise": median_noise,
                    "mean_empty_bin_ratio": ebr,
                    "estimated_contacts": est_contacts,
                    "z_map": z_map_val,
                    "advisor_target_density": adv["target_density"],
                    "advisor_fold_increase": adv["fold_increase"],
                    "advisor_efficiency_index": adv["efficiency_index"],
                    "advisor_recommendation": adv["recommendation"],
                })

            out_df = pd.DataFrame(records)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            out_df.to_csv(args.out, sep="\t", index=False)
            print(f"[sequencing-advisor] Report written to {args.out}")

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
            )

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
        )

        # Phases 3 & 4: Per resolution
        for res in resolutions:
            res_sample_dir = os.path.join(sample_root, str(res))
            zmap_summary = os.path.join(res_sample_dir, "subsample_summary.tsv")

            # Phase 3: Bias vectors (requires subsample_summary.tsv from Phase 1)
            print(f"\n[Phase 3] Building bias vectors at {res} bp (mode={args.mode}) ...")
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
                    mode=args.mode,
                    alpha=args.alpha,
                )
                print(f"  → {os.path.join(vectors_dir, f'{res}.juicervector')}")

            # Phase 4: Blacklist (flags top 1% noise bins — more conservative than the
            # standalone ``blacklist`` command's default of top 5%)
            print(f"\n[Phase 4] Building blacklist (P99 + NaN) at {res} bp ...")
            full_noise_bed = os.path.join(full_dir, f"{res}.bedgraph")
            if os.path.isfile(full_noise_bed):
                out_blacklist = os.path.join(
                    args.out_dir, f"{args.sample_name}_{res}_noise_blacklist.bed"
                )
                filters.build_blacklist_from_bedgraphs(
                    inputs=[full_noise_bed],
                    output_path=out_blacklist,
                    # 0.99 = top 1% threshold; stricter than the standalone command's 0.95 default
                    # because the full-noise bedgraph covers the entire genome and is more reliable.
                    top_quantile=0.99,
                )
                print(f"  → {out_blacklist}")
            else:
                print(f"  [WARN] Full-noise bedgraph not found: {full_noise_bed}")

        print("\n[default-run] Complete.")

    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
