#!/usr/bin/env python
import argparse
import os
import time
from typing import Callable, Dict, List, Optional

from . import figures, noise_fullmap, noise_sampling, filters, noise_to_weights
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
    return list(map(int, value.split(",")))


def _profile_command(name: str, func: Callable[[], Optional[int]]) -> Optional[int]:
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
    if is_cooler_path(path):
        return cooler_resolutions(path, cooler_selection)
    return hic_resolutions(path)


def _coarsest_resolution(path: str, cooler_selection: Optional[str]) -> Optional[int]:
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
    parser = argparse.ArgumentParser(
        prog="jukebox-hic",
        description="Lightweight Hi-C noise analysis toolkit for .hic/.cool matrices",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_sample = sub.add_parser("sample-noise", help="Compute noise on a sampled set of rows")
    sp_sample.add_argument("--hic", required=True, help="Input .hic file path")
    sp_sample.add_argument("--res", required=True, type=int, help="Resolution in bp")
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
        help="Internal group path for .cool/.mcool/.scool URIs (e.g. /resolutions/10000 or cell name)",
    )
    sp_sample.add_argument(
        "--min_mean_score",
        type=float,
        default=0.9,
        help="Minimum distance-normalized mean score required to keep a row (default: 0.9)",
    )
    sp_sample.add_argument(
        "--min_nonzero_frac",
        type=float,
        default=0.05,
        help="Minimum fraction of non-zero bins in the window required to keep a row (default: 0.05)",
    )
    sp_sample.add_argument(
        "--subsample_ratios",
        help="Comma-separated ratios (<=1.0) to simulate (e.g. 1.0,0.5,0.25); defaults to 1.0 only",
    )
    sp_sample.add_argument("--seed", type=int, help="Random seed for subsampling reproducibility")
    sp_sample.add_argument(
        "--profile",
        help="Write per-chromosome profiling metrics (runtime, memory) to the specified CSV",
    )

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
        help="Internal group path for .cool/.mcool/.scool URIs (e.g. /resolutions/10000 or cell name)",
    )
    sp_full.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of worker processes to use for full-noise (default: 1)",
    )

    sp_weights = sub.add_parser(
        "noise-to-weights",
        help="Convert noise bedGraphs into normalization vectors",
    )
    sp_weights.add_argument(
        "--scores_dir",
        required=True,
        help="Directory containing resolution-named bedGraphs or a single bedGraph file",
    )
    sp_weights.add_argument(
        "--res",
        required=True,
        type=int,
        help="Resolution in bp matching the target bedGraph file",
    )
    sp_weights.add_argument(
        "--out_dir",
        required=True,
        help="Directory for output weight bedGraphs and QC summaries",
    )
    sp_weights.add_argument(
        "--coverage",
        help="Optional coverage bedGraph file or directory with resolution-named files",
    )
    sp_weights.add_argument(
        "--chrom_sizes",
        help="Optional chrom sizes TSV (chr\\tsize) for validation and clipping",
    )
    sp_weights.add_argument(
        "--chroms",
        help="Comma-separated list of chromosomes to include",
    )
    sp_weights.add_argument("--tau", type=float, help="Override tau hyperparameter")
    sp_weights.add_argument("--kappa", type=float, help="Override kappa hyperparameter")
    sp_weights.add_argument("--gain", type=float, help="Override gain hyperparameter")
    sp_weights.add_argument("--lambda_", type=float, help="Override shrinkage lambda")
    sp_weights.add_argument("--sigma_target", type=float, help="Target dispersion (omit for default)")
    sp_weights.add_argument("--eps_floor", type=float, help="Minimum epsilon for NaN rows")
    sp_weights.add_argument("--eps_scale", type=float, help="Scaling factor for epsilon computation")
    sp_weights.add_argument("--w_min", type=float, help="Lower clip bound for weights")
    sp_weights.add_argument("--w_max", type=float, help="Upper clip bound for weights")
    sp_weights.add_argument("--max_clip_rate", type=float, help="Maximum tolerated clip rate before relaxation")
    sp_weights.add_argument("--min_neg_spearman", type=float, help="Expected upper bound on Spearman (negative)")
    sp_weights.add_argument("--target_cv_reduction", type=float, help="Target CV reduction for QC expectations")
    sp_weights.add_argument(
        "--stability_ratios",
        help="Comma-separated ratios (0-1) for stability downsampling checks (default: 0.9,0.7,0.5)",
    )
    sp_weights.add_argument(
        "--disable_stability",
        action="store_true",
        help="Disable stability downsampling QC",
    )
    sp_weights.add_argument(
        "--use_depth_heuristics",
        action="store_true",
        help="Enable depth-aware heuristics for lambda and sigma_target",
    )
    sp_weights.add_argument(
        "--depth_reference",
        type=float,
        help="Reference depth constant D0 for heuristic adjustments",
    )
    sp_weights.add_argument(
        "--include_decoys",
        action="store_true",
        help="Include decoy/unplaced chromosomes (default: skip)",
    )
    sp_weights.add_argument(
        "--nan_fraction_threshold",
        type=float,
        help="NaN fraction threshold that triggers epsilon inflation (default: 0.5)",
    )
    sp_weights.add_argument(
        "--high_nan_eps_multiplier",
        type=float,
        help="Multiplier applied to epsilon when NaN fraction exceeds threshold (default: 2.0)",
    )
    sp_weights.add_argument(
        "--rng_seed",
        type=int,
        help="Optional seed for stability downsampling random generator",
    )

    sp_plot = sub.add_parser("plot", help="Plot density of noise values from a bedgraph")
    sp_plot.add_argument("--noise_bed", required=True, help="Noise bedgraph path")
    sp_plot.add_argument("--out_png", required=True, help="Output PNG path")

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
        help="Quantile threshold for blacklisting when z-score cutoff is not provided (default: 0.95)",
    )

    args = parser.parse_args()

    if args.cmd == "sample-noise":
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
            noise_sampling.compute_sampled_noise(
                hic_path=args.hic,
                res=args.res,
                sample_fraction=args.sample_fraction,
                window_bp=args.window_bp,
                chrom_sizes_path=args.chrom_sizes,
                out_dir=args.out_dir,
                norm=args.norm,
                cooler_selection=args.cooler_path,
                min_mean_score=args.min_mean_score,
                min_nonzero_frac=args.min_nonzero_frac,
                subsample_ratios=[float(x) for x in args.subsample_ratios.split(",")] if args.subsample_ratios else None,
                seed=args.seed,
                profile_path=args.profile,
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
            )

        _profile_command("full-noise", run)
    elif args.cmd == "noise-to-weights":
        os.makedirs(args.out_dir, exist_ok=True)
        cfg = noise_to_weights.NoiseWeightConfig()
        if args.tau is not None:
            cfg.tau = float(args.tau)
        if args.kappa is not None:
            cfg.kappa = float(args.kappa)
        if args.gain is not None:
            cfg.gain = float(args.gain)
        if args.lambda_ is not None:
            cfg.shrink_lambda = float(args.lambda_)
        if args.sigma_target is not None:
            cfg.sigma_target = float(args.sigma_target)
        if args.eps_floor is not None:
            cfg.eps_floor = float(args.eps_floor)
        if args.eps_scale is not None:
            cfg.eps_scale = float(args.eps_scale)
        if args.w_min is not None:
            cfg.w_min = float(args.w_min)
        if args.w_max is not None:
            cfg.w_max = float(args.w_max)
        if args.max_clip_rate is not None:
            cfg.max_clip_rate = float(args.max_clip_rate)
        if args.min_neg_spearman is not None:
            cfg.min_neg_spearman = float(args.min_neg_spearman)
        if args.target_cv_reduction is not None:
            cfg.target_cv_reduction = float(args.target_cv_reduction)
        if args.stability_ratios:
            ratios = [float(x) for x in args.stability_ratios.split(",") if x]
            cfg.stability_ratios = tuple(ratios)
        if args.disable_stability:
            cfg.stability_ratios = tuple()
        cfg.use_depth_heuristics = bool(args.use_depth_heuristics)
        if args.depth_reference is not None:
            cfg.depth_reference = float(args.depth_reference)
        cfg.skip_decoys = not bool(args.include_decoys)
        if args.nan_fraction_threshold is not None:
            cfg.nan_fraction_threshold = float(args.nan_fraction_threshold)
        if args.high_nan_eps_multiplier is not None:
            cfg.high_nan_eps_multiplier = float(args.high_nan_eps_multiplier)
        if args.rng_seed is not None:
            cfg.random_seed = int(args.rng_seed)

        chroms = [chrom.strip() for chrom in args.chroms.split(",")] if args.chroms else None

        def run() -> None:
            noise_to_weights.build_weights_from_bedgraphs(
                scores_dir=args.scores_dir,
                res=int(args.res),
                out_dir=args.out_dir,
                config=cfg,
                coverage_path=args.coverage,
                chrom_sizes_path=args.chrom_sizes,
                chrom_allowlist=chroms,
            )

        _profile_command("noise-to-weights", run)
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
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
