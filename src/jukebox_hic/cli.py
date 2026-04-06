#!/usr/bin/env python
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
    return list(map(int, value.split(",")))


def _comma_strs(value: str) -> List[str]:
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
    parser.epilog = (
        'Note: Base install includes "cooler" only. For .hic support, install extras: '
        'pip install "jukebox-hic[straw]" or "jukebox-hic[all]".'
    )

    # ------------------------------------------------------------------ #
    # sample-noise                                                         #
    # ------------------------------------------------------------------ #
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
        help="Internal group path for .cool/.mcool/.scool URIs",
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
    sp_sample.add_argument(
        "--chroms",
        help="Comma-separated list of chromosomes to process (default: all)",
    )

    # ------------------------------------------------------------------ #
    # full-noise                                                           #
    # ------------------------------------------------------------------ #
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
        help="Damping factor for adaptive mode (default: 0.7; valid range 0.5–0.8)",
    )

    # ------------------------------------------------------------------ #
    # plot                                                                 #
    # ------------------------------------------------------------------ #
    sp_plot = sub.add_parser("plot", help="Plot density of noise values from a bedgraph")
    sp_plot.add_argument("--noise_bed", required=True, help="Noise bedgraph path")
    sp_plot.add_argument("--out_png", required=True, help="Output PNG path")

    # ------------------------------------------------------------------ #
    # blacklist                                                            #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # sequencing-advisor                                                   #
    # ------------------------------------------------------------------ #
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
        help="Target z-score to reach (default: 0.0 = 4DN reference level)",
    )
    sp_adv.add_argument(
        "--fold_threshold",
        type=float,
        default=10.0,
        help='Fold-increase above which recommendation is "Stop" (default: 10.0)',
    )

    # ------------------------------------------------------------------ #
    # default-run                                                          #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
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
                chroms=_comma_strs(args.chroms) if args.chroms else None,
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

        # Parse header comments: # resolution=N and # chrom_info lines
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
                    current_rho = est_contacts / (c_len / summary_res)
                    adv = reference.sequencing_advisor(
                        obs_noise=median_noise,
                        current_rho=current_rho,
                        ebr=ebr,
                        z_target=args.z_target,
                        fold_threshold=args.fold_threshold,
                    )
                else:
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
        resolutions = _comma_ints(args.res)
        cooler_sel = getattr(args, "cooler_path", None)
        nan_fill = float("nan") if args.nan_fill == "nan" else 0.0
        chroms = _comma_strs(args.chroms) if args.chroms else None

        sample_root = os.path.join(args.out_dir, "sample")
        full_dir = os.path.join(args.out_dir, "full")
        vectors_dir = os.path.join(args.out_dir, "vectors")
        os.makedirs(sample_root, exist_ok=True)
        os.makedirs(full_dir, exist_ok=True)
        os.makedirs(vectors_dir, exist_ok=True)

        # Dump contacts once (resolution-independent)
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
                norm="none",
                chrom_sizes_path=args.chrom_sizes,
                cooler_selection=cooler_sel,
                chroms=chroms,
            )

        # Phase 2: Full noise for all resolutions at once
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

        # Phase 3 & 4: Per resolution
        for res in resolutions:
            res_sample_dir = os.path.join(sample_root, str(res))
            zmap_summary = os.path.join(res_sample_dir, "subsample_summary.tsv")

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

            print(f"\n[Phase 4] Building blacklist (P99 + NaN) at {res} bp ...")
            full_noise_bed = os.path.join(full_dir, f"{res}.bedgraph")
            if os.path.isfile(full_noise_bed):
                out_blacklist = os.path.join(
                    args.out_dir, f"{args.sample_name}_{res}_noise_blacklist.bed"
                )
                filters.build_blacklist_from_bedgraphs(
                    inputs=[full_noise_bed],
                    output_path=out_blacklist,
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
