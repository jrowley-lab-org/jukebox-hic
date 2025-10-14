#!/usr/bin/env python
import argparse
import os
from typing import List

from . import figures, noise_fullmap, noise_sampling


def _comma_ints(value: str) -> List[int]:
    return list(map(int, value.split(",")))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jukebox-hic",
        description="Lightweight Hi-C noise analysis toolkit (hicstraw only)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_sample = sub.add_parser("sample-noise", help="Compute noise on a sampled set of rows")
    sp_sample.add_argument("--hic", required=True, help="Input .hic file path")
    sp_sample.add_argument("--res", required=True, type=int, help="Resolution in bp")
    sp_sample.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of rows to evaluate (0-1)")
    sp_sample.add_argument("--window_bp", type=int, default=None, help="Half-window size around diagonal in bp")
    sp_sample.add_argument("--chrom_sizes", help="Chrom sizes TSV (chr\\tsize); default: read from .hic")
    sp_sample.add_argument("--out_dir", required=True, help="Directory for outputs")
    sp_sample.add_argument(
        "--subsample_ratios",
        help="Comma-separated ratios (<=1.0) to simulate (e.g. 1.0,0.5,0.25); defaults to 1.0 only",
    )
    sp_sample.add_argument("--seed", type=int, help="Random seed for subsampling reproducibility")

    sp_full = sub.add_parser("full-noise", help="Compute noise genome-wide using hicstraw expected vectors")
    sp_full.add_argument("--hic", required=True, help="Input .hic file path")
    sp_full.add_argument("--res", required=True, help="Comma-separated resolutions in bp")
    sp_full.add_argument("--chrom_sizes", help="Chrom sizes TSV (chr\\tsize); default: read from .hic")
    sp_full.add_argument("--norm", default="NONE", help="Normalization label to request from hicstraw")
    sp_full.add_argument("--bindist_bp", type=int, help="Half-window distance from diagonal in bp")
    sp_full.add_argument("--out_dir", required=True, help="Directory for outputs")

    sp_plot = sub.add_parser("plot", help="Plot density of noise values from a bedgraph")
    sp_plot.add_argument("--noise_bed", required=True, help="Noise bedgraph path")
    sp_plot.add_argument("--out_png", required=True, help="Output PNG path")

    args = parser.parse_args()

    if args.cmd == "sample-noise":
        os.makedirs(args.out_dir, exist_ok=True)
        noise_sampling.compute_sampled_noise(
            hic_path=args.hic,
            res=args.res,
            sample_fraction=args.sample_fraction,
            window_bp=args.window_bp,
            chrom_sizes_path=args.chrom_sizes,
            out_dir=args.out_dir,
            subsample_ratios=[float(x) for x in args.subsample_ratios.split(",")] if args.subsample_ratios else None,
            seed=args.seed,
        )
    elif args.cmd == "full-noise":
        os.makedirs(args.out_dir, exist_ok=True)
        noise_fullmap.compute_full_noise(
            hic_path=args.hic,
            res_list=_comma_ints(args.res),
            chrom_sizes_path=args.chrom_sizes,
            norm=args.norm,
            bindist_bp=args.bindist_bp,
            out_dir=args.out_dir,
        )
    elif args.cmd == "plot":
        figures.plot_noise_density_from_bed(args.noise_bed, args.out_png)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
