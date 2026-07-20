#!/usr/bin/env python
"""
plot_coverage_elbow.py — per-chromosome dual-metric elbow detection for blacklist cutoffs.

Reads the two bedgraph outputs of the ``noise-bedgraph`` command for a single resolution:
  - ``{res}_density.bedgraph``  (per-bin raw contact row sums)
  - ``{res}.bedgraph``          (per-bin noise values, lag-1 autocovariance based)

Outputs:
  - A TSV with per-chromosome threshold values and their percentile positions.
  - A PNG + PDF diagnostic 2×2 figure (overlay curves + per-chromosome dot-plots).

Detection logic and plotting are implemented in the ``jukebox_hic`` package
(``filters.detect_elbow_thresholds`` / ``figures.plot_elbow_figure``).

Usage
-----
python scripts/plot_coverage_elbow.py \\
    --density_bedgraph  full/10000_density.bedgraph \\
    --noise_bedgraph    full/10000.bedgraph \\
    --out_png           elbow.png \\
    --out_tsv           elbow_thresholds.tsv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

_repo_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_repo_src):
    sys.path.insert(0, os.path.abspath(_repo_src))

from jukebox_hic.filters import detect_elbow_thresholds
from jukebox_hic.figures import plot_elbow_figure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-chromosome dual-metric elbow detection for Hi-C blacklist cutoffs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--density_bedgraph", required=True,
                   help="Path to {res}_density.bedgraph from noise-bedgraph")
    p.add_argument("--noise_bedgraph", required=True,
                   help="Path to {res}.bedgraph from noise-bedgraph")
    p.add_argument("--out_png", default="coverage_elbow.png",
                   help="Output figure PNG (a matching .pdf is always written alongside it)")
    p.add_argument("--out_tsv", default="elbow_thresholds.tsv",
                   help="Output per-chromosome thresholds TSV")
    p.add_argument("--smooth_sigma", type=float, default=10.0,
                   help="Gaussian smoothing sigma applied to both metrics")
    p.add_argument("--density_upper_frac", type=float, default=0.01,
                   help="Top fraction of density bins searched for the upper elbow "
                        "(default: 0.01 = top 1%%; increase if the elbow is not detected)")
    p.add_argument("--density_lower_frac", type=float, default=0.01,
                   help="Bottom fraction of density bins searched for the lower elbow "
                        "(default: 0.01 = bottom 1%%)")
    p.add_argument("--noise_upper_frac", type=float, default=0.10,
                   help="Top fraction of noise bins searched for the upper elbow "
                        "(default: 0.10 = top 10%%; increase if elbow is not detected)")
    p.add_argument("--noise_lower_frac", type=float, default=0.01,
                   help="Bottom fraction of noise bins searched for the lower elbow "
                        "(default: 0.01 = bottom 1%%)")
    p.add_argument("--density_transform", default="none",
                   choices=["none", "sqrt", "cbrt", "log1p"],
                   help="Pre-transform applied to density values (default: none)")
    p.add_argument("--noise_transform", default="sqrt",
                   choices=["none", "sqrt", "cbrt", "log1p"],
                   help="Pre-transform applied to noise values (default: sqrt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Running elbow detection ...")
    print(f"  density: {args.density_bedgraph}")
    print(f"  noise:   {args.noise_bedgraph}")

    thresholds_df, per_chrom_curves = detect_elbow_thresholds(
        density_bedgraph=args.density_bedgraph,
        noise_bedgraph=args.noise_bedgraph,
        smooth_sigma=args.smooth_sigma,
        density_upper_frac=args.density_upper_frac,
        density_lower_frac=args.density_lower_frac,
        noise_upper_frac=args.noise_upper_frac,
        noise_lower_frac=args.noise_lower_frac,
        density_transform=args.density_transform,
        noise_transform=args.noise_transform,
    )

    if thresholds_df.empty:
        sys.exit("Elbow detection produced no results.")

    # Print per-chromosome summary
    print(f"\nProcessed {len(thresholds_df)} chromosome(s):")
    for _, row in thresholds_df.iterrows():
        print(
            f"  {row['chrom']:8s}  "
            f"density [{row['density_lower_value']:.0f} – {row['density_upper_value']:.0f}]  "
            f"({row['density_lower_pct']:.1f}th–{row['density_upper_pct']:.1f}th pct)  |  "
            f"noise [{row['noise_lower_value']:.4f} – {row['noise_upper_value']:.4f}]  "
            f"({row['noise_lower_pct']:.1f}th–{row['noise_upper_pct']:.1f}th pct)"
        )

    # Write TSV (public columns only)
    thresholds_df.to_csv(args.out_tsv, sep="\t", index=False, float_format="%.6g")
    print(f"\nThresholds TSV saved to {args.out_tsv}")

    # Write PNG + PDF
    out_prefix = args.out_png[:-4] if args.out_png.lower().endswith(".png") else args.out_png
    plot_elbow_figure(
        per_chrom_curves=per_chrom_curves,
        results=thresholds_df.to_dict("records"),
        out_prefix=out_prefix,
        density_transform=args.density_transform,
        noise_transform=args.noise_transform,
    )
    print(f"Figure saved to {out_prefix}.png  /  {out_prefix}.pdf")


if __name__ == "__main__":
    main()
