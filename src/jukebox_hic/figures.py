#!/usr/bin/env python
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_noise_density_from_bed(noise_bed_path: str, out_png: str) -> None:
    """
    Render a density histogram of noise values from a bedgraph file.
    """
    df = pd.read_csv(
        noise_bed_path,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    values = df["noise"].astype(float).to_numpy()
    transformed = np.log10(np.clip(values, a_min=-0.999, a_max=None) + 1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(transformed, bins=20, density=True, alpha=0.6, color="#4C78A8")
    ax.set_xlabel("log10(noise + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Noise Distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def plot_two_noise_beds(
    noise_bed_a: str,
    noise_bed_b: str,
    labels: Tuple[str, str] = ("A", "B"),
    out_png: str = "noise_compare.png",
) -> None:
    """
    Compare two bedgraph files by plotting overlaid noise histograms.
    """
    df_a = pd.read_csv(
        noise_bed_a,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    df_b = pd.read_csv(
        noise_bed_b,
        sep=r"\s+",
        header=None,
        names=["chr", "start", "stop", "noise"],
        dtype={"chr": str},
    )
    vals_a = df_a["noise"].astype(float).to_numpy()
    vals_b = df_b["noise"].astype(float).to_numpy()
    trans_a = np.log10(np.clip(vals_a, a_min=-0.999, a_max=None) + 1.0)
    trans_b = np.log10(np.clip(vals_b, a_min=-0.999, a_max=None) + 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trans_a, bins=20, density=True, alpha=0.5, label=labels[0])
    ax.hist(trans_b, bins=20, density=True, alpha=0.5, label=labels[1])
    ax.legend()
    ax.set_xlabel("log10(noise + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Noise Distributions")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
