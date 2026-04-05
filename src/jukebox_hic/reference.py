#!/usr/bin/env python
"""Universal 4DN Reference Equation constants and adaptive bias vector logic."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Universal 4DN reference constants (Total Samples baseline)
BETA_0 = 1.1713
BETA_1 = -0.0584
BETA_2 = 1.9838
SIGMA_REF = 0.15  # Proxy for universal residual spread


def compute_zmap(
    contacts: float,
    chrom_len: int,
    resolution: int,
    median_noise: float,
    ebr: float,
) -> Dict[str, float]:
    """
    Calculate the Map Quality Z-score against the 4DN universal reference.

    Parameters
    ----------
    contacts    : total contact pairs for this chromosome
    chrom_len   : chromosome length in bp
    resolution  : bin size in bp
    median_noise: observed median noise value from the sampling phase
    ebr         : mean empty bin ratio from the sampling phase

    Returns
    -------
    dict with keys: z_map, gamma, pred_log_N, obs_log_N, residual

    Logs a warning if z_map > 2.0.
    """
    bins = chrom_len / resolution
    rho = float(contacts) / bins
    pred_log_N = BETA_0 + (BETA_1 * np.log10(rho)) + (BETA_2 * float(ebr))
    obs_log_N = float(np.log10(float(median_noise)))
    residual = obs_log_N - pred_log_N
    z_map = residual / SIGMA_REF
    gamma = 1.0 + max(0.0, float(z_map)) * 0.5

    if float(z_map) > 2.0:
        print(
            f"[WARN] z_map={z_map:.3f} > 2.0: high stochastic noise detected "
            f"(gamma={gamma:.3f})"
        )

    return {
        "z_map": float(z_map),
        "gamma": float(gamma),
        "pred_log_N": float(pred_log_N),
        "obs_log_N": float(obs_log_N),
        "residual": float(residual),
    }


def sequencing_advisor(
    obs_noise: float,
    current_rho: float,
    ebr: float,
    z_target: float = 0.0,
    fold_threshold: float = 10.0,
) -> Dict[str, object]:
    """
    Predict the sequencing density required to reach a target z-score and assess
    whether further sequencing is worthwhile.

    Inverts the 3-term reference model to solve for the density (contacts/bin)
    at which the predicted noise would equal z_target standard deviations from
    the reference.

    Parameters
    ----------
    obs_noise       : observed median noise (from sampling phase)
    current_rho     : current sequencing density in contacts/bin
                      (= total_contacts / (chrom_len / resolution))
    ebr             : mean empty bin ratio from the sampling phase
    z_target        : desired z-score target (default: 0.0 = reference level)
    fold_threshold  : fold-increase above which recommendation is "Stop"
                      (default: 10.0)

    Returns
    -------
    dict with keys:
        target_density     : contacts/bin needed to reach z_target
        fold_increase      : target_density / current_rho
        efficiency_index   : abs(BETA_1) — log-linear sensitivity of noise
                             to sequencing depth (constant)
        recommendation     : "Proceed" or "Stop"
    """
    nan_result: Dict[str, object] = {
        "target_density": float("nan"),
        "fold_increase": float("nan"),
        "efficiency_index": float("nan"),
        "recommendation": "Insufficient data",
    }

    if (
        not np.isfinite(obs_noise)
        or obs_noise <= 0.0
        or not np.isfinite(current_rho)
        or current_rho <= 0.0
        or not np.isfinite(ebr)
    ):
        return nan_result

    try:
        log_target_rho = (
            np.log10(obs_noise) - BETA_0 - BETA_2 * float(ebr) - z_target * SIGMA_REF
        ) / BETA_1
        target_rho = float(10.0 ** log_target_rho)
        fold_increase = target_rho / float(current_rho)
        recommendation = "Stop" if fold_increase > float(fold_threshold) else "Proceed"
    except Exception:
        return nan_result

    return {
        "target_density": target_rho,
        "fold_increase": fold_increase,
        "efficiency_index": abs(BETA_1),
        "recommendation": recommendation,
    }


def process_normalization_vectors(
    noise_track: np.ndarray,
    gamma: float,
    ebr: float,
) -> np.ndarray:
    """
    Transform a raw noise track into a normalization bias vector.

    Steps:
      1. Identify NaN/Inf bins.
      2. Fill masked bins with nanmedian; apply Gaussian smoothing (sigma = 1 + EBR).
      3. Log10-transform and median-centre on valid bins only.
      4. Scale: B_i = 10^(centred_i * gamma).
      5. Re-apply NaN mask to originally masked indices.

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    gamma       : adaptive penalty derived from compute_zmap()
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan = np.isnan(noise_track) | np.isinf(noise_track)

    clean_track = np.copy(noise_track).astype(float)
    median_val = float(np.nanmedian(noise_track))
    if np.isnan(median_val):
        median_val = 0.0
    clean_track[is_nan] = median_val

    sigma = 1.0 + float(ebr)
    smoothed = gaussian_filter1d(clean_track, sigma=sigma)

    log_N = np.log10(np.clip(smoothed, 1e-12, None))
    valid = ~is_nan
    centre = float(np.median(log_N[valid])) if valid.any() else 0.0
    centred = log_N - centre

    bias_vector = np.power(10.0, centred * float(gamma))
    bias_vector[is_nan] = np.nan

    return bias_vector


def generate_blacklist(
    noise_track: np.ndarray,
    chrom: str,
    resolution: int,
) -> List[List]:
    """
    Build BED rows flagging the noisiest 1% of bins and all NaN/Inf bins.

    Parameters
    ----------
    noise_track : per-bin raw noise values
    chrom       : chromosome name
    resolution  : bin size in bp

    Returns
    -------
    List of [chrom, start, end] rows in BED coordinate format.
    """
    finite_vals = noise_track[~np.isnan(noise_track) & ~np.isinf(noise_track)]
    if finite_vals.size > 0:
        p99 = float(np.percentile(finite_vals, 99))
        flagged = np.where(
            (noise_track >= p99) | np.isnan(noise_track) | np.isinf(noise_track)
        )[0]
    else:
        flagged = np.where(np.isnan(noise_track) | np.isinf(noise_track))[0]

    bed_rows: List[List] = []
    for idx in flagged:
        start = int(idx) * resolution
        end = (int(idx) + 1) * resolution
        bed_rows.append([chrom, start, end])

    return bed_rows
