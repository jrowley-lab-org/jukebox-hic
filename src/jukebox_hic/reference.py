#!/usr/bin/env python
"""
Universal 4DN Reference Equation constants and adaptive bias vector logic.

This module implements the statistical reference model used to assess Hi-C data
quality relative to the 4D Nucleome (4DN) consortium's benchmarking standards.

Background
----------
The 4DN consortium established empirical relationships between sequencing depth,
genomic bin size, and expected Hi-C noise levels by profiling many Hi-C experiments.
The key insight is that noise (measured as the lag-1 autocovariance metric from
``noise_sampling``) is predictable from:
- **Sequencing density** (ρ = total contacts / number of bins): more reads → less noise.
- **Empty bin ratio** (EBR): the fraction of diagonal-adjacent bins with zero contacts.
  Higher EBR → sparser data → higher noise.

The reference model is a linear equation in log space:
    log10(N_predicted) = BETA_0 + BETA_1 * log10(ρ) + BETA_2 * EBR

where N is the noise value. The residual between observed and predicted log-noise
is the "z-map" score — a measure of how many standard deviations a sample's noise
deviates from the 4DN reference.

Module contents
---------------
- ``compute_zmap()``                        — compute z-map quality score
- ``sequencing_advisor()``                  — predict depth needed to reach target quality
- ``_preprocess_noise_track()``             — shared noise preprocessing (smooth, log)
- ``process_normalization_vectors_baseline()`` — JUKEBOX-BASELINE bias vector
- ``process_normalization_vectors_adaptive()`` — JUKEBOX-ADAPTIVE bias vector
- ``process_normalization_vectors()``       — legacy gamma-based bias vector (deprecated)
- ``generate_blacklist()``                  — identify noisy/invalid bins
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# 4DN Universal Reference Model Constants
# ---------------------------------------------------------------------------
# These constants parameterise the 3-term linear model that predicts log10(noise)
# from sequencing density (log10(ρ)) and empty bin ratio (EBR):
#
#   log10(N_pred) = BETA_0 + BETA_1 * log10(ρ) + BETA_2 * EBR
#
# BETA_0 : float — model intercept (log10-noise when ρ=1 and EBR=0)
# BETA_1 : float — coefficient for log10(ρ). Negative because deeper sequencing
#                  reduces noise (more contacts → smoother distance-decay patterns).
# BETA_2 : float — coefficient for EBR. Positive because sparser data (higher EBR)
#                  is noisier.
# SIGMA_REF : float — standard deviation of residuals in the 4DN reference dataset.
#                     Used to convert residuals into z-scores (z = residual / SIGMA_REF).
#                     A value of 0.15 corresponds to ~15% variation in log10-noise
#                     around the predicted value across the 4DN training samples.
#
# Source: These constants were calibrated against the 4DN Total Samples benchmark.
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

    The z-map score answers: "Given this chromosome's sequencing density and empty bin
    ratio, how noisy is it relative to what the 4DN reference model predicts?"

    A z-map near 0.0 means the sample matches the reference quality. Positive z-map
    values indicate higher-than-expected noise (worse quality); negative values indicate
    lower-than-expected noise (better than reference).

    The gamma value is a multiplicative penalty that scales up for samples with
    high z-map scores (noisy data gets stronger correction in the bias vector step):
        gamma = 1.0 + max(0.0, z_map) * 0.5

    For example:
    - z_map = 0.0 → gamma = 1.0  (no penalty; data matches reference)
    - z_map = 2.0 → gamma = 2.0  (double-strength correction for noisy data)
    - z_map = -1.0 → gamma = 1.0 (no penalty even for better-than-reference data)

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
    # ρ (rho) = contacts per bin = sequencing density for this chromosome
    rho = float(contacts) / bins
    # Predicted log10(noise) from the 4DN linear reference model
    pred_log_N = BETA_0 + (BETA_1 * np.log10(rho)) + (BETA_2 * float(ebr))
    obs_log_N = float(np.log10(float(median_noise)))
    residual = obs_log_N - pred_log_N
    # z-score: how many reference standard deviations above the predicted noise level
    z_map = residual / SIGMA_REF
    # gamma: adaptive penalty factor for bias vector scaling (clamped at 1.0 minimum)
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

    The inversion works as follows. The reference model predicts:
        log10(N_pred) = BETA_0 + BETA_1 * log10(ρ) + BETA_2 * EBR

    Setting observed noise equal to the noise level at z_target:
        log10(N_obs) = pred_log_N + z_target * SIGMA_REF
                     = BETA_0 + BETA_1 * log10(ρ_target) + BETA_2 * EBR + z_target * SIGMA_REF

    Solving for log10(ρ_target):
        log10(ρ_target) = (log10(N_obs) - BETA_0 - BETA_2*EBR - z_target*SIGMA_REF) / BETA_1

    The fold increase = ρ_target / ρ_current tells how many times more sequencing is needed.
    If fold_increase > fold_threshold (default 10x), the recommendation is "Stop"
    because the required increase is impractically large.

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
                             to sequencing depth (constant; reported for reference)
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
        # Invert the reference model to find the required sequencing density
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


def _preprocess_noise_track(
    noise_track: np.ndarray,
    ebr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared pre-processing for all normalization modes.

    All three normalization functions (baseline, adaptive, legacy) begin with the same
    four-step pipeline:

    1. **Identify invalid bins**: NaN values arise from bins that were too sparse for
       a noise estimate (skipped by ``_compute_row_metrics``); Inf values arise from
       edge cases in the 1/autocovariance formula.

    2. **Fill masked bins with nanmedian**: Bins with NaN/Inf noise are replaced with
       the chromosome's median noise value. This fill is needed before Gaussian smoothing
       (which cannot propagate NaN) and prevents edge bins from pulling the smoothed
       curve to zero. The original NaN positions are restored after bias computation.

    3. **Apply Gaussian smoothing** (sigma = 1 + EBR): Smoothing suppresses bin-level
       noise estimation noise (the noise metric itself has some variance). The sigma is
       proportional to the empty bin ratio because sparse data has noisier estimates
       that benefit from wider smoothing.

    4. **Log10-transform** with a floor at 1e-12: The reference model and bias formulas
       operate in log space. The floor prevents log10(0) = -Inf for bins that were
       filled with median_val near zero.

    Parameters
    ----------
    noise_track : np.ndarray
        Per-bin raw noise values from the full-noise bedgraph (may contain NaN/Inf).
    ebr : float
        Mean empty bin ratio for this chromosome, from the subsample_summary.tsv.

    Returns
    -------
    is_nan : np.ndarray of bool
        Boolean mask: True for bins that were originally NaN or Inf.
        Used to restore NaN at these positions in the final bias vector.
    log_N : np.ndarray of float
        log10 of the Gaussian-smoothed noise values (all bins, including formerly
        masked bins that were filled with median before smoothing).
    smoothed : np.ndarray of float
        The Gaussian-smoothed noise values before the log10 transform. Returned for
        potential diagnostic use (currently not used by callers).
    """
    is_nan = np.isnan(noise_track) | np.isinf(noise_track)

    # Fill invalid bins with the chromosome median so smoothing can proceed
    clean_track = np.copy(noise_track).astype(float)
    median_val = float(np.nanmedian(noise_track))
    if np.isnan(median_val):
        median_val = 0.0
    clean_track[is_nan] = median_val

    # Gaussian smoothing: sigma increases with EBR to account for sparser data
    sigma = 1.0 + float(ebr)
    smoothed = gaussian_filter1d(clean_track, sigma=sigma)
    log_N = np.log10(np.clip(smoothed, 1e-12, None))

    return is_nan, log_N, smoothed


def process_normalization_vectors_baseline(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
) -> np.ndarray:
    """
    JUKEBOX-BASELINE normalization.

    Forces every bin to conform to the 4DN universal baseline by computing
    the exact distance between each bin's noise and the predicted log-noise
    for this chromosome (L̂_target).

    Math
    ----
    log10(B_i) = 0.5 * (log10(N_i) - pred_log_N)
    B_i        = 10 ^ log10(B_i)

    Interpretation
    --------------
    - If a bin's noise equals the predicted value (log10(N_i) == pred_log_N),
      then log10(B_i) = 0 and B_i = 1.0 — no correction applied.
    - If a bin is noisier than predicted (log10(N_i) > pred_log_N),
      then B_i > 1.0 — this bin's contacts will be down-weighted in the matrix.
    - If a bin is quieter than predicted (log10(N_i) < pred_log_N),
      then B_i < 1.0 — slight up-weighting.

    The factor of 0.5 provides a conservative half-strength correction, preventing
    over-correction of bins that may have slightly elevated noise by chance.

    This mode is appropriate when data quality is close to the 4DN reference and
    you want a deterministic, model-anchored correction.

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    pred_log_N  : predicted log10 noise for this chromosome from compute_zmap()
                  (= BETA_0 + BETA_1*log10(rho) + BETA_2*ebr)
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    log_bias = 0.5 * (log_N - float(pred_log_N))
    bias_vector = np.power(10.0, log_bias)
    # Restore NaN for originally invalid bins (balancing engines treat NaN as "ignore bin")
    bias_vector[is_nan] = np.nan

    return bias_vector


def process_normalization_vectors_adaptive(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    JUKEBOX-ADAPTIVE normalization.

    Better suited for sparse or palaeogenomic data. Uses the global Z-score
    to determine the intensity of the correction while dampening local bin
    variance to prevent blow-outs in stochastically noisy regions.

    Math
    ----
    z_map  = (median(log10(N)) - pred_log_N) / SIGMA_REF
    D_i    = log10(N_i) - median(log10(N))          # local deviation from chromosome median
    log10(B_i) = 0.5 * (z_map * SIGMA_REF + alpha * D_i)
    B_i        = 10 ^ log10(B_i)

    Interpretation
    --------------
    The bias has two components:
    1. **Global shift** (``0.5 * z_map * SIGMA_REF``): moves all bins up or down
       proportionally to how far the chromosome's overall noise is from the reference.
       This is the same correction the baseline mode applies but applied uniformly
       rather than per-bin.
    2. **Local deviation** (``0.5 * alpha * D_i``): adjusts each bin relative to the
       chromosome median, dampened by ``alpha`` (default 0.7). This preserves the
       spatial noise pattern while preventing extreme bins from being over-corrected.

    The ``alpha`` parameter (valid range 0.5–0.8) controls how much local variation
    is retained:
    - alpha = 0.5 → 50% of local deviation preserved; aggressive smoothing
    - alpha = 0.8 → 80% of local deviation preserved; conservative smoothing

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    pred_log_N  : predicted log10 noise for this chromosome from compute_zmap()
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)
    alpha       : damping factor for local variance (default 0.7; range 0.5–0.8)

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    # Chromosome-level median of log-noise (for valid bins only)
    valid = ~is_nan
    median_obs = float(np.median(log_N[valid])) if valid.any() else float(pred_log_N)

    # Global z-map: how far is this chromosome's median noise from the 4DN reference?
    z_map = (median_obs - float(pred_log_N)) / SIGMA_REF
    # Local deviation: how far is each bin from the chromosome's own median?
    D_i = log_N - median_obs

    log_bias = 0.5 * (z_map * SIGMA_REF + float(alpha) * D_i)
    bias_vector = np.power(10.0, log_bias)
    bias_vector[is_nan] = np.nan

    return bias_vector


def process_normalization_vectors(
    noise_track: np.ndarray,
    gamma: float,
    ebr: float,
) -> np.ndarray:
    """
    Transform a raw noise track into a normalization bias vector.

    .. deprecated::
        This is the original gamma-based normalization that was superseded by
        ``process_normalization_vectors_baseline()`` and
        ``process_normalization_vectors_adaptive()``. It is retained for
        backwards compatibility but should not be used in new code.

        Key differences from the current modes:
        - This function uses a gamma penalty from ``compute_zmap()`` (which scales
          with z_map severity) to amplify the local correction for noisier samples.
        - The baseline and adaptive modes instead anchor the correction to the 4DN
          reference's predicted log-noise, making the bias vector independent of
          the absolute noise level and easier to interpret.
        - This gamma-based approach can over-correct very noisy samples (large gamma
          amplifies local variance, potentially producing extreme bias values).

    Steps:
      1. Identify NaN/Inf bins.
      2. Fill masked bins with nanmedian; apply Gaussian smoothing (sigma = 1 + EBR).
      3. Log10-transform and median-centre on valid bins only.
      4. Scale: B_i = 10^(centred_i * gamma).
      5. Re-apply NaN mask to originally masked indices.

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    gamma       : adaptive penalty derived from compute_zmap(); higher gamma = stronger
                  correction applied to the median-centred log-noise values.
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    valid = ~is_nan
    # Median-centre the log-noise so that bins at the chromosome median get B_i = 1.0
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

    Blacklisted bins should be excluded from downstream analyses (loop calling,
    compartment detection, etc.) because their noise estimates indicate they are
    unreliable — typically due to repeat elements, mappability issues, or very low
    coverage.

    Two categories of bins are flagged:
    1. **NaN/Inf bins**: no reliable noise estimate could be computed (too sparse).
    2. **Top 1% noise bins**: bins with noise values at or above the 99th percentile
       of the finite noise distribution on this chromosome.

    Parameters
    ----------
    noise_track : per-bin raw noise values
    chrom       : chromosome name
    resolution  : bin size in bp

    Returns
    -------
    List of [chrom, start, end] rows in BED coordinate format (0-based, half-open).
    """
    finite_vals = noise_track[~np.isnan(noise_track) & ~np.isinf(noise_track)]
    if finite_vals.size > 0:
        # 99th percentile threshold: flag the most extreme 1% of bins
        p99 = float(np.percentile(finite_vals, 99))
        flagged = np.where(
            (noise_track >= p99) | np.isnan(noise_track) | np.isinf(noise_track)
        )[0]
    else:
        # All bins are NaN/Inf — flag everything
        flagged = np.where(np.isnan(noise_track) | np.isinf(noise_track))[0]

    bed_rows: List[List] = []
    for idx in flagged:
        start = int(idx) * resolution
        end = (int(idx) + 1) * resolution
        bed_rows.append([chrom, start, end])

    return bed_rows
