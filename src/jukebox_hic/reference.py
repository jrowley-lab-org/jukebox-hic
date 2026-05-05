#!/usr/bin/env python
"""
JUKEBOX Hybrid Spline Noise Model — reference constants and model functions.

This module implements the statistical reference model used to assess Hi-C data
quality relative to the 4D Nucleome (4DN) consortium's benchmarking standards.

Background
----------
The 4DN consortium established empirical relationships between sequencing depth,
genomic bin size, and expected Hi-C noise levels by profiling many Hi-C experiments.
The key insight is that noise (measured as the lag-1 autocovariance metric from
``noise_sampling``) is predictable from:
- **Effective Matrix Coverage** (ρ_win): contacts per bin-pair within the analysis
  window, incorporating both sequencing depth and window size into one variable.
- **Empty bin ratio** (EBR): the fraction of diagonal-adjacent bins with zero contacts.
  Higher EBR → sparser data → higher noise.
- **Depth-sparsity interaction** (ρ_win × EBR): captures how sparsity effects scale
  with coverage.

The reference model is a cubic B-spline in log(ρ_win)-space plus an EBR term and
an interaction term:

    log10(N_pred) = β0 + Σ(j=0..7) βj·sj(Lρ) + β_EBR·EBR + β_inter·(Lρ·EBR)

where Lρ = log10(ρ_win) and sj are B-spline basis functions (degree=3, 5 knots).
Three coefficient sets are hardcoded: OLS mean (central baseline), q=0.75 upper
envelope, and q=0.25 lower envelope — used for quality gating.

Module contents
---------------
- ``compute_noise_model()``                 — compute spline-based quality metrics
- ``compute_zmap()``                        — deprecated wrapper around compute_noise_model
- ``sequencing_advisor()``                  — predict depth needed to reach target quality
- ``_build_design_matrix()``               — build 11-feature design vector for one sample
- ``_preprocess_noise_track()``             — shared noise preprocessing (smooth, log)
- ``process_normalization_vectors_adaptive()`` — JUKEBOX-ADAPTIVE bias vector
- ``process_normalization_vectors()``       — legacy gamma-based bias vector (deprecated)
- ``generate_blacklist()``                  — identify noisy/invalid bins
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# JUKEBOX Hybrid Spline Model Constants
# ---------------------------------------------------------------------------
# Cubic B-spline (degree=3) with 5 interior knots over log10(ρ_win) space.
# Augmented knot vector: [L_MIN]*4 + interior_knots + [L_MAX]*4 → 13 knots,
# 9 basis functions total. The first basis is dropped per patsy bs() convention,
# leaving 8 functions (s0…s7) in the design matrix.
#
# Three coefficient arrays of length 11 each:
#   [intercept, s0, s1, s2, s3, s4, s5, s6, s7, EBR, Lρ·EBR]
#
# Source: calibrated against the 4DN Gold Standard database via OLS (mean) and
# quantile regression (upper=q=0.75, lower=q=0.25).
_INTERIOR_KNOTS: np.ndarray = np.array([-0.705123, 0.259331, 1.042659, 1.782153, 2.886834])
_L_MIN: float = -4.051144
_L_MAX: float = 5.374635
_DEGREE: int = 3

_BETA_MEAN: np.ndarray = np.array([
    -8.908941,   # intercept
     5.209811,   # s0
     5.866539,   # s1
     9.138779,   # s2
    10.189710,   # s3
     9.792181,   # s4
    10.133563,   # s5
    10.204176,   # s6
    10.199352,   # s7
     3.057336,   # EBR
    -2.047100,   # Lρ·EBR (interaction)
])

_BETA_UPPER: np.ndarray = np.array([
    -10.700096,  # intercept
      6.684422,  # s0
      6.519458,  # s1
     10.852507,  # s2
     12.107708,  # s3
     11.714808,  # s4
     12.167096,  # s5
     12.170927,  # s6
     12.210863,  # s7
      3.565983,  # EBR
     -2.338288,  # Lρ·EBR (interaction)
])

_BETA_LOWER: np.ndarray = np.array([
    -9.544795,   # intercept
     7.090679,   # s0
     7.139118,   # s1
     9.895400,   # s2
    10.650998,   # s3
    10.211015,   # s4
    10.424150,   # s5
    10.656722,   # s6
    10.592492,   # s7
     2.558768,   # EBR
    -1.847717,   # Lρ·EBR (interaction)
])

# Pre-computed augmented knot vector and basis count (module-level, computed once)
_KNOTS_AUG: np.ndarray = np.concatenate([
    np.full(_DEGREE + 1, _L_MIN),
    _INTERIOR_KNOTS,
    np.full(_DEGREE + 1, _L_MAX),
])
_N_BASIS_TOTAL: int = len(_KNOTS_AUG) - _DEGREE - 1  # = 9


def _build_design_matrix(L_rho: float, ebr: float) -> np.ndarray:
    """
    Build the 11-element design vector for one (L_rho, EBR) observation.

    Design vector layout:
        [intercept, s0, s1, s2, s3, s4, s5, s6, s7, EBR, L_rho·EBR]

    The 8 spline basis values (s0…s7) are derived from a degree-3 B-spline with
    5 interior knots over [L_MIN, L_MAX]. The augmented knot vector produces 9
    basis functions; the first is dropped per the patsy ``bs(include_intercept=False)``
    convention, leaving 8 that are included in the design.

    L_rho is clipped to [L_MIN, L_MAX] before evaluation so extrapolation is
    handled by clamping rather than by scipy's NaN-extrapolation default.

    Parameters
    ----------
    L_rho : float
        log10(ρ_win) for this chromosome sample.
    ebr : float
        Mean empty bin ratio.

    Returns
    -------
    np.ndarray of shape (11,)
    """
    L = float(np.clip(L_rho, _L_MIN, _L_MAX))
    identity = np.eye(_N_BASIS_TOTAL)
    raw = np.array(
        [BSpline(_KNOTS_AUG, identity[i], _DEGREE)(L) for i in range(_N_BASIS_TOTAL)],
        dtype=float,
    )
    raw = np.nan_to_num(raw, nan=0.0)
    s = raw[1:]  # drop first basis → s0…s7 (8 values)
    return np.array([1.0, *s, float(ebr), L * float(ebr)])


def compute_noise_model(
    contacts: float,
    chrom_len: int,
    resolution: int,
    window_bp: int,
    median_noise: float,
    ebr: float,
) -> Dict[str, object]:
    """
    Evaluate the JUKEBOX hybrid spline noise model for one chromosome sample.

    Computes predicted noise (central, upper envelope, lower envelope), quality
    gating status, and the envelope-normalised z-map score.

    The Effective Matrix Coverage:
        ρ_win = contacts / ((chrom_len/res) × (window_bp/res))

    incorporates both sequencing depth and window size so that the same contacts
    at different resolutions produce the same ρ_win value.

    Quality gating uses the quantile-regression envelopes:
        y_up = max(X·β_upper, X·β_lower)   [ceiling — high noise boundary]
        y_lo = min(X·β_upper, X·β_lower)   [floor   — masking boundary]
        Pass        → y_lo ≤ log10(N_obs) ≤ y_up
        Fail_High   → log10(N_obs) > y_up
        Fail_Masked → log10(N_obs) < y_lo

    The z-map is envelope-normalised:
        z_map = ε / half_envelope   where ε = log10(N_obs) − X·β_mean
        gamma = 1.0 + max(0, z_map) × 0.5

    Parameters
    ----------
    contacts    : total contact pairs for this chromosome at this resolution
    chrom_len   : chromosome length in bp
    resolution  : bin size in bp
    window_bp   : half-window size used during noise sampling (in bp)
    median_noise: observed median noise value from the sampling phase
    ebr         : mean empty bin ratio from the sampling phase

    Returns
    -------
    dict with keys:
        pred_log_N       — X·β_mean (central prediction)
        pred_log_N_upper — y_up (upper envelope boundary)
        pred_log_N_lower — y_lo (lower envelope boundary)
        epsilon          — log10(N_obs) − pred_log_N (raw residual)
        z_map            — ε / half_envelope (envelope-normalised score)
        gamma            — 1.0 + max(0, z_map) × 0.5 (bias-vector penalty)
        quality_status   — "Pass", "Fail_High", or "Fail_Masked"
        obs_log_N        — log10(median_noise)
        rho_win          — computed effective matrix coverage
    """
    rho_win = float(contacts) / ((float(chrom_len) / resolution) * (float(window_bp) / resolution))
    L_rho = float(np.log10(max(rho_win, 1e-300)))

    X = _build_design_matrix(L_rho, ebr)

    pred_mean  = float(X @ _BETA_MEAN)
    pred_upper = float(X @ _BETA_UPPER)
    pred_lower = float(X @ _BETA_LOWER)

    # Clipped envelopes: safe against quantile cross-over at extreme L_rho values
    y_up = max(pred_upper, pred_lower)
    y_lo = min(pred_upper, pred_lower)

    obs_log_N = float(np.log10(max(float(median_noise), 1e-300)))
    epsilon = obs_log_N - pred_mean

    half_envelope = (y_up - y_lo) / 2.0
    z_map = epsilon / half_envelope if half_envelope > 0.0 else 0.0
    gamma = 1.0 + max(0.0, z_map) * 0.5

    if obs_log_N > y_up:
        quality_status = "Fail_High"
    elif obs_log_N < y_lo:
        quality_status = "Fail_Masked"
    else:
        quality_status = "Pass"

    if z_map > 2.0:
        print(
            f"[WARN] z_map={z_map:.3f} > 2.0: high stochastic noise detected "
            f"(quality_status={quality_status}, gamma={gamma:.3f})"
        )

    return {
        "pred_log_N":       pred_mean,
        "pred_log_N_upper": y_up,
        "pred_log_N_lower": y_lo,
        "epsilon":          epsilon,
        "z_map":            z_map,
        "gamma":            gamma,
        "quality_status":   quality_status,
        "obs_log_N":        obs_log_N,
        "rho_win":          rho_win,
    }


def compute_zmap(
    contacts: float,
    chrom_len: int,
    resolution: int,
    median_noise: float,
    ebr: float,
) -> Dict[str, float]:
    """
    Compute z-map quality score (deprecated wrapper).

    .. deprecated::
        Use ``compute_noise_model()`` instead. This wrapper calls ``compute_noise_model``
        with a resolution-derived default window and returns a subset of its output
        for backwards compatibility with callers that relied on the old 3-term linear
        model interface.

        Note: the ``residual`` key is no longer returned; it is superseded by
        ``epsilon`` in ``compute_noise_model()``.

    Parameters
    ----------
    contacts     : total contact pairs for this chromosome
    chrom_len    : chromosome length in bp
    resolution   : bin size in bp
    median_noise : observed median noise value from the sampling phase
    ebr          : mean empty bin ratio from the sampling phase

    Returns
    -------
    dict with keys: z_map, gamma, pred_log_N, obs_log_N
    """
    from .noise_sampling import _default_window_bp  # avoid circular import at module level
    result = compute_noise_model(
        contacts=contacts,
        chrom_len=chrom_len,
        resolution=resolution,
        window_bp=_default_window_bp(int(resolution)),
        median_noise=median_noise,
        ebr=ebr,
    )
    return {
        "z_map":      result["z_map"],
        "gamma":      result["gamma"],
        "pred_log_N": result["pred_log_N"],
        "obs_log_N":  result["obs_log_N"],
    }


def sequencing_advisor(
    obs_noise: float,
    current_rho_win: float,
    ebr: float,
    window_bp: int,
    z_target: float = 0.0,
    fold_threshold: float = 10.0,
) -> Dict[str, object]:
    """
    Predict the sequencing depth needed to reach a target quality level.

    The spline model is not analytically invertible, so this function uses
    ``scipy.optimize.brentq`` to numerically find the ρ_win value at which the
    model's central prediction equals the target noise level.

    Target derivation
    -----------------
    The target log-noise is anchored at the current sample's position in the
    model:
        half_envelope = (y_up(current) − y_lo(current)) / 2
        target_log_N  = X(current) · β_mean + z_target × half_envelope

    Then ``brentq`` finds L_target ∈ [L_MIN, L_MAX] such that:
        X(L_target) · β_mean = target_log_N

    fold_increase = ρ_win_target / ρ_win_current.

    Because ρ_win = contacts / ((chrom_len/res) × (window_bp/res)), a fold
    increase in ρ_win maps directly to the same fold increase in contacts
    (window_bp and resolution are fixed).

    Parameters
    ----------
    obs_noise       : observed median noise from the sampling phase
    current_rho_win : current effective matrix coverage
                      (= contacts / ((chrom_len/res) × (window_bp/res)))
    ebr             : mean empty bin ratio from the sampling phase
    window_bp       : analysis window size in bp (used to compute rho_win)
    z_target        : desired envelope-normalised z-score (default: 0.0)
    fold_threshold  : fold-increase above which recommendation is "Stop"
                      (default: 10.0)

    Returns
    -------
    dict with keys:
        target_density  — ρ_win value needed to reach z_target
        fold_increase   — target_density / current_rho_win
        recommendation  — "Proceed" or "Stop"
    """
    nan_result: Dict[str, object] = {
        "target_density": float("nan"),
        "fold_increase": float("nan"),
        "recommendation": "Insufficient data",
    }

    if (
        not np.isfinite(obs_noise)
        or obs_noise <= 0.0
        or not np.isfinite(current_rho_win)
        or current_rho_win <= 0.0
        or not np.isfinite(ebr)
    ):
        return nan_result

    try:
        L_current = float(np.log10(max(current_rho_win, 1e-300)))
        X_current = _build_design_matrix(L_current, ebr)
        pred_upper_c = float(X_current @ _BETA_UPPER)
        pred_lower_c = float(X_current @ _BETA_LOWER)
        half_envelope = (max(pred_upper_c, pred_lower_c) - min(pred_upper_c, pred_lower_c)) / 2.0
        target_log_N = float(X_current @ _BETA_MEAN) + z_target * half_envelope

        # Numerically invert: find L such that X(L)·β_mean = target_log_N
        f = lambda L: float(_build_design_matrix(float(L), ebr) @ _BETA_MEAN) - target_log_N
        f_lo = f(_L_MIN)
        f_hi = f(_L_MAX)

        if f_lo * f_hi > 0:
            # No sign change → target_log_N is outside the model's predictable range
            return nan_result

        L_target = brentq(f, _L_MIN, _L_MAX, xtol=1e-6, maxiter=200)
        rho_win_target = float(10.0 ** L_target)
        fold_increase = rho_win_target / float(current_rho_win)
        recommendation = "Stop" if fold_increase > float(fold_threshold) else "Proceed"
    except Exception:
        return nan_result

    return {
        "target_density": rho_win_target,
        "fold_increase":  fold_increase,
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


def process_normalization_vectors_adaptive(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    JUKEBOX-ADAPTIVE normalization.

    Better suited for sparse or palaeogenomic data. Uses the global shift from the
    model prediction to determine correction intensity while dampening local bin
    variance to prevent blow-outs in stochastically noisy regions.

    Math
    ----
    global_shift = median(log10(N)) - pred_log_N
    D_i          = log10(N_i) - median(log10(N))   # local deviation from chromosome median
    log10(B_i)   = 0.5 * (global_shift + alpha * D_i)
    B_i          = 10 ^ log10(B_i)

    Interpretation
    --------------
    The bias has two components:
    1. **Global shift** (``0.5 * global_shift``): moves all bins up or down
       proportionally to how far the chromosome's overall noise is from the model
       prediction. Applied uniformly across all bins.
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
    pred_log_N  : predicted log10 noise for this chromosome from compute_noise_model()
                  (= X·β_mean from the spline model)
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

    # Global shift: how far is this chromosome's median noise from the model prediction?
    global_shift = median_obs - float(pred_log_N)
    # Local deviation: how far is each bin from the chromosome's own median?
    D_i = log_N - median_obs

    log_bias = 0.5 * (global_shift + float(alpha) * D_i)
    bias_vector = np.power(10.0, log_bias)
    bias_vector[is_nan] = np.nan

    return bias_vector


def process_normalization_vectors_powerlaw(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
    p: float = 3.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    JUKEBOX-POWERLAW normalization.

    Applies a signed power-law (root) transformation to the log-space residual,
    compressing the long tail more aggressively than a linear scalar.

    Math
    ----
    ε_i  = log10(N_i) − pred_log_N
    ε_i′ = sign(ε_i) · |ε_i|^(1/p) · α
    B_i  = 10 ^ ε_i′

    With p=3 (cube root) and α=0.5: large residuals are pulled in relative to the
    linear baseline while the sign of the correction is preserved, preventing
    blowout from extreme outlier bins.

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    pred_log_N  : predicted log10 noise from compute_noise_model() (= X·β_mean)
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)
    p           : compression factor; p=3 → cube root, p=2 → square root (default 3.0)
    alpha       : scaling constant to keep center stable (default 0.5)

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    residual = log_N - float(pred_log_N)
    compressed = np.sign(residual) * np.power(np.abs(residual), 1.0 / float(p)) * float(alpha)
    bias_vector = np.power(10.0, compressed)
    bias_vector[is_nan] = np.nan

    return bias_vector


def process_normalization_vectors_bayesian(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
    density_track: np.ndarray,
    k_density: Optional[float] = None,
) -> np.ndarray:
    """
    JUKEBOX-BAYESIAN normalization.

    Evidence-weighted Bayesian shrinkage: bins with low local contact density
    receive less correction (shrink toward B_i = 1.0).

    Math
    ----
    ρ_i  = per-bin contact density (row sum of the contact matrix)
    K    = half-saturation constant (default: median non-zero density across all bins)
    ε_i  = log10(N_i) − pred_log_N
    w_i  = ρ_i / (ρ_i + K)
    B_i  = 10 ^ (w_i · 0.5 · ε_i)

    When ρ_i >> K: w_i → 1.0 → full baseline correction.
    When ρ_i = K:  w_i = 0.5 → half correction.
    When ρ_i → 0:  w_i → 0.0 → B_i = 1.0 (no correction; sparse bin shrinks to identity).

    Parameters
    ----------
    noise_track   : per-bin raw noise values (may contain NaN/Inf)
    pred_log_N    : predicted log10 noise from compute_noise_model() (= X·β_mean)
    ebr           : mean empty bin ratio (Gaussian sigma offset)
    density_track : per-bin contact density (row sums written by noise_fullmap)
    k_density     : half-saturation constant K; if None, auto-set to the median
                    of all non-zero density values in density_track

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    residual = log_N - float(pred_log_N)

    rho = np.clip(density_track.astype(float), 0.0, None)

    if k_density is None or not np.isfinite(k_density) or k_density <= 0:
        finite_rho = rho[np.isfinite(rho) & (rho > 0)]
        k = float(np.median(finite_rho)) if finite_rho.size > 0 else 1.0
    else:
        k = float(k_density)

    weight = rho / (rho + k)
    log_bias = weight * 0.5 * residual
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
