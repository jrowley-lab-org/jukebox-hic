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
- ``select_advisor_resolution()``           — ENCODE-tier resolution selection from total contacts
- ``aggregate_genome_wide_noise_ebr()``     — contact-weighted genome-wide noise/EBR aggregation
- ``map_sequencing_advisor()``              — whole-map sequencing-sufficiency recommendation
- ``_build_design_matrix()``               — build 11-feature design vector for one sample
- ``_preprocess_noise_track()``             — shared noise preprocessing (smooth, log)
- ``process_normalization_vectors_jukebox()`` — JUKEBOX bias vector
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d

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

# ---------------------------------------------------------------------------
# JUKEBOX Decile Diagnostic Constants (q10–q90)
# ---------------------------------------------------------------------------
# Higher-precision knot values calibrated against the 4DN Gold Standard database
# via quantile regression at deciles 10–90.
#
# Beta vector layout (11 elements): [const, s0, s1, s2, s3, s4, s5, s6, s7, EBR, Lρ·EBR]
_DECILE_INTERIOR_KNOTS: np.ndarray = np.array([
    -0.70512324, 0.25933136, 1.04265875, 1.78215330, 2.88683441
])
_DECILE_L_MIN: float = -4.05114379
_DECILE_L_MAX: float =  5.37463540

_DECILE_KNOTS_AUG: np.ndarray = np.concatenate([
    np.full(_DEGREE + 1, _DECILE_L_MIN),
    _DECILE_INTERIOR_KNOTS,
    np.full(_DEGREE + 1, _DECILE_L_MAX),
])

_DECILE_BETAS: Dict[int, np.ndarray] = {
    10: np.array([ -9.7430, 6.8718, 7.3579, 10.0377, 10.7694, 10.3195, 10.4304, 10.7010, 10.5921,  2.4502, -1.8769]),
    20: np.array([ -9.4172, 6.9640, 7.0586,  9.7539, 10.4912, 10.0538, 10.2463, 10.4742, 10.4272,  2.5183, -1.8338]),
    30: np.array([ -9.6105, 7.2113, 7.1612,  9.9671, 10.7384, 10.3050, 10.5358, 10.7649, 10.7080,  2.6019, -1.8435]),
    40: np.array([ -9.4982, 6.8836, 6.8846,  9.8332, 10.6857, 10.2387, 10.5280, 10.7166, 10.6876,  2.7279, -1.8874]),
    50: np.array([ -9.5537, 6.6355, 6.6372,  9.8448, 10.7966, 10.3451, 10.6819, 10.8515, 10.8268,  2.9097, -1.9698]),
    60: np.array([ -9.6776, 6.3950, 6.3759,  9.9090, 10.9728, 10.5357, 10.9102, 11.0527, 11.0105,  3.1340, -2.0709]),
    70: np.array([-10.5867, 6.6732, 6.5670, 10.7372, 11.9476, 11.5479, 11.9546, 12.0301, 12.0440,  3.4744, -2.2855]),
    80: np.array([-10.6838, 6.4941, 6.3865, 10.8508, 12.1604, 11.7751, 12.2638, 12.1899, 12.2361,  3.6430, -2.4016]),
    90: np.array([-10.0198, 5.9371, 6.0107, 10.3712, 11.7477, 11.3243, 12.0291, 11.5292, 11.8412,  3.5812, -2.4003]),
}

# ---------------------------------------------------------------------------
# ENCODE-standards Whole-Map Sequencing Advisor Tiers
# ---------------------------------------------------------------------------
# Resolution (bin size, in bp) at which the whole-map residual (epsilon) should
# be evaluated, selected by total genome-wide contact count. Each tuple is
# (lower_bound_inclusive, upper_bound_exclusive, resolution_bp). A resolution
# of None in the last tier means "use the finest (smallest bp) resolution
# actually requested/run in this invocation" rather than a fixed value.
_ADVISOR_RESOLUTION_TIERS: List[Tuple[float, float, Optional[int]]] = [
    (0.0,              50_000_000.0,    100_000),
    (50_000_000.0,     100_000_000.0,   25_000),
    (100_000_000.0,    250_000_000.0,   10_000),
    (250_000_000.0,    500_000_000.0,   5_000),
    (500_000_000.0,    1_000_000_000.0, 1_000),
    (1_000_000_000.0,  2_000_000_000.0, None),
]

# ENCODE guidance: 2 billion total contacts is considered sufficient for Hi-C.
# At or above this, no further sequencing is ever advised, regardless of the
# residual — this must be checked before any residual is computed/consulted.
_ENCODE_SUFFICIENT_CONTACTS: float = 2_000_000_000.0

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


def _build_decile_design_row(L_rho: float, ebr: float) -> np.ndarray:
    """
    Build the 11-element design vector for the decile diagnostic model.

    Identical structure to ``_build_design_matrix()`` but uses the higher-precision
    decile knot geometry (``_DECILE_KNOTS_AUG``, ``_DECILE_L_MIN/MAX``).

    Parameters
    ----------
    L_rho : float
        log10(ρ_win) for this chromosome sample.
    ebr : float
        Mean empty bin ratio.

    Returns
    -------
    np.ndarray of shape (11,)
        [const, s0, s1, s2, s3, s4, s5, s6, s7, EBR, L_rho·EBR]
    """
    L = float(np.clip(L_rho, _DECILE_L_MIN, _DECILE_L_MAX))
    identity = np.eye(_N_BASIS_TOTAL)
    raw = np.array(
        [BSpline(_DECILE_KNOTS_AUG, identity[i], _DEGREE)(L) for i in range(_N_BASIS_TOTAL)],
        dtype=float,
    )
    raw = np.nan_to_num(raw, nan=0.0)
    s = raw[1:]  # drop first basis — patsy bs(include_intercept=False) convention
    return np.array([1.0, *s, float(ebr), L * float(ebr)])


def classify_quality_percentile(L_rho: float, ebr: float, obs_noise_log10: float) -> int:
    """
    Classify a sample's noise quality as a percentile decile relative to the 4DN landscape.

    Evaluates each of the nine decile boundary predictions (q10–q90) and returns the
    lowest decile whose predicted boundary is at or above the observed log10 noise.
    Returns 95 when the observed noise exceeds the q90 boundary (worse than the 90th
    percentile of the 4DN noise distribution).

    A monotony guardrail is applied before classification: spline flexibility at extreme
    L_rho coordinates can cause pred(qn) < pred(qn-1). A running-maximum enforcement
    ensures boundaries are non-decreasing from q10 to q90, preventing a sample from
    being incorrectly classified as high-quality in a "pinched" coordinate region.

    Parameters
    ----------
    L_rho : float
        log10(ρ_win) — log-effective matrix coverage for this chromosome.
    ebr : float
        Mean empty bin ratio from the sampling phase.
    obs_noise_log10 : float
        log10(median_noise) — observed noise in log space.

    Returns
    -------
    int
        Percentile decile: one of {10, 20, 30, 40, 50, 60, 70, 80, 90, 95}.
    """
    design_row = _build_decile_design_row(L_rho, ebr)

    raw_predictions: Dict[int, float] = {
        q: float(np.dot(design_row, beta))
        for q, beta in _DECILE_BETAS.items()
    }

    # Enforce monotonically non-decreasing boundaries q10 → q90
    monotone_predictions: Dict[int, float] = {}
    running_max = float("-inf")
    for q in sorted(raw_predictions):
        running_max = max(running_max, raw_predictions[q])
        monotone_predictions[q] = running_max

    for q in sorted(monotone_predictions):
        if obs_noise_log10 <= monotone_predictions[q]:
            return q

    return 95  # observed noise exceeds the 90th percentile boundary


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


def select_advisor_resolution(
    total_contacts: float,
    available_resolutions: Sequence[int],
) -> Optional[int]:
    """
    Choose which resolution's whole-map data the sequencing advisor should
    consult, based on total genome-wide contact count (ENCODE-standards tiers,
    see ``_ADVISOR_RESOLUTION_TIERS``).

    Returns the target resolution in bp, or ``None`` if ``total_contacts``
    already meets/exceeds the ENCODE-sufficient threshold
    (``_ENCODE_SUFFICIENT_CONTACTS``) — callers must treat ``None`` as an
    unconditional "Sufficient" short-circuit and must NOT evaluate the
    residual in that case.

    Parameters
    ----------
    total_contacts         : total contact pairs across the whole genome
    available_resolutions  : resolutions (bp) actually requested/run in this
                              invocation; used to resolve the "finest requested
                              resolution" tier (1B–2B contacts)

    Raises
    ------
    ValueError
        If the finest-requested-resolution tier is selected but
        ``available_resolutions`` is empty.
    """
    if total_contacts >= _ENCODE_SUFFICIENT_CONTACTS:
        return None

    for lower, upper, res in _ADVISOR_RESOLUTION_TIERS:
        if lower <= total_contacts < upper:
            if res is None:
                if not available_resolutions:
                    raise ValueError(
                        "total_contacts falls in the finest-requested-resolution "
                        "tier (1B–2B contacts), but no resolutions were provided."
                    )
                return min(int(r) for r in available_resolutions)
            return res

    # total_contacts < 0 (should not occur in practice) — fall back to the
    # coarsest tier's resolution.
    return _ADVISOR_RESOLUTION_TIERS[0][2]


def aggregate_genome_wide_noise_ebr(
    chrom_median_noise: Dict[str, float],
    chrom_mean_ebr: Dict[str, float],
    chrom_contacts: Dict[str, float],
) -> Tuple[float, float]:
    """
    Aggregate per-chromosome ``median_noise`` and ``mean_empty_bin_ratio`` into
    single genome-wide values, for use as inputs to ``map_sequencing_advisor()``.

    Uses a contact-weighted mean:
        agg_x = Σ(x_c · contacts_c) / Σ(contacts_c)

    This mirrors how ρ_win (the model's core covariate) is itself an implicit
    contact-weighted density over the genome, so chromosomes contributing more
    contacts dominate the aggregate proportionally. Falls back to a plain
    (unweighted) median across chromosomes only if total contact weight is
    zero (degenerate case).

    Parameters
    ----------
    chrom_median_noise : per-chromosome median noise, at one resolution
    chrom_mean_ebr      : per-chromosome mean empty bin ratio, at the same resolution
    chrom_contacts      : per-chromosome contacts, at the same resolution

    Returns
    -------
    (agg_median_noise, agg_ebr) : Tuple[float, float]
    """
    chroms = [
        c for c in chrom_median_noise
        if c in chrom_contacts and c in chrom_mean_ebr
        and np.isfinite(chrom_contacts[c]) and chrom_contacts[c] > 0
        and np.isfinite(chrom_median_noise[c]) and np.isfinite(chrom_mean_ebr[c])
    ]
    if not chroms:
        raise ValueError("No chromosomes with valid contacts/noise/ebr to aggregate.")

    weights = np.array([chrom_contacts[c] for c in chroms], dtype=float)
    total_w = float(weights.sum())
    if total_w <= 0.0:
        return (
            float(np.median([chrom_median_noise[c] for c in chroms])),
            float(np.median([chrom_mean_ebr[c] for c in chroms])),
        )

    noise_vals = np.array([chrom_median_noise[c] for c in chroms], dtype=float)
    ebr_vals   = np.array([chrom_mean_ebr[c] for c in chroms], dtype=float)
    return (
        float(np.sum(noise_vals * weights) / total_w),
        float(np.sum(ebr_vals * weights) / total_w),
    )


def map_sequencing_advisor(
    total_contacts: float,
    genome_len: int,
    resolution: int,
    window_bp: int,
    median_noise: float,
    ebr: float,
) -> Dict[str, object]:
    """
    Whole-map sequencing-sufficiency check (ENCODE/4DN rule).

    Rule
    ----
    1. If ``total_contacts >= _ENCODE_SUFFICIENT_CONTACTS`` (2 billion, per
       ENCODE Hi-C guidance): short-circuit to "Sufficient" WITHOUT evaluating
       any residual.
    2. Otherwise call ``compute_noise_model()`` at whole-genome scale and use
       the sign of ``epsilon`` (raw log10 residual, obs_log_N − pred_log_N):
           epsilon > 0  → "Sequence Further" (quality has not yet reached the
                          reference standard at this depth/resolution)
           epsilon <= 0 → "Sufficient" (quality has met the reference standard)

    Parameters
    ----------
    total_contacts : total contact pairs across the whole genome
    genome_len     : total genome length in bp (sum of chromosome sizes)
    resolution     : bin size in bp selected by ``select_advisor_resolution()``
    window_bp      : analysis window size in bp for this resolution
    median_noise   : genome-wide aggregated median noise (see
                     ``aggregate_genome_wide_noise_ebr()``)
    ebr            : genome-wide aggregated mean empty bin ratio

    Returns
    -------
    dict with keys:
        total_contacts, genome_len, resolution, window_bp, epsilon,
        quality_status, recommendation
        (``resolution``/``window_bp``/``epsilon``/``quality_status`` are
        ``None``/NaN/"" when short-circuited at the 2B-contacts tier)
    """
    if total_contacts >= _ENCODE_SUFFICIENT_CONTACTS:
        return {
            "total_contacts":  float(total_contacts),
            "genome_len":      int(genome_len),
            "resolution":      None,
            "window_bp":       None,
            "epsilon":         float("nan"),
            "quality_status":  "",
            "recommendation":  "Sufficient",
        }

    model_result = compute_noise_model(
        contacts=total_contacts,
        chrom_len=genome_len,
        resolution=resolution,
        window_bp=window_bp,
        median_noise=median_noise,
        ebr=ebr,
    )
    epsilon = model_result["epsilon"]
    recommendation = "Sequence Further" if epsilon > 0.0 else "Sufficient"

    return {
        "total_contacts":  float(total_contacts),
        "genome_len":      int(genome_len),
        "resolution":      int(resolution),
        "window_bp":       int(window_bp),
        "epsilon":         epsilon,
        "quality_status":  model_result["quality_status"],
        "recommendation":  recommendation,
    }


def _preprocess_noise_track(
    noise_track: np.ndarray,
    ebr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared pre-processing for the JUKEBOX bias-vector computation.

    ``process_normalization_vectors_jukebox()`` begins with the same four-step pipeline:

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
        Per-bin raw noise values from the noise-bedgraph output (may contain NaN/Inf).
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


def _center_log_bias(log_bias: np.ndarray, is_nan: np.ndarray) -> None:
    """Subtract mean of valid log-bias values so geometric_mean(10^log_bias) = 1.0."""
    valid = ~is_nan
    if not valid.any():
        return
    shift = float(np.mean(log_bias[valid]))
    if np.isfinite(shift):
        log_bias[valid] -= shift


def process_normalization_vectors_jukebox(
    noise_track: np.ndarray,
    pred_log_N: float,
    ebr: float,
    p: float = 3.0,
    alpha: float = 0.5,
    max_log2_fold: Optional[float] = 3.0,
) -> np.ndarray:
    """
    JUKEBOX normalization.

    Applies a signed power-law (root) transformation to the log-space residual,
    compressing the long tail more aggressively than a linear scalar.

    Math
    ----
    ε_i  = log10(N_i) − pred_log_N
    ε_i′ = sign(ε_i) · |ε_i|^(1/p) · α
    B_i  = 10 ^ ε_i′

    With p=2 (square root, the CLI default) or p=3 (cube root): large residuals
    are pulled in relative to the linear baseline while the sign of the
    correction is preserved, preventing blowout from extreme outlier bins.

    Parameters
    ----------
    noise_track : per-bin raw noise values (may contain NaN/Inf)
    pred_log_N  : predicted log10 noise from compute_noise_model() (= X·β_mean)
    ebr         : mean empty bin ratio (used as Gaussian sigma offset)
    p           : compression factor; p=2 → square root (CLI default), p=3 → cube root
    alpha       : scaling constant to keep center stable (default 0.5)
    max_log2_fold : clamp the bias to +/- this many log2 units around 1.0
                    (default 3.0 -> weights confined to [0.125, 8]).  None
                    disables clamping.

    Returns
    -------
    bias_vector : same shape as noise_track; NaN where original was NaN/Inf.
                  Finite entries have geometric mean exactly 1.0.
    """
    is_nan, log_N, _ = _preprocess_noise_track(noise_track, ebr)

    residual = log_N - float(pred_log_N)

    # Re-centre the residual on its own median BEFORE the non-linear compression.
    #
    # The signed root x -> sign(x)*|x|^(1/p) is steep near zero and flat far from
    # it.  If the bulk of the residual distribution does not sit near zero, the
    # bulk lands on the flat part of the curve (its spread is squashed) while the
    # few bins whose residual crosses zero land on the near-singular part (their
    # spread explodes).  The result is a bias vector with a near-constant bulk and
    # a runaway tail -- weights of 50-100x that blank out whole rows in Juicer.
    #
    # Centring first makes the transform symmetric about the chromosome's own
    # typical bin, so the vector encodes *relative* within-chromosome noise, which
    # is what a matrix-balancing bias vector should carry.  It also makes the
    # output robust to any scale offset in pred_log_N.
    valid = ~is_nan
    if valid.any():
        anchor = float(np.median(residual[valid]))
        if np.isfinite(anchor):
            residual = residual - anchor

    compressed = np.sign(residual) * np.power(np.abs(residual), 1.0 / float(p)) * float(alpha)

    # Clamp the log-bias.  Juicer divides the observed count by bias[i]*bias[j], so
    # an unclamped weight of 100 removes a row entirely (a pair of them by 1e4).
    # Clamping keeps the vector in a range comparable to KR/VC while preserving the
    # ranking of noisy bins.
    if max_log2_fold is not None and np.isfinite(max_log2_fold) and max_log2_fold > 0:
        lim = float(max_log2_fold) * np.log10(2.0)
        np.clip(compressed, -lim, lim, out=compressed)

    _center_log_bias(compressed, is_nan)
    bias_vector = np.power(10.0, compressed)
    bias_vector[is_nan] = np.nan

    return bias_vector
