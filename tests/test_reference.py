"""
Tests for src/jukebox_hic/reference.py.

All functions here are pure math (numpy/scipy/scipy.interpolate only) with no
file I/O, so every test runs without any Hi-C fixture files.
"""
import numpy as np
import pytest

from jukebox_hic.reference import (
    _ENCODE_SUFFICIENT_CONTACTS,
    aggregate_genome_wide_noise_ebr,
    classify_quality_percentile,
    compute_noise_model,
    map_sequencing_advisor,
    process_normalization_vectors_jukebox,
    select_advisor_resolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPICAL_MODEL_KWARGS = dict(
    contacts=50_000_000,
    chrom_len=248_956_422,
    resolution=10_000,
    window_bp=1_000_000,
    median_noise=2.0,
    ebr=0.1,
)

_VALID_PERCENTILES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95}

# ---------------------------------------------------------------------------
# compute_noise_model
# ---------------------------------------------------------------------------

def test_compute_noise_model_returns_expected_keys():
    result = compute_noise_model(**_TYPICAL_MODEL_KWARGS)
    required = {
        "pred_log_N", "pred_log_N_upper", "pred_log_N_lower",
        "epsilon", "z_map", "gamma", "quality_status", "obs_log_N", "rho_win",
    }
    assert required.issubset(result.keys())


def test_compute_noise_model_quality_status_is_valid_string():
    result = compute_noise_model(**_TYPICAL_MODEL_KWARGS)
    assert result["quality_status"] in ("Pass", "Fail_High", "Fail_Masked")


def test_compute_noise_model_rho_win_positive():
    result = compute_noise_model(**_TYPICAL_MODEL_KWARGS)
    assert result["rho_win"] > 0.0


def test_compute_noise_model_gamma_at_least_one():
    # gamma = 1.0 + max(0, z_map) * 0.5, so minimum is 1.0
    result = compute_noise_model(**_TYPICAL_MODEL_KWARGS)
    assert result["gamma"] >= 1.0


def test_compute_noise_model_obs_log_N_matches_input():
    # When median_noise = 10.0, obs_log_N = log10(10.0) = 1.0
    result = compute_noise_model(
        contacts=50_000_000,
        chrom_len=248_956_422,
        resolution=10_000,
        window_bp=1_000_000,
        median_noise=10.0,
        ebr=0.1,
    )
    assert abs(result["obs_log_N"] - 1.0) < 1e-9


def test_compute_noise_model_epsilon_is_obs_minus_pred():
    # epsilon = obs_log_N - pred_log_N (verify the arithmetic holds)
    result = compute_noise_model(**_TYPICAL_MODEL_KWARGS)
    assert abs(result["epsilon"] - (result["obs_log_N"] - result["pred_log_N"])) < 1e-9


def test_compute_noise_model_high_noise_fails_high():
    # Very high median noise should produce Fail_High quality status
    result = compute_noise_model(
        contacts=1_000_000,
        chrom_len=248_956_422,
        resolution=10_000,
        window_bp=1_000_000,
        median_noise=1e8,
        ebr=0.9,
    )
    assert result["quality_status"] == "Fail_High"


# ---------------------------------------------------------------------------
# classify_quality_percentile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obs_log_N", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_classify_quality_percentile_always_in_valid_set(obs_log_N):
    p = classify_quality_percentile(L_rho=1.0, ebr=0.2, obs_noise_log10=obs_log_N)
    assert p in _VALID_PERCENTILES


def test_classify_quality_percentile_extreme_high_noise_returns_95():
    # log10 noise of 10.0 is astronomically high — must exceed all decile boundaries
    p = classify_quality_percentile(L_rho=0.0, ebr=0.9, obs_noise_log10=10.0)
    assert p == 95


def test_classify_quality_percentile_ordering():
    # Higher observed noise should give an equal-or-worse (higher) percentile
    low  = classify_quality_percentile(L_rho=2.0, ebr=0.05, obs_noise_log10=0.5)
    high = classify_quality_percentile(L_rho=2.0, ebr=0.05, obs_noise_log10=3.0)
    assert low <= high


# ---------------------------------------------------------------------------
# select_advisor_resolution
# ---------------------------------------------------------------------------

def test_select_advisor_resolution_at_2B_returns_none():
    result = select_advisor_resolution(
        total_contacts=float(_ENCODE_SUFFICIENT_CONTACTS),
        available_resolutions=[10_000],
    )
    assert result is None


def test_select_advisor_resolution_above_2B_returns_none():
    result = select_advisor_resolution(
        total_contacts=5_000_000_000.0,
        available_resolutions=[10_000],
    )
    assert result is None


@pytest.mark.parametrize("contacts,expected_res", [
    (25_000_000.0,   100_000),   # tier 0: [0, 50M)  → 100 kb
    (75_000_000.0,    25_000),   # tier 1: [50M, 100M) → 25 kb
    (150_000_000.0,   10_000),   # tier 2: [100M, 250M) → 10 kb
    (375_000_000.0,    5_000),   # tier 3: [250M, 500M) → 5 kb
    (750_000_000.0,    1_000),   # tier 4: [500M, 1B) → 1 kb
])
def test_select_advisor_resolution_tiers(contacts, expected_res):
    result = select_advisor_resolution(
        total_contacts=contacts,
        available_resolutions=[1_000, 5_000, 10_000, 25_000, 100_000],
    )
    assert result == expected_res


def test_select_advisor_resolution_finest_tier_picks_smallest_available():
    # 1B–2B contacts → returns min(available_resolutions)
    result = select_advisor_resolution(
        total_contacts=1_500_000_000.0,
        available_resolutions=[10_000, 5_000, 25_000],
    )
    assert result == 5_000


def test_select_advisor_resolution_finest_tier_empty_list_raises():
    with pytest.raises(ValueError):
        select_advisor_resolution(
            total_contacts=1_500_000_000.0,
            available_resolutions=[],
        )


# ---------------------------------------------------------------------------
# aggregate_genome_wide_noise_ebr
# ---------------------------------------------------------------------------

def test_aggregate_genome_wide_noise_ebr_contact_weighted():
    # chr1 has 2× contacts; its noise (2.0) should dominate the weighted mean
    agg_noise, agg_ebr = aggregate_genome_wide_noise_ebr(
        chrom_median_noise={"chr1": 2.0, "chr2": 4.0},
        chrom_mean_ebr={"chr1": 0.1, "chr2": 0.3},
        chrom_contacts={"chr1": 200.0, "chr2": 100.0},
    )
    # (2*200 + 4*100) / 300 = 800/300
    assert abs(agg_noise - 800 / 300) < 1e-9
    # (0.1*200 + 0.3*100) / 300 = 50/300
    assert abs(agg_ebr - 50 / 300) < 1e-9


def test_aggregate_genome_wide_noise_ebr_single_chrom():
    agg_noise, agg_ebr = aggregate_genome_wide_noise_ebr(
        chrom_median_noise={"chr1": 3.0},
        chrom_mean_ebr={"chr1": 0.2},
        chrom_contacts={"chr1": 100.0},
    )
    assert abs(agg_noise - 3.0) < 1e-9
    assert abs(agg_ebr - 0.2) < 1e-9


def test_aggregate_genome_wide_noise_ebr_zero_contacts_raises():
    # Chromosomes with contacts=0 are filtered out, leaving no valid chroms
    with pytest.raises(ValueError, match="No chromosomes"):
        aggregate_genome_wide_noise_ebr(
            chrom_median_noise={"chr1": 2.0},
            chrom_mean_ebr={"chr1": 0.1},
            chrom_contacts={"chr1": 0.0},
        )


# ---------------------------------------------------------------------------
# map_sequencing_advisor
# ---------------------------------------------------------------------------

def test_map_sequencing_advisor_sufficient_at_2B():
    result = map_sequencing_advisor(
        total_contacts=float(_ENCODE_SUFFICIENT_CONTACTS),
        genome_len=3_000_000_000,
        resolution=10_000,
        window_bp=1_000_000,
        median_noise=2.0,
        ebr=0.1,
    )
    assert result["recommendation"] == "Sufficient"
    assert result["resolution"] is None


def test_map_sequencing_advisor_returns_all_keys():
    result = map_sequencing_advisor(
        total_contacts=100_000_000.0,
        genome_len=3_000_000_000,
        resolution=10_000,
        window_bp=1_000_000,
        median_noise=2.0,
        ebr=0.1,
    )
    for key in ("total_contacts", "genome_len", "resolution", "window_bp",
                "epsilon", "quality_status", "recommendation"):
        assert key in result


def test_map_sequencing_advisor_recommendation_is_string():
    result = map_sequencing_advisor(
        total_contacts=100_000_000.0,
        genome_len=3_000_000_000,
        resolution=10_000,
        window_bp=1_000_000,
        median_noise=2.0,
        ebr=0.1,
    )
    assert result["recommendation"] in ("Sufficient", "Sequence Further")


# ---------------------------------------------------------------------------
# process_normalization_vectors_jukebox
# ---------------------------------------------------------------------------

def test_process_normalization_vectors_nan_preserved():
    # NaN bins in input must remain NaN in output
    track = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    result = process_normalization_vectors_jukebox(track, pred_log_N=0.5, ebr=0.1)
    assert np.isnan(result[1])
    assert np.isnan(result[3])


def test_process_normalization_vectors_inf_preserved():
    # Inf bins should also be masked to NaN
    track = np.array([1.0, np.inf, 2.0])
    result = process_normalization_vectors_jukebox(track, pred_log_N=0.5, ebr=0.1)
    assert np.isnan(result[1])


def test_process_normalization_vectors_valid_bins_finite():
    track = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = process_normalization_vectors_jukebox(track, pred_log_N=0.5, ebr=0.1)
    assert np.all(np.isfinite(result))


def test_process_normalization_vectors_preserves_length():
    track = np.array([1.0, 2.0, np.nan, 4.0])
    result = process_normalization_vectors_jukebox(track, pred_log_N=0.5, ebr=0.1)
    assert len(result) == 4
