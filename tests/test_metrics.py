"""
Tests for src/jukebox_hic/metrics.py.

estimate_worker_memory_mb is a pure arithmetic formula (no OS calls).
collect_memory_usage_mb wraps optional system calls and must not crash
regardless of whether psutil/resource are installed.
"""
import pytest

from jukebox_hic.metrics import collect_memory_usage_mb, estimate_worker_memory_mb


# ---------------------------------------------------------------------------
# estimate_worker_memory_mb
# ---------------------------------------------------------------------------

def test_estimate_worker_memory_positive():
    mb = estimate_worker_memory_mb(dump_factor=10.0, noise_bins=100)
    assert mb > 0.0


def test_estimate_worker_memory_is_float():
    mb = estimate_worker_memory_mb(dump_factor=10.0, noise_bins=100)
    assert isinstance(mb, float)


def test_estimate_worker_memory_scales_with_dump_factor():
    # Doubling dump_factor increases fetch_bins and therefore MB, but the
    # relationship is quadratic (fetch_bins^2 in the formula).
    # With dump_factor=10 and noise_bins=100:
    #   fetch_bins = 10*100 + 2*100 = 1200
    # With dump_factor=20 and noise_bins=100:
    #   fetch_bins = 20*100 + 2*100 = 2200
    mb_low  = estimate_worker_memory_mb(dump_factor=10.0,  noise_bins=100)
    mb_high = estimate_worker_memory_mb(dump_factor=20.0,  noise_bins=100)
    assert mb_high > mb_low


def test_estimate_worker_memory_formula():
    # Verify the formula directly:
    #   fetch_bins = int(dump_factor * noise_bins) + 2 * noise_bins
    #   mb = (fetch_bins ** 2) * density_estimate * 24 / 1e6
    dump_factor     = 10.0
    noise_bins      = 100
    density_estimate = 0.05
    fetch_bins = int(dump_factor * noise_bins) + 2 * noise_bins
    expected = (fetch_bins ** 2) * density_estimate * 24 / 1e6
    mb = estimate_worker_memory_mb(dump_factor, noise_bins, density_estimate)
    assert abs(mb - expected) < 1e-10


def test_estimate_worker_memory_custom_density():
    # Higher density estimate → more RAM predicted
    mb_sparse = estimate_worker_memory_mb(dump_factor=10.0, noise_bins=100, density_estimate=0.01)
    mb_dense  = estimate_worker_memory_mb(dump_factor=10.0, noise_bins=100, density_estimate=0.30)
    assert mb_dense > mb_sparse


# ---------------------------------------------------------------------------
# collect_memory_usage_mb
# ---------------------------------------------------------------------------

def test_collect_memory_usage_returns_two_element_tuple():
    result = collect_memory_usage_mb()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_collect_memory_usage_values_are_non_negative():
    current_mb, peak_mb = collect_memory_usage_mb()
    # Values may be 0.0 if dependencies are absent, but never negative
    assert current_mb >= 0.0
    assert peak_mb >= 0.0


def test_collect_memory_usage_values_are_floats():
    current_mb, peak_mb = collect_memory_usage_mb()
    assert isinstance(current_mb, float)
    assert isinstance(peak_mb, float)
