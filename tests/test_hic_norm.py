"""Tests for writing a JUKEBOX normalization into a .hic file."""
import numpy as np
import pytest

from jukebox_hic.hic_norm import SCHEMES, bias_from_noise


def test_exponent_sets_strength_and_direction():
    """bias = (noise/median)^exponent, so the exponent is the only knob needed."""
    track = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    lin = bias_from_noise(track, exponent=1.0)
    sqrt = bias_from_noise(track, exponent=0.5)
    inv = bias_from_noise(track, exponent=-0.5)
    noop = bias_from_noise(track, exponent=0.0)

    # median is 4; linear gives the ratio itself
    np.testing.assert_allclose(lin, [0.25, 0.5, 1.0, 2.0, 4.0], rtol=1e-12)
    # sqrt is the square root of the linear weights
    np.testing.assert_allclose(sqrt, np.sqrt(lin), rtol=1e-12)
    # inverse sqrt is its reciprocal
    np.testing.assert_allclose(inv, 1.0 / np.sqrt(lin), rtol=1e-12)
    # exponent 0 is a genuine no-op
    np.testing.assert_allclose(noop, np.ones_like(track), rtol=1e-12)


@pytest.mark.parametrize("exponent", [-1.0, -0.5, 0.5, 1.0, 2.0])
def test_median_weight_is_one(exponent):
    """
    The typical bin must be left alone at any intensity. The anchor is the median,
    not the geometric mean: the noise metric is heavy-tailed enough that
    mean-centring lets a handful of degenerate bins rescale the whole map.
    """
    rng = np.random.default_rng(0)
    # odd length so the median is a real element rather than an average of two
    track = 10.0 ** rng.normal(1.0, 0.8, size=3001)
    b = bias_from_noise(track, exponent=exponent, max_log2_fold=None)
    f = b[np.isfinite(b)]
    assert np.isclose(np.median(f), 1.0, rtol=1e-9)


def test_heavy_tail_does_not_shift_the_bulk():
    """A few 1e9 bins must not move the weight of a typical bin."""
    rng = np.random.default_rng(7)
    track = 10.0 ** rng.normal(1.0, 0.4, size=2000)
    clean = bias_from_noise(track, exponent=1.0)
    track[:5] = 1e9
    tailed = bias_from_noise(track, exponent=1.0)
    ok = np.isfinite(clean) & np.isfinite(tailed)
    ok[:5] = False
    # the anchor shifts by at most one order statistic, so the bulk is untouched
    # to well within a percent -- mean-centring would move it by tens of percent
    np.testing.assert_allclose(clean[ok], tailed[ok], rtol=5e-3)


def test_clamp_is_a_hard_bound_even_when_it_binds():
    rng = np.random.default_rng(1)
    track = 10.0 ** rng.normal(1.0, 0.8, size=2000)
    track[:20] = 1e12          # pathological bins
    b = bias_from_noise(track, exponent=1.0, max_log2_fold=3.0)
    f = b[np.isfinite(b)]
    assert f.max() <= 8.0 + 1e-9
    assert f.min() >= 0.125 - 1e-9


def test_unmeasurable_bins_stay_nan():
    track = np.array([5.0, np.nan, 7.0, np.inf, 6.0, -1.0])
    b = bias_from_noise(track, exponent=1.0)
    assert np.isnan(b[[1, 3, 5]]).all(), "NaN/Inf/non-positive bins must be masked"
    assert np.isfinite(b[[0, 2, 4]]).all()


def test_all_nan_track_yields_all_nan():
    b = bias_from_noise(np.full(50, np.nan), exponent=1.0)
    assert np.isnan(b).all()


def test_residual_root_scheme_still_available():
    rng = np.random.default_rng(2)
    track = 10.0 ** rng.normal(1.0, 0.5, size=500)
    b = bias_from_noise(track, scheme="residual-root")
    f = b[np.isfinite(b)]
    assert np.isclose(np.exp(np.mean(np.log(f))), 1.0, rtol=1e-9)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        bias_from_noise(np.array([1.0, 2.0]), scheme="nope")
    assert "power" in SCHEMES and "residual-root" in SCHEMES


def test_scheme_is_invariant_to_a_rescaled_track():
    """
    Multiplying every bin's noise by a constant is not a change in relative noise,
    so the vector must not move. This is what makes each resolution's vector
    comparable even though absolute noise differs between resolutions.
    """
    rng = np.random.default_rng(3)
    track = 10.0 ** rng.normal(1.0, 0.6, size=800)
    a = bias_from_noise(track, exponent=1.0)
    b = bias_from_noise(track * 137.0, exponent=1.0)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_input_file_is_never_overwritten(tmp_path):
    from jukebox_hic.hic_norm import add_norm_to_hic
    f = tmp_path / "a.hic"
    f.write_bytes(b"HIC\0not-a-real-file")
    with pytest.raises(ValueError, match="refusing"):
        add_norm_to_hic(str(f), str(f), juicer_tools=str(f), noise_dir=str(tmp_path))
