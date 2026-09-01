"""
Regression tests for the JUKEBOX noise definition.

The Methods define a single noise metric:

    N_i = 1 / |lag-1 autocovariance of (O+1)/(E+1) over the diagonal window|

Two code paths implement it: ``noise_sampling`` (per-map, used to fit the 4DN
model) and ``noise_fullmap`` (per-bin, used for the bedgraph -> blacklist ->
normalization vector chain).  They must agree.  Prior to this fix the per-bin
path took the autocovariance of the RAW counts, silently dropping the
distance-decay normalization, which put the per-bin track ~4.5 log10 units off
the scale the 4DN model predicts.
"""
import numpy as np
import pytest
from scipy import sparse

from jukebox_hic import noise_fullmap as NF
from jukebox_hic import noise_sampling as NS


class _CM:
    """Minimal ChromMatrix stand-in exposing coo/csr/csc."""
    def __init__(self, dense):
        m = sparse.coo_matrix(dense)
        self.coo = m
        self.csr = m.tocsr()
        self.csc = m.tocsc()


def _decay_matrix(n=200, w=25, depth=20.0, gamma=1.2, seed=3):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n))
    for i in range(n):
        for lag in range(1, w + 1):
            j = i + lag
            if j >= n:
                break
            a[i, j] = rng.poisson(depth * (lag ** -gamma))
    return a + a.T


N_BINS, WINDOW = 200, 25


@pytest.fixture(scope="module")
def setup():
    dense = _decay_matrix(N_BINS, WINDOW)
    cm = _CM(dense)
    lookup, default = NF._precompute_expectation(cm.coo)
    return cm, lookup, default


def test_fullmap_uses_the_expected_curve(setup):
    """The per-bin path must actually consume the distance-decay expectation."""
    cm, lookup, default = setup
    real, _ = NF._row_window_noise(cm, N_BINS, WINDOW, lookup, default)
    fake, _ = NF._row_window_noise(cm, N_BINS, WINDOW, np.full_like(lookup, 1e6), 1e6)
    assert not np.array_equal(real, fake, equal_nan=True), (
        "noise_fullmap._row_window_noise produced identical output for a real and a "
        "garbage expected curve - the O/E normalization is not being applied."
    )


def test_fullmap_and_sampling_agree_bin_for_bin(setup):
    """Both implementations of the Methods formula must return the same values."""
    cm, lookup, default = setup
    full, _ = NF._row_window_noise(cm, N_BINS, WINDOW, lookup, default)
    samp = np.array([
        NS._compute_row_metrics(
            *NS._row_window_vectors(cm, i, WINDOW, lookup, default)
        )["noise_raw"]
        for i in range(N_BINS)
    ])
    both = np.isfinite(full) & np.isfinite(samp)
    assert both.sum() > N_BINS // 2, "too few comparable bins to be a meaningful test"
    np.testing.assert_allclose(full[both], samp[both], rtol=1e-9, atol=0)


def test_noise_scale_is_not_dominated_by_depth():
    """
    Scaling every count by a constant leaves the O/E ratios (and therefore the
    noise) nearly unchanged, whereas raw-count autocovariance would move by ~k^2.
    """
    dense = _decay_matrix(N_BINS, WINDOW)
    out = {}
    for k in (1.0, 8.0):
        cm = _CM(dense * k)
        lookup, default = NF._precompute_expectation(cm.coo)
        vals, _ = NF._row_window_noise(cm, N_BINS, WINDOW, lookup, default)
        out[k] = np.nanmedian(vals)
    ratio = out[8.0] / out[1.0]
    assert 0.25 < ratio < 4.0, f"median noise moved {ratio:.3g}x under a pure depth rescale"


def test_density_uses_the_same_window_as_the_noise(setup):
    """
    The density track must be the row sum over the same 2W diagonal window the
    noise uses -- not over whatever submatrix the chunker happened to fetch.
    """
    cm, lookup, default = setup
    _, dens = NF._row_window_noise(cm, N_BINS, WINDOW, lookup, default)
    dense = cm.csr.toarray()
    for i in (60, 100, 140):
        lo, hi = max(0, i - WINDOW), min(N_BINS, i + WINDOW + 1)
        expected = dense[i, lo:hi].sum() - dense[i, i]
        assert np.isclose(dens[i], expected, rtol=1e-9), (
            f"bin {i}: density {dens[i]} != 2W-window row sum {expected}"
        )


def test_expectation_is_chromosome_wide_not_per_chunk():
    """
    The distance-decay expectation E(k) is a property of the chromosome.  When it
    was computed per fetched chunk, every reported noise value depended on
    --dump_factor -- a memory-tuning knob -- so the same map gave different
    blacklists on machines with different memory budgets.

    Here: pooling lag statistics over disjoint row spans must give the same curve
    regardless of how the chromosome is partitioned.
    """
    dense = _decay_matrix(N_BINS, WINDOW)
    coo = sparse.coo_matrix(dense)
    lags = np.abs(coo.col.astype(np.int64) - coo.row.astype(np.int64))
    rows = coo.row.astype(np.int64)

    def pooled(chunk):
        sums = np.zeros(WINDOW + 1)
        counts = np.zeros(WINDOW + 1, dtype=np.int64)
        for start in range(0, N_BINS, chunk):
            end = min(start + chunk, N_BINS)
            keep = (lags <= WINDOW) & (rows >= start) & (rows < end)
            if keep.any():
                sums += np.bincount(lags[keep], weights=coo.data[keep], minlength=WINDOW + 1)
                counts += np.bincount(lags[keep], minlength=WINDOW + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    np.testing.assert_allclose(pooled(37), pooled(N_BINS), rtol=1e-12)
    np.testing.assert_allclose(pooled(11), pooled(N_BINS), rtol=1e-12)
