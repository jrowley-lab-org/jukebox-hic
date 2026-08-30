"""
Regression tests for the JUKEBOX normalization (bias) vector.

Juicer applies a custom vector as normalized[i][j] = observed[i][j] / (v_i * v_j)
and reads each chromosome block positionally.  That imposes three requirements
the pre-fix code did not enforce:

1. the block must have one entry per bin of the chromosome, gaps included;
2. finite weights must have geometric mean 1 so the vector does not rescale the
   whole matrix;
3. weights must stay in a usable range - an outlier weight of 100 divides a row
   by 100 (and a pair of such bins by 1e4), erasing it.
"""
import numpy as np
import pandas as pd

from jukebox_hic import reference
from jukebox_hic.noise_to_weights import _reindex_to_full_grid


RES = 10_000


def test_reindex_fills_gaps_and_sizes_to_the_chromosome():
    """A gapped bedgraph must become a full-length grid with NaN in the holes."""
    df = pd.DataFrame({
        "chrom": ["chr1"] * 3,
        "start": [0, 20_000, 50_000],       # bins 0, 2, 5 - bins 1,3,4 missing
        "end":   [10_000, 30_000, 60_000],
        "value": [1.0, 2.0, 3.0],
    })
    out = _reindex_to_full_grid(df, RES, chrom_len=80_000)
    assert len(out) == 8, "vector length must be ceil(chrom_len / res)"
    assert out[0] == 1.0 and out[2] == 2.0 and out[5] == 3.0, "values landed on wrong bins"
    assert np.isnan(out[[1, 3, 4, 6, 7]]).all(), "gaps must be NaN, not dropped"


def test_bias_vector_has_unit_geometric_mean():
    rng = np.random.default_rng(0)
    track = 10.0 ** rng.normal(0.9, 0.4, size=2000)
    v = reference.process_normalization_vectors_jukebox(track, pred_log_N=0.95, ebr=0.05)
    f = v[np.isfinite(v)]
    assert np.isclose(np.exp(np.mean(np.log(f))), 1.0, rtol=1e-9)


def test_bias_vector_respects_the_clamp():
    rng = np.random.default_rng(1)
    track = 10.0 ** rng.normal(0.9, 0.4, size=2000)
    track[:20] = 1e12          # pathological bins that previously blew the vector up
    v = reference.process_normalization_vectors_jukebox(
        track, pred_log_N=0.95, ebr=0.05, max_log2_fold=3.0
    )
    f = v[np.isfinite(v)]
    assert f.max() <= 8.0 + 1e-9, f"weight {f.max():.3g} exceeds the 2^3 clamp"
    assert f.min() >= 0.125 - 1e-9, f"weight {f.min():.3g} below the 2^-3 clamp"


def test_bias_vector_is_robust_to_a_mis_scaled_anchor():
    """
    The vector encodes relative within-chromosome noise, so shifting pred_log_N by
    a constant (as a scale mismatch would) must not change the result.
    """
    rng = np.random.default_rng(2)
    track = 10.0 ** rng.normal(0.9, 0.4, size=1500)
    a = reference.process_normalization_vectors_jukebox(track, pred_log_N=0.95, ebr=0.05)
    b = reference.process_normalization_vectors_jukebox(track, pred_log_N=5.40, ebr=0.05)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_nan_bins_are_preserved_as_nan():
    track = np.array([5.0, np.nan, 7.0, np.inf, 6.0] * 100)
    v = reference.process_normalization_vectors_jukebox(track, pred_log_N=0.9, ebr=0.1)
    assert np.isnan(v[1]) and np.isnan(v[3])
    assert np.isfinite(v[0]) and np.isfinite(v[2]) and np.isfinite(v[4])
