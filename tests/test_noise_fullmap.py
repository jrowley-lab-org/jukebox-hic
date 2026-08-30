"""
Tests for private helpers in src/jukebox_hic/noise_fullmap.py.

No multiprocessing or real Hi-C files are used — only the pure arithmetic and
file-parsing functions that make up the pre- and post-computation plumbing.
"""
import numpy as np
import scipy.sparse as sparse
import pytest

from jukebox_hic.noise_fullmap import (
    _compute_dump_bins,
    _parse_extended_chrom_sizes,
    _precompute_expectation,
)


# ---------------------------------------------------------------------------
# _compute_dump_bins
# ---------------------------------------------------------------------------

def test_compute_dump_bins_positive():
    result = _compute_dump_bins(n_bins=1000, noise_bins=100, dump_factor=10.0)
    assert result > 0


def test_compute_dump_bins_never_exceeds_n_bins():
    # The result is capped at n_bins
    result = _compute_dump_bins(n_bins=50, noise_bins=100, dump_factor=10.0)
    assert result <= 50


def test_compute_dump_bins_minimum_is_one():
    # With n_bins=1, the result must be 1 (not 0 or negative)
    result = _compute_dump_bins(n_bins=1, noise_bins=100, dump_factor=10.0)
    assert result == 1


def test_compute_dump_bins_scales_with_dump_factor():
    # With n_bins large enough that the cap doesn't apply, doubling dump_factor
    # should roughly double the chunk size.
    # dump_factor=10: min(10000, max(200, 1000)) = 1000
    # dump_factor=20: min(10000, max(200, 2000)) = 2000
    low  = _compute_dump_bins(n_bins=10_000, noise_bins=100, dump_factor=10.0)
    high = _compute_dump_bins(n_bins=10_000, noise_bins=100, dump_factor=20.0)
    assert high == 2 * low


def test_compute_dump_bins_floor_is_two_noise_bins():
    # When dump_factor * noise_bins < 2 * noise_bins, floor kicks in
    # dump_factor=0.5, noise_bins=100: int(0.5*100)=50 < 2*100=200 → max = 200
    result = _compute_dump_bins(n_bins=10_000, noise_bins=100, dump_factor=0.5)
    assert result == 200


def test_compute_dump_bins_returns_int():
    result = _compute_dump_bins(n_bins=1000, noise_bins=100)
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _precompute_expectation
# ---------------------------------------------------------------------------

def test_precompute_expectation_returns_tuple(tiny_coo):
    means, default = _precompute_expectation(tiny_coo)
    assert isinstance(means, np.ndarray)
    assert isinstance(default, float)


def test_precompute_expectation_means_is_1d(tiny_coo):
    means, _ = _precompute_expectation(tiny_coo)
    assert means.ndim == 1
    assert len(means) > 0


def test_precompute_expectation_default_is_finite(tiny_coo):
    # tiny_coo has off-diagonal entries so default should be positive
    _, default = _precompute_expectation(tiny_coo)
    assert np.isfinite(default)


def test_precompute_expectation_diagonal_only_is_finite():
    # All entries on the diagonal (lag=0 only) — default must still be finite
    row  = np.array([0, 1, 2])
    col  = np.array([0, 1, 2])
    data = np.array([5.0, 3.0, 7.0])
    diag_coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
    means, default = _precompute_expectation(diag_coo)
    assert np.isfinite(default)
    assert default > 0.0


def test_precompute_expectation_empty_matrix_returns_zero():
    empty = sparse.coo_matrix((3, 3))
    means, default = _precompute_expectation(empty)
    assert default == 0.0
    assert len(means) == 1


# ---------------------------------------------------------------------------
# _parse_extended_chrom_sizes
# ---------------------------------------------------------------------------

def test_parse_extended_chrom_sizes_basic_two_columns(tmp_path):
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\nchr2\t242193529\n")
    result = _parse_extended_chrom_sizes(str(f))
    assert result["chr1"] == (248956422, None)
    assert result["chr2"] == (242193529, None)


def test_parse_extended_chrom_sizes_on_diagonal_region(tmp_path):
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\t10000000-50000000\n")
    result = _parse_extended_chrom_sizes(str(f))
    chrom_len, specs = result["chr1"]
    assert chrom_len == 248956422
    assert specs is not None
    assert len(specs) == 1
    # On-diagonal spec is (start, end) — a flat int tuple
    spec = specs[0]
    assert isinstance(spec[0], int)
    assert spec == (10_000_000, 50_000_000)


def test_parse_extended_chrom_sizes_off_diagonal_region(tmp_path):
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\t0-10000000:50000000-60000000\n")
    result = _parse_extended_chrom_sizes(str(f))
    _, specs = result["chr1"]
    assert specs is not None
    assert len(specs) == 1
    spec = specs[0]
    # Off-diagonal spec is ((l_start, l_end), (r_start, r_end))
    assert not isinstance(spec[0], int)
    assert spec == ((0, 10_000_000), (50_000_000, 60_000_000))


def test_parse_extended_chrom_sizes_multiple_regions(tmp_path):
    # Two on-diagonal regions in separate columns
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\t0-10000000\t20000000-30000000\n")
    result = _parse_extended_chrom_sizes(str(f))
    _, specs = result["chr1"]
    assert specs is not None
    assert len(specs) == 2


def test_parse_extended_chrom_sizes_bad_colon_count_is_skipped(tmp_path, capsys):
    # A column with two colons is ambiguous — skipped with a [WARN]
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\t0:10000:50000\n")
    result = _parse_extended_chrom_sizes(str(f))
    # The bad region is skipped; no valid specs remain → whole chrom skipped
    assert "chr1" not in result


def test_parse_extended_chrom_sizes_region_exceeds_size_is_skipped(tmp_path):
    # Region end (200000) > chrom size (100000) → skipped
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t100000\t0-200000\n")
    result = _parse_extended_chrom_sizes(str(f))
    # No valid specs for chr1 → chrom skipped entirely
    assert "chr1" not in result


def test_parse_extended_chrom_sizes_comments_and_blanks_ignored(tmp_path):
    f = tmp_path / "sizes.tsv"
    f.write_text("# header comment\n\nchr1\t248956422\n")
    result = _parse_extended_chrom_sizes(str(f))
    assert "chr1" in result


def test_parse_extended_chrom_sizes_last_row_wins_on_duplicate(tmp_path):
    # Duplicate chromosome entries: last row wins
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t100000\nchr1\t200000\n")
    result = _parse_extended_chrom_sizes(str(f))
    assert result["chr1"][0] == 200000


# ---------------------------------------------------------------------------
# Robustness: short chromosomes must not reach the native reader
# ---------------------------------------------------------------------------

def test_short_chromosome_guard_threshold():
    """
    A chromosome shorter than the noise window cannot carry a 2W diagonal window,
    and these are exactly the entries that crash the native .hic reader (chrM at
    fine resolutions makes hicstraw die with SIGFPE, which used to hang or kill a
    whole genome-wide run). The guard compares chromosome length to the window.
    """
    from jukebox_hic.noise_fullmap import _default_bindist_bp

    # chrM against the windows JUKEBOX actually uses
    chrM = 16_569
    for res in (10_000, 25_000, 250_000):
        assert chrM < _default_bindist_bp(res), (
            f"chrM would not be filtered at {res} bp"
        )
    # real chromosomes must survive the same test
    for res, length in ((250_000, 46_709_983), (25_000, 46_709_983), (10_000, 46_709_983)):
        assert length >= _default_bindist_bp(res), (
            f"chr21 would be wrongly filtered at {res} bp"
        )
