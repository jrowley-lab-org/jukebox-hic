"""
Tests for private helpers in src/jukebox_hic/cli.py.

_comma_ints and _comma_strs are the argparse type-converter functions.
_parse_subsample_summary reads the custom comment-header TSV format that
sample-noise produces.  All tests use tmp_path or in-memory strings; no
Hi-C files or network access is required.
"""
import pytest

from jukebox_hic.cli import _comma_ints, _comma_strs, _parse_subsample_summary


# ---------------------------------------------------------------------------
# _comma_ints
# ---------------------------------------------------------------------------

def test_comma_ints_single_value():
    assert _comma_ints("10000") == [10000]


def test_comma_ints_multiple_values():
    assert _comma_ints("5000,10000,25000") == [5000, 10000, 25000]


def test_comma_ints_preserves_order():
    assert _comma_ints("25000,5000,10000") == [25000, 5000, 10000]


def test_comma_ints_invalid_raises():
    # Non-integer tokens cause int() to raise ValueError
    with pytest.raises(ValueError):
        _comma_ints("5000,abc")


def test_comma_ints_mixed_valid_invalid_raises():
    with pytest.raises(ValueError):
        _comma_ints("10000,10.5")


# ---------------------------------------------------------------------------
# _comma_strs
# ---------------------------------------------------------------------------

def test_comma_strs_single_value():
    assert _comma_strs("chr1") == ["chr1"]


def test_comma_strs_multiple_values():
    assert _comma_strs("chr1,chr2,chrX") == ["chr1", "chr2", "chrX"]


def test_comma_strs_strips_whitespace():
    # Each token is stripped of leading/trailing whitespace
    assert _comma_strs("chr1, chr2 , chrX") == ["chr1", "chr2", "chrX"]


def test_comma_strs_empty_tokens_dropped():
    # Trailing comma or double-comma produces empty token, which is silently dropped
    assert _comma_strs("chr1,,chr2,") == ["chr1", "chr2"]


# ---------------------------------------------------------------------------
# _parse_subsample_summary
# ---------------------------------------------------------------------------

_SUMMARY_CONTENT = """\
# resolution=10000
# window_bp=1000000
# chrom_info\tchr1\tcontacts=5000000\tsize_bp=248956422
# chrom_info\tchr2\tcontacts=3000000\tsize_bp=242193529
chrom\tratio\trows_evaluated\tmean_noise\tmedian_noise\tstd_noise\tmean_row_mean\tmean_row_std\tmean_acf\tmean_empty_bin_ratio\testimated_contacts\tz_map\tgamma\tpred_log_N\tpred_log_N_upper\tpred_log_N_lower\tepsilon\tquality_status\tquality_percentile
chr1\t1.0\t200\t2.5\t2.3\t0.8\t1.2\t0.3\t0.05\t0.10\t5000000\t0.3\t1.2\t1.5\t2.0\t1.0\t0.1\tPass\t50
chr1\t0.5\t200\t3.0\t2.8\t0.9\t1.3\t0.4\t0.06\t0.15\t2500000\t0.5\t1.3\t1.5\t2.0\t1.0\t0.2\tPass\t60
chr2\t1.0\t180\t2.1\t2.0\t0.7\t1.1\t0.2\t0.04\t0.08\t3000000\t0.2\t1.1\t1.4\t1.9\t0.9\t0.1\tPass\t40
"""


def test_parse_subsample_summary_returns_four_tuple(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    result = _parse_subsample_summary(str(f))
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_parse_subsample_summary_resolution(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    resolution, _, _, _ = _parse_subsample_summary(str(f))
    assert resolution == 10_000


def test_parse_subsample_summary_window_bp(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    _, window_bp, _, _ = _parse_subsample_summary(str(f))
    assert window_bp == 1_000_000


def test_parse_subsample_summary_chrom_sizes(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    _, _, chrom_sizes, _ = _parse_subsample_summary(str(f))
    assert chrom_sizes["chr1"] == 248_956_422
    assert chrom_sizes["chr2"] == 242_193_529


def test_parse_subsample_summary_df_filtered_to_ratio_one(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    _, _, _, df = _parse_subsample_summary(str(f))
    # Only ratio=1.0 rows should appear; the ratio=0.5 chr1 row is excluded
    assert (df["ratio"] == 1.0).all()
    assert len(df) == 2   # one for chr1 and one for chr2


def test_parse_subsample_summary_no_resolution_comment(tmp_path):
    # Missing the # resolution= line → resolution is None (not an error)
    content = (
        "# window_bp=1000000\n"
        "chrom\tratio\tmean_noise\n"
        "chr1\t1.0\t2.5\n"
    )
    f = tmp_path / "summary.tsv"
    f.write_text(content)
    resolution, window_bp, _, _ = _parse_subsample_summary(str(f))
    assert resolution is None
    assert window_bp == 1_000_000


def test_parse_subsample_summary_df_has_expected_columns(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SUMMARY_CONTENT)
    _, _, _, df = _parse_subsample_summary(str(f))
    assert "chrom" in df.columns
    assert "ratio" in df.columns
    assert "mean_noise" in df.columns
