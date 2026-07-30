"""
Tests for private helpers in src/jukebox_hic/noise_to_weights.py.

These functions are importable directly despite the leading underscore — the
underscore is a convention, not a language restriction.  Tests cover:
  - _is_decoy_chrom      — string-only, pure function
  - _load_bedgraph       — reads a 4-column file
  - _load_chrom_sizes    — reads a 2-column file, or returns {} for None
  - _validate_grid       — DataFrame validation with 7 error paths
  - _load_zmap_summary   — comment-skipping TSV parser
"""
import pandas as pd
import pytest

from jukebox_hic.noise_to_weights import (
    _is_decoy_chrom,
    _load_bedgraph,
    _load_chrom_sizes,
    _load_zmap_summary,
    _validate_grid,
)


# ---------------------------------------------------------------------------
# _is_decoy_chrom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chrom", ["chr1", "chr2", "chr22", "chrX", "chrY", "chrM"])
def test_is_decoy_chrom_standard_chroms_return_false(chrom):
    assert _is_decoy_chrom(chrom) is False


@pytest.mark.parametrize("chrom", [
    "chrUn_gl000220",    # starts with chrUn
    "chr1_random",       # contains _
    "chr6_ssto_hap7",    # contains _ + hap token
    "HG2_PATCH",         # contains _ + patch token
    "GL000191.1",        # starts with gl (NCBI prefix)
    "KI270762v1",        # starts with ki (NCBI prefix)
    "chrZ",              # sex chromosome alias used in some assemblies
    "chrW",              # sex chromosome alias
])
def test_is_decoy_chrom_decoy_sequences_return_true(chrom):
    assert _is_decoy_chrom(chrom) is True


# ---------------------------------------------------------------------------
# _load_bedgraph
# ---------------------------------------------------------------------------

def test_load_bedgraph_valid(tmp_path):
    f = tmp_path / "test.bedgraph"
    f.write_text("chr1\t0\t10000\t1.5\nchr1\t10000\t20000\t2.3\n")
    df = _load_bedgraph(str(f))
    assert list(df.columns) == ["chrom", "start", "end", "value"]
    assert len(df) == 2
    assert df.iloc[0]["chrom"] == "chr1"


def test_load_bedgraph_empty_raises(tmp_path):
    f = tmp_path / "empty.bedgraph"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _load_bedgraph(str(f))


def test_load_bedgraph_tab_and_space_separated(tmp_path):
    # _load_bedgraph uses sep=r"\s+" so both delimiters are accepted
    f = tmp_path / "space.bedgraph"
    f.write_text("chr1 0 10000 1.5\nchr1 10000 20000 2.3\n")
    df = _load_bedgraph(str(f))
    assert len(df) == 2


# ---------------------------------------------------------------------------
# _load_chrom_sizes
# ---------------------------------------------------------------------------

def test_load_chrom_sizes_none_returns_empty_dict():
    assert _load_chrom_sizes(None) == {}


def test_load_chrom_sizes_empty_string_returns_empty_dict():
    assert _load_chrom_sizes("") == {}


def test_load_chrom_sizes_valid(tmp_path):
    f = tmp_path / "sizes.tsv"
    f.write_text("chr1\t248956422\nchr2\t242193529\n")
    result = _load_chrom_sizes(str(f))
    assert result == {"chr1": 248956422, "chr2": 242193529}


def test_load_chrom_sizes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_chrom_sizes(str(tmp_path / "nonexistent.tsv"))


# ---------------------------------------------------------------------------
# _validate_grid
# ---------------------------------------------------------------------------

def _make_grid(starts, ends, chrom="chr1"):
    """Build a minimal bedgraph DataFrame with the given bin boundaries."""
    return pd.DataFrame({
        "chrom": [chrom] * len(starts),
        "start": starts,
        "end":   ends,
        "value": [1.0] * len(starts),
    })


def test_validate_grid_clean_input_passes():
    df = _make_grid([0, 10_000, 20_000], [10_000, 20_000, 30_000])
    result = _validate_grid(df, res=10_000, chrom_len=30_000)
    assert len(result) == 3


def test_validate_grid_negative_start_raises():
    df = _make_grid([-1], [9_999])
    with pytest.raises(ValueError, match="negative"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_negative_end_raises():
    df = _make_grid([0], [-1])
    with pytest.raises(ValueError, match="negative"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_zero_length_interval_raises():
    # end == start → zero-length interval
    df = _make_grid([0], [0])
    with pytest.raises(ValueError, match="zero or negative"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_inverted_interval_raises():
    # end < start → negative-length interval
    df = _make_grid([10_000], [5_000])
    with pytest.raises(ValueError, match="zero or negative"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_duplicate_bins_raises():
    df = _make_grid([0, 0], [10_000, 10_000])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_misaligned_start_raises():
    # start=5000 is not a multiple of res=10000
    df = _make_grid([5_000], [15_000])
    with pytest.raises(ValueError, match="[Aa]lign"):
        _validate_grid(df, res=10_000, chrom_len=248_956_422)


def test_validate_grid_overflow_raises():
    # Bin ends at 30000, but chrom_len=20000
    df = _make_grid([0, 10_000, 20_000], [10_000, 20_000, 30_000])
    with pytest.raises(ValueError, match="exceed"):
        _validate_grid(df, res=10_000, chrom_len=20_000)


def test_validate_grid_unsorted_gets_sorted_silently():
    # Unsorted input should be silently sorted, not rejected
    df = _make_grid([20_000, 0, 10_000], [30_000, 10_000, 20_000])
    result = _validate_grid(df, res=10_000, chrom_len=30_000)
    assert list(result["start"]) == [0, 10_000, 20_000]


def test_validate_grid_short_final_bin_allowed():
    # A final bin shorter than res is allowed when it reaches the chromosome end
    df = _make_grid([0, 10_000, 20_000], [10_000, 20_000, 25_000])
    # chrom_len=25000 so the final bin [20000,25000) is legitimately short
    result = _validate_grid(df, res=10_000, chrom_len=25_000)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# _load_zmap_summary
# ---------------------------------------------------------------------------

_SAMPLE_SUMMARY_TSV = """\
# comment line — should be ignored
chrom\tratio\tmean_empty_bin_ratio\tpred_log_N\tpred_log_N_upper\tpred_log_N_lower
chr1\t1.0\t0.1\t1.5\t2.0\t1.0
chr1\t0.5\t0.15\t1.6\t2.1\t1.1
chr2\t1.0\t0.2\t1.8\t2.3\t1.3
"""


def test_load_zmap_summary_extracts_ratio_one_rows(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SAMPLE_SUMMARY_TSV)
    result = _load_zmap_summary(str(f))
    # Only ratio=1.0 rows should be in the result
    assert "chr1" in result
    assert "chr2" in result


def test_load_zmap_summary_correct_values(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SAMPLE_SUMMARY_TSV)
    result = _load_zmap_summary(str(f))
    assert abs(result["chr1"]["ebr"] - 0.1) < 1e-9
    assert abs(result["chr1"]["pred_log_N"] - 1.5) < 1e-9
    assert abs(result["chr2"]["ebr"] - 0.2) < 1e-9
    assert abs(result["chr2"]["pred_log_N"] - 1.8) < 1e-9


def test_load_zmap_summary_comment_lines_excluded(tmp_path):
    f = tmp_path / "summary.tsv"
    f.write_text(_SAMPLE_SUMMARY_TSV)
    result = _load_zmap_summary(str(f))
    # "comment line" would have parsed as a chromosome named "#" — must not appear
    assert "#" not in result


def test_load_zmap_summary_all_comment_returns_empty(tmp_path):
    f = tmp_path / "empty.tsv"
    f.write_text("# only comments\n# nothing else\n")
    result = _load_zmap_summary(str(f))
    assert result == {}
