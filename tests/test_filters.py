"""
Tests for src/jukebox_hic/filters.py.

Covers Kneedle elbow detection, interval merging, and the detect_elbow_thresholds
integration path using synthetic bedgraph files written to tmp_path.
"""
import numpy as np
import pandas as pd
import pytest

from jukebox_hic.filters import (
    _kneedle,
    _merge_intervals,
    detect_lower_elbow,
    detect_upper_elbow,
    detect_elbow_thresholds,
)


# ---------------------------------------------------------------------------
# _kneedle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform", ["none", "sqrt", "log1p", "cbrt"])
def test_kneedle_supported_transforms_do_not_crash(transform):
    values = np.linspace(0.0, 10.0, 50)
    idx = _kneedle(values, smooth_sigma=2.0, transform=transform)
    assert 0 <= idx < len(values)


def test_kneedle_returns_int():
    values = np.linspace(0.0, 5.0, 20)
    idx = _kneedle(values, smooth_sigma=2.0)
    assert isinstance(idx, int)


def test_kneedle_flat_input_does_not_crash():
    # All-equal array: span < 1e-12, should return last index
    values = np.ones(10)
    idx = _kneedle(values, smooth_sigma=2.0)
    assert idx == len(values) - 1


# ---------------------------------------------------------------------------
# detect_upper_elbow
# ---------------------------------------------------------------------------

def test_detect_upper_elbow_valid_index():
    values = np.sort(np.random.default_rng(0).exponential(scale=2.0, size=200))
    idx = detect_upper_elbow(values)
    assert 0 <= idx < len(values)


def test_detect_upper_elbow_monotonic_input():
    values = np.linspace(0.0, 1.0, 100)
    idx = detect_upper_elbow(values)
    assert 0 <= idx < len(values)


def test_detect_upper_elbow_flat_input():
    values = np.ones(50)
    idx = detect_upper_elbow(values)
    assert 0 <= idx < len(values)


# ---------------------------------------------------------------------------
# detect_lower_elbow
# ---------------------------------------------------------------------------

def test_detect_lower_elbow_valid_index():
    values = np.sort(np.random.default_rng(1).exponential(scale=2.0, size=200))
    idx = detect_lower_elbow(values)
    assert 0 <= idx < len(values)


def test_detect_lower_elbow_monotonic_input():
    values = np.linspace(0.0, 1.0, 100)
    idx = detect_lower_elbow(values)
    assert 0 <= idx < len(values)


# ---------------------------------------------------------------------------
# _merge_intervals
# ---------------------------------------------------------------------------

def test_merge_intervals_overlapping():
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [0,     5000,  8000],
        "end":   [6000,  9000, 12000],
    })
    result = _merge_intervals(df)
    assert len(result) == 1
    assert result.iloc[0]["start"] == 0
    assert result.iloc[0]["end"] == 12000


def test_merge_intervals_adjacent():
    # Abutting intervals should merge
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [0,     10000],
        "end":   [10000, 20000],
    })
    result = _merge_intervals(df)
    assert len(result) == 1
    assert result.iloc[0]["end"] == 20000


def test_merge_intervals_non_overlapping():
    # Gap between intervals → kept separate
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [0,     20000],
        "end":   [10000, 30000],
    })
    result = _merge_intervals(df)
    assert len(result) == 2


def test_merge_intervals_empty_input():
    df = pd.DataFrame(columns=["chrom", "start", "end"])
    result = _merge_intervals(df)
    assert list(result.columns) == ["chrom", "start", "end"]
    assert len(result) == 0


def test_merge_intervals_multi_chrom():
    # Each chromosome is merged independently
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chr2"],
        "start": [0,     5000,   0,      5000],
        "end":   [6000, 10000,   4000,   9000],
    })
    result = _merge_intervals(df)
    chr1 = result[result["chrom"] == "chr1"]
    chr2 = result[result["chrom"] == "chr2"]
    # chr1: [0,6000) and [5000,10000) overlap → merged to [0,10000)
    assert len(chr1) == 1
    assert chr1.iloc[0]["end"] == 10000
    # chr2: [0,4000) and [5000,9000) are disjoint → kept separate
    assert len(chr2) == 2


def test_merge_intervals_coordinates_are_int():
    df = pd.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end":   [10000],
    })
    result = _merge_intervals(df)
    assert result["start"].dtype in (np.int64, np.int32, int)
    assert result["end"].dtype in (np.int64, np.int32, int)


# ---------------------------------------------------------------------------
# detect_elbow_thresholds (integration — uses synthetic bedgraph files)
# ---------------------------------------------------------------------------

def _write_bedgraph(path, chrom, values, res=10_000):
    """Write a minimal 4-column bedgraph to *path*."""
    lines = []
    for i, v in enumerate(values):
        lines.append(f"{chrom}\t{i*res}\t{(i+1)*res}\t{v}")
    path.write_text("\n".join(lines) + "\n")


def test_detect_elbow_thresholds_returns_correct_columns(tmp_path):
    density = tmp_path / "density.bedgraph"
    noise   = tmp_path / "noise.bedgraph"
    # 20 bins is enough for the search_frac logic to work cleanly
    d_vals = list(np.linspace(0.5, 5.0, 20))
    n_vals = list(np.linspace(0.3, 8.0, 20))
    _write_bedgraph(density, "chr1", d_vals)
    _write_bedgraph(noise,   "chr1", n_vals)

    df, curves = detect_elbow_thresholds(str(density), str(noise))

    assert "chrom" in df.columns
    assert "density_upper_value" in df.columns
    assert "noise_upper_value" in df.columns
    assert "n_bins" in df.columns


def test_detect_elbow_thresholds_one_row_per_chrom(tmp_path):
    density = tmp_path / "density.bedgraph"
    noise   = tmp_path / "noise.bedgraph"
    d_vals = list(np.linspace(0.5, 5.0, 20))
    n_vals = list(np.linspace(0.3, 8.0, 20))
    _write_bedgraph(density, "chr1", d_vals)
    _write_bedgraph(noise,   "chr1", n_vals)

    df, _ = detect_elbow_thresholds(str(density), str(noise))
    assert len(df) == 1
    assert df.iloc[0]["chrom"] == "chr1"


def test_detect_elbow_thresholds_curves_keyed_by_chrom(tmp_path):
    density = tmp_path / "density.bedgraph"
    noise   = tmp_path / "noise.bedgraph"
    _write_bedgraph(density, "chr1", list(np.linspace(0.5, 5.0, 20)))
    _write_bedgraph(noise,   "chr1", list(np.linspace(0.3, 8.0, 20)))

    _, curves = detect_elbow_thresholds(str(density), str(noise))
    assert "chr1" in curves
    # Each entry is (d_sorted, n_sorted, d_lo_idx, d_hi_idx, n_lo_idx, n_hi_idx)
    assert len(curves["chr1"]) == 6


def test_detect_elbow_thresholds_skips_chrom_with_too_few_bins(tmp_path):
    # chr2 has only 3 finite bins — below the 4-bin minimum, so it should be skipped
    density = tmp_path / "density.bedgraph"
    noise   = tmp_path / "noise.bedgraph"
    content_d = (
        "chr1\t0\t10000\t0.5\nchr1\t10000\t20000\t1.0\nchr1\t20000\t30000\t1.5\n"
        "chr1\t30000\t40000\t2.0\nchr1\t40000\t50000\t2.5\n"
        "chr2\t0\t10000\t0.1\nchr2\t10000\t20000\t0.2\nchr2\t20000\t30000\t0.3\n"
    )
    density.write_text(content_d)
    noise.write_text(content_d)

    df, _ = detect_elbow_thresholds(str(density), str(noise))
    assert "chr1" in df["chrom"].values
    assert "chr2" not in df["chrom"].values


# ---------------------------------------------------------------------------
# Blacklist flagging rules
# ---------------------------------------------------------------------------

def _tiny_tracks(tmp_path):
    """Six bins: one unmappable, one extreme-high noise, one extreme-low noise,
    one extreme-high density, and two ordinary."""
    noise = tmp_path / "n.bedgraph"
    dens = tmp_path / "d.bedgraph"
    #        bin      noise     density
    rows = [(0, "nan", "nan"),      # unmappable
            (1, "1e6", "500"),      # extreme HIGH noise
            (2, "1e-6", "500"),     # extreme LOW noise (extreme order)
            (3, "10", "1e6"),       # extreme HIGH density
            (4, "10", "500"),       # ordinary
            (5, "11", "510")]       # ordinary
    noise.write_text("".join(f"chr1 {i*10000} {(i+1)*10000} {n}\n" for i, n, _ in rows))
    dens.write_text("".join(f"chr1 {i*10000} {(i+1)*10000} {d}\n" for i, _, d in rows))
    return str(noise), str(dens)


def _flagged(tmp_path, rule):
    import pandas as pd
    from jukebox_hic.filters import build_blacklist_from_elbow_thresholds
    n, d = _tiny_tracks(tmp_path)
    th = pd.DataFrame([{
        "chrom": "chr1", "n_bins": 6,
        "density_lower_value": 1.0, "density_lower_pct": 0.0,
        "density_upper_value": 1e5, "density_upper_pct": 100.0,
        "noise_lower_value": 1e-3, "noise_lower_pct": 0.0,
        "noise_upper_value": 1e5, "noise_upper_pct": 100.0,
    }])
    out = tmp_path / f"{rule}.bed"
    build_blacklist_from_elbow_thresholds(
        density_bedgraph=d, noise_bedgraph=n, output_path=str(out),
        thresholds_df=th, rule=rule,
    )
    bins = set()
    for line in open(out):
        f = line.split()
        if len(f) >= 3:
            bins.update(range(int(f[1]) // 10000, int(f[2]) // 10000))
    return bins


def test_noise_high_rule_takes_only_the_disorder_tail(tmp_path):
    """
    The recommended rule: unmappable bins plus extreme-HIGH noise, and nothing
    else. The low tail flags unusually *smooth* bins and belongs in neither a
    noise blacklist nor the loop filter.
    """
    got = _flagged(tmp_path, "noise-high")
    assert got == {0, 1}, f"expected the unmappable bin and the high-noise bin, got {got}"


def test_legacy_rules_pull_in_the_low_tail_and_density(tmp_path):
    union = _flagged(tmp_path, "union")
    assert 2 in union, "union should flag the extreme-LOW noise bin"
    assert 3 in union, "union should flag the extreme-HIGH density bin"


def test_mask_rule_is_the_unmappability_mask_alone(tmp_path):
    assert _flagged(tmp_path, "mask") == {0}


def test_unmappable_bins_are_flagged_under_every_rule(tmp_path):
    for rule in ("noise-high", "union", "intersection", "mask"):
        assert 0 in _flagged(tmp_path, rule), f"{rule} dropped the unmappable bin"


def test_unknown_rule_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="unknown blacklist rule"):
        _flagged(tmp_path, "nonsense")
