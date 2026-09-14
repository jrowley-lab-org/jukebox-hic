"""
Tests for src/jukebox_hic/filters.py.

Covers Kneedle elbow detection, interval merging, and the detect_elbow_thresholds
integration path using synthetic bedgraph files written to tmp_path.
"""
import numpy as np
import pandas as pd
import pytest

from jukebox_hic.filters import (
    _density_conditioned_residuals,
    _robust_residual_threshold,
    _kneedle,
    _merge_intervals,
    _stratified_noise_thresholds,
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


# ---------------------------------------------------------------------------
# _stratified_noise_thresholds / "density-stratified" rule
# ---------------------------------------------------------------------------

def _two_density_populations(seed=0):
    """
    Synthetic genome with two density populations, each with its own realistic
    (exponential-tailed) noise distribution:

    - "well_sampled": 270 bins, high density, low baseline noise.
    - "sparse": 30 bins, low density, noise uniformly elevated (representing
      normalization-amplified noise typical of a sparsely-sequenced-but-real
      region) but with its own natural spread, not a single flat value.

    The sparse population is sized to exactly 10% of the genome (matching the
    default noise_upper_frac), so a *global* elbow search sees the entire
    sparse population as its top-fraction search window, while a *per-stratum*
    search (with density_strata chosen so this population is its own stratum)
    additionally restricts to only the top fraction *within* that population.
    """
    rng = np.random.default_rng(seed)
    well_sampled_density = np.full(270, 10_000.0)
    well_sampled_noise = rng.exponential(scale=0.3, size=270)

    sparse_density = np.full(30, 100.0)
    sparse_noise = 5.0 + rng.exponential(scale=0.4, size=30)

    density = np.concatenate([well_sampled_density, sparse_density])
    noise = np.concatenate([well_sampled_noise, sparse_noise])
    sparse_slice = slice(270, 300)
    return density, noise, sparse_slice


def test_density_stratified_flags_fewer_sparse_bins_than_global_noise_high():
    """
    The core regression test for the PI's complaint: bins whose noise is
    merely typical for their (low) density should not be penalized just for
    being low-density. A global noise-high elbow, searched only within its
    top fraction, ends up searching across the *entire* sparse population in
    this construction (see _two_density_populations) and over-flags it;
    density-stratification narrows that fraction down within the sparse
    stratum itself, flagging fewer of its "typical for this density" bins.
    """
    density, noise, sparse_slice = _two_density_populations()

    # Mirrors what the "noise-high" rule computes: one global elbow.
    sorted_noise = np.sort(noise)
    global_idx = detect_upper_elbow(sorted_noise, 0.10, 10.0, "sqrt")
    global_threshold = sorted_noise[global_idx]
    noise_high_flagged = noise >= global_threshold

    # density_strata=10 splits the 300 bins into groups of 30, aligned with
    # the sparse population's own size, so it forms exactly one stratum.
    thresholds = _stratified_noise_thresholds(
        density, noise, n_strata=10,
        noise_upper_frac=0.10, smooth_sigma=10.0, noise_transform="sqrt",
    )
    stratified_flagged = noise >= thresholds

    n_sparse_flagged_by_noise_high = int(noise_high_flagged[sparse_slice].sum())
    n_sparse_flagged_by_stratified = int(stratified_flagged[sparse_slice].sum())

    assert n_sparse_flagged_by_noise_high > n_sparse_flagged_by_stratified, (
        f"expected density-stratification to flag fewer of the 30 sparse bins "
        f"than plain noise-high (noise-high flagged {n_sparse_flagged_by_noise_high}, "
        f"density-stratified flagged {n_sparse_flagged_by_stratified})"
    )
    # And density-stratification should not be flagging most of a population
    # whose noise is, internally, unremarkable for its own density.
    assert n_sparse_flagged_by_stratified < 15, (
        f"density-stratified flagged {n_sparse_flagged_by_stratified}/30 sparse bins "
        "— expected it to isolate a minority, not the bulk of the population"
    )


def test_density_stratified_still_flags_a_true_local_outlier():
    """
    A bin whose noise is anomalous *even relative to its own density peers*
    should still be flagged — density-stratification narrows the comparison
    group, it doesn't disable flagging altogether.
    """
    density, noise, sparse_slice = _two_density_populations()
    # Inject one bin far outside the sparse population's own range (~5-6.5).
    noise[270] = 500.0

    thresholds = _stratified_noise_thresholds(
        density, noise, n_strata=10,
        noise_upper_frac=0.10, smooth_sigma=10.0, noise_transform="sqrt",
    )
    assert noise[270] >= thresholds[270]


def test_density_stratified_rule_integration(tmp_path):
    """
    Same population design, run through the full build_blacklist_from_elbow_thresholds
    pipeline (bedgraph files in, BED out) rather than calling the helper directly,
    to confirm the "density-stratified" rule is wired up end-to-end.
    """
    density, noise, sparse_slice = _two_density_populations()
    density_path = tmp_path / "d.bedgraph"
    noise_path = tmp_path / "n.bedgraph"
    _write_bedgraph(density_path, "chr1", density)
    _write_bedgraph(noise_path, "chr1", noise)

    from jukebox_hic.filters import build_blacklist_from_elbow_thresholds
    th = pd.DataFrame([{
        "chrom": "chr1", "n_bins": len(density),
        "density_lower_value": 0.0, "density_lower_pct": 0.0,
        "density_upper_value": 1e9, "density_upper_pct": 100.0,
        "noise_lower_value": -1e9, "noise_lower_pct": 0.0,
        "noise_upper_value": 1e9, "noise_upper_pct": 100.0,
    }])
    out = tmp_path / "density_stratified.bed"
    build_blacklist_from_elbow_thresholds(
        density_bedgraph=str(density_path), noise_bedgraph=str(noise_path),
        output_path=str(out), thresholds_df=th, rule="density-stratified",
        density_strata=10,
    )
    flagged_bins = set()
    for line in open(out):
        f = line.split()
        if len(f) >= 3:
            flagged_bins.update(range(int(f[1]) // 10_000, int(f[2]) // 10_000))
    # Should flag some, but not most, of the 30 sparse bins (indices 270-299).
    n_flagged_sparse = len(flagged_bins & set(range(270, 300)))
    assert 0 < n_flagged_sparse < 15


# ---------------------------------------------------------------------------
# _density_conditioned_residuals / "density-residual" rule
# ---------------------------------------------------------------------------

def _on_trend_genome(seed=0, n=600):
    """
    Synthetic genome where noise follows a clean power-law in density, with
    realistic scatter but no genuine outliers. Every bin sits on the
    density→noise trend, so a density-conditioned rule should find almost
    nothing to flag here.
    """
    rng = np.random.default_rng(seed)
    density = 10.0 ** rng.uniform(1.0, 4.0, size=n)         # 10 … 10,000
    # noise decreases with density (deeper bins are less noisy), times lognormal scatter
    noise = 50.0 * density ** -0.5 * np.exp(rng.normal(0.0, 0.15, size=n))
    return density, noise


def test_density_residual_leaves_on_trend_bins_alone():
    """
    The regression test for the failure that killed "density-stratified": a
    population sitting on the density→noise trend must not be flagged wholesale
    just because it exists. Residuals here are all small, so the flagged
    fraction should be far below the 10% search fraction that a per-stratum
    elbow would mechanically hand back.
    """
    density, noise = _on_trend_genome()
    residual = _density_conditioned_residuals(density, noise)

    # The trend is tracked well enough that residuals stay small and centred.
    assert abs(float(np.median(residual))) < 0.05
    assert float(np.std(residual)) < 0.25

    n_flagged = int((residual >= _robust_residual_threshold(residual, 4.0)).sum())

    # Contrast with the per-stratum elbow, which hands back a share of every
    # stratum no matter how well-behaved the data is.
    stratified = _stratified_noise_thresholds(density, noise, 5, 0.10, 10.0, "sqrt")
    n_stratified = int((noise >= stratified).sum())

    assert n_flagged < n_stratified, (
        f"density-residual flagged {n_flagged} on-trend bins, density-stratified "
        f"flagged {n_stratified} — the residual rule should be the conservative one"
    )
    assert n_flagged == 0, (
        f"flagged {n_flagged}/{len(density)} on-trend bins — a population with no "
        "genuine outliers should contribute nothing at all to the blacklist"
    )


def test_density_residual_flags_bins_noisy_for_their_density():
    """
    A bin that is noisy relative to the trend at its own density must still be
    flagged, including one whose absolute noise is unremarkable because it sits
    at high density where the expected noise is low.
    """
    density, noise = _on_trend_genome()
    # Two injected outliers, each 10x the noise expected at its own density.
    low_density_idx = int(np.argmin(density))
    high_density_idx = int(np.argmax(density))
    noise[low_density_idx] *= 10.0
    noise[high_density_idx] *= 10.0

    residual = _density_conditioned_residuals(density, noise)
    flagged = residual >= _robust_residual_threshold(residual, 4.0)

    assert flagged[low_density_idx]
    assert flagged[high_density_idx], (
        "a bin 10x noisier than its density predicts was missed — its absolute "
        "noise is low, which is exactly what a density-blind rule gets wrong"
    )


def test_density_residual_ignores_absolute_noise_level():
    """
    Sparse regions are noisier in absolute terms. A density-conditioned rule
    should not flag them for that alone, which is the PI's original objection.
    """
    density, noise = _on_trend_genome()
    residual = _density_conditioned_residuals(density, noise)

    # The lowest-density decile has much higher raw noise than the highest, but
    # after conditioning its residuals should be no larger.
    order = np.argsort(density)
    lowest = order[: len(order) // 10]
    highest = order[-len(order) // 10:]

    assert np.median(noise[lowest]) > 5 * np.median(noise[highest]), "test setup"
    assert abs(float(np.median(residual[lowest]))) < 0.1
    assert abs(float(np.median(residual[highest]))) < 0.1


def test_density_residual_rule_integration(tmp_path):
    """Run the rule end-to-end through the BED-writing pipeline."""
    density, noise = _on_trend_genome()
    noise[int(np.argmax(density))] *= 20.0      # one unambiguous outlier

    density_path = tmp_path / "d.bedgraph"
    noise_path = tmp_path / "n.bedgraph"
    _write_bedgraph(density_path, "chr1", density)
    _write_bedgraph(noise_path, "chr1", noise)

    from jukebox_hic.filters import build_blacklist_from_elbow_thresholds
    th = pd.DataFrame([{
        "chrom": "chr1", "n_bins": len(density),
        "density_lower_value": 0.0, "density_lower_pct": 0.0,
        "density_upper_value": 1e9, "density_upper_pct": 100.0,
        "noise_lower_value": -1e9, "noise_lower_pct": 0.0,
        "noise_upper_value": 1e9, "noise_upper_pct": 100.0,
    }])
    out = tmp_path / "density_residual.bed"
    build_blacklist_from_elbow_thresholds(
        density_bedgraph=str(density_path), noise_bedgraph=str(noise_path),
        output_path=str(out), thresholds_df=th, rule="density-residual",
    )
    flagged_bins = set()
    for line in open(out):
        f = line.split()
        if len(f) >= 3:
            flagged_bins.update(range(int(f[1]) // 10_000, int(f[2]) // 10_000))

    assert int(np.argmax(density)) in flagged_bins
    # On otherwise on-trend data the blacklist should stay small.
    assert len(flagged_bins) < 0.05 * len(density)
