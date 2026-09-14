"""
Tests for src/jukebox_hic/noise_gradient.py.

The load-bearing test here is test_threshold_at_k_reproduces_density_residual_blacklist:
the gradient and the "density-residual" blacklist rule are meant to be two views
of one quantity, and that only holds if both derive their robust centre and
scale identically.
"""
import numpy as np
import pandas as pd
import pytest

from jukebox_hic.filters import build_blacklist_from_elbow_thresholds
from jukebox_hic.noise_gradient import (
    _infer_res,
    build_noise_gradient,
    compute_chrom_gradient,
)

_RES = 10_000


def _write_bedgraph(path, chrom, values, res=_RES, starts=None):
    """Write a minimal 4-column bedgraph; `starts` allows deliberate gaps."""
    if starts is None:
        starts = [i * res for i in range(len(values))]
    lines = [f"{chrom}\t{s}\t{s + res}\t{v}" for s, v in zip(starts, values)]
    path.write_text("\n".join(lines) + "\n")


def _on_trend_genome(seed=0, n=600):
    """
    Noise follows a clean power law in density with realistic scatter and no
    genuine outliers — every bin sits on the density→noise trend.
    """
    rng = np.random.default_rng(seed)
    density = 10.0 ** rng.uniform(1.0, 4.0, size=n)
    noise = 50.0 * density ** -0.5 * np.exp(rng.normal(0.0, 0.15, size=n))
    return density, noise


def _tracks(tmp_path, density, noise, chrom="chr1", starts=None):
    d = tmp_path / "d.bedgraph"
    n = tmp_path / "n.bedgraph"
    _write_bedgraph(d, chrom, density, starts=starts)
    _write_bedgraph(n, chrom, noise, starts=starts)
    return str(d), str(n)


# ---------------------------------------------------------------------------
# The z scale itself
# ---------------------------------------------------------------------------

def test_z_is_centred_and_unit_scaled():
    """On-trend data should produce z centred near 0 with robust scale near 1."""
    density, noise = _on_trend_genome()
    frame = compute_chrom_gradient(density, noise)
    z = frame["z_self"].to_numpy()

    assert abs(float(np.median(z))) < 0.1
    mad = float(np.median(np.abs(z - np.median(z))))
    assert 0.8 < 1.4826 * mad < 1.25


def test_sparse_region_not_penalised():
    """
    The whole point of conditioning on density: sparse bins are noisier in
    absolute terms but must not score higher for that reason alone.
    """
    density, noise = _on_trend_genome()
    frame = compute_chrom_gradient(density, noise)

    order = np.argsort(density)
    lowest = order[: len(order) // 10]
    highest = order[-len(order) // 10:]

    assert np.median(noise[lowest]) > 5 * np.median(noise[highest]), "test setup"
    z = frame["z_max"].to_numpy()
    assert abs(float(np.median(z[lowest])) - float(np.median(z[highest]))) < 0.5


# ---------------------------------------------------------------------------
# The consistency contract with the density-residual blacklist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [3.0, 4.0, 5.0])
def test_threshold_at_k_reproduces_density_residual_blacklist(tmp_path, k):
    """
    Selecting gradient bins with z_self >= k must give exactly the bins the
    density-residual rule flags at density_residual_k = k. If this breaks, the
    gradient and the blacklist are telling different stories about the same data.
    """
    density, noise = _on_trend_genome()
    # Inject outliers so both paths actually select something.
    for idx in (17, 140, 411):
        noise[idx] *= 25.0
    d_path, n_path = _tracks(tmp_path, density, noise)

    gradient = build_noise_gradient(d_path, n_path, str(tmp_path / f"grad{k}"), res=_RES)
    from_gradient = set(
        (gradient.loc[gradient["z_self"] >= k, "start"] // _RES).astype(int)
    )

    thresholds = pd.DataFrame([{
        "chrom": "chr1", "n_bins": len(density),
        "density_lower_value": 0.0, "density_lower_pct": 0.0,
        "density_upper_value": 1e9, "density_upper_pct": 100.0,
        "noise_lower_value": -1e9, "noise_lower_pct": 0.0,
        "noise_upper_value": 1e9, "noise_upper_pct": 100.0,
    }])
    bed = tmp_path / f"bl{k}.bed"
    build_blacklist_from_elbow_thresholds(
        density_bedgraph=d_path, noise_bedgraph=n_path, output_path=str(bed),
        thresholds_df=thresholds, rule="density-residual", density_residual_k=k,
    )
    from_blacklist = set()
    for line in open(bed):
        f = line.split()
        if len(f) >= 3:
            from_blacklist.update(range(int(f[1]) // _RES, int(f[2]) // _RES))

    assert from_gradient == from_blacklist, (
        f"k={k}: gradient selected {len(from_gradient)} bins, blacklist "
        f"{len(from_blacklist)}; symmetric difference "
        f"{sorted(from_gradient ^ from_blacklist)[:10]}"
    )


def test_tied_residuals_do_not_explode_the_z_scale():
    """
    Regression for the K562 chrY blowup. A near-empty chromosome has few
    measurable bins whose counts take a handful of discrete values, so over half
    the residuals land on exactly 0.0 and the MAD becomes floating-point dust
    rather than zero. Dividing by that produced z ~ 1e15.
    """
    rng = np.random.default_rng(0)
    n = 400
    # Discrete, heavily tied noise with a real spread in a minority of bins.
    noise = np.full(n, 9950.25)
    noise[:40] = 39801.0
    noise[40:60] = rng.choice([4422.33, 2487.56, 1592.04], size=20)
    density = np.full(n, 100.0)

    frame = compute_chrom_gradient(density, noise)
    z = frame["z_self"].to_numpy()
    z = z[np.isfinite(z)]

    assert z.size > 0
    assert np.abs(z).max() < 100.0, f"z blew up to {np.abs(z).max():.3g}"


def test_fully_degenerate_residuals_flag_nothing_absurd():
    """All-identical input has no spread at all; z must stay finite-or-inf, not dust-driven."""
    density = np.full(200, 100.0)
    noise = np.full(200, 5.0)
    frame = compute_chrom_gradient(density, noise)
    z = frame["z_self"].to_numpy()
    # Every residual is identical, so nothing is above the centre.
    assert not np.any(z[np.isfinite(z)] > 0)


# ---------------------------------------------------------------------------
# Neighbourhood behaviour
# ---------------------------------------------------------------------------

def test_neighbour_columns_are_shifted_by_one_bin():
    density, noise = _on_trend_genome()
    noise[300] *= 30.0
    frame = compute_chrom_gradient(density, noise)

    assert frame["z_plus1"].iloc[299] == pytest.approx(frame["z_self"].iloc[300])
    assert frame["z_minus1"].iloc[301] == pytest.approx(frame["z_self"].iloc[300])


def test_spike_raises_the_summary_of_both_neighbours():
    """The reason neighbours are included at all: a noisy bin taints its flanks."""
    density, noise = _on_trend_genome()
    noise[300] *= 30.0
    frame = compute_chrom_gradient(density, noise)

    for neighbour in (299, 301):
        assert frame["z_max"].iloc[neighbour] == pytest.approx(frame["z_self"].iloc[300])
        assert frame["z_max"].iloc[neighbour] > frame["z_self"].iloc[neighbour]


def test_first_and_last_bin_have_one_neighbour():
    density, noise = _on_trend_genome()
    frame = compute_chrom_gradient(density, noise)

    assert np.isnan(frame["z_minus1"].iloc[0])
    assert np.isnan(frame["z_plus1"].iloc[-1])
    assert frame["n_finite"].iloc[0] == 2
    assert frame["n_finite"].iloc[-1] == 2
    assert np.isfinite(frame["z_max"].iloc[0])


def test_nan_neighbour_is_skipped_not_poisoning():
    density, noise = _on_trend_genome()
    noise[200] = np.nan                      # unmappable bin
    frame = compute_chrom_gradient(density, noise)

    # Its neighbours lose one contributor but still get a finite summary.
    assert frame["n_finite"].iloc[199] == 2
    assert np.isfinite(frame["z_max"].iloc[199])
    assert np.isfinite(frame["z_max"].iloc[201])


def test_unmappable_centre_is_marked_but_still_reports_neighbourhood():
    density, noise = _on_trend_genome()
    noise[200] = np.nan
    frame = compute_chrom_gradient(density, noise)

    assert frame["unmappable"].iloc[200] == 1
    assert np.isnan(frame["z_self"].iloc[200])
    # z_max comes from the two flanking bins, so the locus is still scored.
    assert frame["n_finite"].iloc[200] == 2
    assert np.isfinite(frame["z_max"].iloc[200])
    assert frame["unmappable"].iloc[199] == 0


# ---------------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------------

def test_gapped_bedgraph_uses_bin_index_not_row_order(tmp_path):
    """
    A missing bin must become a NaN hole, not close up. If the gap closed, the
    bins either side would wrongly become each other's neighbours.
    """
    density, noise = _on_trend_genome(n=300)
    keep = [i for i in range(300) if i != 150]
    starts = [i * _RES for i in keep]
    d_path, n_path = _tracks(tmp_path, density[keep], noise[keep], starts=starts)

    out = build_noise_gradient(d_path, n_path, str(tmp_path / "g"), res=_RES)
    bins = dict(zip((out["start"] // _RES).astype(int), range(len(out))))

    assert out["unmappable"].iloc[bins[150]] == 1
    assert np.isnan(out["z_self"].iloc[bins[150]])
    # Bin 149's +1 neighbour is the hole, not bin 151.
    assert np.isnan(out["z_plus1"].iloc[bins[149]])
    assert out["n_finite"].iloc[bins[149]] == 2


def test_infer_res_uses_modal_width():
    df = pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "start": [0, 10_000, 20_000, 30_000],
        "end":   [10_000, 20_000, 30_000, 34_321],   # short final bin
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    assert _infer_res(df) == 10_000


def test_short_chrom_skipped(tmp_path):
    density, noise = _on_trend_genome(n=300)
    d = tmp_path / "d.bedgraph"
    n = tmp_path / "n.bedgraph"
    # chr1 is usable; chr2 has only 3 bins, below the 4-bin minimum.
    d.write_text(
        "\n".join(f"chr1\t{i*_RES}\t{(i+1)*_RES}\t{v}" for i, v in enumerate(density))
        + "\n" + "\n".join(f"chr2\t{i*_RES}\t{(i+1)*_RES}\t{i+1.0}" for i in range(3)) + "\n"
    )
    n.write_text(
        "\n".join(f"chr1\t{i*_RES}\t{(i+1)*_RES}\t{v}" for i, v in enumerate(noise))
        + "\n" + "\n".join(f"chr2\t{i*_RES}\t{(i+1)*_RES}\t{i+1.0}" for i in range(3)) + "\n"
    )
    out = build_noise_gradient(str(d), str(n), str(tmp_path / "g"), res=_RES)
    assert set(out["chrom"]) == {"chr1"}


def test_decoy_chrom_excluded(tmp_path):
    density, noise = _on_trend_genome(n=300)
    d = tmp_path / "d.bedgraph"
    n = tmp_path / "n.bedgraph"
    for path, vals in ((d, density), (n, noise)):
        rows = [f"chr1\t{i*_RES}\t{(i+1)*_RES}\t{v}" for i, v in enumerate(vals)]
        rows += [f"chr1_KI270762v1_alt\t{i*_RES}\t{(i+1)*_RES}\t{v}"
                 for i, v in enumerate(vals)]
        path.write_text("\n".join(rows) + "\n")

    out = build_noise_gradient(str(d), str(n), str(tmp_path / "g"), res=_RES)
    assert set(out["chrom"]) == {"chr1"}


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def test_bedgraph_is_four_finite_columns_without_header(tmp_path):
    density, noise = _on_trend_genome()
    noise[200] = np.nan
    d_path, n_path = _tracks(tmp_path, density, noise)
    out_dir = tmp_path / "g"
    build_noise_gradient(d_path, n_path, str(out_dir), res=_RES)

    lines = (out_dir / "gradient.bedgraph").read_text().strip().split("\n")
    assert lines, "bedgraph is empty"
    for line in lines[:20]:
        fields = line.split()
        assert len(fields) == 4
        assert np.isfinite(float(fields[3]))
    # No header: the first field must be a chromosome name, not "chrom".
    assert lines[0].split()[0] == "chr1"


def test_tsv_keeps_every_bin_including_unmappable(tmp_path):
    density, noise = _on_trend_genome()
    noise[200] = np.nan
    d_path, n_path = _tracks(tmp_path, density, noise)
    out_dir = tmp_path / "g"
    build_noise_gradient(d_path, n_path, str(out_dir), res=_RES)

    tsv = pd.read_csv(out_dir / "gradient.tsv", sep="\t")
    assert len(tsv) == len(density)
    assert tsv["unmappable"].sum() == 1
    assert "z_minus1" in tsv.columns and "z_plus1" in tsv.columns


def test_summary_choice_changes_the_bedgraph_column(tmp_path):
    density, noise = _on_trend_genome()
    noise[300] *= 30.0
    d_path, n_path = _tracks(tmp_path, density, noise)

    values = {}
    for summary in ("max", "self"):
        out_dir = tmp_path / summary
        build_noise_gradient(d_path, n_path, str(out_dir), res=_RES, summary=summary)
        rows = (out_dir / "gradient.bedgraph").read_text().strip().split("\n")
        values[summary] = {int(r.split()[1]) // _RES: float(r.split()[3]) for r in rows}

    # Bin 299 neighbours the spike: high under "max", ordinary under "self".
    assert values["max"][299] > 5.0
    assert values["self"][299] < values["max"][299]


def test_unknown_summary_is_rejected(tmp_path):
    density, noise = _on_trend_genome(n=300)
    d_path, n_path = _tracks(tmp_path, density, noise)
    with pytest.raises(ValueError, match="unknown summary"):
        build_noise_gradient(d_path, n_path, str(tmp_path / "g"), res=_RES, summary="median")
