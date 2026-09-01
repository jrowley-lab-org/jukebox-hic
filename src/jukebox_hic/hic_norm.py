#!/usr/bin/env python
"""
Write a JUKEBOX normalization directly into a ``.hic`` file.

Juicer applies a normalization vector as ``normalized[i][j] = observed[i][j] /
(v_i * v_j)``, so a per-bin weight derived from the JUKEBOX noise track can be
stored alongside KR/VC/SCALE and selected in Juicebox like any other
normalization.  Each resolution present in the file is normalized independently:
noise is a property of a matrix at a given bin size, so a 5 kb map and a 250 kb
map get their own tracks, their own medians, and their own vectors.

The input file is never modified.  ``add_norm_to_hic()`` copies it first and adds
the vector to the copy.

How intense should the correction be?
------------------------------------
That is an open question, so the mapping from noise to weight is a parameter.
With ``scheme="power"`` the weight is

    B_i = (N_i / median(N))^exponent

so ``exponent`` alone controls both strength and direction:

    +1.0   linear      — down-weight noisy bins in proportion to their noise
    +0.5   sqrt        — gentler
    -0.5   inverse sqrt — *up*-weight noisy bins
     0.0   identity    — no-op, useful as a control

``scheme="residual-root"`` keeps the older signed-root-of-log-residual form for
comparison.  Either way the finished vector is clamped and rescaled so its finite
entries have geometric mean 1, which keeps the vector from rescaling the map as a
whole.

Main entry point: ``add_norm_to_hic()``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import reference
from .noise_to_weights import (
    _is_decoy_chrom,
    _load_bedgraph,
    _load_chrom_sizes,
    _reindex_to_full_grid,
    _validate_grid,
)

SCHEMES = ("power", "residual-root")


def bias_from_noise(
    noise_track: np.ndarray,
    scheme: str = "power",
    exponent: float = 1.0,
    max_log2_fold: Optional[float] = 3.0,
    ebr: float = 0.05,
    alpha: float = 0.5,
    p: float = 2.0,
) -> np.ndarray:
    """
    Turn one chromosome's noise track into a Juicer bias vector.

    Parameters
    ----------
    noise_track : per-bin noise from ``noise-bedgraph`` (NaN where unmeasurable).
    scheme : ``"power"`` (default) or ``"residual-root"``.
    exponent : power-scheme exponent. Positive down-weights noisy bins, negative
        up-weights them, 0 is a no-op. Ignored by ``residual-root``.
    max_log2_fold : clamp weights to +/- this many log2 units around 1.0
        (default 3 -> [0.125, 8]). None disables.
    ebr, alpha, p : passed through to the residual-root scheme.

    Returns
    -------
    Bias vector, same length as ``noise_track``, NaN where the noise was NaN.
    The median finite weight is exactly 1 (for a non-zero exponent), and every
    weight is inside the clamp.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    if scheme == "residual-root":
        # pred_log_N is irrelevant here: the residual is re-centred on its own
        # median inside process_normalization_vectors_jukebox().
        return reference.process_normalization_vectors_jukebox(
            noise_track, pred_log_N=0.0, ebr=ebr, p=p, alpha=alpha,
            max_log2_fold=max_log2_fold,
        )

    track = np.asarray(noise_track, dtype=float)
    bad = ~np.isfinite(track) | (track <= 0)
    if bad.all():
        return np.full(track.shape, np.nan)

    med = float(np.median(track[~bad]))
    if not np.isfinite(med) or med <= 0:
        return np.full(track.shape, np.nan)

    # log-space throughout: the noise metric spans many orders of magnitude
    log_ratio = np.zeros(track.shape, dtype=float)
    log_ratio[~bad] = np.log10(track[~bad] / med)
    log_bias = float(exponent) * log_ratio

    # The ratio is already taken against the chromosome median, so the typical bin
    # sits at weight 1 by construction and no further centring is wanted. Anchoring
    # on the median rather than the geometric mean matters here because the noise
    # metric is extremely heavy-tailed -- 1/|autocovariance| runs to 1e9 at
    # degenerate bins. Mean-centring lets that tail drag the whole vector down and
    # silently rescales the map (measured: geometric mean 0.79-0.93, median weight
    # 0.57-0.81). Median-anchoring leaves the bulk at 1 and confines the tail to
    # the clamp, which is applied last so it is a hard bound on what is emitted.
    if max_log2_fold is not None and np.isfinite(max_log2_fold) and max_log2_fold > 0:
        lim = float(max_log2_fold) * np.log10(2.0)
        np.clip(log_bias, -lim, lim, out=log_bias)

    bias = np.power(10.0, log_bias)
    bias[bad] = np.nan
    return bias


def _blocks_for_resolution(
    noise_bedgraph: str,
    res: int,
    name: str,
    chrom_sizes: Dict[str, int],
    skip_decoys: bool,
    **bias_kw,
) -> List[str]:
    """Render every chromosome block for one resolution as Juicer vector text."""
    df = _load_bedgraph(noise_bedgraph)
    out: List[str] = []
    for chrom, group in df.groupby("chrom", sort=True):
        if skip_decoys and _is_decoy_chrom(chrom):
            continue
        chrom_len = chrom_sizes.get(chrom)
        try:
            group = _validate_grid(group.copy(), res, chrom_len)
        except ValueError as exc:
            print(f"[WARN] {chrom} @ {res}: skipping — {exc}")
            continue
        track = _reindex_to_full_grid(group, res, chrom_len)
        if track.size == 0:
            continue
        bias = bias_from_noise(track, **bias_kw)
        lines = [f"vector {name} {chrom} {res} BP"]
        lines.extend("NaN" if np.isnan(v) else repr(float(v)) for v in bias)
        out.append("\n".join(lines))
    return out


def add_norm_to_hic(
    hic_path: str,
    out_path: str,
    juicer_tools: str,
    noise_dir: str,
    resolutions: Optional[Sequence[int]] = None,
    name: str = "JUKEBOX",
    chrom_sizes_path: Optional[str] = None,
    skip_decoys: bool = True,
    java: str = "java",
    java_mem: str = "32g",
    overwrite: bool = False,
    keep_vector: Optional[str] = None,
    **bias_kw,
) -> str:
    """
    Copy *hic_path* to *out_path* and add a JUKEBOX normalization to the copy.

    One vector file is built covering every requested resolution — Juicer's format
    carries the resolution in each block header, so a single ``addNorm`` call adds
    them all, each derived independently from that resolution's own noise track.

    Parameters
    ----------
    hic_path : input .hic. Never modified.
    out_path : destination for the copy that receives the new normalization.
    juicer_tools : path to a juicer_tools jar (1.22.x is known to preserve v8
        files; 2.20 was observed to corrupt them).
    noise_dir : directory of ``{res}.bedgraph`` noise tracks, as written by
        ``noise-bedgraph`` (or by the Rust ``jukebox-rs``).
    resolutions : which resolutions to add. Default: every ``{res}.bedgraph``
        found in *noise_dir*.
    name : normalization label as it will appear in Juicebox.
    bias_kw : forwarded to ``bias_from_noise`` (``scheme``, ``exponent``,
        ``max_log2_fold``, ...).

    Returns
    -------
    Path to the written .hic.
    """
    if not os.path.isfile(hic_path):
        raise FileNotFoundError(hic_path)
    if not os.path.isfile(juicer_tools):
        raise FileNotFoundError(f"juicer_tools jar not found: {juicer_tools}")
    if os.path.abspath(hic_path) == os.path.abspath(out_path):
        raise ValueError("refusing to write the normalization into the input file")
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} exists; pass overwrite=True to replace it"
        )

    if resolutions is None:
        resolutions = []
        for entry in os.listdir(noise_dir):
            stem, ext = os.path.splitext(entry)
            if ext.lower() in (".bedgraph", ".bg") and stem.isdigit():
                resolutions.append(int(stem))
        resolutions.sort()
    if not resolutions:
        raise ValueError(f"no {{res}}.bedgraph tracks found in {noise_dir}")

    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)

    blocks: List[str] = []
    for res in resolutions:
        path = os.path.join(noise_dir, f"{res}.bedgraph")
        if not os.path.isfile(path):
            print(f"[WARN] no noise track for {res} bp — skipping that resolution")
            continue
        got = _blocks_for_resolution(
            path, int(res), name, chrom_sizes, skip_decoys, **bias_kw
        )
        print(f"  {res:>8} bp: {len(got)} chromosome blocks")
        blocks.extend(got)
    if not blocks:
        raise ValueError("no vector blocks were produced")

    print(f"copying {hic_path} -> {out_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    shutil.copyfile(hic_path, out_path)

    vec_path = keep_vector
    tmp = None
    if vec_path is None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jukebox.vector", delete=False)
        vec_path = tmp.name
        tmp.close()
    with open(vec_path, "w") as fh:
        fh.write("\n".join(blocks))
        fh.write("\n")

    cmd = [java, f"-Xmx{java_mem}", "-jar", juicer_tools, "addNorm", out_path, vec_path]
    print("  " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"juicer_tools addNorm failed ({proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    if tmp is not None:
        os.unlink(vec_path)
    print(f"added '{name}' normalization to {out_path}")
    return out_path
