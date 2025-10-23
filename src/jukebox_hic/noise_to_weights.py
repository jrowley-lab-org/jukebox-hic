#!/usr/bin/env python
"""Transform noise bedGraphs into normalization weight vectors."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class NoiseWeightConfig:
    tau: float = 0.7
    kappa: float = 0.12
    gain: float = 0.8
    shrink_lambda: float = 0.4
    sigma_target: Optional[float] = None
    eps_floor: float = 1e-3
    eps_scale: float = 0.1
    w_min: float = 0.5
    w_max: float = 2.0
    max_clip_rate: float = 0.10
    min_neg_spearman: float = -0.6
    target_cv_reduction: float = 0.2
    stability_ratios: Tuple[float, ...] = (0.9, 0.7, 0.5)
    use_depth_heuristics: bool = False
    depth_reference: float = 3e7
    random_seed: Optional[int] = None
    skip_decoys: bool = True
    high_nan_eps_multiplier: float = 2.0
    nan_fraction_threshold: float = 0.5

    def stability_enabled(self) -> bool:
        return any(0.0 < r < 1.0 for r in self.stability_ratios)


def _resolve_bedgraph_path(path_or_dir: str, res: int) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"No such file or directory: {path_or_dir}")
    candidates = [
        os.path.join(path_or_dir, f"{res}.bedgraph"),
        os.path.join(path_or_dir, f"{res}.bedGraph"),
        os.path.join(path_or_dir, f"{res}.bg"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate bedGraph for resolution {res} in directory {path_or_dir}"
    )


def _load_bedgraph(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"chrom": str},
        engine="c",
    )
    if df.empty:
        raise ValueError(f"BedGraph {path} is empty")
    return df


def _load_chrom_sizes(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chrom sizes file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["chrom", "size"],
        dtype={"chrom": str, "size": int},
    )
    return dict(zip(df["chrom"], df["size"]))


def _load_contacts_overview(path: str) -> Dict[str, float]:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=0,
        names=["chrom", "contacts"],
        dtype={"chrom": str, "contacts": float},
    )
    contacts: Dict[str, float] = {}
    for _, row in df.iterrows():
        chrom = str(row["chrom"])
        if chrom.upper() == "TOTAL":
            continue
        contacts[chrom] = float(row["contacts"])
    if not contacts:
        raise ValueError(f"No chromosome rows found in contacts overview: {path}")
    return contacts


def _is_decoy_chrom(chrom: str) -> bool:
    lowered = chrom.lower()
    if lowered.startswith("chrun") or lowered.startswith("un_"):
        return True
    if "_" in chrom:
        return True
    decoy_tokens = ("random", "alt", "fix", "decoy", "hap", "patch")
    if any(token in lowered for token in decoy_tokens):
        return True
    if lowered.startswith(("gl", "ki", "jh", "hs", "chrz", "chrw")):
        return True
    return False


def _validate_grid(df: pd.DataFrame, res: int, chrom_len: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    if (df["start"] < 0).any() or (df["end"] < 0).any():
        raise ValueError("BedGraph contains negative coordinates")
    if (df["end"] <= df["start"]).any():
        raise ValueError("BedGraph contains zero or negative length intervals")
    if (df["start"].diff().fillna(res) < 0).any():
        df = df.sort_values("start").reset_index(drop=True)
    if df["start"].duplicated().any():
        dup_starts = df.loc[df["start"].duplicated(), "start"].tolist()
        raise ValueError(f"Duplicate bins detected at starts: {dup_starts[:5]}")
    if (df["start"] % res != 0).any():
        bad = df.loc[(df["start"] % res) != 0, "start"].iloc[0]
        raise ValueError(f"Start {bad} not aligned to resolution {res}")
    lengths = (df["end"] - df["start"]).astype(int)
    approx_res = np.isclose(lengths, res, atol=1)
    final_idx = len(df) - 1
    shorter_final = np.zeros(len(df), dtype=bool)
    if final_idx >= 0:
        final_end = int(df.iloc[final_idx]["end"])
        within_terminal = True
        if chrom_len is not None:
            within_terminal = abs(final_end - chrom_len) <= res
        shorter_final[final_idx] = lengths.iloc[final_idx] <= res and within_terminal
    valid_lengths = approx_res | shorter_final
    valid_lengths &= lengths > 0
    if not valid_lengths.all():
        bad_row = df.iloc[np.where(~valid_lengths)[0][0]]
        raise ValueError(
            f"Intervals do not match expected resolution grid (start={int(bad_row['start'])} "
            f"end={int(bad_row['end'])} length={int(bad_row['end']) - int(bad_row['start'])}, expected {res})"
        )
    if chrom_len is not None:
        max_end = int(df["end"].max())
        if max_end > chrom_len:
            raise ValueError(f"Intervals exceed declared length for chromosome (end={max_end} > {chrom_len})")
    return df.reset_index(drop=True)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    denom = max(len(values), 1)
    if denom == 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (denom - 1.0)


def _robust_sd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad == 0.0:
        return float(np.std(finite, ddof=0))
    return 1.4826 * mad


def _coefficient_of_variation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    mean_val = float(np.mean(finite))
    if mean_val == 0.0:
        return float("nan")
    std_val = float(np.std(finite, ddof=0))
    return std_val / mean_val


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_eps(config: NoiseWeightConfig, n_ok: int, nan_fraction: float) -> float:
    if n_ok <= 0:
        return config.eps_floor
    eps = max(config.eps_floor, config.eps_scale / math.sqrt(max(n_ok, 1)))
    if nan_fraction >= config.nan_fraction_threshold:
        eps *= config.high_nan_eps_multiplier
    return eps


def _apply_pipeline_once(
    ranks: np.ndarray,
    present_mask: np.ndarray,
    scores: np.ndarray,
    coverage: np.ndarray,
    config: NoiseWeightConfig,
    gain_eff: float,
    lambda_eff: float,
    sigma_target_eff: Optional[float],
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    weights = np.full(scores.shape, eps, dtype=float)
    if present_mask.any():
        w0 = 1.0 + gain_eff * (2.0 * _sigmoid((config.tau - ranks) / config.kappa) - 1.0)
        weights[present_mask] = w0
        current_sum = float(np.sum(weights[present_mask]))
        if current_sum > 0.0:
            scale = present_mask.sum() / current_sum
            weights[present_mask] *= scale
    weights[present_mask] = (1.0 - lambda_eff) * weights[present_mask] + lambda_eff
    if sigma_target_eff is not None and sigma_target_eff > 0.0:
        sd = _robust_sd(weights[present_mask])
        if sd > 0.0:
            weights[present_mask] = 1.0 + (weights[present_mask] - 1.0) * (sigma_target_eff / sd)
    weights = np.clip(weights, config.w_min, config.w_max)
    clip_mask = (weights == config.w_min) | (weights == config.w_max)
    if present_mask.any():
        renorm_sum = float(np.sum(weights[present_mask]))
        if renorm_sum > 0.0:
            weights[present_mask] *= present_mask.sum() / renorm_sum
    clip_rate = float(np.count_nonzero(clip_mask & present_mask)) / max(int(present_mask.sum()), 1)
    coverage_present = coverage[present_mask]
    coverage_present = np.nan_to_num(coverage_present, nan=0.0, posinf=0.0, neginf=0.0)
    cv_before = _coefficient_of_variation(coverage_present)
    cv_after = _coefficient_of_variation(weights[present_mask] * coverage_present)
    sigma_obs = _robust_sd(weights[present_mask]) if present_mask.any() else 0.0
    corr_mask = present_mask & np.isfinite(scores)
    if np.count_nonzero(corr_mask) >= 2:
        corr_val, _ = spearmanr(weights[corr_mask], scores[corr_mask])
        spearman_val = float(corr_val) if np.isfinite(corr_val) else float("nan")
    else:
        spearman_val = float("nan")
    qc = {
        "clip_rate": clip_rate,
        "cv_before": cv_before,
        "cv_after": cv_after,
        "sigma_observed": sigma_obs,
        "spearman": spearman_val,
    }
    return weights, qc


def _auto_relax(
    config: NoiseWeightConfig,
    lambda_eff: float,
    gain_eff: float,
    clip_rate: float,
) -> Tuple[float, float, bool]:
    if clip_rate <= config.max_clip_rate:
        return lambda_eff, gain_eff, False
    new_lambda = min(0.95, lambda_eff + 0.1)
    new_gain = gain_eff * 0.85
    return new_lambda, new_gain, True


def _compute_weights_for_chrom(
    chrom: str,
    scores: pd.Series,
    coverage: Optional[pd.Series],
    res: int,
    config: NoiseWeightConfig,
    depth_total_override: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, Dict[str, float]]:
    values = scores.to_numpy(dtype=float, copy=False)
    present_mask = np.isfinite(values)
    n = len(values)
    n_ok = int(np.count_nonzero(present_mask))
    n_nan = n - n_ok
    nan_fraction = n_nan / max(n, 1)
    coverage_array = (
        coverage.to_numpy(dtype=float, copy=False)
        if coverage is not None
        else np.full(n, 1.0, dtype=float)
    )
    coverage_array = np.where(
        np.isfinite(coverage_array),
        coverage_array,
        np.where(present_mask, 1.0, 0.0),
    )
    if depth_total_override is not None:
        depth_total = float(depth_total_override)
    else:
        depth_total = float(np.sum(np.nan_to_num(coverage_array[present_mask], nan=0.0)))
    gain_eff = config.gain * (0.5 if n_ok < 20 else 1.0)
    lambda_eff = config.shrink_lambda
    sigma_target_eff = config.sigma_target
    if config.use_depth_heuristics:
        D0 = config.depth_reference
        lambda_eff = float(np.clip(0.2 + 0.5 * math.exp(-depth_total / D0), 0.2, 0.6))
        sigma_target_eff = 0.35 * D0 / (depth_total + D0)
    if n_ok == 0:
        eps = _compute_eps(config, n_ok, nan_fraction)
        weights = np.full(n, eps, dtype=float)
        qc = {
            "clip_rate": 0.0,
            "cv_before": float("nan"),
            "cv_after": float("nan"),
            "sigma_observed": 0.0,
            "spearman": float("nan"),
        }
        stats = {
            "n": n,
            "n_ok": n_ok,
            "n_nan": n_nan,
            "depth_total": depth_total,
            "eps": eps,
            "lambda": lambda_eff,
            "gain": gain_eff,
            "sigma_target": sigma_target_eff if sigma_target_eff is not None else float("nan"),
        }
        return weights, qc, present_mask, stats
    ranks = _percentile_rank(values[present_mask])
    eps = _compute_eps(config, n_ok, nan_fraction)
    weights, qc = _apply_pipeline_once(
        ranks=ranks,
        present_mask=present_mask,
        scores=values,
        coverage=coverage_array,
        config=config,
        gain_eff=gain_eff,
        lambda_eff=lambda_eff,
        sigma_target_eff=sigma_target_eff,
        eps=eps,
    )
    lambda_final = lambda_eff
    gain_final = gain_eff
    relaxed = False
    if qc["clip_rate"] > config.max_clip_rate:
        lambda_final, gain_final, relaxed = _auto_relax(config, lambda_eff, gain_eff, qc["clip_rate"])
        if relaxed:
            weights, qc = _apply_pipeline_once(
                ranks=ranks,
                present_mask=present_mask,
                scores=values,
                coverage=coverage_array,
                config=config,
                gain_eff=gain_final,
                lambda_eff=lambda_final,
                sigma_target_eff=sigma_target_eff,
                eps=eps,
            )
    stats = {
        "n": n,
        "n_ok": n_ok,
        "n_nan": n_nan,
        "depth_total": depth_total,
        "eps": eps,
        "lambda": lambda_final,
        "gain": gain_final,
        "sigma_target": sigma_target_eff if sigma_target_eff is not None else float("nan"),
        "relaxed": relaxed,
    }
    return weights, qc, present_mask, stats


def _stability_check(
    weights: np.ndarray,
    present_mask: np.ndarray,
    scores: pd.Series,
    coverage: Optional[pd.Series],
    res: int,
    config: NoiseWeightConfig,
) -> Dict[str, float]:
    if not config.stability_enabled():
        return {"stability_spearman_mean": float("nan")}
    rng = np.random.default_rng(config.random_seed)
    coverage_array = (
        coverage.to_numpy(dtype=float, copy=False)
        if coverage is not None
        else np.full(len(scores), np.nan, dtype=float)
    )
    coverage_array = np.where(
        np.isfinite(coverage_array),
        coverage_array,
        np.where(present_mask, 1.0, 0.0),
    )
    stabilities: List[float] = []
    for ratio in config.stability_ratios:
        if not (0.0 < ratio < 1.0):
            continue
        lam = np.clip(coverage_array * ratio, 0.0, None)
        downsampled = rng.poisson(lam)
        down_cov = pd.Series(downsampled, index=scores.index, dtype=float)
        down_weights, _, down_mask, _ = _compute_weights_for_chrom(
            chrom="__tmp__",
            scores=scores,
            coverage=down_cov,
            res=res,
            config=config,
            depth_total_override=None,
        )
        joint_mask = present_mask & down_mask
        if np.count_nonzero(joint_mask) < 2:
            continue
        corr_val, _ = spearmanr(weights[joint_mask], down_weights[joint_mask])
        if np.isfinite(corr_val):
            stabilities.append(float(corr_val))
    mean_val = float(np.mean(stabilities)) if stabilities else float("nan")
    return {"stability_spearman_mean": mean_val}


def _write_bedgraph(path: str, chrom: str, res: int, chrom_len: Optional[int], weights: np.ndarray, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        for idx, weight in enumerate(weights):
            start = int(df.iloc[idx]["start"])
            end = int(df.iloc[idx]["end"])
            if chrom_len is not None:
                end = min(end, chrom_len)
            handle.write(f"{chrom} {start} {end} {weight}\n")


def _write_qc_json(path: str, qc: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(qc, handle, indent=2, sort_keys=True)


def build_weights_from_bedgraphs(
    scores_dir: str,
    res: int,
    out_dir: str,
    config: NoiseWeightConfig,
    coverage_path: Optional[str] = None,
    chrom_sizes_path: Optional[str] = None,
    chrom_allowlist: Optional[Sequence[str]] = None,
) -> None:
    score_path = _resolve_bedgraph_path(scores_dir, res)
    scores_df = _load_bedgraph(score_path)
    coverage_df: Optional[pd.DataFrame] = None
    contacts_lookup: Optional[Dict[str, float]] = None
    if coverage_path:
        if os.path.isdir(coverage_path):
            resolved_cov = _resolve_bedgraph_path(coverage_path, res)
            coverage_df = _load_bedgraph(resolved_cov)
        elif coverage_path.lower().endswith((".tsv", ".csv")):
            contacts_lookup = _load_contacts_overview(coverage_path)
        else:
            coverage_df = _load_bedgraph(coverage_path)
    elif os.path.isdir(scores_dir):
        default_contacts = os.path.join(scores_dir, "contacts_overview.tsv")
        if os.path.isfile(default_contacts):
            contacts_lookup = _load_contacts_overview(default_contacts)
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)
    allowset = set(chrom_allowlist) if chrom_allowlist else None
    per_chrom_records: List[Dict[str, float]] = []
    for chrom, group in scores_df.groupby("chrom", sort=True):
        if allowset is not None and chrom not in allowset:
            continue
        if config.skip_decoys and allowset is None and _is_decoy_chrom(chrom):
            continue
        chrom_len = chrom_sizes.get(chrom)
        try:
            group = _validate_grid(group.copy(), res, chrom_len)
        except ValueError as exc:
            raise ValueError(f"{chrom}: {exc}") from exc
        coverage_series: Optional[pd.Series] = None
        if coverage_df is not None:
            cov_group = coverage_df[coverage_df["chrom"] == chrom].copy()
            if not cov_group.empty:
                cov_group = _validate_grid(cov_group, res, chrom_len)
                merged = group[["chrom", "start"]].merge(
                    cov_group[["chrom", "start", "value"]],
                    on=["chrom", "start"],
                    how="left",
                    suffixes=("", "_cov"),
                )
                coverage_series = pd.Series(merged["value"], index=group.index, dtype=float)
        depth_override = contacts_lookup.get(chrom) if contacts_lookup else None
        weights, qc, present_mask, stats = _compute_weights_for_chrom(
            chrom=chrom,
            scores=group["value"],
            coverage=coverage_series,
            res=res,
            config=config,
            depth_total_override=depth_override,
        )
        stability = _stability_check(
            weights=weights,
            present_mask=present_mask,
            scores=group["value"],
            coverage=coverage_series,
            res=res,
            config=config,
        )
        qc_all = {
            "chrom": chrom,
            "resolution": res,
            **stats,
            **qc,
            **stability,
        }
        warning_msgs: List[str] = []
        spearman_val = qc_all.get("spearman")
        if spearman_val is not None and np.isfinite(spearman_val):
            if spearman_val > config.min_neg_spearman:
                warning_msgs.append(
                    f"Spearman {spearman_val:.3f} exceeds expected upper bound {config.min_neg_spearman}"
                )
        cv_before = qc_all.get("cv_before")
        cv_after = qc_all.get("cv_after")
        if cv_before is not None and cv_after is not None and np.isfinite(cv_before) and np.isfinite(cv_after):
            cv_reduction = cv_before - cv_after
            qc_all["cv_reduction"] = cv_reduction
            if cv_reduction < config.target_cv_reduction:
                warning_msgs.append(
                    f"CV reduction {cv_reduction:.3f} below target {config.target_cv_reduction}"
                )
        else:
            qc_all["cv_reduction"] = float("nan")
        if warning_msgs:
            for message in warning_msgs:
                print(f"[WARN] {chrom}@{res}: {message}")
        qc_all["warnings"] = "; ".join(warning_msgs) if warning_msgs else ""
        per_chrom_records.append(qc_all)
        bed_out = os.path.join(out_dir, f"{chrom}_{res}.weights.bedgraph")
        qc_out = os.path.join(out_dir, f"{chrom}_{res}.weights.qc.json")
        _write_bedgraph(bed_out, chrom, res, chrom_len, weights, group)
        _write_qc_json(qc_out, qc_all)
    if not per_chrom_records:
        raise RuntimeError("No chromosomes produced weight vectors; check filters or inputs")
    summary_df = pd.DataFrame(per_chrom_records)
    summary_df.sort_values(["chrom"], inplace=True)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"weights_{res}_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
