#!/usr/bin/env python
"""Common helpers for accessing Hi-C data from .hic and cooler formats."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import cooler  # type: ignore
    from cooler.fileops import list_coolers  # type: ignore
except ImportError:
    cooler = None  # type: ignore
    list_coolers = None  # type: ignore


COOLER_SUFFIXES: Tuple[str, ...] = (".cool", ".mcool", ".scool")
HICSTRAW_HELP = (
    "hicstraw is required for .hic inputs. Install it with "
    'pip install "jukebox-hic[straw]" or convert the .hic file to .mcool using hic2cool.'
)


def is_cooler_path(path: str) -> bool:
    return "::" in path or any(path.lower().endswith(ext) for ext in COOLER_SUFFIXES)


def hic_resolutions(path: str) -> List[int]:
    try:
        import hicstraw  # type: ignore
    except ImportError:
        return []
    try:
        hic = hicstraw.HiCFile(path)
    except Exception:
        return []
    try:
        resolutions = hic.getResolutions()
    except Exception:
        return []
    try:
        return sorted(int(res) for res in resolutions)
    except Exception:
        return []


def cooler_resolutions(path: str, selection: Optional[str]) -> List[int]:
    if cooler is None:
        return []
    try:
        if "::" in path:
            obj = cooler.Cooler(path)
            bin_size = obj.info.get("bin-size")
            return [int(bin_size)] if bin_size is not None else []
        lower = path.lower()
        if lower.endswith(".cool"):
            obj = cooler.Cooler(path)
            bin_size = obj.info.get("bin-size")
            return [int(bin_size)] if bin_size is not None else []
        if lower.endswith(".mcool"):
            if list_coolers is None:
                return []
            resolutions: set[int] = set()
            for uri in list(list_coolers(path)):
                full_uri = uri if "::" in uri else f"{path}::{uri}"
                try:
                    obj = cooler.Cooler(full_uri)
                    bin_size = obj.info.get("bin-size")
                    if bin_size is not None:
                        resolutions.add(int(bin_size))
                except Exception:
                    continue
            return sorted(resolutions)
        if lower.endswith(".scool"):
            if not selection:
                return []
            obj = cooler.Cooler(f"{path}::{selection}")
            bin_size = obj.info.get("bin-size")
            return [int(bin_size)] if bin_size is not None else []
    except Exception:
        return []
    return []


def normalize_hic_norm(norm: str) -> str:
    lower = norm.lower()
    if lower == "none":
        return "NONE"
    if lower == "balance":
        return "KR"
    if os.path.exists(norm):
        raise ValueError("Custom normalization vectors are not supported for .hic inputs")
    return norm.upper()


def resolve_cooler_uri(path: str, res: int, selection: Optional[str]) -> str:
    if "::" in path:
        return path
    lower = path.lower()
    if lower.endswith(".cool"):
        return path
    if lower.endswith(".mcool"):
        if list_coolers is None:
            raise RuntimeError("cooler is required for .mcool handling")
        target_group = f"/resolutions/{res}"
        candidate_full = f"{path}::{target_group}"
        available = list(list_coolers(path))
        full_set = set(available)
        group_set = set(u.split("::", 1)[1] if "::" in u else u for u in available)
        if candidate_full in full_set or target_group in group_set:
            return candidate_full
        pretty = sorted(group_set)
        raise ValueError(
            f"Resolution {res} bp not found in {path}. Available groups: {pretty}"
        )
    if lower.endswith(".scool"):
        if not selection:
            raise ValueError("Provide --cooler_path=<group> for .scool files (e.g. cell name)")
        return f"{path}::{selection}"
    raise ValueError(f"Unrecognized cooler-like suffix in {path}")


def load_custom_weights(path: str, bins_df: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 1:
        if len(df) != len(bins_df):
            raise ValueError(
                f"Custom vector length {len(df)} does not match number of bins {len(bins_df)}"
            )
        return df.iloc[:, 0].to_numpy(dtype=float)
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["chrom", "start", "end", "value"]
        merge = bins_df[["chrom", "start"]].merge(
            df[["chrom", "start", "value"]], on=["chrom", "start"], how="left"
        )
        weights = merge["value"].fillna(0.0).to_numpy(dtype=float)
        if np.count_nonzero(np.isnan(weights)):
            raise ValueError("Custom normalization has NaNs after alignment; please verify formatting")
        return weights
    raise ValueError("Custom normalization file must have either 1 column or chrom/start/end/value columns")


@dataclass
class ChromMatrix:
    chrom: str
    chrom_len: int
    coo: sparse.coo_matrix
    csr: sparse.csr_matrix
    csc: sparse.csc_matrix


def total_contacts(matrix: ChromMatrix) -> float:
    coo = matrix.coo
    if coo.nnz == 0:
        return 0.0
    row = coo.row
    col = coo.col
    data = coo.data
    lo = np.minimum(row, col).astype(np.uint64, copy=False)
    hi = np.maximum(row, col).astype(np.uint64, copy=False)
    keys = (lo << np.uint64(32)) | hi
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    data_sorted = data[order]
    _, idx, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    summed = np.add.reduceat(data_sorted, idx)
    averages = summed / counts
    return float(averages.sum())


class BaseProvider:
    def chrom_sizes(self) -> Dict[str, int]:
        raise NotImplementedError

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        raise NotImplementedError


class HiCProvider(BaseProvider):
    def __init__(self, path: str, res: int, norm: str) -> None:
        try:
            import hicstraw  # type: ignore
        except ImportError as exc:  # pragma: no cover - surfaced at runtime
            raise RuntimeError(HICSTRAW_HELP) from exc
        self.path = path
        self.res = int(res)
        self.norm = normalize_hic_norm(norm)
        self._hicstraw: Any = hicstraw
        self._hic = self._hicstraw.HiCFile(path)
        self._chrom_sizes = {
            str(chrom.name): int(chrom.length)
            for chrom in self._hic.getChromosomes()
            if chrom.name != "ALL"
        }

    def chrom_sizes(self) -> Dict[str, int]:
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        chrom_len = self._chrom_sizes.get(chrom)
        if chrom_len is None:
            return None
        records = self._hicstraw.straw(
            "observed",
            self.norm,
            self.path,
            chrom,
            chrom,
            "BP",
            self.res,
        )
        if not records:
            return None
        rows = []
        cols = []
        data = []
        for rec in records:
            rows.append(rec.binX // self.res)
            cols.append(rec.binY // self.res)
            data.append(float(rec.counts))
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        data_arr = np.asarray(data, dtype=float)

        mask = row_arr != col_arr
        row_sym = np.concatenate([row_arr, col_arr[mask]])
        col_sym = np.concatenate([col_arr, row_arr[mask]])
        data_sym = np.concatenate([data_arr, data_arr[mask]])

        rowbins = (chrom_len + self.res - 1) // self.res
        if row_sym.size:
            max_idx = int(max(row_sym.max(), col_sym.max()))
            rowbins = max(rowbins, max_idx + 1)
        coo = sparse.coo_matrix((data_sym, (row_sym, col_sym)), shape=(rowbins, rowbins))
        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)


class CoolerProvider(BaseProvider):
    def __init__(self, path: str, res: int, norm: str, selection: Optional[str]) -> None:
        if cooler is None:
            raise RuntimeError("cooler is required for .cool/.mcool/.scool inputs")
        self.res = int(res)
        self.uri = resolve_cooler_uri(path, self.res, selection)
        self._cooler = cooler.Cooler(self.uri)

        bin_size = int(self._cooler.info.get("bin-size", self.res))
        if bin_size != self.res:
            raise ValueError(
                f"Input resolution {self.res} bp does not match cooler bin size {bin_size} bp"
            )

        self._chrom_sizes = dict(self._cooler.chromsizes)
        self._bins = self._cooler.bins()[:]
        self._bins["chrom"] = self._bins["chrom"].astype(str)

        norm_lower = norm.lower()
        self._norm_mode = "none"
        self._custom_weights: Optional[np.ndarray] = None
        if norm_lower == "none":
            self._norm_mode = "none"
        elif norm_lower == "balance":
            if "weight" not in self._bins.columns or self._bins["weight"].isna().all():
                raise ValueError("Cooler file does not contain balance weights")
            self._norm_mode = "balance"
        elif os.path.exists(norm):
            self._norm_mode = "custom"
            self._custom_weights = load_custom_weights(norm, self._bins)
        else:
            raise ValueError("Unsupported cooler normalization; choose none, balance, or provide vector path")

        self._raw_matrix = self._cooler.matrix(balance=False, sparse=True)
        self._balanced_matrix = (
            self._cooler.matrix(balance=True, sparse=True)
            if self._norm_mode == "balance"
            else None
        )

    def chrom_sizes(self) -> Dict[str, int]:
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        if chrom not in self._chrom_sizes:
            return None
        start, end = self._cooler.extent(chrom)
        if end <= start:
            return None
        chrom_len = int(self._chrom_sizes[chrom])
        if self._norm_mode == "balance":
            sub = self._balanced_matrix.fetch(chrom)  # type: ignore[union-attr]
        else:
            sub = self._raw_matrix.fetch(chrom)
        coo = sub.tocoo()
        finite_mask = np.isfinite(coo.data)
        if not np.all(finite_mask):
            coo = sparse.coo_matrix(
                (coo.data[finite_mask], (coo.row[finite_mask], coo.col[finite_mask])),
                shape=coo.shape,
            )

        if self._norm_mode == "custom":
            weights = self._custom_weights[start:end]
            factor = weights[coo.row] * weights[coo.col]
            data = coo.data * factor
            coo = sparse.coo_matrix((data, (coo.row, coo.col)), shape=coo.shape)

        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)


def select_provider(
    path: str,
    res: int,
    norm: str,
    cooler_selection: Optional[str],
) -> Tuple[BaseProvider, str]:
    if is_cooler_path(path):
        provider = CoolerProvider(path, res, norm, cooler_selection)
        return provider, "cooler"
    provider = HiCProvider(path, res, norm)
    return provider, "hic"


def read_chrom_sizes(
    provider: BaseProvider,
    chrom_sizes_path: Optional[str],
) -> Dict[str, int]:
    if chrom_sizes_path and os.path.exists(chrom_sizes_path):
        df = pd.read_csv(
            chrom_sizes_path,
            sep="\t",
            header=None,
            names=["chr", "size"],
            dtype={"chr": str, "size": int},
        )
        sizes = dict(zip(df["chr"], df["size"]))
        provider_sizes = provider.chrom_sizes()
        return {chrom: provider_sizes[chrom] for chrom in provider_sizes if chrom in sizes}
    return provider.chrom_sizes()
