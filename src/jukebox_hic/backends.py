#!/usr/bin/env python
"""
Common helpers for accessing Hi-C contact data from .hic (Juicer) and cooler file formats.

This module provides a unified abstraction layer (the "provider" pattern) so that the
rest of jukebox-hic can load contact matrices without caring whether the underlying
file is a Juicer .hic file or a cooler .cool/.mcool/.scool file.

Architecture overview
---------------------
``BaseProvider``  — abstract interface specifying three methods every backend must supply:
  - ``chrom_sizes()`` → dict of chromosome name → length in bp
  - ``fetch_chrom()`` → a ``ChromMatrix`` for a single chromosome
  - ``fetch_region()`` → a ``ChromMatrix`` for a bp-bounded sub-region (local 0-based indices)

``HiCProvider``   — implements BaseProvider via the ``hicstraw`` library for .hic files.
``CoolerProvider``— implements BaseProvider via the ``cooler`` library for cooler files.

``ChromMatrix``   — a dataclass holding the sparse contact matrix for one chromosome in
                    three representations (COO, CSR, CSC) so downstream code can pick
                    whichever sparse format is most convenient for its access pattern.

Top-level helper functions handle format detection, resolution discovery, URI
construction, weight loading, contact counting, and provider construction.
"""

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


# File-extension suffixes that identify cooler-format files.
# .cool  — single-resolution cooler (one bin size embedded in the file).
# .mcool — multi-resolution cooler (multiple /resolutions/<N> groups in one HDF5 file).
# .scool — single-cell cooler (one cooler group per cell in the file).
COOLER_SUFFIXES: Tuple[str, ...] = (".cool", ".mcool", ".scool")

# Human-readable error messages shown when optional dependencies are missing.
HICSTRAW_HELP = (
    "hicstraw is required for .hic inputs. "
    "It is included in the default install (pip install jukebox-hic) "
    'or can be added with pip install "jukebox-hic[straw]". '
    "Alternatively, convert the .hic file to .mcool using hic2cool and use the cooler backend."
)

COOLER_HELP = (
    "cooler is required for .cool/.mcool/.scool inputs. "
    "It is included in the default install (pip install jukebox-hic) "
    'or can be added with pip install "jukebox-hic[cooler]".'
)


def is_cooler_path(path: str) -> bool:
    """
    Return True if *path* looks like a cooler file path rather than a Juicer .hic file.

    Detection is done in two ways:
    1. If the path contains ``::`` it is already a cooler URI (e.g. ``file.mcool::resolutions/10000``).
    2. If the filename ends with a known cooler suffix (.cool, .mcool, .scool) — case-insensitive.

    Everything else is assumed to be a Juicer .hic file.
    """
    return "::" in path or any(path.lower().endswith(ext) for ext in COOLER_SUFFIXES)


def hic_resolutions(path: str) -> List[int]:
    """
    Return the list of available bin sizes (resolutions) stored in a Juicer .hic file.

    The hicstraw library is queried via ``HiCFile.getResolutions()``. Each resolution
    is the bin size in base pairs (e.g. 5000, 10000, 25000, 50000, 100000, 250000, ...).

    Returns an empty list rather than raising if:
    - hicstraw is not installed (optional dependency).
    - The file cannot be opened (wrong path or corrupted file).
    - The resolution list cannot be retrieved or converted to integers.

    The four nested try/except blocks mirror the hicstraw API's multiple failure modes:
    the import itself can fail, the file open can fail, getResolutions() can fail, and
    the int-conversion of resolution strings can fail — all must be handled gracefully
    so callers receive a clean empty list instead of a traceback.
    """
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
    """
    Return the list of available bin sizes (resolutions) stored in a cooler file.

    Handles the three distinct cooler file types differently:

    - **Already a URI** (contains ``::``): open directly and read ``bin-size`` from metadata.
    - **.cool** (single-resolution): open directly, return the single ``bin-size``.
    - **.mcool** (multi-resolution): iterate over all groups in the file with
      ``list_coolers()``, open each one, and collect all distinct bin sizes.
    - **.scool** (single-cell): use ``selection`` as the group path for a specific cell
      (e.g. ``"cell_1"``). Returns empty list if ``selection`` is not provided.

    ``selection`` is the ``--cooler_path`` CLI argument, used to address a specific
    group inside a multi-level HDF5 cooler hierarchy.

    Returns an empty list on any error rather than raising, for the same reason as
    ``hic_resolutions()``: callers use the result to show available options, and a
    missing cooler library or corrupt file should not crash the tool.
    """
    if cooler is None:
        return []
    try:
        if "::" in path:
            # Already a fully-qualified URI — open it and read the single bin-size.
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
                # ``list_coolers`` can return either full URIs or bare group paths
                # depending on the cooler version — normalise to a full URI.
                full_uri = uri if "::" in uri else f"{path}::{uri}"
                try:
                    obj = cooler.Cooler(full_uri)
                    bin_size = obj.info.get("bin-size")
                    if bin_size is not None:
                        resolutions.add(int(bin_size))
                except Exception:
                    # Some groups in an mcool may be metadata or index groups,
                    # not valid cooler objects — skip them silently.
                    continue
            return sorted(resolutions)
        if lower.endswith(".scool"):
            # .scool files contain one cooler per cell; a cell group path is required.
            if not selection:
                return []
            obj = cooler.Cooler(f"{path}::{selection}")
            bin_size = obj.info.get("bin-size")
            return [int(bin_size)] if bin_size is not None else []
    except Exception:
        return []
    return []


def normalize_hic_norm(norm: str) -> str:
    """
    Translate a user-facing normalization name to the string expected by hicstraw.

    The .hic format uses its own normalization labels (e.g. "KR" for Knight-Ruiz
    balancing) that differ from what jukebox-hic exposes to users.

    Mapping:
    - ``"none"``    → ``"NONE"``  — no normalization; use raw observed counts.
    - ``"balance"`` → ``"KR"``   — Knight-Ruiz matrix balancing (default in Juicer).
    - Any file path → raises ValueError; .hic files do not support external weight
      vectors (the hicstraw API cannot apply custom bin weights).
    - Anything else → upper-cased and passed through (e.g. "VC", "VC_SQRT", "SCALE").
    """
    lower = norm.lower()
    if lower == "none":
        return "NONE"
    if lower == "balance":
        return "KR"
    if os.path.exists(norm):
        raise ValueError("Custom normalization vectors are not supported for .hic inputs")
    return norm.upper()


def resolve_cooler_uri(path: str, res: int, selection: Optional[str]) -> str:
    """
    Build the cooler URI string needed to open a specific resolution from a cooler file.

    Cooler files use HDF5 group paths to address individual contact matrices:
    - ``path::resolutions/10000`` selects the 10 kb resolution from an .mcool file.
    - A bare ``.cool`` path already points to a single matrix and needs no suffix.
    - An ``.scool`` file uses the ``selection`` argument (cell name) as the HDF5 group.

    Raises ``ValueError`` if:
    - The requested resolution is not present in an .mcool file.
    - ``selection`` is missing for an .scool file.
    - The suffix is not recognised as a valid cooler variant.

    The standard .mcool group layout is ``/resolutions/<bin_size_bp>``, as written by
    ``cooler zoomify``. The function checks both the full URI form and bare group form
    to handle minor differences between cooler versions.
    """
    if "::" in path:
        # Already a fully-qualified URI — use as-is.
        return path
    lower = path.lower()
    if lower.endswith(".cool"):
        # Single-resolution file; the path is the URI.
        return path
    if lower.endswith(".mcool"):
        if list_coolers is None:
            raise RuntimeError(COOLER_HELP)
        target_group = f"/resolutions/{res}"
        candidate_full = f"{path}::{target_group}"
        available = list(list_coolers(path))
        full_set = set(available)
        # Strip the leading ``path::`` prefix to get bare group paths for comparison.
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
    """
    Load a custom per-bin normalization weight vector from a file.

    The weight file may be in one of two formats:

    **1-column format** — one weight value per line, in the same order as the bins in
    the cooler file. The file length must exactly match the number of bins.

    **4-column bedgraph format** — columns: chrom, start, end, value. The weights are
    aligned to the cooler bins by (chrom, start) position using a left join. Bins with
    no matching entry receive weight 0.0 (i.e. they will be zeroed out in the matrix).

    These weights are applied as multiplicative scaling factors in ``CoolerProvider``:
    each matrix entry (i, j) is multiplied by ``weight[i] * weight[j]``, which is the
    same convention as cooler's built-in balancing (the "weight" column stores
    ``1 / sqrt(normalisation_factor)`` per bin).

    Raises ``ValueError`` if the column count is not 1 or ≥4.
    """
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
    """
    Sparse contact matrix for a single chromosome, stored in three formats simultaneously.

    Hi-C contact data is extremely sparse (typically <1% of entries are non-zero), so
    scipy sparse matrices are used throughout. Three representations are kept because
    different operations are most efficient with different formats:

    Attributes
    ----------
    chrom : str
        Chromosome name (e.g. ``"chr1"``).
    chrom_len : int
        Chromosome length in base pairs (from the file header or chrom sizes file).
    coo : scipy.sparse.coo_matrix
        **COO (Coordinate)** format — stores three arrays: ``row``, ``col``, ``data``.
        Efficient for construction and for iterating over all non-zero entries.
        Used in ``_precompute_expectation()`` and ``total_contacts()``.
    csr : scipy.sparse.csr_matrix
        **CSR (Compressed Sparse Row)** format — efficient for row slicing.
        ``matrix.getrow(i)`` is O(nnz_in_row). Used in ``_row_window_noise()`` and
        ``_row_window_vectors()`` to extract downstream contacts for each row bin.
    csc : scipy.sparse.csc_matrix
        **CSC (Compressed Sparse Column)** format — efficient for column slicing.
        ``matrix.getcol(j)`` is O(nnz_in_col). Used in ``_row_window_noise()`` and
        ``_row_window_vectors()`` to extract upstream contacts for each column bin.
    """
    chrom: str
    chrom_len: int
    coo: sparse.coo_matrix
    csr: sparse.csr_matrix
    csc: sparse.csc_matrix


def total_contacts(matrix: ChromMatrix) -> float:
    """
    Count the total number of unique contact pairs in a chromosome's sparse matrix.

    Hi-C contact matrices are symmetric: the entry at (i, j) and the entry at (j, i)
    represent the same contact pair. Backends that return already-symmetrised matrices
    (like hicstraw after mirror-filling) store this pair twice, so naively summing
    all values would double-count off-diagonal contacts.

    This function deduplicates using a bit-packing trick:
    - For each (row, col) pair in the COO matrix, sort so the smaller index is ``lo``
      and the larger is ``hi``.
    - Encode the pair as a single 64-bit integer: ``key = (lo << 32) | hi``.
      This works because bin indices fit in 32 bits for any real genome.
    - Entries with the same key are the same contact seen from both sides.
    - Group by key, take the average value within each group (upper and lower triangle
      entries should be identical, but averaging protects against floating-point drift),
      then sum all averages.

    Diagonal entries (i == i) are not deduplicated separately because ``lo == hi``
    for them, so the key is unique and they are counted once automatically.

    Parameters
    ----------
    matrix : ChromMatrix
        The chromosome contact matrix.

    Returns
    -------
    float
        Total unique contact count (sum of upper-triangle + diagonal values).
    """
    coo = matrix.coo
    if coo.nnz == 0:
        return 0.0
    row = coo.row
    col = coo.col
    data = coo.data

    # --- Bit-packing deduplication ---
    # Ensure lower index is always in `lo` so symmetric duplicates hash to the same key.
    lo = np.minimum(row, col).astype(np.uint64, copy=False)
    hi = np.maximum(row, col).astype(np.uint64, copy=False)
    # Pack (lo, hi) into one uint64: upper 32 bits = lo, lower 32 bits = hi.
    # This is valid as long as the matrix dimension is < 2^32 ≈ 4 billion bins.
    keys = (lo << np.uint64(32)) | hi

    # Sort by key so np.unique can find groups efficiently.
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    data_sorted = data[order]

    # np.unique returns the starting index of each unique key group and the group size.
    _, idx, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    # Sum the values within each duplicate group, then divide by group size (average).
    summed = np.add.reduceat(data_sorted, idx)
    averages = summed / counts
    return float(averages.sum())


class BaseProvider:
    """
    Abstract base class for Hi-C data providers.

    Defines the two-method interface that all concrete providers (``HiCProvider``,
    ``CoolerProvider``) must implement, so the rest of jukebox-hic can load contact
    data without knowing which file format is in use.

    Do not instantiate this class directly — use ``select_provider()`` instead.
    """

    def chrom_sizes(self) -> Dict[str, int]:
        """Return a mapping of chromosome name → length in base pairs."""
        raise NotImplementedError

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        """
        Load the full contact matrix for *chrom* and return it as a ``ChromMatrix``.

        Returns ``None`` if the chromosome is not present in the file or has no data.
        """
        raise NotImplementedError

    def fetch_region(self, chrom: str, start_bp: int, end_bp: int) -> Optional[ChromMatrix]:
        """
        Load the contact sub-matrix for the diagonal block [start_bp, end_bp) of *chrom*.

        The returned ``ChromMatrix`` uses **local 0-based indices**: bin 0 corresponds to
        the genomic position ``start_bp``.  ``ChromMatrix.chrom_len`` is set to
        ``end_bp - start_bp``.

        Both ``start_bp`` and ``end_bp`` should be bin-aligned (multiples of the
        provider's resolution) for predictable results.

        Returns ``None`` if the region is empty, outside the chromosome, or has no contacts.
        """
        raise NotImplementedError

    def fetch_region_pair(
        self,
        chrom: str,
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> Optional[sparse.csr_matrix]:
        """
        Load the off-diagonal rectangular contact block for region_a × region_b.

        Both regions must be on the same chromosome (*chrom*).  Row indices in the
        returned matrix correspond to bins in region_a (0-based, local), and column
        indices correspond to bins in region_b (0-based, local).

        Returns a CSR matrix of shape ``(n_a_bins, n_b_bins)``, or ``None`` if the
        block is empty or outside the chromosome.
        """
        raise NotImplementedError


class HiCProvider(BaseProvider):
    """
    Provider that reads contact data from a Juicer .hic file via the hicstraw library.

    Juicer .hic files are a binary format produced by the Juicer pipeline. They store
    contact matrices at multiple resolutions in a compact indexed format. The hicstraw
    Python library wraps the Java Straw API to extract chromosome-level matrices.

    Normalisation is handled at load time: the ``norm`` argument selects which
    pre-computed balancing vector (stored inside the .hic file) to apply when
    extracting counts. Common choices are ``"NONE"`` (raw counts) and ``"KR"``
    (Knight-Ruiz balancing).

    The matrix returned by hicstraw contains only the upper triangle; ``fetch_chrom``
    mirrors the off-diagonal entries to fill the lower triangle and produce a full
    symmetric matrix, which is what the noise algorithms expect.

    Attributes
    ----------
    path : str
        Absolute or relative path to the .hic file.
    res : int
        Bin size in base pairs for this provider.
    norm : str
        Normalisation label in hicstraw's convention (after ``normalize_hic_norm()``).
    _hicstraw : module
        The imported hicstraw module (stored so worker processes don't need to re-import).
    _hic : hicstraw.HiCFile
        Open HiCFile handle used for all subsequent queries.
    _chrom_sizes : dict
        Chromosome name → length in bp, read from the file header on construction.
    """

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
        # The "ALL" pseudo-chromosome aggregates the whole genome; exclude it from
        # per-chromosome processing since it's not a real chromosome.
        self._chrom_sizes = {
            str(chrom.name): int(chrom.length)
            for chrom in self._hic.getChromosomes()
            if chrom.name != "ALL"
        }

    def chrom_sizes(self) -> Dict[str, int]:
        """Return chromosome name → length in bp from the .hic file header."""
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        """
        Extract the full symmetric contact matrix for *chrom* at the configured resolution.

        hicstraw returns upper-triangle records (binX ≤ binY) as a list of
        (binX, binY, counts) triples. This method:
        1. Converts bin coordinates from bp to bin indices (divide by resolution).
        2. Mirrors off-diagonal entries to fill the lower triangle (symmetric matrix).
        3. Packs the result into a COO matrix, then derives CSR and CSC from it.

        Returns None if the chromosome is missing or has no contacts at this resolution.
        """
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
            # hicstraw bin coordinates are in bp; divide by resolution to get bin index.
            rows.append(rec.binX // self.res)
            cols.append(rec.binY // self.res)
            data.append(float(rec.counts))
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        data_arr = np.asarray(data, dtype=float)

        # Mirror off-diagonal entries: for every (i, j) where i ≠ j, also add (j, i).
        # Diagonal entries (i == j) are not duplicated.
        mask = row_arr != col_arr
        row_sym = np.concatenate([row_arr, col_arr[mask]])
        col_sym = np.concatenate([col_arr, row_arr[mask]])
        data_sym = np.concatenate([data_arr, data_arr[mask]])

        # Compute the expected number of bins from chromosome length; also guard against
        # the case where the data contains an index beyond the expected range.
        rowbins = (chrom_len + self.res - 1) // self.res
        if row_sym.size:
            max_idx = int(max(row_sym.max(), col_sym.max()))
            rowbins = max(rowbins, max_idx + 1)
        coo = sparse.coo_matrix((data_sym, (row_sym, col_sym)), shape=(rowbins, rowbins))
        # sum_duplicates() collapses any repeated (i, j) entries that arise from the
        # mirroring step or from hicstraw returning overlapping records.
        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)

    def fetch_region(self, chrom: str, start_bp: int, end_bp: int) -> Optional[ChromMatrix]:
        """
        Extract a diagonal contact sub-matrix for [start_bp, end_bp) of *chrom*.

        hicstraw accepts region strings in the form ``"chrom:start:end"`` for both
        query regions.  Returned records have ``binX``/``binY`` in absolute bp; the
        local bin offset (``start_bp // res``) is subtracted so the result is 0-based.
        """
        chrom_len = self._chrom_sizes.get(chrom)
        if chrom_len is None:
            return None
        start_bp = max(0, int(start_bp))
        end_bp = min(int(chrom_len), int(end_bp))
        if end_bp <= start_bp:
            return None

        region = f"{chrom}:{start_bp}:{end_bp}"
        try:
            records = self._hicstraw.straw(
                "observed", self.norm, self.path, region, region, "BP", self.res,
            )
        except Exception:
            return None
        if not records:
            return None

        bin_offset = start_bp // self.res
        region_bins = (end_bp - start_bp + self.res - 1) // self.res

        rows_list: List[int] = []
        cols_list: List[int] = []
        data_list: List[float] = []
        for rec in records:
            r = rec.binX // self.res - bin_offset
            c = rec.binY // self.res - bin_offset
            if r < 0 or r >= region_bins or c < 0 or c >= region_bins:
                continue
            rows_list.append(r)
            cols_list.append(c)
            data_list.append(float(rec.counts))

        if not rows_list:
            return None

        row_arr = np.asarray(rows_list, dtype=np.int32)
        col_arr = np.asarray(cols_list, dtype=np.int32)
        data_arr = np.asarray(data_list, dtype=float)

        mask = row_arr != col_arr
        row_sym = np.concatenate([row_arr, col_arr[mask]])
        col_sym = np.concatenate([col_arr, row_arr[mask]])
        data_sym = np.concatenate([data_arr, data_arr[mask]])

        coo = sparse.coo_matrix((data_sym, (row_sym, col_sym)), shape=(region_bins, region_bins))
        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=end_bp - start_bp, coo=coo, csr=csr, csc=csc)

    def fetch_region_pair(
        self,
        chrom: str,
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> Optional[sparse.csr_matrix]:
        """
        Fetch the off-diagonal rectangular block region_a × region_b via hicstraw.

        hicstraw's ``straw()`` accepts two independent region strings, so this calls
        ``straw(region_a, region_b)`` where the two strings are different.  Records are
        filtered to the rectangle and assembled into a CSR matrix with local 0-based
        row indices (offset = start_a // res) and local 0-based col indices
        (offset = start_b // res).  No symmetrization is applied — the block is
        inherently non-square.
        """
        chrom_len = self._chrom_sizes.get(chrom)
        if chrom_len is None:
            return None
        start_a = max(0, int(start_a))
        end_a = min(int(chrom_len), int(end_a))
        start_b = max(0, int(start_b))
        end_b = min(int(chrom_len), int(end_b))
        if end_a <= start_a or end_b <= start_b:
            return None

        region_a = f"{chrom}:{start_a}:{end_a}"
        region_b = f"{chrom}:{start_b}:{end_b}"
        try:
            records = self._hicstraw.straw(
                "observed", self.norm, self.path, region_a, region_b, "BP", self.res,
            )
        except Exception:
            return None
        if not records:
            return None

        offset_a = start_a // self.res
        offset_b = start_b // self.res
        n_a = (end_a - start_a + self.res - 1) // self.res
        n_b = (end_b - start_b + self.res - 1) // self.res

        rows_list: List[int] = []
        cols_list: List[int] = []
        data_list: List[float] = []
        for rec in records:
            r = rec.binX // self.res - offset_a
            c = rec.binY // self.res - offset_b
            if r < 0 or r >= n_a or c < 0 or c >= n_b:
                continue
            rows_list.append(r)
            cols_list.append(c)
            data_list.append(float(rec.counts))

        if not rows_list:
            return None

        csr = sparse.csr_matrix(
            (np.asarray(data_list, dtype=float),
             (np.asarray(rows_list, dtype=np.int32),
              np.asarray(cols_list, dtype=np.int32))),
            shape=(n_a, n_b),
        )
        csr.sum_duplicates()
        return csr


class CoolerProvider(BaseProvider):
    """
    Provider that reads contact data from a cooler file (.cool, .mcool, .scool).

    Cooler is an HDF5-based format for Hi-C contact matrices, produced by tools such
    as ``cooler cload`` and ``cooler zoomify``. Unlike .hic files, cooler files store
    the full symmetric matrix (both upper and lower triangles) using compressed sparse
    row format internally.

    Normalisation modes
    -------------------
    Three normalisation modes are supported, selected by the ``norm`` argument:

    - ``"none"``    — raw observed counts with no normalisation applied.
    - ``"balance"`` — use the ``weight`` column pre-computed by ``cooler balance``
                      (iterative correction / Knight-Ruiz). Requires the cooler file
                      to have been balanced before use.
    - path string   — load a custom weight vector from a bedgraph or single-column
                      file via ``load_custom_weights()``. Applied as element-wise
                      multiplication: entry (i,j) *= weight[i] * weight[j].

    Attributes
    ----------
    res : int
        Bin size in base pairs.
    uri : str
        The fully-qualified cooler URI used to open the file (e.g.
        ``"data.mcool::resolutions/10000"``).
    _cooler : cooler.Cooler
        Open Cooler object for metadata and matrix queries.
    _chrom_sizes : dict
        Chromosome name → length in bp from the cooler metadata.
    _bins : pd.DataFrame
        Full bin table (chrom, start, end, weight, ...) for the file, loaded once
        at construction to avoid repeated I/O.
    _norm_mode : str
        One of ``"none"``, ``"balance"``, or ``"custom"`` — controls which matrix
        accessor is used in ``fetch_chrom()``.
    _custom_weights : np.ndarray or None
        Per-bin weight array when ``_norm_mode == "custom"``; otherwise None.
    _raw_matrix : cooler matrix accessor
        Matrix accessor for raw (un-normalised) counts; always initialised.
    _balanced_matrix : cooler matrix accessor or None
        Matrix accessor for balance-weighted counts; only initialised when
        ``_norm_mode == "balance"`` to avoid loading weights unnecessarily.
    """

    def __init__(self, path: str, res: int, norm: str, selection: Optional[str]) -> None:
        if cooler is None:
            raise RuntimeError(COOLER_HELP)
        self.res = int(res)
        self.uri = resolve_cooler_uri(path, self.res, selection)
        self._cooler = cooler.Cooler(self.uri)

        # Validate that the file's bin size matches the requested resolution.
        bin_size = int(self._cooler.info.get("bin-size", self.res))
        if bin_size != self.res:
            raise ValueError(
                f"Input resolution {self.res} bp does not match cooler bin size {bin_size} bp"
            )

        self._chrom_sizes = dict(self._cooler.chromsizes)
        # Load the complete bin table once; coerce chrom column to plain str to
        # avoid category dtype complications in downstream merge operations.
        self._bins = self._cooler.bins()[:]
        self._bins["chrom"] = self._bins["chrom"].astype(str)

        # --- Determine normalisation mode ---
        norm_lower = norm.lower()
        self._norm_mode = "none"
        self._custom_weights: Optional[np.ndarray] = None
        if norm_lower == "none":
            self._norm_mode = "none"
        elif norm_lower == "balance":
            # The "weight" column is added by ``cooler balance``. Fail early if absent.
            if "weight" not in self._bins.columns or self._bins["weight"].isna().all():
                raise ValueError("Cooler file does not contain balance weights")
            self._norm_mode = "balance"
        elif os.path.exists(norm):
            self._norm_mode = "custom"
            self._custom_weights = load_custom_weights(norm, self._bins)
        else:
            raise ValueError("Unsupported cooler normalization; choose none, balance, or provide vector path")

        # Pre-open matrix accessors. The cooler library returns a view object; actual
        # data is only fetched when ``fetch()`` is called on it.
        self._raw_matrix = self._cooler.matrix(balance=False, sparse=True)
        self._balanced_matrix = (
            self._cooler.matrix(balance=True, sparse=True)
            if self._norm_mode == "balance"
            else None
        )

    def chrom_sizes(self) -> Dict[str, int]:
        """Return chromosome name → length in bp from the cooler metadata."""
        return self._chrom_sizes

    def fetch_chrom(self, chrom: str) -> Optional[ChromMatrix]:
        """
        Extract the full symmetric contact matrix for *chrom* at the configured resolution.

        For ``_norm_mode == "balance"``, uses the cooler library's built-in balanced
        matrix accessor (which multiplies each entry by weight[i] * weight[j]
        automatically and sets entries to NaN where either bin has a NaN weight).

        For ``_norm_mode == "custom"``, fetches raw counts and applies the custom
        weight vector manually: entry (i,j) *= custom_weight[i] * custom_weight[j].
        The weight slice is indexed by absolute bin indices (``start:end``) obtained
        from ``cooler.extent(chrom)``.

        Infinite and NaN values (from balancing with zero-weight bins) are removed
        from the sparse matrix before it is returned.

        Returns None if the chromosome is absent or the extent is empty.
        """
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

        # Remove non-finite entries (NaN or Inf) that arise from balancing bins
        # that have zero coverage — these bins get weight NaN in the cooler format,
        # which propagates into the balanced matrix as NaN contact values.
        finite_mask = np.isfinite(coo.data)
        if not np.all(finite_mask):
            coo = sparse.coo_matrix(
                (coo.data[finite_mask], (coo.row[finite_mask], coo.col[finite_mask])),
                shape=coo.shape,
            )

        if self._norm_mode == "custom":
            # Apply custom weights: the bin indices in coo.row/coo.col are relative to
            # the chromosome start (0-based within the chrom block). The custom_weights
            # array is indexed in absolute bin coordinates, so slice [start:end] to get
            # the per-chromosome sub-array aligned to coo's local indices.
            weights = self._custom_weights[start:end]
            factor = weights[coo.row] * weights[coo.col]
            data = coo.data * factor
            coo = sparse.coo_matrix((data, (coo.row, coo.col)), shape=coo.shape)

        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=chrom_len, coo=coo, csr=csr, csc=csc)

    def fetch_region(self, chrom: str, start_bp: int, end_bp: int) -> Optional[ChromMatrix]:
        """
        Extract a diagonal contact sub-matrix for [start_bp, end_bp) of *chrom*.

        Uses cooler's region-fetch API (``"chrom:start-end"`` format).  The returned
        matrix has local 0-based indices aligned to ``start_bp``.  Custom weight
        vectors are sliced to the region using the chromosome's absolute bin offset.
        """
        if chrom not in self._chrom_sizes:
            return None
        chrom_len = int(self._chrom_sizes[chrom])
        start_bp = max(0, int(start_bp))
        end_bp = min(chrom_len, int(end_bp))
        if end_bp <= start_bp:
            return None

        region = f"{chrom}:{start_bp}-{end_bp}"
        try:
            if self._norm_mode == "balance":
                sub = self._balanced_matrix.fetch(region, region)  # type: ignore[union-attr]
            else:
                sub = self._raw_matrix.fetch(region, region)
        except Exception:
            return None

        coo = sub.tocoo()

        finite_mask = np.isfinite(coo.data)
        if not np.all(finite_mask):
            coo = sparse.coo_matrix(
                (coo.data[finite_mask], (coo.row[finite_mask], coo.col[finite_mask])),
                shape=coo.shape,
            )

        if self._norm_mode == "custom":
            chrom_abs_start, _ = self._cooler.extent(chrom)
            region_start_abs = chrom_abs_start + start_bp // self.res
            region_bins = coo.shape[0]
            weights = self._custom_weights[region_start_abs:region_start_abs + region_bins]
            if coo.nnz > 0:
                factor = weights[coo.row] * weights[coo.col]
                data = coo.data * factor
                coo = sparse.coo_matrix((data, (coo.row, coo.col)), shape=coo.shape)

        if coo.nnz == 0:
            return None

        coo.sum_duplicates()
        csr = coo.tocsr()
        csc = coo.tocsc()
        return ChromMatrix(chrom=chrom, chrom_len=end_bp - start_bp, coo=coo, csr=csr, csc=csc)

    def fetch_region_pair(
        self,
        chrom: str,
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> Optional[sparse.csr_matrix]:
        """
        Fetch the off-diagonal rectangular block region_a × region_b via cooler.

        cooler's matrix accessor accepts two distinct region strings, so this calls
        ``matrix().fetch(region_a, region_b)``.  The returned matrix is already
        local-indexed (rows 0..n_a-1, cols 0..n_b-1).  Custom weight vectors are
        applied using separate absolute bin offsets for the row and column dimensions.
        """
        if chrom not in self._chrom_sizes:
            return None
        chrom_len = int(self._chrom_sizes[chrom])
        start_a = max(0, int(start_a))
        end_a = min(chrom_len, int(end_a))
        start_b = max(0, int(start_b))
        end_b = min(chrom_len, int(end_b))
        if end_a <= start_a or end_b <= start_b:
            return None

        region_a = f"{chrom}:{start_a}-{end_a}"
        region_b = f"{chrom}:{start_b}-{end_b}"
        try:
            if self._norm_mode == "balance":
                sub = self._balanced_matrix.fetch(region_a, region_b)  # type: ignore[union-attr]
            else:
                sub = self._raw_matrix.fetch(region_a, region_b)
        except Exception:
            return None

        coo = sub.tocoo()

        finite_mask = np.isfinite(coo.data)
        if not np.all(finite_mask):
            coo = sparse.coo_matrix(
                (coo.data[finite_mask], (coo.row[finite_mask], coo.col[finite_mask])),
                shape=coo.shape,
            )

        if self._norm_mode == "custom":
            # Row weights: absolute offset for region_a; col weights: offset for region_b.
            chrom_abs_start, _ = self._cooler.extent(chrom)
            row_offset = chrom_abs_start + start_a // self.res
            col_offset = chrom_abs_start + start_b // self.res
            n_a = coo.shape[0]
            n_b = coo.shape[1]
            row_weights = self._custom_weights[row_offset:row_offset + n_a]
            col_weights = self._custom_weights[col_offset:col_offset + n_b]
            if coo.nnz > 0:
                factor = row_weights[coo.row] * col_weights[coo.col]
                coo = sparse.coo_matrix(
                    (coo.data * factor, (coo.row, coo.col)), shape=coo.shape
                )

        if coo.nnz == 0:
            return None

        coo.sum_duplicates()
        return coo.tocsr()


def select_provider(
    path: str,
    res: int,
    norm: str,
    cooler_selection: Optional[str],
) -> Tuple[BaseProvider, str]:
    """
    Construct and return the appropriate provider for *path*, auto-detecting the format.

    Returns a 2-tuple of ``(provider, backend_name)`` where ``backend_name`` is either
    ``"cooler"`` or ``"hic"`` — this string is used in profiling output and log messages
    to tell the user which format was detected.

    Format detection:
    - Cooler is detected by ``is_cooler_path()`` (URI ``::`` or known extension).
    - Everything else is treated as a Juicer .hic file.
    """
    if is_cooler_path(path):
        provider = CoolerProvider(path, res, norm, cooler_selection)
        return provider, "cooler"
    provider = HiCProvider(path, res, norm)
    return provider, "hic"


def read_chrom_sizes(
    provider: BaseProvider,
    chrom_sizes_path: Optional[str],
) -> Dict[str, int]:
    """
    Return the chromosome name → size (bp) mapping to use for analysis.

    By default, chromosome sizes are read directly from the Hi-C file header via
    ``provider.chrom_sizes()``. An optional override file may be supplied to:
    - Restrict analysis to a subset of chromosomes (useful when only some chroms
      should be processed, e.g. canonical chromosomes only).
    - Provide sizes for formats that do not embed them reliably.

    The override is applied as an **intersection**: only chromosomes that appear in
    *both* the file header and the override file are included. This means that
    chromosomes present in the file but absent from the override file are dropped,
    and chromosomes in the override file but absent from the Hi-C file are also dropped.

    Parameters
    ----------
    provider : BaseProvider
        An open Hi-C/cooler provider whose ``chrom_sizes()`` method returns the
        available chromosomes and their lengths from the file itself.
    chrom_sizes_path : str or None
        Optional path to a two-column TSV file: ``chromosome_name<tab>size_in_bp``.
        If None or the file does not exist, the provider's own sizes are returned.

    Returns
    -------
    dict
        Chromosome name → length in bp.
    """
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
        # Keep only chromosomes that the file actually contains.
        return {chrom: provider_sizes[chrom] for chrom in provider_sizes if chrom in sizes}
    return provider.chrom_sizes()
