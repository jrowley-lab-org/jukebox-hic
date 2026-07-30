"""
Tests for src/jukebox_hic/backends.py — pure helper functions only.

All tests here avoid opening any real .hic or .cool files.  Tests that would
need a real file (CoolerProvider, HiCProvider instance methods, resolve_cooler_uri
against a real .mcool) are left out of scope and can be added as integration tests
once a small fixture file is committed to the repo.
"""
import numpy as np
import scipy.sparse as sparse
import pytest

from jukebox_hic.backends import (
    ChromMatrix,
    is_cooler_path,
    normalize_hic_norm,
    resolve_cooler_uri,
    total_contacts,
)


# ---------------------------------------------------------------------------
# is_cooler_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "sample.cool",
    "sample.mcool",
    "sample.MCOOL",        # case-insensitive
    "sample.scool",
    "sample.mcool::resolutions/10000",   # already a URI
    "/abs/path/to/data.cool",
])
def test_is_cooler_path_true(path):
    assert is_cooler_path(path) is True


@pytest.mark.parametrize("path", [
    "sample.hic",
    "sample.HIC",
    "sample.txt",
    "sample.bam",
    "/abs/path/to/data.hic",
])
def test_is_cooler_path_false(path):
    assert is_cooler_path(path) is False


# ---------------------------------------------------------------------------
# normalize_hic_norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm,expected", [
    ("none",    "NONE"),
    ("NONE",    "NONE"),
    ("balance", "KR"),
    ("BALANCE", "KR"),
    ("vc",      "VC"),
    ("VC_SQRT", "VC_SQRT"),
    ("SCALE",   "SCALE"),
])
def test_normalize_hic_norm_mappings(norm, expected):
    assert normalize_hic_norm(norm) == expected


def test_normalize_hic_norm_unknown_uppercases():
    # Unknown labels are upper-cased and returned as-is
    assert normalize_hic_norm("custom") == "CUSTOM"


# ---------------------------------------------------------------------------
# resolve_cooler_uri — path-construction only (no file I/O)
# ---------------------------------------------------------------------------

def test_resolve_cooler_uri_already_uri():
    # If path already contains ::, it is returned unchanged
    uri = "sample.mcool::resolutions/10000"
    assert resolve_cooler_uri(uri, res=10_000, selection=None) == uri


def test_resolve_cooler_uri_plain_cool():
    # .cool files are single-resolution; the path is the URI
    path = "/data/sample.cool"
    assert resolve_cooler_uri(path, res=10_000, selection=None) == path


def test_resolve_cooler_uri_scool_with_selection():
    # .scool: the URI is path::selection (no file-existence check needed)
    path = "/data/sample.scool"
    result = resolve_cooler_uri(path, res=10_000, selection="cell_1")
    assert result == "/data/sample.scool::cell_1"


def test_resolve_cooler_uri_scool_no_selection_raises():
    with pytest.raises(ValueError, match="cooler_path"):
        resolve_cooler_uri("/data/sample.scool", res=10_000, selection=None)


def test_resolve_cooler_uri_unrecognized_suffix_raises():
    with pytest.raises(ValueError, match="[Uu]nrecognized"):
        resolve_cooler_uri("/data/sample.unknown", res=10_000, selection=None)


# ---------------------------------------------------------------------------
# total_contacts — synthetic ChromMatrix
# ---------------------------------------------------------------------------

def test_total_contacts_no_symmetric_duplicates(tiny_chrommatrix):
    # tiny_coo has (0,0)=5, (0,1)=3, (1,2)=7 — no symmetric duplicates
    # so total = 5 + 3 + 7 = 15
    assert total_contacts(tiny_chrommatrix) == pytest.approx(15.0)


def test_total_contacts_symmetric_matrix_deduplicates():
    # Build an explicitly symmetric matrix (i,j) and (j,i) both stored
    # The function should count each pair once via the bit-packing trick
    row  = np.array([0, 1])   # (0,1) and (1,0) are the same pair
    col  = np.array([1, 0])
    data = np.array([3.0, 3.0])
    coo = sparse.coo_matrix((data, (row, col)), shape=(2, 2))
    mat = ChromMatrix(
        chrom="chr1", chrom_len=20_000,
        coo=coo, csr=coo.tocsr(), csc=coo.tocsc(),
    )
    # The pair (0,1) should be counted once (average of 3.0 and 3.0 = 3.0)
    assert total_contacts(mat) == pytest.approx(3.0)


def test_total_contacts_empty_matrix_returns_zero():
    coo = sparse.coo_matrix((3, 3))
    mat = ChromMatrix(
        chrom="chr1", chrom_len=30_000,
        coo=coo, csr=coo.tocsr(), csc=coo.tocsc(),
    )
    assert total_contacts(mat) == 0.0


def test_total_contacts_diagonal_only():
    # Only self-contacts (i==i), each key is unique, so all are counted
    row  = np.array([0, 1, 2])
    col  = np.array([0, 1, 2])
    data = np.array([4.0, 5.0, 6.0])
    coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
    mat = ChromMatrix(
        chrom="chr1", chrom_len=30_000,
        coo=coo, csr=coo.tocsr(), csc=coo.tocsc(),
    )
    assert total_contacts(mat) == pytest.approx(15.0)
