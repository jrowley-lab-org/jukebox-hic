"""
Shared pytest fixtures for the jukebox-hic test suite.

Fixtures defined here are automatically available to every test file in this
directory — no import needed.  A test function requests them by parameter name.
"""
import numpy as np
import scipy.sparse as sparse
import pytest

from jukebox_hic.backends import ChromMatrix


@pytest.fixture
def tiny_coo():
    """
    3×3 upper-triangle contact matrix in COO format.

    Entries: (0,0)=5.0, (0,1)=3.0, (1,2)=7.0
    Total unique contacts: 5+3+7 = 15.0 (no symmetric duplicates).
    """
    row  = np.array([0, 0, 1])
    col  = np.array([0, 1, 2])
    data = np.array([5.0, 3.0, 7.0])
    return sparse.coo_matrix((data, (row, col)), shape=(3, 3))


@pytest.fixture
def tiny_chrommatrix(tiny_coo):
    """
    ChromMatrix wrapping tiny_coo, suitable for passing to total_contacts().
    chrom_len is 30000 (3 bins × 10000 bp).
    """
    return ChromMatrix(
        chrom="chr1",
        chrom_len=30000,
        coo=tiny_coo,
        csr=tiny_coo.tocsr(),
        csc=tiny_coo.tocsc(),
    )


@pytest.fixture
def noise_track_1d():
    """
    10-bin noise array with two NaN values, representing typical sparse-bin output.
    """
    return np.array([0.5, 1.2, 3.4, 0.9, np.nan, 0.7, 2.1, 0.3, np.nan, 1.5])
