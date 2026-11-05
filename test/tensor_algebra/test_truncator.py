"""
Unit tests of tensap.Truncator.

Truncations are checked through their guarantees: the relative error in
Frobenius norm is below the tolerance, the ranks are below max_rank, and a
tensor of exact low rank is recovered with its ranks.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

SHAPE = [4, 5, 4, 5]
TREE = tensap.DimensionTree.balanced(4)


def _relative_error(y, a):
    return np.linalg.norm(y - a) / np.linalg.norm(a)


def _low_rank_matrix(rng, singular_values, shape=(8, 6)):
    r = len(singular_values)
    u = np.linalg.qr(rng.standard_normal((shape[0], r)))[0]
    v = np.linalg.qr(rng.standard_normal((shape[1], r)))[0]
    return u @ np.diag(singular_values) @ v.T


@pytest.fixture
def x():
    """A tree-based tensor of ranks 3 (np.random is seeded)."""
    return tensap.TreeBasedTensor.randn(TREE, np.r_[1, np.full(6, 3)], SHAPE)


# --- truncated svd of a matrix ---------------------------------------------------


SINGULAR_VALUES = [10.0, 1.0, 0.1, 0.01]


def _frobenius_rank(tolerance):
    """The smallest rank with a relative error below the tolerance."""
    s = np.array(SINGULAR_VALUES)
    tails = [np.linalg.norm(s[k:]) / np.linalg.norm(s) for k in range(1, s.size)]
    return 1 + int(np.nonzero(np.array(tails + [0.0]) < tolerance)[0][0])


# --- svd of an order-2 tensor ---------------------------------------------------------


# --- truncation of higher-order tensors ------------------------------------------------


@pytest.mark.parametrize("method", ["hosvd", "hsvd"])
def test_max_rank_per_node_is_respected_and_kept(x, method):
    full = tensap.FullTensor(x.full().numpy())
    if method == "hosvd":
        max_rank = np.array([2, 3, 1, 2])
        expected = np.r_[1, max_rank]
    else:
        max_rank = np.array([1, 2, 2, 3, 1, 3, 2])
        expected = max_rank
    truncator = tensap.Truncator(1e-12, max_rank)
    call = (lambda: truncator.hosvd(full)) if method == "hosvd" else (
        lambda: truncator.hsvd(full, TREE))
    npt.assert_array_equal(call().ranks, expected)
    npt.assert_array_equal(truncator.max_rank, max_rank)
    npt.assert_array_equal(call().ranks, expected)  # same result twice
