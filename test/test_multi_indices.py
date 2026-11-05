"""
Unit tests of tensap.MultiIndices.

The sets of multi-indices built by the static constructors are compared with
a brute-force enumeration of a box of N^d; the other methods are checked on
small sets written by hand.
"""

import itertools

import numpy as np
import numpy.testing as npt
import pytest
from tensap import MultiIndices


def _set(mi):
    return sorted(map(tuple, mi.array.tolist()))


def _box(d, n):
    return [np.array(i) for i in itertools.product(range(n + 1), repeat=d)]


def _brute(d, n, keep):
    """The multi-indices of {0, ..., n}^d such that keep(i)."""
    return sorted(tuple(i.tolist()) for i in _box(d, n) if keep(i))


@pytest.fixture
def mi():
    return MultiIndices(np.array([[2, 1], [1, 2], [2, 2], [1, 1]]))


# --- construction and properties ---------------------------------------------


def test_comparisons_do_not_modify_the_argument(mi):
    J = MultiIndices([2, 1])
    npt.assert_array_equal(mi == J, [True, False, False, False])
    npt.assert_array_equal(mi <= J, [True, False, False, True])
    assert J.array.shape == (1, 2)
    npt.assert_array_equal(mi == MultiIndices(np.flip(mi.array, 0)), [False] * 4)
    assert not np.any(mi == mi.array)


def test_arithmetic(mi):
    npt.assert_array_equal((mi + 1).array, mi.array + 1)
    npt.assert_array_equal((mi - 1).array, mi.array - 1)
    npt.assert_array_equal((1 + mi).array, mi.array + 1)
    npt.assert_array_equal((5 - mi).array, 5 - mi.array)
    npt.assert_array_equal(mi.array, [[2, 1], [1, 2], [2, 2], [1, 1]])  # unchanged


# --- norms and sorting ---------------------------------------------------------------


@pytest.mark.parametrize(
    "columns, mode, expected",
    [
        (None, "ascend", [[1, 1], [2, 1], [1, 2], [2, 2]]),
        (None, "descend", [[2, 2], [1, 2], [2, 1], [1, 1]]),
        ([0, 1], "ascend", [[1, 1], [1, 2], [2, 1], [2, 2]]),
        ([0, 1], "descend", [[2, 2], [2, 1], [1, 2], [1, 1]]),
    ],
)
def test_sort_is_lexicographic(mi, columns, mode, expected):
    # by default, sort by the last column first
    npt.assert_array_equal(mi.sort(columns, mode).array, expected)


# --- set operations ----------------------------------------------------------------------


def test_remove_and_keep_dims(mi):
    J = MultiIndices.bounded_by([1, 2, 3])
    assert _set(J.remove_dims([0, 2])) == [(0,), (1,), (2,)]
    assert _set(J.keep_dims([0, 2])) == _set(MultiIndices.bounded_by([1, 3]))
    assert _set(mi.keep_dims(1)) == [(1,), (2,)]


def test_product_set():
    # first index varies fastest
    P = MultiIndices.product_set([[0, 1], [5, 6, 7]])
    npt.assert_array_equal(P.array[:3], [[0, 5], [1, 5], [0, 6]])
    assert P.cardinal() == 6
    sets = [np.array([0, 1]), MultiIndices(np.array([[2, 3], [4, 5]]))]
    P = MultiIndices.product_set(sets)
    assert isinstance(sets[1], MultiIndices)  # the input is not modified
    assert _set(P) == [(0, 2, 3), (0, 4, 5), (1, 2, 3), (1, 4, 5)]
    # one set of indices, d times
    assert _set(MultiIndices.product_set(np.arange(3), 2)) == _brute(2, 2, lambda i: True)
    assert _set(MultiIndices.product_set([np.arange(3)], 2)) == _brute(2, 2, lambda i: True)
    with pytest.raises(ValueError):
        MultiIndices.product_set([np.zeros((2, 2, 2))])


# --- downward closed sets ---------------------------------------------------------------------


L_SHAPE = MultiIndices(np.array([[0, 0], [1, 0], [0, 1], [2, 0]]))


def test_envelope():
    mi = MultiIndices(np.array([[0], [1], [2]]))
    npt.assert_array_equal(mi.envelope([3, -1, 2]), [3, 2, 2])
    # in dimension 2, max over the multi-indices j >= i
    npt.assert_array_equal(L_SHAPE.envelope(np.array([1, 5, -6, 2])), [6, 5, 6, 2])
    with pytest.raises(AssertionError):
        mi.envelope([1, 2])


# --- static constructors of sets ---------------------------------------------------------------


@pytest.mark.parametrize(
    "d, p, m", [(2, 1, 3), (3, 1, 2), (2, 2, 3), (3, 2, 2), (2, 2, 1.5), (3, 3, 2),
                (2, 0.5, 2.5), (2, np.inf, 2), (3, np.inf, 1)]
)
def test_with_bounded_norm(d, p, m):
    mi = MultiIndices.with_bounded_norm(d, p, m)
    assert _set(mi) == _brute(d, int(m), lambda i: MultiIndices(i).norm(p)[0] <= m)


@pytest.mark.parametrize("d, p, w", [(2, 3, [1, 1]), (2, 5, [1, 2]), (3, 4, [1, 1, 1]),
                                     (2, 7, [0.5, 2])])
def test_weighted_hyperbolic_cross_set(d, p, w):
    mi = MultiIndices.weighted_hyperbolic_cross_set(d, p, np.array(w))
    n = int((p + 1) ** (1 / min(w)))  # largest index in one dimension
    assert _set(mi) == _brute(d, n, lambda i: np.prod((i + 1.0) ** np.array(w)) <= p + 1)


@pytest.mark.parametrize("d, p, m", [(3, 2, 1), (3, 2, 2), (2, 3, 1), (4, 1, 2), (3, 1, 3)])
def test_low_order_interactions(d, p, m):
    mi = MultiIndices.low_order_interactions(d, p, m)
    assert _set(mi) == _brute(d, p, lambda i: np.count_nonzero(i) <= m)
