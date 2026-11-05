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


def test_constructor(mi):
    assert mi.cardinal() == 4 and mi.ndim() == 2
    npt.assert_array_equal(MultiIndices(mi).array, mi.array)
    assert MultiIndices([1, 2, 3]).array.shape == (1, 3)
    npt.assert_array_equal(mi.to_list()[1], [1, 2, 2, 1])
    npt.assert_array_equal(mi.get_indices([0, 3]), [[2, 1], [1, 1]])
    npt.assert_array_equal(mi.keep_indices([0, 2]).array, [[2, 1], [2, 2]])
    assert mi.plot() is None  # not implemented


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


def test_sub2ind_and_ind2sub_use_fortran_order():
    mi = MultiIndices.ind2sub([3, 4], [0, 1, 3, 11])
    npt.assert_array_equal(mi.array, [[0, 0], [1, 0], [0, 1], [2, 3]])
    npt.assert_array_equal(mi.sub2ind([3, 4]), [0, 1, 3, 11])


# --- norms and sorting ---------------------------------------------------------------


def test_norms(mi):
    a = mi.array
    npt.assert_allclose(mi.norm(), np.linalg.norm(a, axis=1))
    npt.assert_allclose(mi.norm(1), a.sum(1))
    npt.assert_allclose(mi.norm(np.inf), a.max(1))
    npt.assert_allclose(mi.norm(1, [0, 1]), [3, 3])
    w = np.array([1, 2])
    npt.assert_allclose(mi.weighted_norm(2, w), np.linalg.norm(a * w, axis=1))
    npt.assert_allclose(mi.weighted_norm(np.inf, w, [0, 1]), [2, 4])


@pytest.mark.parametrize("mode", ["ascend", "descend"])
def test_sort_by_norms(mi, mode):
    sign = 1 if mode == "ascend" else -1
    assert np.all(np.diff(mi.sort_by_norm(2, mode).norm(2)) * sign >= 0)
    w = [3, 1]
    y = mi.sort_by_weighted_norm(1, w, mode)
    assert np.all(np.diff(y.weighted_norm(1, w)) * sign >= 0)
    assert _set(y) == _set(mi)


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


def test_add_and_remove_indices(mi):
    J = MultiIndices(np.array([[3, 3], [1, 1]]))
    union = mi.add_indices(J)
    npt.assert_array_equal(union.array, np.vstack((mi.array, [[3, 3]])))  # order kept
    assert _set(mi.remove_indices(J)) == [(1, 2), (2, 1), (2, 2)]
    assert _set(mi.remove_indices([0, 1])) == [(1, 1), (2, 2)]
    assert _set(mi.remove_indices(2)) == [(1, 1), (1, 2), (2, 1)]


def test_intersect_indices(mi):
    J = MultiIndices(np.array([[1, 1], [9, 9], [2, 1]]))
    K, ind_I, ind_J = mi.intersect_indices(J)
    assert _set(K) == [(1, 1), (2, 1)]
    npt.assert_array_equal(mi.array[ind_I], J.array[ind_J])


def test_remove_and_keep_dims(mi):
    J = MultiIndices.bounded_by([1, 2, 3])
    assert _set(J.remove_dims([0, 2])) == [(0,), (1,), (2,)]
    assert _set(J.keep_dims([0, 2])) == _set(MultiIndices.bounded_by([1, 3]))
    assert _set(mi.keep_dims(1)) == [(1,), (2,)]


def test_product(mi):
    J = MultiIndices(np.array([[0], [1]]))
    P = J.product(MultiIndices(np.array([[5, 6], [7, 8]])))
    npt.assert_array_equal(P.array, [[0, 5, 6], [1, 5, 6], [0, 7, 8], [1, 7, 8]])
    npt.assert_array_equal(mi.product(None).array, mi.array)


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


def test_bounded_by():
    mi = MultiIndices.bounded_by([2, 1])
    assert _set(mi) == [(i, j) for i in range(3) for j in range(2)]
    assert _set(MultiIndices.bounded_by([2, 2], 1)) == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_is_downward_closed():
    assert MultiIndices.bounded_by([2, 3]).is_downward_closed()
    assert not MultiIndices(np.array([[0, 0], [1, 1]])).is_downward_closed()
    J = MultiIndices.bounded_by([2, 2], 1)
    assert J.is_downward_closed(1) and not J.is_downward_closed()


L_SHAPE = MultiIndices(np.array([[0, 0], [1, 0], [0, 1], [2, 0]]))


def test_margins_of_a_box():
    mi = MultiIndices.bounded_by([2, 2])
    margin = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]
    assert _set(mi.get_margin()) == margin
    assert _set(mi.get_reduced_margin()) == [(0, 3), (3, 0)]
    assert _set(mi.get_maximal_indices()) == [(2, 2)]


def test_margins_of_an_l_shaped_set():
    assert _set(L_SHAPE.get_margin()) == [(0, 2), (1, 1), (2, 1), (3, 0)]
    assert _set(L_SHAPE.get_reduced_margin()) == [(0, 2), (1, 1), (3, 0)]
    assert _set(L_SHAPE.get_maximal_indices()) == [(0, 1), (2, 0)]


@pytest.mark.parametrize("d", [1, 2, 3])
def test_margin_definitions(d):
    # margin: i not in mi with i - e_k in mi for some k; reduced margin: for
    # every k with i_k > 0
    mi = MultiIndices.hyperbolic_cross_set(d, 4)
    S = set(_set(mi))

    def below(i):
        return [tuple((i - e).tolist()) for e, ik in zip(np.eye(d, dtype=int), i) if ik > 0]

    candidates = [i for i in _box(d, 6) if tuple(i.tolist()) not in S]
    margin = sorted(tuple(i.tolist()) for i in candidates if any(b in S for b in below(i)))
    reduced = sorted(tuple(i.tolist()) for i in candidates if all(b in S for b in below(i)))
    assert _set(mi.get_margin()) == margin
    assert _set(mi.get_reduced_margin()) == reduced


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


@pytest.mark.parametrize(
    "d, p, m, w", [(2, 2, 3, [1, 2]), (2, 1, 4, [1, 2]), (2, np.inf, 3, [1, 1.5]),
                   (3, 1, 3, [1, 2, 3])]
)
def test_with_bounded_weighted_norm(d, p, m, w):
    mi = MultiIndices.with_bounded_weighted_norm(d, p, m, w)
    assert _set(mi) == _brute(d, m, lambda i: MultiIndices(i).weighted_norm(p, w)[0] <= m)
    assert np.all(np.diff(mi.weighted_norm(p, w)) >= 0)  # sorted by norm
    with pytest.raises(AssertionError):
        MultiIndices.with_bounded_weighted_norm(d + 1, p, m, w)


@pytest.mark.parametrize("d, p", [(1, 3), (2, 1), (2, 3), (3, 4)])
def test_hyperbolic_cross_set(d, p):
    mi = MultiIndices.hyperbolic_cross_set(d, p)
    assert _set(mi) == _brute(d, p, lambda i: np.prod(i + 1) <= p + 1)


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
