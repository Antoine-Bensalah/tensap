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


def test_repr():
    assert "tolerance = 1e-08" in repr(tensap.Truncator())


# --- truncated svd of a matrix ---------------------------------------------------


SINGULAR_VALUES = [10.0, 1.0, 0.1, 0.01]


def _frobenius_rank(tolerance):
    """The smallest rank with a relative error below the tolerance."""
    s = np.array(SINGULAR_VALUES)
    tails = [np.linalg.norm(s[k:]) / np.linalg.norm(s) for k in range(1, s.size)]
    return 1 + int(np.nonzero(np.array(tails + [0.0]) < tolerance)[0][0])


@pytest.mark.parametrize("tolerance", [0.5, 1e-1, 5e-3, 1e-12])
def test_trunc_svd_in_frobenius_norm(rng, tolerance):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    rank = _frobenius_rank(tolerance)
    y = tensap.Truncator(tolerance).trunc_svd(m)
    assert y.representation_rank() == rank
    npt.assert_allclose(y.core.data, SINGULAR_VALUES[:rank])
    assert _relative_error(y.numpy(), m) < tolerance


@pytest.mark.parametrize("power", [1, np.inf, "inf"])
def test_trunc_svd_in_schatten_norms(rng, power):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    s = np.array(SINGULAR_VALUES)
    y = tensap.Truncator().trunc_svd(m, tolerance=0.05, power=power)
    r = y.representation_rank()
    # the smallest rank with a relative error below the tolerance
    if power == 1:
        errors = [np.sum(s[k:]) / np.sum(s) for k in range(1, 5)]
    else:
        errors = [s[k] / s[0] if k < 4 else 0.0 for k in range(1, 5)]
    assert r == 1 + int(np.nonzero(np.array(errors) < 0.05)[0][0])


def test_trunc_svd_max_rank(rng):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    assert tensap.Truncator(1e-12, max_rank=2).trunc_svd(m).representation_rank() == 2


@pytest.mark.parametrize("kind, expected", [("hard", [10.0, 1.0]), ("soft", [9.5, 0.5])])
def test_trunc_svd_thresholding(rng, kind, expected):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    truncator = tensap.Truncator(1e-12, thresholding_type=kind, thresholding_parameter=0.5)
    npt.assert_allclose(truncator.trunc_svd(m).core.data, expected)


@pytest.mark.xfail(
    strict=True,
    reason="trunc_svd fails when every singular value is below the thresholding "
    "parameter (a rank 0 result is not handled)",
)
def test_trunc_svd_thresholding_of_every_singular_value(rng):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    tensap.Truncator(thresholding_parameter=100.0).trunc_svd(m)


# --- svd of an order-2 tensor ---------------------------------------------------------


@pytest.mark.parametrize("wrap", [lambda m: m, tensap.FullTensor])
def test_svd_of_a_matrix(rng, wrap):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    y = tensap.Truncator(1e-2).svd(wrap(m))
    assert y.representation_rank() == _frobenius_rank(1e-2)
    assert _relative_error(y.numpy(), m) < 1e-2


def test_svd_of_a_canonical_tensor(rng):
    space = [rng.standard_normal((6, 3)), rng.standard_normal((5, 3))]
    x = tensap.CanonicalTensor(space, np.array([1.0, 1e-3, 1e-6]))
    a = x.numpy()
    y = tensap.Truncator(1e-2).svd(x)
    assert y.representation_rank() < 3
    npt.assert_array_equal(y.shape, [6, 5])
    assert _relative_error(y.numpy(), a) < 1e-2


def test_svd_of_a_tree_based_tensor():
    x = tensap.TreeBasedTensor.randn(tensap.DimensionTree.trivial(2), [1, 2, 2], [3, 4])
    y = tensap.Truncator().svd(x)
    assert isinstance(y, tensap.TreeBasedTensor)
    npt.assert_allclose(y.full().numpy(), x.full().numpy())


def test_svd_rejects_other_formats(rng):
    x = tensap.SparseTensor(rng.standard_normal((3, 4)))
    with pytest.raises(NotImplementedError):
        tensap.Truncator().svd(x)


# --- truncation of higher-order tensors ------------------------------------------------


@pytest.mark.parametrize("method", ["truncate", "hosvd", "hsvd", "ttsvd"])
def test_exact_recovery_of_a_low_rank_tensor(x, method):
    a = x.full().numpy()
    truncator = tensap.Truncator(1e-10)
    full = tensap.FullTensor(a)
    if method == "hsvd":
        y = truncator.hsvd(full, TREE)
        npt.assert_array_equal(y.ranks, x.ranks)
    else:
        y = getattr(truncator, method)(full)
    npt.assert_allclose(y.full().numpy(), a, atol=1e-10 * np.linalg.norm(a))
    if method == "hosvd":
        assert y.tree == tensap.DimensionTree.trivial(4)
        npt.assert_array_equal(y.ranks, [1, 3, 3, 3, 3])


def test_hosvd_of_an_array(x):
    a = x.full().numpy()
    npt.assert_allclose(tensap.Truncator(1e-10).hosvd(a).full().numpy(), a, atol=1e-8)


def test_hosvd_of_an_order_two_tensor(rng):
    m = _low_rank_matrix(rng, SINGULAR_VALUES)
    assert tensap.Truncator(1e-2).hosvd(m).representation_rank() == _frobenius_rank(1e-2)


def test_hosvd_rejects_other_formats(x):
    with pytest.raises(ValueError):
        tensap.Truncator().hosvd(x)


@pytest.mark.parametrize("tolerance", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("method", ["hosvd", "hsvd", "ttsvd"])
def test_truncation_error_of_a_full_tensor_is_below_the_tolerance(rng, method, tolerance):
    a = rng.standard_normal(SHAPE)
    truncator = tensap.Truncator(tolerance)
    full = tensap.FullTensor(a)
    y = truncator.hsvd(full, TREE) if method == "hsvd" else getattr(truncator, method)(full)
    assert _relative_error(y.full().numpy(), a) <= tolerance


@pytest.mark.parametrize("hsvd_type", [1, 2])
@pytest.mark.parametrize("tolerance", [1e-10, 0.1, 0.3])
def test_hsvd_of_a_tree_based_tensor(x, hsvd_type, tolerance):
    a = x.full().numpy()
    truncator = tensap.Truncator(tolerance)
    truncator._hsvd_type = hsvd_type
    y = truncator.hsvd(x)
    assert y.tree == x.tree and np.all(y.ranks <= x.ranks)
    assert _relative_error(y.full().numpy(), a) <= tolerance + 1e-12
    if tolerance < 1e-8:
        npt.assert_array_equal(y.ranks, x.ranks)
    elif tolerance == 0.3:
        assert np.any(y.ranks < x.ranks)


@pytest.mark.parametrize("hsvd_type", [1, 2])
def test_hsvd_of_a_tree_based_tensor_with_an_inactive_leaf(x, hsvd_type):
    x = x.inactivate_nodes([TREE.dim2ind[0]])
    a = x.full().numpy()
    truncator = tensap.Truncator(0.1)
    truncator._hsvd_type = hsvd_type
    y = truncator.hsvd(x)
    npt.assert_array_equal(y.is_active_node, x.is_active_node)
    assert _relative_error(y.full().numpy(), a) <= 0.1


def test_truncate_a_tree_based_tensor(x):
    a = x.full().numpy()
    y = tensap.Truncator(1e-10).truncate(x)
    npt.assert_allclose(y.full().numpy(), a, atol=1e-10 * np.linalg.norm(a))


def test_hsvd_ignores_the_tree_of_a_tree_based_tensor(x):
    with pytest.warns(UserWarning):
        y = tensap.Truncator().hsvd(x, tensap.DimensionTree.linear(4))
    assert y.tree == x.tree


def test_hsvd_with_a_root_rank(rng):
    # a full tensor with one more dimension than the tree: the root rank
    b = rng.standard_normal(SHAPE + [2])
    y = tensap.Truncator(1e-12).hsvd(tensap.FullTensor(b), TREE)
    assert y.ranks[TREE.root - 1] == 2
    npt.assert_allclose(y.full().numpy(), b, atol=1e-10)


def test_hsvd_with_inactive_leaves(x):
    a = x.full().numpy()
    active = np.full(TREE.nb_nodes, True)
    active[TREE.dim2ind[[0, 3]] - 1] = False
    y = tensap.Truncator(1e-10).hsvd(tensap.FullTensor(a), TREE, active)
    npt.assert_array_equal(y.is_active_node, active)
    npt.assert_allclose(y.full().numpy(), a, atol=1e-8)


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


def test_hsvd_wrong_arguments(x):
    truncator = tensap.Truncator()
    truncator._hsvd_type = 3
    with pytest.raises(ValueError):
        truncator.hsvd(x)
    with pytest.raises(NotImplementedError):
        tensap.Truncator().hsvd(tensap.CanonicalTensor.randn(2, SHAPE), TREE)


def test_truncate_wrong_arguments():
    truncator = tensap.Truncator()
    with pytest.raises(ValueError):
        truncator.truncate(np.ones(3))
    with pytest.raises(NotImplementedError):
        truncator.truncate(tensap.CanonicalTensor.randn(2, SHAPE))

