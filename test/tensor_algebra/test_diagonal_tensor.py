"""
Unit tests of tensap.DiagonalTensor.

A diagonal tensor of order d and rank r has entries x(i, ..., i) = data[i].
Every operation is checked against the same computation on its dense array.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _diag(rng, rank=3, order=3):
    data = rng.standard_normal(rank)
    dense = np.zeros((rank,) * order)
    for i in range(rank):
        dense[(i,) * order] = data[i]
    return tensap.DiagonalTensor(data, order), dense


# --- construction and properties ---------------------------------------------


def test_constructor_and_properties(rng):
    x, a = _diag(rng, 3, 4)
    assert x.order == x.ndim == 4
    npt.assert_array_equal(x.shape, [3, 3, 3, 3])
    assert x.storage() == 3
    assert x.is_orth
    npt.assert_array_equal(x.numpy(), a)
    npt.assert_array_equal(x.full().numpy(), a)


def test_copy_constructor(rng):
    x, a = _diag(rng)
    y = tensap.DiagonalTensor(x)
    npt.assert_array_equal(y.numpy(), a)
    y.data[0] = 0.0
    assert x.data[0] != 0.0


def test_sparse_storage_counts_nonzeros():
    assert tensap.DiagonalTensor(np.array([1.0, 0.0, 2.0]), 2).sparse_storage() == 2


@pytest.mark.parametrize(
    "constructor, check",
    [
        (tensap.DiagonalTensor.zeros, lambda d: np.all(d == 0)),
        (tensap.DiagonalTensor.ones, lambda d: np.all(d == 1)),
        (tensap.DiagonalTensor.rand, lambda d: np.all((d >= 0) & (d <= 1))),
        (tensap.DiagonalTensor.randn, lambda d: d.size == 4),
    ],
)
def test_constructors(constructor, check):
    x = constructor(4, 3)
    assert x.order == 3 and x.data.size == 4
    assert check(x.data)


# --- arithmetic ---------------------------------------------------------------


def test_arithmetic_between_diagonal_tensors(rng):
    x, a = _diag(rng)
    y, b = _diag(rng)
    npt.assert_allclose((x + y).numpy(), a + b)
    npt.assert_allclose((x - y).numpy(), a - b)
    npt.assert_allclose((-x).numpy(), -a)
    npt.assert_allclose((x * y).numpy(), a * b)


@pytest.mark.xfail(
    strict=True,
    reason="DiagonalTensor.__mul__ is the Hadamard product with a tensor: "
    "multiplication by a scalar fails",
)
def test_multiplication_by_a_scalar(rng):
    x, a = _diag(rng)
    npt.assert_allclose((2.0 * x).numpy(), 2.0 * a)


def test_dot_and_norm(rng):
    x, a = _diag(rng)
    y, b = _diag(rng)
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))


@pytest.mark.xfail(
    strict=True,
    reason="DiagonalTensor.dot assumes a diagonal argument (see the migration "
    "journal of 21 October 2026)",
)
def test_dot_with_a_full_tensor(rng):
    x, a = _diag(rng)
    y = tensap.FullTensor(rng.standard_normal(a.shape))
    npt.assert_allclose(x.dot(y), np.sum(a * y.numpy()))


# --- structure ---------------------------------------------------------------------


def test_cat_is_block_diagonal(rng):
    x, a = _diag(rng, 2, 3)
    y, b = _diag(rng, 3, 3)
    npt.assert_allclose(x.cat(y).numpy(), x.full().cat(y.full()).numpy())


def test_kron(rng):
    x, _ = _diag(rng, 2, 3)
    y, _ = _diag(rng, 3, 3)
    npt.assert_allclose(x.kron(y).numpy(), x.full().kron(y.full()).numpy())


@pytest.mark.xfail(
    strict=True,
    reason="DiagonalTensor.kron assumes a diagonal argument (see the migration "
    "journal of 22 October 2026)",
)
def test_kron_with_a_full_tensor(rng):
    x, _ = _diag(rng, 2, 2)
    y = tensap.FullTensor(rng.standard_normal((3, 3)))
    npt.assert_allclose(x.kron(y).numpy(), x.full().kron(y).numpy())


def test_transpose_keeps_the_tensor(rng):
    x, a = _diag(rng)
    npt.assert_array_equal(x.transpose([2, 0, 1]).numpy(), a)
    npt.assert_array_equal(x.transpose([2, 0, 1]).itranspose([2, 0, 1]).numpy(), a)


def test_reshape_changes_only_the_shape(rng):
    x, _ = _diag(rng, 3, 2)
    npt.assert_array_equal(x.reshape([9, 1]).shape, [9, 1])


def test_orth_returns_a_copy(rng):
    x, a = _diag(rng)
    y = x.orth()
    npt.assert_array_equal(y.numpy(), a)
    assert y is not x


def test_update_attributes(rng):
    x, _ = _diag(rng, 3, 2)
    x.data = np.ones(5)
    x.update_attributes()
    npt.assert_array_equal(x.shape, [5, 5])


def test_sub_tensor(rng):
    x, a = _diag(rng)
    s = x.sub_tensor([0, 2], ":", [1, 2])
    npt.assert_array_equal(s.numpy(), a[np.ix_([0, 2], range(3), [1, 2])])


# --- contractions ---------------------------------------------------------------


def test_tensor_matrix_product(rng):
    x, a = _diag(rng)
    M = [rng.standard_normal((4, 3)) for _ in range(3)]
    npt.assert_allclose(
        x.tensor_matrix_product(M).numpy(), np.einsum("ijk,ai,bj,ck->abc", a, *M)
    )


def test_tensor_vector_product_on_all_dimensions(rng):
    x, a = _diag(rng)
    v = [rng.standard_normal(3) for _ in range(3)]
    npt.assert_allclose(x.tensor_vector_product(v), np.einsum("ijk,i,j,k->", a, *v))


def test_tensor_diagonal_matrix_product(rng):
    x, a = _diag(rng)
    d = [rng.standard_normal(3) for _ in range(3)]
    expected = np.einsum("ijk,i,j,k->ijk", a, *d)
    npt.assert_allclose(x.tensor_diagonal_matrix_product(d).numpy(), expected)


def test_tensordot(rng):
    x, a = _diag(rng)
    y = tensap.FullTensor(rng.standard_normal((3, 4, 3)))
    npt.assert_allclose(
        x.tensordot(y, [0, 2], [2, 0]).numpy(),
        np.tensordot(a, y.numpy(), [[0, 2], [2, 0]]),
    )


def test_dot_with_rank_one_metric(rng):
    x, a = _diag(rng, 3, 2)
    y, b = _diag(rng, 4, 2)
    M = [rng.standard_normal((3, 4)) for _ in range(2)]
    npt.assert_allclose(
        x.dot_with_rank_one_metric(y, M), np.einsum("ij,kl,ik,jl->", a, b, *M)
    )


def test_tensordot_matrix_product_except_dim(rng):
    x, a = _diag(rng, 3, 3)
    y, b = _diag(rng, 4, 3)
    M = [rng.standard_normal((3, 4)) for _ in range(3)]
    out = x.tensordot_matrix_product_except_dim(y, M, 1)
    expected = np.einsum("ijk,lmn,il,kn->jm", a, b, M[0], M[2])
    npt.assert_allclose(out, expected)


def test_tensor_matrix_product_eval_diag(rng):
    x, a = _diag(rng)
    H = [rng.standard_normal((5, 3)) for _ in range(3)]
    out = x.tensor_matrix_product_eval_diag(H)
    npt.assert_allclose(np.ravel(out), np.einsum("ijk,ai,aj,ak->a", a, *H))


# --- diagonal and evaluations ----------------------------------------------------


def test_eval_diag_on_all_dimensions(rng):
    x, _ = _diag(rng)
    npt.assert_array_equal(x.eval_diag(), x.data)


def test_eval_diag_on_contiguous_dimensions(rng):
    x, _ = _diag(rng, 3, 3)
    npt.assert_allclose(x.eval_diag([0, 1]).numpy(), x.full().eval_diag([0, 1]).numpy())


@pytest.mark.xfail(
    strict=True,
    reason="DiagonalTensor.eval_diag drops the dimensions between non contiguous "
    "dims: the order of the result is wrong",
)
def test_eval_diag_on_non_contiguous_dimensions(rng):
    x, _ = _diag(rng, 3, 3)
    npt.assert_allclose(x.eval_diag([0, 2]).numpy(), x.full().eval_diag([0, 2]).numpy())


def test_eval_at_indices(rng):
    x, a = _diag(rng)
    indices = np.array([[0, 0, 0], [1, 2, 1], [2, 2, 2]])
    npt.assert_allclose(x.eval_at_indices(indices), a[tuple(indices.T)])
    with pytest.raises(NotImplementedError):
        x.eval_at_indices(np.array([[0, 0]]), [0, 1])


# --- conversions ------------------------------------------------------------------


def test_sparse_conversion(rng):
    x, a = _diag(rng)
    npt.assert_allclose(x.sparse().full().numpy(), a)


@pytest.mark.parametrize(
    "tree", [None, tensap.DimensionTree.balanced(4), tensap.DimensionTree.trivial(4)]
)
def test_tree_based_tensor_conversion(rng, tree):
    x, a = _diag(rng, 3, 4)
    npt.assert_allclose(x.tree_based_tensor(tree).full().numpy(), a)


def test_tree_based_tensor_with_active_leaves(rng):
    x, a = _diag(rng, 3, 3)
    tree = tensap.DimensionTree.linear(3)
    active = np.full(tree.nb_nodes, True)
    npt.assert_allclose(x.tree_based_tensor(tree, active).full().numpy(), a)


def test_tree_based_tensor_requires_active_internal_nodes(rng):
    x, _ = _diag(rng, 3, 4)
    tree = tensap.DimensionTree.balanced(4)
    active = np.full(tree.nb_nodes, False)
    active[0] = True  # root only
    with pytest.raises(ValueError):
        x.tree_based_tensor(tree, active)


def test_tensor_vector_product_on_some_dimensions(rng):
    x, a = _diag(rng)
    v = [rng.standard_normal(3) for _ in range(2)]
    y = x.tensor_vector_product(v)
    assert isinstance(y, tensap.DiagonalTensor) and y.order == 1
    npt.assert_allclose(y.numpy(), np.einsum("ijk,i,j->k", a, *v))


def test_repr(rng):
    x, _ = _diag(rng, 3, 2)
    assert "DiagonalTensor" in repr(x) and "order = 2" in repr(x)
