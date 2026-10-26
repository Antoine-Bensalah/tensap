"""
Unit tests of tensap.SparseTensor.

Every operation is checked against the same computation on the dense array.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _sparse(rng, shape=(3, 4, 5), density=0.3):
    dense = rng.standard_normal(shape) * (rng.random(shape) < density)
    return tensap.SparseTensor(dense), dense


# --- construction and properties ---------------------------------------------


def test_constructor_from_a_dense_array(rng):
    x, a = _sparse(rng)
    assert x.order == x.ndim == 3
    npt.assert_array_equal(x.shape, a.shape)
    assert x.size == a.size
    assert x.storage() == x.sparse_storage() == x.count_non_zero() == np.count_nonzero(a)
    npt.assert_array_equal(x.full().numpy(), a)


def test_constructor_from_data_indices_and_shape():
    indices = tensap.MultiIndices(np.array([[0, 1], [2, 0]]))
    x = tensap.SparseTensor(np.array([5.0, 7.0]), indices, [3, 2])
    expected = np.zeros((3, 2))
    expected[0, 1], expected[2, 0] = 5.0, 7.0
    npt.assert_array_equal(x.full().numpy(), expected)


def test_copy_constructor(rng):
    x, a = _sparse(rng)
    npt.assert_array_equal(tensap.SparseTensor(x).full().numpy(), a)


def test_empty_constructor():
    x = tensap.SparseTensor()
    assert x.order == 0


@pytest.mark.parametrize(
    "args",
    [("not a tensor",), (np.ones(2), tensap.MultiIndices(np.array([[0], [1]])))],
)
def test_constructor_rejects_wrong_arguments(args):
    with pytest.raises(ValueError):
        tensap.SparseTensor(*args)


def test_numpy_for_matrices(rng):
    x, a = _sparse(rng, (4, 5))
    npt.assert_array_equal(x.numpy().todense(), a)


# --- arithmetic ---------------------------------------------------------------


def test_addition(rng):
    x, a = _sparse(rng)
    y, b = _sparse(rng)
    npt.assert_allclose((x + y).full().numpy(), a + b)


def test_hadamard_product(rng):
    x, a = _sparse(rng, density=0.6)
    y, b = _sparse(rng, density=0.6)
    npt.assert_allclose((x * y).full().numpy(), a * b)


def test_dot_and_norm(rng):
    x, a = _sparse(rng, density=0.6)
    y, b = _sparse(rng, density=0.6)
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))


def test_dot_with_a_full_tensor(rng):
    x, a = _sparse(rng)
    y = tensap.FullTensor(rng.standard_normal(a.shape))
    npt.assert_allclose(x.dot(y), np.sum(a * y.numpy()))


def test_dot_rejects_an_unconvertible_argument(rng):
    x, _ = _sparse(rng)
    with pytest.raises(ValueError):
        x.dot(object())
    with pytest.raises(ValueError):
        x * object()


# --- evaluations ---------------------------------------------------------------


def test_eval_at_indices(rng):
    x, a = _sparse(rng, density=0.5)
    indices = np.array([[0, 0, 0], [1, 2, 3], [2, 3, 4], [0, 1, 2]])
    out = x.eval_at_indices(indices)
    npt.assert_allclose(np.ravel(out.todense()), a[tuple(indices.T)])
    out = x.eval_at_indices(tensap.MultiIndices(indices))
    npt.assert_allclose(np.ravel(out.todense()), a[tuple(indices.T)])


def test_eval_diag(rng):
    x, a = _sparse(rng, (4, 4, 4), density=0.8)
    out = x.eval_diag()
    npt.assert_allclose(np.ravel(out.todense()), [a[i, i, i] for i in range(4)])
    assert x.eval_diag([1]) is x


# --- structure ---------------------------------------------------------------------


def test_transpose_and_itranspose(rng):
    x, a = _sparse(rng)
    perm = [2, 0, 1]
    npt.assert_array_equal(x.transpose(perm).full().numpy(), np.transpose(a, perm))
    npt.assert_array_equal(x.transpose(perm).itranspose(perm).full().numpy(), a)


def test_reshape_uses_fortran_order(rng):
    x, a = _sparse(rng)
    npt.assert_array_equal(
        x.reshape([12, 5]).full().numpy(), np.reshape(a, (12, 5), order="F")
    )


# --- contractions ---------------------------------------------------------------


def test_tensor_vector_product_on_all_dimensions(rng):
    x, a = _sparse(rng, density=0.5)
    v = [rng.standard_normal(n) for n in a.shape]
    npt.assert_allclose(
        np.sum(x.tensor_vector_product(v)), np.einsum("ijk,i,j,k->", a, *v)
    )


def test_tensor_vector_product_on_some_dimensions(rng):
    x, a = _sparse(rng, density=0.5)
    v = rng.standard_normal(4)
    out = x.tensor_vector_product(v, [1])
    npt.assert_allclose(out.full().numpy(), np.einsum("ijk,j->ik", a, v))


def test_tensor_matrix_product(rng):
    x, a = _sparse(rng)
    M = [rng.standard_normal((2, n)) for n in a.shape]
    npt.assert_allclose(
        x.tensor_matrix_product(M).full().numpy(),
        np.einsum("ijk,ai,bj,ck->abc", a, *M),
    )
    npt.assert_allclose(
        x.tensor_matrix_product(M[2], 2).full().numpy(),
        np.einsum("ijk,ck->ijc", a, M[2]),
    )


def test_tensor_matrix_product_of_a_vector(rng):
    x, a = _sparse(rng, (5,), density=0.6)
    M = rng.standard_normal((3, 5))
    npt.assert_allclose(np.ravel(x.tensor_matrix_product([M]).full().numpy()), M @ a)


def test_tensor_matrix_product_eval_diag(rng):
    x, a = _sparse(rng)
    H = [rng.standard_normal((6, n)) for n in a.shape]
    npt.assert_allclose(
        x.tensor_matrix_product_eval_diag(H), np.einsum("ijk,ai,aj,ak->a", a, *H)
    )


def test_tensor_diagonal_matrix_product(rng):
    x, a = _sparse(rng)
    d = [rng.standard_normal(n) for n in a.shape]
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product(d).full().numpy(),
        np.einsum("ijk,i,j,k->ijk", a, *d),
    )
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product([d[1]], [1]).full().numpy(),
        a * d[1][None, :, None],
    )


def test_dot_with_rank_one_metric(rng):
    x, a = _sparse(rng, (3, 4), density=0.6)
    y, b = _sparse(rng, (3, 4), density=0.6)
    M = [rng.standard_normal((3, 3)), rng.standard_normal((4, 4))]
    npt.assert_allclose(
        x.dot_with_rank_one_metric(y, M), np.einsum("ij,kl,ik,jl->", a, b, *M)
    )


# --- conversions from other formats --------------------------------------------------


def test_full_and_diagonal_tensors_convert_to_sparse(rng):
    a = rng.standard_normal((2, 3)) * (rng.random((2, 3)) < 0.5)
    npt.assert_array_equal(tensap.FullTensor(a).sparse().full().numpy(), a)
    d = tensap.DiagonalTensor(np.array([1.0, 0.0, 2.0]), 2)
    npt.assert_array_equal(d.sparse().full().numpy(), np.diag([1.0, 0.0, 2.0]))


# --- not implemented ------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda x: -x,
        lambda x: x - x,
        lambda x: x.squeeze([0]),
        lambda x: x.orth(0),
        lambda x: x.cat(x, [0]),
        lambda x: x.sub_tensor([0], ":", ":"),
        lambda x: x.tensordot_matrix_product_except_dim(x, [None] * 3, 0),
        lambda x: x.eval_at_indices(np.array([[0, 0]]), [0, 1]),
    ],
)
def test_not_implemented_methods(rng, call):
    x, _ = _sparse(rng)
    with pytest.raises((NotImplementedError, AssertionError)):
        call(x)


@pytest.mark.parametrize("shapes", [((3, 4, 2), (2, 3, 2)), ((3, 4, 2), (2, 3))])
def test_kron(rng, shapes):
    x, a = _sparse(rng, shapes[0])
    y, b = _sparse(rng, shapes[1])
    z = x.kron(y)
    expected = tensap.FullTensor(a).kron(tensap.FullTensor(b)).numpy()
    npt.assert_allclose(z.full().numpy(), expected)
    assert z.storage() == x.storage() * y.storage()


def test_tensor_diagonal_matrix_product_with_diagonals_as_columns(rng):
    x, a = _sparse(rng, (3, 3, 3))
    D = rng.standard_normal((3, 3))  # column k: diagonal for dimension k
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product(D).full().numpy(),
        np.einsum("ijk,i,j,k->ijk", a, D[:, 0], D[:, 1], D[:, 2]),
    )
