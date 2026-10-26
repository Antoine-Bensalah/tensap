"""
Unit tests of tensap.FullTensor.

Every operation is checked against the same computation done directly on the
numpy array (the oracle), on small tensors.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

SHAPES = [(3,), (3, 4), (2, 3, 4)]


def _tensor(rng, shape):
    data = rng.standard_normal(shape)
    return tensap.FullTensor(data), data


# --- construction and properties ---------------------------------------------


def test_constructor_and_properties(rng):
    x, a = _tensor(rng, (2, 3, 4))
    assert x.order == x.ndim == 3
    assert tuple(x.shape) == (2, 3, 4)
    assert x.storage() == 24
    npt.assert_array_equal(x.numpy(), a)
    npt.assert_array_equal(x.full().numpy(), a)


@pytest.mark.xfail(
    strict=True,
    reason="FullTensor.size returns the shape, its docstring announces the number of elements",
)
def test_size_is_the_number_of_elements(rng):
    x, _ = _tensor(rng, (2, 3, 4))
    assert np.ndim(x.size) == 0 and x.size == 24


def test_constructor_with_shape_uses_fortran_order():
    x = tensap.FullTensor(np.arange(6.0), shape=[2, 3])
    npt.assert_array_equal(x.numpy(), np.reshape(np.arange(6.0), (2, 3), order="F"))


def test_constructor_with_order_appends_singleton_dimensions(rng):
    x = tensap.FullTensor(rng.standard_normal((2, 3)), order=4)
    assert tuple(x.shape) == (2, 3, 1, 1)


def test_sparse_storage_counts_nonzeros():
    x = tensap.FullTensor(np.array([[0.0, 1.0], [2.0, 0.0]]))
    assert x.sparse_storage() == 2


@pytest.mark.parametrize(
    "constructor, value",
    [(tensap.FullTensor.zeros, 0.0), (tensap.FullTensor.ones, 1.0)],
)
def test_constant_constructors(constructor, value):
    x = constructor([2, 3])
    assert tuple(x.shape) == (2, 3)
    npt.assert_array_equal(x.numpy(), value)


def test_random_constructors_have_the_right_shape():
    for constructor in (tensap.FullTensor.rand, tensap.FullTensor.randn):
        assert tuple(constructor([2, 3, 4]).shape) == (2, 3, 4)
    assert np.all((tensap.FullTensor.rand([50]).numpy() >= 0))


def test_diag():
    x = tensap.FullTensor.diag([1.0, 2.0, 3.0], 3)
    expected = np.zeros((3, 3, 3))
    for i, v in enumerate([1.0, 2.0, 3.0]):
        expected[i, i, i] = v
    npt.assert_array_equal(x.numpy(), expected)


# --- arithmetic ---------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_arithmetic(rng, shape):
    x, a = _tensor(rng, shape)
    y, b = _tensor(rng, shape)
    npt.assert_allclose((x + y).numpy(), a + b)
    npt.assert_allclose((x - y).numpy(), a - b)
    npt.assert_allclose((-x).numpy(), -a)
    npt.assert_allclose((x * y).numpy(), a * b)
    npt.assert_allclose(x.hadamard_product(y).numpy(), a * b)
    npt.assert_allclose((x / 2.0).numpy(), a / 2.0)
    npt.assert_allclose((2.0 * x).numpy(), 2.0 * a)
    npt.assert_allclose((x**2).numpy(), a**2)
    npt.assert_allclose((1.0 + x).numpy(), 1.0 + a)
    npt.assert_allclose((1.0 - x).numpy(), 1.0 - a)
    assert x == tensap.FullTensor(a.copy())


@pytest.mark.parametrize("shape", SHAPES)
def test_dot_and_norm(rng, shape):
    x, a = _tensor(rng, shape)
    y, b = _tensor(rng, shape)
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))


# --- indexing ---------------------------------------------------------------------


def test_sub_tensor(rng):
    x, a = _tensor(rng, (3, 4, 5))
    s = x.sub_tensor([0, 2], ":", [1, 3, 4])
    npt.assert_array_equal(s.numpy(), a[np.ix_([0, 2], range(4), [1, 3, 4])])
    # a single index keeps the dimension
    assert tuple(x.sub_tensor(1, ":", ":").shape) == (1, 4, 5)


def test_eval_at_indices_all_dimensions(rng):
    x, a = _tensor(rng, (3, 4, 5))
    indices = np.array([[0, 1, 2], [2, 3, 4], [1, 0, 0]])
    npt.assert_array_equal(x.eval_at_indices(indices), a[tuple(indices.T)])


def test_eval_at_indices_one_dimension(rng):
    x, a = _tensor(rng, (3, 4, 5))
    s = x.eval_at_indices(np.array([[1], [3]]), [1])
    npt.assert_array_equal(s.numpy(), a[:, [1, 3], :])


def test_eval_at_indices_several_dimensions(rng):
    x, a = _tensor(rng, (3, 4, 5))
    indices = np.array([[0, 1], [2, 4]])  # pairs (i0, i2)
    s = x.eval_at_indices(indices, [0, 2])
    expected = np.stack([a[0, :, 1], a[2, :, 4]], axis=0)
    npt.assert_allclose(s.numpy(), expected)


def test_eval_diag(rng):
    x, a = _tensor(rng, (3, 3, 3))
    npt.assert_allclose(x.eval_diag(), [a[i, i, i] for i in range(3)])
    y, b = _tensor(rng, (3, 2, 3))
    s = y.eval_diag([0, 2])
    npt.assert_allclose(s.numpy(), np.stack([b[i, :, i] for i in range(3)]))


# --- reshaping ----------------------------------------------------------------


def test_reshape_uses_fortran_order(rng):
    x, a = _tensor(rng, (2, 3, 4))
    npt.assert_array_equal(x.reshape([6, 4]).numpy(), np.reshape(a, (6, 4), order="F"))


def test_transpose_and_itranspose(rng):
    x, a = _tensor(rng, (2, 3, 4))
    perm = [2, 0, 1]
    npt.assert_array_equal(x.transpose(perm).numpy(), np.transpose(a, perm))
    npt.assert_array_equal(x.transpose(perm).itranspose(perm).numpy(), a)


def test_squeeze(rng):
    x = tensap.FullTensor(rng.standard_normal((2, 1, 3, 1)))
    assert tuple(x.squeeze().shape) == (2, 3)
    assert tuple(x.squeeze([1]).shape) == (2, 3, 1)
    assert np.isscalar(tensap.FullTensor(np.ones((1, 1))).squeeze()) or np.ndim(
        tensap.FullTensor(np.ones((1, 1))).squeeze()
    ) == 0


def test_matricize(rng):
    x, a = _tensor(rng, (2, 3, 4))
    m = x.matricize([1])
    npt.assert_array_equal(
        m.numpy(), np.reshape(np.transpose(a, (1, 0, 2)), (3, 8), order="F")
    )
    m = x.matricize([0, 2])
    npt.assert_array_equal(
        m.numpy(), np.reshape(np.transpose(a, (0, 2, 1)), (8, 3), order="F")
    )


# --- concatenation and reductions ---------------------------------------------


def test_cat_all_dimensions_is_block_diagonal(rng):
    x, a = _tensor(rng, (2, 3))
    y, b = _tensor(rng, (1, 2))
    expected = np.zeros((3, 5))
    expected[:2, :3] = a
    expected[2:, 3:] = b
    npt.assert_array_equal(x.cat(y).numpy(), expected)


def test_cat_one_dimension_is_concatenation(rng):
    x, a = _tensor(rng, (2, 3))
    y, b = _tensor(rng, (2, 4))
    npt.assert_array_equal(x.cat(y, [1]).numpy(), np.concatenate([a, b], axis=1))


def test_reductions(rng):
    x, a = _tensor(rng, (2, 3, 4))
    npt.assert_allclose(x.reduce_sum(1).numpy(), a.sum(1, keepdims=True))
    npt.assert_allclose(x.reduce_mean([0, 2]).numpy(), a.mean((0, 2), keepdims=True))


# --- products and contractions ------------------------------------------------


def test_kron(rng):
    x, a = _tensor(rng, (2, 3))
    y, b = _tensor(rng, (4, 2))
    npt.assert_allclose(x.kron(y).numpy(), np.kron(a, b))
    x3, a3 = _tensor(rng, (2, 2, 2))
    y3, b3 = _tensor(rng, (3, 1, 2))
    expected = np.einsum("ijk,lmn->iljmkn", a3, b3).reshape(6, 2, 4)
    npt.assert_allclose(x3.kron(y3).numpy(), expected)
    # different orders: the missing dimensions of the second tensor have size 1
    z = x3.kron(x)
    assert tuple(z.shape) == (4, 6, 2)
    npt.assert_allclose(z.numpy(), np.einsum("ijk,lm->iljmk", a3, a).reshape(4, 6, 2))


def test_tensordot(rng):
    x, a = _tensor(rng, (2, 3, 4))
    y, b = _tensor(rng, (4, 3, 5))
    npt.assert_allclose(
        x.tensordot(y, [1, 2], [1, 0]).numpy(), np.tensordot(a, b, [[1, 2], [1, 0]])
    )
    # dims1 = 0 and no dims2: outer product
    npt.assert_allclose(x.tensordot(y, 0).numpy(), np.multiply.outer(a, b))


def test_tensor_matrix_product(rng):
    x, a = _tensor(rng, (2, 3, 4))
    M = [rng.standard_normal((5, n)) for n in (2, 3, 4)]
    npt.assert_allclose(
        x.tensor_matrix_product(M).numpy(),
        np.einsum("ijk,ai,bj,ck->abc", a, *M),
    )
    npt.assert_allclose(
        x.tensor_matrix_product(M[1], 1).numpy(), np.einsum("ijk,bj->ibk", a, M[1])
    )


def test_tensor_vector_product(rng):
    x, a = _tensor(rng, (2, 3, 4))
    v = [rng.standard_normal(n) for n in (2, 3, 4)]
    npt.assert_allclose(
        np.asarray(x.tensor_vector_product(v)), np.einsum("ijk,i,j,k->", a, *v)
    )
    npt.assert_allclose(
        x.tensor_vector_product(v[1], 1).numpy(), np.einsum("ijk,j->ik", a, v[1])
    )


def test_tensor_diagonal_matrix_product(rng):
    x, a = _tensor(rng, (2, 3, 4))
    d = rng.standard_normal(3)
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product(d, 1).numpy(), a * d[None, :, None]
    )


def test_dot_with_rank_one_metric(rng):
    x, a = _tensor(rng, (2, 3))
    y, b = _tensor(rng, (2, 3))
    M = [rng.standard_normal((n, n)) for n in (2, 3)]
    npt.assert_allclose(
        x.dot_with_rank_one_metric(y, M), np.einsum("ij,kl,ik,jl->", a, b, *M)
    )


def test_tensordot_matrix_product_except_dim(rng):
    x, a = _tensor(rng, (2, 3, 4))
    y, b = _tensor(rng, (5, 6, 7))
    M = [rng.standard_normal((2, 5)), None, rng.standard_normal((4, 7))]
    out = x.tensordot_matrix_product_except_dim(y, M, 1)
    expected = np.einsum("ijk,lmn,il,kn->jm", a, b, M[0], M[2])
    npt.assert_allclose(out.numpy(), expected)


def test_tensor_matrix_product_eval_diag(rng):
    x, a = _tensor(rng, (2, 3, 4))
    H = [rng.standard_normal((5, n)) for n in (2, 3, 4)]
    out = x.tensor_matrix_product_eval_diag(H)
    npt.assert_allclose(np.ravel(out.numpy()), np.einsum("ijk,ai,aj,ak->a", a, *H))


def test_tensordot_eval_diag(rng):
    # z(i1, k, j2) = sum_{l1, l2} x(i1, l1, k, l2) y(k, j2, l1, l2)
    x, a = _tensor(rng, (2, 3, 4, 2))
    y, b = _tensor(rng, (4, 5, 3, 2))
    z = x.tensordot_eval_diag(y, [1, 3], [2, 3], 2, 0)
    npt.assert_allclose(z.numpy(), np.einsum("iakb,kjab->ikj", a, b))


def test_outer_product_eval_diag(rng):
    x, a = _tensor(rng, (3, 2))
    y, b = _tensor(rng, (4, 3))
    z = x.outer_product_eval_diag(y, 0, 1)
    npt.assert_allclose(z.numpy(), np.einsum("ia,bi->iab", a, b))


# --- orthogonalization and decompositions -------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_orth(rng, dim):
    x, a = _tensor(rng, (3, 4, 5))
    q, r = x.orth(dim)
    assert q.is_orth and q.orth_dim == dim
    m = np.reshape(np.moveaxis(q.numpy(), dim, -1), (-1, q.shape[dim]))
    npt.assert_allclose(m.T @ m, np.eye(q.shape[dim]), atol=1e-12)
    npt.assert_allclose(q.tensor_matrix_product(r.T, dim).numpy(), a, atol=1e-12)


def test_orth_without_dim_returns_a_copy(rng):
    x, a = _tensor(rng, (2, 3))
    q, r = x.orth()
    npt.assert_array_equal(q.numpy(), a)
    assert r.size == 0


def test_singular_values(rng):
    x, a = _tensor(rng, (3, 4))
    npt.assert_allclose(x.singular_values(), np.linalg.svd(a, compute_uv=False))
    y, b = _tensor(rng, (2, 3, 4))
    sv = y.singular_values()
    for mu in range(3):
        m = np.moveaxis(b, mu, 0).reshape(b.shape[mu], -1)
        npt.assert_allclose(sv[mu], np.linalg.svd(m, compute_uv=False))


def test_principal_components(rng):
    a = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 8))  # rank 3
    x = tensap.FullTensor(a)
    V, s = x.principal_components()
    npt.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-12)
    npt.assert_allclose(V @ V.T @ a, a, atol=1e-10)
    V2, _ = x.principal_components(2)
    assert V2.shape[1] == 2
    V3, _ = x.principal_components(1e-8)
    assert V3.shape[1] == 3


def test_alpha_principal_components(rng):
    x, a = _tensor(rng, (2, 3, 4))
    V, _ = x.alpha_principal_components(1)
    m = np.moveaxis(a, 1, 0).reshape(3, -1)
    npt.assert_allclose(V @ V.T @ m, m, atol=1e-10)


# --- conversions ------------------------------------------------------------------


def test_conversions_preserve_the_tensor(rng):
    x, a = _tensor(rng, (2, 3, 4))
    npt.assert_allclose(x.tree_based_tensor().full().numpy(), a)
    npt.assert_allclose(x.sparse().full().numpy(), a)


# --- remaining call variants ---------------------------------------------------


def test_constructor_from_a_list():
    x = tensap.FullTensor([[1.0, 2.0], [3.0, 4.0]])
    npt.assert_array_equal(x.numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_division_by_a_tensor(rng):
    x, a = _tensor(rng, (2, 3))
    y = tensap.FullTensor(rng.uniform(1, 2, (2, 3)))
    npt.assert_allclose((x / y).numpy(), a / y.numpy())


def test_eval_at_indices_accepts_transposed_indices(rng):
    x, a = _tensor(rng, (3, 4, 5))
    indices = np.array([[0, 2, 1], [1, 4, 3]])  # one column per evaluation
    s = x.eval_at_indices(indices, [0, 2])
    expected = np.stack([a[0, :, 1], a[2, :, 4], a[1, :, 3]], axis=0)
    npt.assert_allclose(s.numpy(), expected)


def test_eval_diag_on_one_dimension_returns_the_tensor(rng):
    x, a = _tensor(rng, (3, 4))
    npt.assert_array_equal(x.eval_diag([1]).numpy(), a)


def test_sub_tensor_on_a_tensor_with_a_singleton_dimension(rng):
    x, a = _tensor(rng, (1, 4, 3))
    s = x.sub_tensor([0], [1, 3], ":")
    npt.assert_array_equal(s.numpy(), a[np.ix_([0], [1, 3], range(3))])


def test_orth_on_the_last_dimension(rng):
    x, a = _tensor(rng, (3, 4, 5))
    q, r = x.orth(-1)
    assert q.orth_dim == 2
    npt.assert_allclose(q.tensor_matrix_product(r.T, 2).numpy(), a, atol=1e-12)


def test_matricize_last_dimension_and_explicit_columns(rng):
    x, a = _tensor(rng, (2, 3, 4))
    npt.assert_array_equal(
        x.matricize(-1).numpy(),
        np.reshape(np.transpose(a, (2, 0, 1)), (4, 6), order="F"),
    )
    npt.assert_array_equal(
        x.matricize([1], [2, 0]).numpy(),
        np.reshape(np.transpose(a, (1, 2, 0)), (3, 8), order="F"),
    )


def test_tensordot_eval_diag_with_several_diagonals(rng):
    # z(k, l, i5, j1) = sum_{l1, l2} x(k, l1, l, l2, i5) y(j1, k, l1, l2, l)
    x, a = _tensor(rng, (3, 2, 4, 2, 5))
    y, b = _tensor(rng, (6, 3, 2, 2, 4))
    z = x.tensordot_eval_diag(y, [1, 3], [2, 3], [0, 2], [1, 4], diag=True)
    npt.assert_allclose(z.numpy(), np.einsum("kalbi,jkabl->klij", a, b))


def test_tensor_matrix_product_eval_diag_on_some_dimensions(rng):
    x, a = _tensor(rng, (2, 3, 4))
    H = [rng.standard_normal((5, n)) for n in (2, 4)]
    out = x.tensor_matrix_product_eval_diag(H, [0, 2])
    npt.assert_allclose(out.numpy(), np.einsum("ijk,ai,ak->aj", a, *H))
    out = x.tensor_matrix_product_eval_diag(H[1], 2)
    npt.assert_allclose(out.numpy(), np.einsum("ijk,ak->aij", a, H[1]))


def test_tensor_diagonal_matrix_product_on_all_dimensions(rng):
    x, a = _tensor(rng, (2, 3))
    d = [rng.standard_normal(2), rng.standard_normal(3)]
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product(d).numpy(), a * np.outer(d[0], d[1])
    )
