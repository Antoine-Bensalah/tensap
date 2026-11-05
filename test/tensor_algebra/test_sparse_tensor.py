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


# --- arithmetic ---------------------------------------------------------------


# --- evaluations ---------------------------------------------------------------


# --- structure ---------------------------------------------------------------------


# --- contractions ---------------------------------------------------------------


def test_tensor_vector_product_on_all_dimensions(rng):
    x, a = _sparse(rng, density=0.5)
    v = [rng.standard_normal(n) for n in a.shape]
    npt.assert_allclose(
        np.sum(x.tensor_vector_product(v)), np.einsum("ijk,i,j,k->", a, *v)
    )


def test_tensor_matrix_product_of_a_vector(rng):
    x, a = _sparse(rng, (5,), density=0.6)
    M = rng.standard_normal((3, 5))
    npt.assert_allclose(np.ravel(x.tensor_matrix_product([M]).full().numpy()), M @ a)


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


# --- conversions from other formats --------------------------------------------------


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
