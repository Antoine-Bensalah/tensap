"""
Unit tests of tensap.CanonicalTensor.

A canonical tensor x = sum_k sigma_k v1_k x ... x vd_k is stored as its
factor matrices space[nu] (n_nu x r) and its diagonal core sigma. Every
operation is checked against the same computation on the dense array.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _dense(space, sigma):
    order = len(space)
    letters = "abcdefgh"[:order]
    expr = ",".join(f"{c}r" for c in letters) + ",r->" + letters
    return np.einsum(expr, *space, sigma)


def _canonical(rng, shape=(3, 4, 5), rank=2):
    space = [rng.standard_normal((n, rank)) for n in shape]
    sigma = rng.standard_normal(rank)
    return tensap.CanonicalTensor(space, sigma), _dense(space, sigma)


# --- construction and properties ---------------------------------------------


def test_constructor_and_properties(rng):
    x, a = _canonical(rng)
    assert x.order == x.ndim == 3
    npt.assert_array_equal(x.shape, [3, 4, 5])
    assert x.representation_rank() == 2
    assert x.storage() == 2 + 2 * (3 + 4 + 5)
    assert x.sparse_storage() == x.storage()
    assert isinstance(x.core, tensap.DiagonalTensor)
    npt.assert_allclose(x.full().numpy(), a)
    npt.assert_allclose(x.numpy(), a)
    assert "CanonicalTensor" in repr(x)


def test_constructor_with_a_diagonal_core(rng):
    space = [rng.standard_normal((n, 2)) for n in (2, 3)]
    core = tensap.DiagonalTensor(np.array([1.0, -2.0]), 2)
    x = tensap.CanonicalTensor(space, core)
    npt.assert_allclose(x.numpy(), _dense(space, core.data))


@pytest.mark.parametrize(
    "constructor, value",
    [(tensap.CanonicalTensor.zeros, 0.0), (tensap.CanonicalTensor.ones, 2.0)],
)
def test_constant_constructors(constructor, value):
    # rank 2: the entries of ones are 1 + 1
    x = constructor(2, [2, 3])
    npt.assert_allclose(x.numpy(), value)


def test_random_constructors():
    for constructor in (tensap.CanonicalTensor.rand, tensap.CanonicalTensor.randn):
        x = constructor(3, [2, 4, 5])
        assert x.representation_rank() == 3
        npt.assert_array_equal(x.shape, [2, 4, 5])


# --- arithmetic ---------------------------------------------------------------


def test_arithmetic(rng):
    x, a = _canonical(rng)
    y, b = _canonical(rng, rank=3)
    npt.assert_allclose((x + y).numpy(), a + b)
    assert (x + y).representation_rank() == 5
    npt.assert_allclose((x - y).numpy(), a - b)
    npt.assert_allclose((-x).numpy(), -a)


def test_dot_and_norm(rng):
    x, a = _canonical(rng)
    y, b = _canonical(rng, rank=3)
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))
    with pytest.raises(NotImplementedError):
        x.norm(np.eye(3))


def test_dot_requires_a_canonical_tensor(rng):
    x, a = _canonical(rng)
    with pytest.raises(AssertionError):
        x.dot(tensap.FullTensor(a))


# --- contractions ---------------------------------------------------------------


def test_tensor_matrix_product(rng):
    x, a = _canonical(rng)
    M = [rng.standard_normal((2, n)) for n in (3, 4, 5)]
    y = x.tensor_matrix_product(M)
    assert isinstance(y, tensap.CanonicalTensor)
    npt.assert_allclose(y.numpy(), np.einsum("ijk,ai,bj,ck->abc", a, *M))
    npt.assert_allclose(
        x.tensor_matrix_product(M[1], 1).numpy(), np.einsum("ijk,bj->ibk", a, M[1])
    )


def test_tensor_vector_product(rng):
    x, a = _canonical(rng)
    v = [rng.standard_normal(n) for n in (3, 4, 5)]
    npt.assert_allclose(x.tensor_vector_product(v), np.einsum("ijk,i,j,k->", a, *v))
    y = x.tensor_vector_product(v[1], 1)
    assert isinstance(y, tensap.CanonicalTensor) and y.order == 2
    npt.assert_allclose(y.numpy(), np.einsum("ijk,j->ik", a, v[1]))


def test_squeeze(rng):
    space = [rng.standard_normal((n, 2)) for n in (3, 1, 4)]
    sigma = rng.standard_normal(2)
    x = tensap.CanonicalTensor(space, sigma)
    a = _dense(space, sigma)
    npt.assert_allclose(x.squeeze().numpy(), a[:, 0, :])
    npt.assert_allclose(x.squeeze([1]).numpy(), a[:, 0, :])
    with pytest.raises(AssertionError):
        x.squeeze([0])


def test_squeeze_to_a_scalar(rng):
    space = [rng.standard_normal((1, 2)) for _ in range(2)]
    sigma = rng.standard_normal(2)
    x = tensap.CanonicalTensor(space, sigma)
    npt.assert_allclose(x.squeeze(), _dense(space, sigma).item())


def test_reduce_sum(rng):
    x, a = _canonical(rng)
    npt.assert_allclose(x.reduce_sum(), a.sum())
    npt.assert_allclose(x.reduce_sum([0, 2]).numpy(), a.sum((0, 2)))


# --- diagonal ---------------------------------------------------------------------


def test_eval_diag(rng):
    x, a = _canonical(rng, (3, 3, 3))
    npt.assert_allclose(x.eval_diag().numpy(), [a[i, i, i] for i in range(3)])


def test_eval_diag_on_one_dimension_returns_a_copy(rng, capsys):
    x, a = _canonical(rng, (3, 3, 3))
    npt.assert_allclose(x.eval_diag([1]).numpy(), a)


def test_eval_diag_does_not_modify_the_tensor(rng):
    x, a = _canonical(rng, (3, 3, 3))
    x.eval_diag()
    npt.assert_allclose(x.numpy(), a)


def test_eval_diag_on_some_dimensions(rng):
    x, a = _canonical(rng, (3, 4, 3))
    expected = np.stack([a[i, :, i] for i in range(3)])
    npt.assert_allclose(x.eval_diag([0, 2]).numpy(), expected)


def test_tensor_matrix_product_eval_diag(rng):
    x, a = _canonical(rng)
    H = [rng.standard_normal((6, n)) for n in (3, 4, 5)]
    out = x.tensor_matrix_product_eval_diag(H)
    npt.assert_allclose(np.ravel(out.numpy()), np.einsum("ijk,ai,aj,ak->a", a, *H))


# --- orthogonalization and conversions ----------------------------------------


def test_orth_preserves_the_tensor(rng):
    x, a = _canonical(rng)
    y = x.orth()
    npt.assert_allclose(y.numpy(), a, atol=1e-12)
    for factor in y.space:
        npt.assert_allclose(factor.T @ factor, np.eye(factor.shape[1]), atol=1e-12)


@pytest.mark.parametrize(
    "tree",
    [tensap.DimensionTree.linear(3), tensap.DimensionTree.balanced(3),
     tensap.DimensionTree.trivial(3)],
)
def test_tree_based_tensor(rng, tree):
    x, a = _canonical(rng)
    npt.assert_allclose(x.tree_based_tensor(tree).full().numpy(), a)


def test_tree_based_tensor_with_inactive_leaves(rng):
    x, a = _canonical(rng)
    tree = tensap.DimensionTree.balanced(3)
    active = np.full(tree.nb_nodes, True)
    active[tree.dim2ind[0] - 1] = False
    out = x.tree_based_tensor(tree, active)
    assert not out.is_active_node[tree.dim2ind[0] - 1]
    npt.assert_allclose(out.full().numpy(), a)


# --- gradient of the diagonal evaluation ------------------------------------------


def test_parameter_gradient_eval_diag_with_respect_to_the_core(rng):
    # evaluations: space[nu] are N x r matrices, f(k) = sum_j s_j prod_nu F_nu(k, j)
    N, r = 6, 2
    space = [rng.standard_normal((N, r)) for _ in range(3)]
    x = tensap.CanonicalTensor(space, rng.standard_normal(r))
    out = x.parameter_gradient_eval_diag(4)
    npt.assert_allclose(out.numpy(), space[0] * space[1] * space[2])


def test_parameter_gradient_eval_diag_with_respect_to_a_factor(rng):
    N, r = 6, 2
    space = [rng.standard_normal((N, r)) for _ in range(3)]
    x = tensap.CanonicalTensor(space, rng.standard_normal(r))
    f_2 = space[0] * space[2]  # product of the factors but the second one
    out = x.parameter_gradient_eval_diag(2)
    npt.assert_allclose(out.numpy(), np.einsum("kj,ab->kjab", f_2, np.eye(N)))
    H = rng.standard_normal((N, 4))
    out = x.parameter_gradient_eval_diag(2, [None, H, None])
    npt.assert_allclose(out.numpy(), np.einsum("kj,ka->kja", f_2, H))


def test_norm_of_an_orthonormalized_tensor(rng):
    x, a = _canonical(rng)
    y = x.orth()
    y.is_orth = True
    npt.assert_allclose(y.norm(), np.linalg.norm(a))


@pytest.mark.xfail(
    strict=True,
    reason="CanonicalTensor.__radd__ adds the scalar as a tensor: sum() and "
    "0 + x fail",
)
def test_sum_of_canonical_tensors(rng):
    x, a = _canonical(rng)
    y, b = _canonical(rng)
    npt.assert_allclose(sum([x, y]).numpy(), a + b)
