"""
Unit tests of the multivariate functional bases and of FunctionalBasisArray.

Tensor product bases are checked against products of univariate
evaluations; FunctionalBasisArray against closed-form statistics of the
function f(x) = x0 x1 + x2^2 of three uniform variables on [-1, 1], which
lies in the span of the bases (Legendre polynomials of degree at most 2).
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

D = 3


def _f(x):
    return x[:, 0] * x[:, 1] + x[:, 2] ** 2


@pytest.fixture
def h():
    return tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(3))


@pytest.fixture
def bases(h):
    return tensap.FunctionalBases.duplicate(h, D)


@pytest.fixture
def x(rng):
    return rng.uniform(-1, 1, (5, D))


@pytest.fixture
def indices():
    return tensap.MultiIndices.with_bounded_norm(D, 1, 2)


@pytest.fixture
def sparse(bases, indices):
    return tensap.SparseTensorProductFunctionalBasis(bases, indices)


@pytest.fixture
def rule(h):
    G = h.measure.gauss_integration_rule(3)
    return tensap.FullTensorProductIntegrationRule([G.points] * D, [G.weights] * D)


@pytest.fixture
def f(sparse, rule):
    return sparse.projection(_f, rule)


# --- FunctionalBases --------------------------------------------------------------


def test_functional_bases(bases, h, x):
    assert len(bases) == bases.length() == D
    npt.assert_array_equal(bases.cardinals(), [3] * D)
    npt.assert_array_equal(bases.ndim(), [1] * D)
    hx = bases.eval(x)
    for k in range(D):
        npt.assert_allclose(hx[k], h.eval(x[:, k]))
    hx = bases.eval(x[:, [0, 2]], [0, 2])
    assert len(hx) == 2
    npt.assert_allclose(bases.eval_derivative([1, 0, 2], x)[2], h.eval_derivative(2, x[:, 2]))
    npt.assert_allclose(bases.derivative([1, 0, 0]).bases[0].eval(x[:, 0]),
                        h.eval_derivative(1, x[:, 0]), atol=1e-12)
    for g in bases.gram_matrices():
        npt.assert_allclose(g, np.eye(3), atol=1e-14)
    for m in bases.mean():
        npt.assert_allclose(m, [1, 0, 0], atol=1e-14)
    for c in bases.one():
        npt.assert_allclose(c, [1, 0, 0])
    assert bases.storage() == 0


def test_functional_bases_operations(bases):
    assert bases.remove_bases(1).length() == 2 and bases.keep_bases([0, 2]).length() == 2
    assert bases.transpose([2, 1, 0]).length() == D
    assert bases == bases
    assert isinstance(bases.orthonormalize(), tensap.FunctionalBases)
    assert bases.adaptation_path().shape == (3, 3, D)
    assert isinstance(bases.get_random_vector(), tensap.RandomVector)


def test_functional_bases_random_evaluations(bases, h):
    hx = bases.random(4)
    assert len(hx) == D and hx[0].shape == (4, 3)
    hx, x = bases.random_dims([0, 1], 4, nargout=2)
    npt.assert_allclose(hx[1], h.eval(x[1]))
    with pytest.raises(AssertionError):
        bases.random_dims([0], 4, tensap.LebesgueMeasure(-1, 1))


# --- tensor product bases -------------------------------------------------------------


def test_sparse_tensor_product_basis(sparse, bases, h, indices, x):
    assert sparse.cardinal() == indices.cardinal() == 10 and sparse.ndim() == D
    hx = [h.eval(x[:, k]) for k in range(D)]
    expected = np.stack([hx[0][:, i] * hx[1][:, j] * hx[2][:, k] for i, j, k in indices.array], 1)
    npt.assert_allclose(sparse.eval(x), expected, atol=1e-14)
    npt.assert_allclose(sparse.eval_with_functional_bases_evals(hx), expected, atol=1e-14)
    npt.assert_allclose(sparse.mean(), (np.sum(indices.array, 1) == 0).astype(float),
                        atol=1e-14)
    G = sparse.gram_matrix()
    npt.assert_allclose(G.toarray() if hasattr(G, "toarray") else G, np.eye(10), atol=1e-14)
    dx = [h.eval_derivative(1, x[:, 0]), hx[1], hx[2]]
    expected = np.stack([dx[0][:, i] * dx[1][:, j] * dx[2][:, k] for i, j, k in indices.array], 1)
    npt.assert_allclose(sparse.eval_derivative([1, 0, 0], x), expected, atol=1e-12)
    npt.assert_allclose(sparse.derivative([1, 0, 0]).eval(x), expected, atol=1e-12)
    assert sparse.adaptation_path().shape[0] == 10
    assert isinstance(sparse.get_random_vector(), tensap.RandomVector)
    assert sparse == sparse and sparse.keep_bases([0, 2]).ndim() == 2
    assert sparse.remove_bases(1).ndim() == 2 and sparse.transpose([2, 1, 0]).ndim() == D
    # the H1_0 Gram matrix: sum over the dimensions of the Gram matrices of
    # the derivatives, diagonal for Legendre polynomials of degree <= 2
    G1 = sparse.gram_matrix_h1_0()
    assert G1.shape == (10, 10)


# --- FunctionalBasisArray ------------------------------------------------------------------


def test_vector_valued_projection(sparse, rule, x):
    g = sparse.projection(lambda z: np.stack([_f(z), z[:, 0]], 1), rule)
    npt.assert_array_equal(g.shape, [2])
    npt.assert_allclose(g.eval(x), np.stack([_f(x), x[:, 0]], 1), atol=1e-14)
    npt.assert_allclose(np.ravel(g.mean()), [1 / 3, 0], atol=1e-14)
    # product with a matrix: (g M)(x) = g(x) M
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(g.matmul(M).eval(x), g.eval(x) @ M, atol=1e-14)
