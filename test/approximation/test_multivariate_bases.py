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


def test_functional_bases_interpolation(bases):
    points = bases.interpolation_points()
    assert len(points) == D and np.size(points[0]) == 3
    points, ind = bases.magic_points([np.linspace(-1, 1, 10)] * D)
    for p, i in zip(points, ind):
        npt.assert_allclose(np.ravel(p), np.linspace(-1, 1, 10)[i])
    g, output = bases.tensor_product_interpolation(lambda z: z[:, 0] * z[:, 1] ** 2)
    z = np.random.uniform(-1, 1, (4, D))
    npt.assert_allclose(g(z), z[:, 0] * z[:, 1] ** 2, atol=1e-12)


# --- tensor product bases -------------------------------------------------------------


def test_full_tensor_product_basis(bases, h, x):
    F = tensap.FullTensorProductFunctionalBasis(bases)
    assert F.cardinal() == 27 and F.ndim() == D and len(F) == D
    # first dimension varies fastest
    hx = [h.eval(x[:, k]) for k in range(D)]
    expected = np.einsum("ni,nj,nk->nkji", *hx).reshape(5, -1)
    npt.assert_allclose(F.eval(x), expected, atol=1e-14)
    npt.assert_allclose(F.mean(), np.eye(27)[0], atol=1e-14)
    dx = [h.eval_derivative(1, x[:, 0]), hx[1], hx[2]]
    npt.assert_allclose(F.eval_derivative([1, 0, 0], x),
                        np.einsum("ni,nj,nk->nkji", *dx).reshape(5, -1), atol=1e-12)
    assert F == F and F.remove_bases(0).ndim() == 2 and F.keep_bases([1]).ndim() == 1
    assert F.transpose([2, 1, 0]).ndim() == D
    npt.assert_array_equal(F.domain(), [[-1, 1]] * D)
    points, grid = F.interpolation_points()
    assert points.shape == (27, D) and len(grid) == D
    g, _ = F.tensor_product_interpolation(_f)
    npt.assert_allclose(g(x), _f(x), atol=1e-12)


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


def test_sparse_tensor_product_interpolation(sparse, x):
    g, output = sparse.tensor_product_interpolation(_f)
    npt.assert_allclose(g(x), _f(x), atol=1e-12)
    assert output["number_of_evaluations"] == 10


# --- FunctionalBasisArray ------------------------------------------------------------------


def test_projection_reproduces_the_function(f, x):
    assert isinstance(f, tensap.FunctionalBasisArray)
    npt.assert_allclose(f.eval(x).ravel(), _f(x), atol=1e-14)
    npt.assert_allclose(f(x).ravel(), _f(x), atol=1e-14)


def test_statistics(f):
    # E f = 1/3, Var f = Var(x0 x1) + Var(x2^2) = 1/9 + 4/45, E f^2 = 14/45
    npt.assert_allclose(f.mean(), [[1 / 3]])
    npt.assert_allclose(f.expectation(), [[1 / 3]])
    npt.assert_allclose(f.variance(), [[1 / 5]])
    npt.assert_allclose(f.std(), [[np.sqrt(1 / 5)]])
    npt.assert_allclose(f.dot_product_expectation(f), [[14 / 45]])
    npt.assert_allclose(f.norm_expectation(), [[np.sqrt(14 / 45)]])
    npt.assert_allclose(f.dot(f), 14 / 45)
    npt.assert_allclose(f.norm(), np.sqrt(14 / 45))
    assert f.storage() == 10 and 3 <= f.sparse_storage() <= 10
    assert f.is_random()


def test_conditional_expectations(f, x):
    # E(f | x0, x1) = x0 x1 + 1/3, E(f | x2) = x2^2
    g = f.conditional_expectation([0, 1])
    npt.assert_allclose(g.eval(x[:, :2]).ravel(), x[:, 0] * x[:, 1] + 1 / 3, atol=1e-14)
    g = f.conditional_expectation(np.array([False, False, True]))
    npt.assert_allclose(g.eval(x[:, [2]]).ravel(), x[:, 2] ** 2, atol=1e-14)
    # variances of the conditional expectations: 0, 1/9 and 4/45
    v = f.variance_conditional_expectation(np.array([[True, False, False],
                                                     [True, True, False],
                                                     [False, False, True]]))
    npt.assert_allclose(np.ravel(v), [0, 1 / 9, 4 / 45], atol=1e-14)


def test_derivatives(f, x):
    npt.assert_allclose(f.eval_derivative([1, 0, 0], x).ravel(), x[:, 1], atol=1e-12)
    npt.assert_allclose(f.derivative([0, 0, 1]).eval(x).ravel(), 2 * x[:, 2], atol=1e-12)
    npt.assert_allclose(f.eval_gradient(x), np.stack([x[:, 1], x[:, 0], 2 * x[:, 2]], 1),
                        atol=1e-12)


def test_arithmetic(f, x):
    npt.assert_allclose((f + f).eval(x).ravel(), 2 * _f(x))
    npt.assert_allclose((f - f).eval(x).ravel(), 0, atol=1e-14)
    npt.assert_allclose((-f).eval(x).ravel(), -_f(x))
    npt.assert_allclose((f * 3).eval(x).ravel(), 3 * _f(x))
    npt.assert_allclose(f.get_coefficients(), f.data)


def test_vector_valued_projection(sparse, rule, x):
    g = sparse.projection(lambda z: np.stack([_f(z), z[:, 0]], 1), rule)
    npt.assert_array_equal(g.shape, [2])
    npt.assert_allclose(g.eval(x), np.stack([_f(x), x[:, 0]], 1), atol=1e-14)
    npt.assert_allclose(np.ravel(g.mean()), [1 / 3, 0], atol=1e-14)
    # product with a matrix: (g M)(x) = g(x) M
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(g.matmul(M).eval(x), g.eval(x) @ M, atol=1e-14)


def test_random_evaluations(f):
    fx, z = f.random(6)
    assert fx.shape == (6, 1)
    npt.assert_allclose(fx.ravel(), _f(z))
    assert isinstance(f.get_random_vector(), tensap.RandomVector)


def test_projection_on_another_basis(f, bases, x):
    full = tensap.FullTensorProductFunctionalBasis(bases)
    g = f.projection(full)
    npt.assert_allclose(g.eval(x).ravel(), _f(x), atol=1e-14)
    g = f.sub_functional_basis()
    assert isinstance(g, tensap.SubFunctionalBasis) and g.cardinal() == 1
