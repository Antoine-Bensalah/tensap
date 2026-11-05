"""
Unit tests of the functions of tensap: user-defined, tensorized and
compositional functions, functional tensors, tensor grids.

Functional tensors interpolate f(x) = x0 x1 + x2^2 of three uniform variables
on [-1, 1] with Legendre polynomials of degree at most 2, so that they are
exact and their statistics have closed forms.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _f(x):
    return x[:, 0] * x[:, 1] + x[:, 2] ** 2


@pytest.fixture
def x(rng):
    return rng.uniform(-1, 1, (5, 3))


@pytest.fixture
def bases():
    h = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(3))
    return tensap.FunctionalBases.duplicate(h, 3)


@pytest.fixture(params=["full", "tree-based"])
def g(bases, request):
    g, _ = tensap.FullTensorProductFunctionalBasis(bases).tensor_product_interpolation(_f)
    if request.param == "tree-based":
        tree = tensap.DimensionTree.linear(3)
        g = tensap.FunctionalTensor(tensap.Truncator(1e-12).hsvd(g.tensor, tree), bases)
    return g


# --- user-defined functions -------------------------------------------------------


def test_stored_evaluations_are_reused(x):
    calls = []

    def f(z):
        calls.append(z.shape[0])
        return _f(z)

    u = tensap.UserDefinedFunction(f, 3)
    u.store = True
    npt.assert_allclose(u(x), _f(x))
    npt.assert_allclose(u(x[[1, 3]]), _f(x[[1, 3]]))
    npt.assert_allclose(u(np.vstack((x[:2], x[:2] + 1))), _f(np.vstack((x[:2], x[:2] + 1))))
    assert sum(calls) == 7  # only the new points are evaluated
    assert u._x_stored.shape == (7, 3)


def test_random_evaluations(x):
    measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    u = tensap.UserDefinedFunction(_f, 3)
    with pytest.raises(ValueError):
        u.random(3)
    with pytest.raises(ValueError):
        u.random(3, tensap.LebesgueMeasure(0, 1))
    fx, z = u.random(4, measure)
    npt.assert_allclose(np.ravel(fx), _f(z))
    u.measure = measure
    fx, z = u.random(4)
    assert z.shape == (4, 3)


def test_partial_evaluation(x):
    u = tensap.UserDefinedFunction(_f, 3)
    v = u.partial_evaluation([2], np.array([[2.0]]))
    npt.assert_allclose(v(x[:, :2]), x[:, 0] * x[:, 1] + 4)
    u.measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    v = u.partial_evaluation([0], np.array([[0.5]]))
    assert v.measure.ndim() == 2
    npt.assert_allclose(v(x[:, 1:]), 0.5 * x[:, 1] + x[:, 2] ** 2)
    u.dim = []
    with pytest.raises(ValueError):
        u.partial_evaluation([0], np.array([[0.5]]))


def test_evaluations_on_tensor_grids():
    u = tensap.UserDefinedFunction(_f, 3)
    grids = [np.array([0.0, 1.0]), np.array([0.0, 2.0, 3.0]), np.array([1.0, 2.0])]
    y = u.eval_on_tensor_grid(tensap.FullTensorGrid(grids))
    assert isinstance(y, tensap.FullTensor)
    a, b, c = np.meshgrid(*grids, indexing="ij")
    npt.assert_allclose(y.numpy(), a * b + c**2)
    indices = tensap.MultiIndices(np.array([[0, 0, 0], [1, 2, 1]]))
    y = u.eval_on_tensor_grid(tensap.SparseTensorGrid(grids, indices))
    assert isinstance(y, tensap.SparseTensor)
    npt.assert_allclose(y.full().numpy()[1, 2, 1], 1 * 3 + 4)
    with pytest.raises(ValueError):
        u.eval_on_tensor_grid(np.zeros((2, 3)))


# --- tensorizer -------------------------------------------------------------------------


# --- compositional functions and sets ---------------------------------------------------


# --- tensor grids ----------------------------------------------------------------------


# --- functional tensors -------------------------------------------------------------------


def test_functional_tensor_evaluations(g, x, bases):
    npt.assert_allclose(g(x), _f(x), atol=1e-14)
    npt.assert_allclose(g.eval_with_bases_evals(bases.eval(x)), _f(x), atol=1e-14)
    y = g.eval_on_grid([np.linspace(-1, 1, 4)] * 3)
    a, b, c = np.meshgrid(*[np.linspace(-1, 1, 4)] * 3, indexing="ij")
    npt.assert_allclose(y.full().numpy() if hasattr(y, "full") else y.numpy(),
                        a * b + c**2, atol=1e-13)
    fx, z = g.random(4, nargout=2)
    npt.assert_allclose(fx, _f(np.hstack(z)), atol=1e-14)
    assert isinstance(g.get_random_vector(), tensap.RandomVector)
    assert g.storage() == g.tensor.storage()


def test_functional_tensor_statistics(g):
    # E f = 1/3, Var f = 1/5, E f^2 = 14/45
    npt.assert_allclose(g.mean(), 1 / 3)
    npt.assert_allclose(g.expectation(), 1 / 3)
    npt.assert_allclose(g.variance(), 1 / 5)
    npt.assert_allclose(g.std(), np.sqrt(1 / 5))
    npt.assert_allclose(g.dot_product_expectation(g), 14 / 45)
    npt.assert_allclose(g.norm(), np.sqrt(14 / 45))


def test_functional_tensor_conditional_expectations(g, x):
    h = g.conditional_expectation([0, 1])
    npt.assert_allclose(h(x[:, :2]), x[:, 0] * x[:, 1] + 1 / 3, atol=1e-14)
    h = g.conditional_expectation(np.array([False, False, True]))
    npt.assert_allclose(h(x[:, [2]]), x[:, 2] ** 2, atol=1e-14)
    v = g.variance_conditional_expectation(
        np.array([[True, False, False], [True, True, False], [False, False, True]]))
    npt.assert_allclose(np.ravel(v), [0, 1 / 9, 4 / 45], atol=1e-14)


def test_functional_tensor_derivatives(g, x):
    npt.assert_allclose(g.eval_derivative([1, 0, 0], x), x[:, 1], atol=1e-13)
    npt.assert_allclose(g.derivative([0, 0, 1])(x), 2 * x[:, 2], atol=1e-13)
    npt.assert_allclose(g.eval_gradient(x), np.stack([x[:, 1], x[:, 0], 2 * x[:, 2]], 1),
                        atol=1e-13)


def test_functional_tensor_product(g, x):
    h = g * g
    npt.assert_allclose(h(x), _f(x) ** 2, atol=1e-13)
    npt.assert_allclose((g * 2.0)(x), 2 * _f(x), atol=1e-14)
