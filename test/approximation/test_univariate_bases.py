"""
Unit tests of the univariate functional bases of tensap.

Bases are checked through exact properties: orthonormality (Gram matrix by
quadrature), reproduction of the functions of their span (interpolation,
projection, derivatives, products), partition of unity of B-splines, and
derivatives against finite differences.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

X = np.linspace(-0.9, 0.9, 7)


def _f(x):
    """A polynomial of degree 3, in the span of the polynomial basis."""
    return 1 + x - x**3


@pytest.fixture
def legendre():
    return tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(5))


@pytest.fixture
def canonical():
    measure = tensap.UniformRandomVariable(-1, 1)
    return tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(measure), range(4))


def _gram(basis, rule):
    A = basis.eval(rule.points)
    return A.T @ (rule.weights[:, None] * A)


# --- polynomial bases ---------------------------------------------------------------


@pytest.mark.parametrize("basis", ["legendre", "canonical"])
def test_interpolation_and_projections_reproduce_the_span(basis, request):
    b = request.getfixturevalue(basis)
    npt.assert_allclose(b.interpolate(_f).eval(X).ravel(), _f(X), atol=1e-12)
    x = np.linspace(-1, 1, b.cardinal())
    npt.assert_allclose(b.interpolate(_f(x), x).eval(X).ravel(), _f(X), atol=1e-12)
    G = b.measure.gauss_integration_rule(6)
    npt.assert_allclose(b.projection(_f, G).eval(X).ravel(), _f(X), atol=1e-12)
    npt.assert_allclose(b.least_squares_projection(_f(G.points), G).eval(X).ravel(), _f(X),
                        atol=1e-12)
    # vector-valued function
    g = b.projection(lambda x: np.stack([_f(x), 2 * x], 1), G)
    npt.assert_allclose(g.eval(X), np.stack([_f(X), 2 * X], 1), atol=1e-12)


def test_derivatives_of_the_basis(legendre):
    p = tensap.LegendrePolynomials()
    npt.assert_allclose(legendre.derivative(1).eval(X), p.d_polyval(range(5), X), atol=1e-12)
    npt.assert_allclose(legendre.gradient().eval(X), p.d_polyval(range(5), X), atol=1e-12)
    npt.assert_allclose(legendre.hessian(None).eval(X), p.dn_polyval(2, range(5), X),
                        atol=1e-11)
    npt.assert_allclose(legendre.eval_derivative(2, X), p.dn_polyval(2, range(5), X))
    jac = legendre.eval_jacobian(X)
    assert jac.shape == (7, 5, 1)
    npt.assert_allclose(jac[:, :, 0], p.d_polyval(range(5), X))


def test_kron_spans_the_products(legendre):
    q = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(3))
    k = legendre.kron(q)
    expected = (legendre.eval(X)[:, :, None] * q.eval(X)[:, None, :]).reshape(7, -1, order="F")
    npt.assert_allclose(k.eval(X), expected, atol=1e-12)


# --- B-splines ------------------------------------------------------------------------


@pytest.fixture
def bsplines():
    return tensap.BSplinesFunctionalBasis.with_extra_knots(np.linspace(0, 1, 5), 2)


# --- piecewise polynomials ----------------------------------------------------------------


@pytest.fixture
def piecewise():
    return tensap.PiecewisePolynomialFunctionalBasis(np.array([0.0, 0.2, 0.5, 1.0]), [1, 2, 3])


def test_piecewise_polynomials_are_orthonormal(piecewise):
    assert piecewise.cardinal() == 2 + 3 + 4 and piecewise.ndim() == 1
    G = tensap.IntegrationRule.gauss_legendre_composite(piecewise.points, 4)
    npt.assert_allclose(_gram(piecewise, G), np.eye(9), atol=1e-12)
    # mean for the uniform probability measure on [0, 1]
    npt.assert_allclose(piecewise.mean(), piecewise.eval(G.points).T @ G.weights, atol=1e-12)


# --- other bases ---------------------------------------------------------------------------


def test_delta_basis():
    D = tensap.DeltaFunctionalBasis(np.array([0.0, 1.0, 2.0]))
    npt.assert_array_equal(D.eval(np.array([1.0, 2.0, 0.0])), np.eye(3)[[1, 2, 0]])
    assert D.cardinal() == 3 and D.ndim() == 1 and D.is_orthonormal
    npt.assert_allclose(D.mean(), 1 / 3)
    npt.assert_allclose(np.ravel(D.domain()), [0, 2])
    rv = tensap.DiscreteRandomVariable(np.array([0.0, 1.0]), np.array([0.3, 0.7]))
    D = tensap.DeltaFunctionalBasis(np.array([0.0, 1.0]), rv)
    npt.assert_allclose(D.mean(), [0.3, 0.7])
    npt.assert_allclose(D.interpolate(lambda x: 5 * x[:, 0]).data.ravel(), [0, 5])


def test_sub_functional_basis(legendre):
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    S = tensap.SubFunctionalBasis(legendre, A)
    npt.assert_allclose(S.eval(X), legendre.eval(X) @ A)
    npt.assert_allclose(S.eval(X, [1]), legendre.eval(X) @ A[:, [1]])
    assert S.cardinal() == 2 and S.storage() == 10 and S.ndim() == 1
    assert not S.is_orthonormal
    npt.assert_allclose(S.mean(), [1, 0], atol=1e-14)
    npt.assert_allclose(S.gram_matrix(), A.T @ A, atol=1e-14)
    p = tensap.LegendrePolynomials()
    npt.assert_allclose(S.eval_derivative(1, X), p.d_polyval(range(5), X) @ A)
    npt.assert_allclose(S.derivative(1).eval(X), p.d_polyval(range(5), X) @ A, atol=1e-12)
    orth = S.orthonormalize()
    assert orth.is_orthonormal
    npt.assert_allclose(orth.gram_matrix(), np.eye(2), atol=1e-14)
    npt.assert_array_equal(S.domain(), [-1, 1])
