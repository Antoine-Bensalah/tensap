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


def test_polynomial_basis_properties(legendre):
    p = tensap.LegendrePolynomials()
    npt.assert_allclose(legendre.eval(X), p.polyval(range(5), X))
    assert legendre.cardinal() == 5 and legendre.ndim() == 1 and legendre.is_orthonormal
    npt.assert_allclose(legendre.gram_matrix(), np.eye(5), atol=1e-14)
    npt.assert_allclose(legendre.mean(), [1, 0, 0, 0, 0], atol=1e-14)
    npt.assert_allclose(legendre.expectation(), legendre.mean())
    npt.assert_allclose(legendre.one(), [1, 0, 0, 0, 0])
    npt.assert_array_equal(legendre.domain(), [-1, 1])
    assert legendre.storage() == 0
    npt.assert_array_equal(legendre.adaptation_path(), np.triu(np.ones((5, 5), bool)))
    assert legendre == tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(5))
    assert not legendre == tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), [0])
    assert not legendre == tensap.LegendrePolynomials()
    assert "PolynomialFunctionalBasis" in repr(legendre)


def test_one_requires_the_constant(legendre):
    with pytest.raises(ValueError):
        tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), [1, 2]).one()


def test_canonical_basis_gram_matrix_and_orthonormalization(canonical):
    # int x^(i+j) dx / 2 on [-1, 1]
    expected = [[(1 + (-1) ** (i + j)) / 2 / (i + j + 1) for j in range(4)] for i in range(4)]
    npt.assert_allclose(canonical.gram_matrix(), expected, atol=1e-15)
    assert not canonical.is_orthonormal
    o = canonical.orthonormalize()
    assert isinstance(o, tensap.SubFunctionalBasis) and o.is_orthonormal
    G = canonical.measure.gauss_integration_rule(5)
    npt.assert_allclose(_gram(o, G), np.eye(4), atol=1e-12)


def test_orthonormalize_an_orthonormal_basis_keeps_it(legendre):
    assert legendre.orthonormalize() is legendre


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


def test_random_evaluations_and_magic_points(legendre):
    hx, x = legendre.random(4)
    assert hx.shape == (4, 5)
    npt.assert_allclose(hx, legendre.eval(x))
    with pytest.raises(ValueError):
        legendre.random(4, tensap.LebesgueMeasure(-1, 1))
    points, ind, output = legendre.magic_points(np.linspace(-1, 1, 50))
    assert points.shape == (5, 1) and np.unique(ind).size == 5
    # the interpolation matrix at the magic points is invertible
    assert np.linalg.cond(legendre.eval(points)) < 1e3
    assert output["lebesgue_constant"] >= 1
    assert legendre.interpolation_points().shape == (5, 1)


def test_not_implemented_methods(legendre):
    for call in (legendre.optimal_sampling_measure, legendre.christoffel):
        with pytest.raises(NotImplementedError):
            call()


# --- B-splines ------------------------------------------------------------------------


@pytest.fixture
def bsplines():
    return tensap.BSplinesFunctionalBasis.with_extra_knots(np.linspace(0, 1, 5), 2)


def test_bsplines_partition_of_unity(bsplines):
    x = np.linspace(0, 1, 23)
    assert bsplines.cardinal() == 6 and bsplines.ndim() == 1
    hx = bsplines.eval(x)
    assert np.all(hx >= -1e-15)
    npt.assert_allclose(hx.sum(1), 1)


def test_bsplines_are_piecewise_polynomials(bsplines):
    # on each knot interval, a degree 2 polynomial: the third differences vanish
    x = 0.3 + 0.01 * np.arange(4)  # inside [0.25, 0.5]
    npt.assert_allclose(np.diff(bsplines.eval(x), 3, axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
def test_bsplines_derivatives(bsplines, n):
    x, h = np.array([0.05, 0.3, 0.6, 0.8, 0.95]), 1e-5  # not at the knots
    f = bsplines.eval if n == 1 else lambda z: bsplines.eval_derivative(1, z)
    fd = (f(x + h) - f(x - h)) / (2 * h)
    npt.assert_allclose(bsplines.eval_derivative(n, x), fd, rtol=1e-5, atol=1e-4)


def test_cardinal_bspline():
    b = tensap.BSplinesFunctionalBasis.cardinal_bspline(3)
    npt.assert_array_equal(b.knots, np.arange(5))
    assert b.cardinal() == 1
    # the cubic cardinal B-spline at its center
    npt.assert_allclose(b.eval(np.array([2.0])), [[2 / 3]])


def test_dilated_bsplines():
    basis = tensap.DilatedBSplinesFunctionalBasis.with_level_bounded_by(2, 2, 2)
    x = np.linspace(0.01, 0.99, 9)
    hx = basis.eval(x)
    assert hx.shape == (9, basis.cardinal())
    x, h = np.linspace(0.1, 0.9, 4) + 1e-3, 1e-6
    fd = (basis.eval(x + h) - basis.eval(x - h)) / (2 * h)
    npt.assert_allclose(basis.eval_derivative(1, x), fd, rtol=1e-4, atol=1e-4)


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


def test_piecewise_polynomials_support_and_elements(piecewise):
    npt.assert_array_equal(piecewise.element_number([0.0, 0.1, 0.2, 0.7, 1.0]), [0, 0, 1, 2, 2])
    hx = piecewise.eval(np.array([0.05]))
    npt.assert_array_equal(np.nonzero(hx[0])[0], [0, 1])  # functions of the first element
    npt.assert_allclose(piecewise.eval(X[X > 0], indices=[0, 3]),
                        piecewise.eval(X[X > 0])[:, [0, 3]])


def test_piecewise_polynomials_derivatives(piecewise):
    x, h = np.array([0.1, 0.3, 0.7, 0.9]), 1e-6
    fd = (piecewise.eval(x + h) - piecewise.eval(x - h)) / (2 * h)
    npt.assert_allclose(piecewise.eval_derivative(1, x), fd, rtol=1e-6, atol=1e-5)
    npt.assert_allclose(piecewise.eval_derivative(1, x, indices=[1]), fd[:, [1]], atol=1e-5)


@pytest.mark.parametrize("method", ["interpolation_points", "magic_points"])
def test_piecewise_polynomials_interpolation(piecewise, method):
    x = getattr(piecewise, method)()
    assert x.size == 9
    # p + 1 points in each element of degree p
    npt.assert_array_equal(np.bincount(piecewise.element_number(x)), [2, 3, 4])
    assert np.linalg.cond(piecewise.eval(x)) < 1e4


def test_piecewise_polynomials_constructors():
    P = tensap.PiecewisePolynomialFunctionalBasis
    assert P.hp(0, 1, 0.25, 2).cardinal() == 4 * 3
    assert P.np(0, 1, 4, 2).cardinal() == 4 * 3
    b = P.singularityhp_adapted(0, 1, [0.0], 0.25)
    assert b.points[0] == 0 and b.points[-1] == 1
    assert np.all(np.diff(b.points) > 0)
    # geometric refinement towards the singularity, with increasing degrees
    assert np.all(np.diff(np.diff(b.points)) >= 0) and np.all(np.diff(b.p) >= 0)
    b = P.singularityhp_adapted(0, 1, [1.0], 0.25)
    assert np.all(np.diff(b.p) <= 0)
    b = P.singularityhp_adapted(0, 1, [0.5], 0.25)
    assert 0.5 in b.points


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


def test_user_defined_basis():
    U = tensap.UserDefinedFunctionalBasis(
        [lambda x: x[:, 0], lambda x: x[:, 0] ** 2], tensap.UniformRandomVariable(), 1)
    npt.assert_allclose(U.eval(np.array([[1.0], [2.0]])), [[1, 1], [2, 4]])
    assert U.cardinal() == 2 and U.ndim() == 1
    npt.assert_array_equal(U.domain(), [-1, 1])


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
