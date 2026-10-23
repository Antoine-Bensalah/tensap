"""
Unit tests of the univariate polynomials of tensap.

Orthonormal polynomials are checked through their Gram matrix computed with
an independent quadrature, their derivatives through finite differences
and their monomial coefficients, their roots through the polynomial values.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

DEGREES = np.arange(6)


def _gauss_hermite(n):
    """Gauss rule of the standard normal measure."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


def _gauss_legendre(n, a=-1.0, b=1.0, mass=1.0):
    """Gauss rule on [a, b] of a uniform measure of total mass `mass`."""
    x, w = np.polynomial.legendre.leggauss(n)
    return a + (b - a) * (x + 1) / 2, w / 2 * mass


def _empirical_rule(p, n=20):
    return p.measure.gauss_integration_rule(n).points, p.measure.gauss_integration_rule(n).weights


# family -> (polynomials, quadrature of their measure (points, weights))
FAMILIES = {
    "hermite": (tensap.HermitePolynomials, lambda p: _gauss_hermite(20)),
    "legendre": (tensap.LegendrePolynomials, lambda p: _gauss_legendre(20)),
    "legendre lebesgue": (
        tensap.LegendrePolynomialsLebesgue,
        lambda p: _gauss_legendre(20, mass=2.0),
    ),
    "uniform on [0, 2]": (
        lambda: tensap.UniformRandomVariable(0, 2).orthonormal_polynomials(),
        lambda p: _gauss_legendre(20, 0.0, 2.0),
    ),
    "normal(1, 2)": (
        lambda: tensap.NormalRandomVariable(1, 2).orthonormal_polynomials(),
        lambda p: (1 + 2 * _gauss_hermite(20)[0], _gauss_hermite(20)[1]),
    ),
    "discrete": (
        lambda: tensap.DiscreteRandomVariable(np.arange(8.0)).orthonormal_polynomials(),
        lambda p: (np.arange(8.0), np.full(8, 1 / 8)),
    ),
    "empirical": (
        lambda: tensap.EmpiricalPolynomials(np.random.randn(30)),
        _empirical_rule,
    ),
}


@pytest.fixture(params=list(FAMILIES))
def family(request):
    constructor, rule = FAMILIES[request.param]
    p = constructor()
    x, w = rule(p)
    return request.param, p, np.ravel(x), np.ravel(w)


def test_orthonormality(family):
    _, p, x, w = family
    P = p.polyval(DEGREES, x)
    npt.assert_allclose(P.T @ (w[:, None] * P), np.eye(DEGREES.size), atol=1e-10)
    assert p.is_orthonormal()


def test_moments(family):
    name, p, x, w = family
    P = p.polyval(DEGREES, x)
    # moments of products of 1, 2 and 3 polynomials
    ind = np.array([[0, 0, 0], [1, 2, 3], [2, 2, 0], [1, 1, 2]])
    expected = [np.sum(w * P[:, i] * P[:, j] * P[:, k]) for i, j, k in ind]
    npt.assert_allclose(p.moment(ind), expected, atol=1e-10)
    npt.assert_allclose(p.moment([[1, 1], [1, 2]]), [1, 0], atol=1e-12)
    expected = [np.sum(w * P[:, i]) for i in range(3)]
    npt.assert_allclose(p.moment(np.array([[0], [1], [2]])), expected, atol=1e-10)
    npt.assert_allclose(p.mean([0, 1, 2]), np.array(expected) / np.sum(w), atol=1e-10)


def test_derivatives_by_finite_differences(family):
    _, p, _, _ = family
    x, h = np.linspace(-0.5, 0.7, 5), 1e-5
    d1 = (p.polyval(DEGREES, x + h) - p.polyval(DEGREES, x - h)) / (2 * h)
    npt.assert_allclose(p.d_polyval(DEGREES, x), d1, rtol=1e-6, atol=1e-6)
    d2 = (p.d_polyval(DEGREES, x + h) - p.d_polyval(DEGREES, x - h)) / (2 * h)
    npt.assert_allclose(p.dn_polyval(2, DEGREES, x), d2, rtol=1e-6, atol=1e-5)
    npt.assert_allclose(p.dn_polyval(6, [5], x), 0, atol=1e-8)


def test_roots(family):
    _, p, _, _ = family
    for deg in (1, 2, 5):
        roots = p.roots(deg)
        assert roots.size == deg and np.all(np.diff(roots) > 0)
        npt.assert_allclose(p.polyval(deg, roots), 0, atol=1e-10)


def test_monomial_coefficients(family):
    name, p, _, _ = family
    if isinstance(p, tensap.ShiftedOrthonormalPolynomials):
        with pytest.raises(NotImplementedError):
            p.poly_coeff()
        return
    x = np.linspace(-0.5, 0.7, 5)
    coef = p.poly_coeff(DEGREES) / np.sqrt(p.measure.mass())
    npt.assert_allclose(
        np.polynomial.polynomial.polyval(x, coef.T).T, p.polyval(DEGREES, x), atol=1e-12
    )
    # coefficients of the derivatives
    npt.assert_allclose(
        np.polynomial.polynomial.polyval(x, p.d_poly_coef(DEGREES).T).T
        / np.sqrt(p.measure.mass()),
        p.d_polyval(DEGREES, x),
        atol=1e-10,
    )
    npt.assert_allclose(
        np.polynomial.polynomial.polyval(x, p.dn_poly_coeff(2, DEGREES).T).T
        / np.sqrt(p.measure.mass()),
        p.dn_polyval(2, DEGREES, x),
        atol=1e-10,
    )


def test_random_evaluations(family):
    name, p, _, _ = family
    if name == "legendre lebesgue":
        with pytest.raises(AssertionError):  # not a probability measure
            p.random([0, 1], 4)
        return
    out, x = p.random([0, 1, 2], 7)
    assert out.shape == (7, 3) and np.size(x) == 7
    npt.assert_allclose(out, p.polyval([0, 1, 2], x))


def test_domains(family):
    _, p, x, _ = family
    domain, truncated = np.ravel(p.domain()), np.ravel(p.truncated_domain())
    assert domain[0] <= truncated[0] < truncated[1] <= domain[1]
    assert truncated[0] <= np.min(x) and np.max(x) <= truncated[1]


def test_equality_and_repr():
    assert tensap.LegendrePolynomials() == tensap.LegendrePolynomials()
    assert not tensap.LegendrePolynomials() == tensap.HermitePolynomials()
    assert not tensap.LegendrePolynomials() == tensap.CanonicalPolynomials()
    p = tensap.UniformRandomVariable(0, 2).orthonormal_polynomials()
    assert p == tensap.UniformRandomVariable(0, 2).orthonormal_polynomials()
    assert not p == tensap.UniformRandomVariable(0, 3).orthonormal_polynomials()
    assert not p == tensap.LegendrePolynomials()
    assert "LegendrePolynomials" in repr(tensap.LegendrePolynomials())
    assert "Shifted" in repr(p)
    assert p.one() == (1, 0) and tensap.LegendrePolynomials().one() == (1, 0)
    assert p.ndim() == 1


def test_shifted_polynomials_moments_with_another_measure():
    # orthonormal on [0, 2]: moments with respect to the uniform measure on [0, 4]
    p = tensap.UniformRandomVariable(0, 2).orthonormal_polynomials()
    x, w = _gauss_legendre(20, 0.0, 4.0)
    P = p.polyval([0, 1, 2], x)
    expected = [np.sum(w * P[:, 1] * P[:, 2]), np.sum(w * P[:, 1] * P[:, 1])]
    m = tensap.UniformRandomVariable(0, 4)
    npt.assert_allclose(p.moment([[1, 2], [1, 1]], m), expected, atol=1e-12)
    npt.assert_allclose(p.moment([[1, 1]], p.measure), [1.0])


# --- canonical polynomials ---------------------------------------------------------


def test_canonical_polynomials():
    p = tensap.CanonicalPolynomials(tensap.UniformRandomVariable(-1, 1))
    x = np.array([0.5, 2.0, -1.5])
    npt.assert_allclose(p.polyval([0, 1, 3], x), np.stack([x**0, x, x**3], 1))
    npt.assert_allclose(p.d_polyval([0, 1, 3], x), np.stack([0 * x, x**0, 3 * x**2], 1))
    npt.assert_allclose(p.dn_polyval(2, [0, 1, 3], x), np.stack([0 * x, 0 * x, 6 * x], 1))
    # moments of x^k for the uniform measure on [-1, 1]
    npt.assert_allclose(p.moment([[2], [1], [4]]), [1 / 3, 0, 1 / 5], atol=1e-15)
    npt.assert_allclose(p.moment([[1, 3]]), [1 / 5])
    npt.assert_allclose(p.mean([2, 4]), [1 / 3, 1 / 5])
    lebesgue = tensap.LebesgueMeasure(0, 2)
    npt.assert_allclose(p.mean([1, 2], lebesgue), [1, 4 / 3])
    assert not p.is_orthonormal() and p.one() == (1, 0)
    npt.assert_array_equal(p.domain(), [-np.inf, np.inf])
    assert p == tensap.CanonicalPolynomials() and not p == tensap.LegendrePolynomials()
    with pytest.raises(ValueError):
        tensap.CanonicalPolynomials("not a measure")
    with pytest.raises(AssertionError):
        tensap.CanonicalPolynomials().mean([1])


# --- integration rules ------------------------------------------------------------------


@pytest.mark.parametrize(
    "measure",
    [tensap.UniformRandomVariable(-1, 2), tensap.NormalRandomVariable(1, 2),
     tensap.LebesgueMeasure(0, 3)],
)
def test_gauss_rules_are_exact_up_to_degree_2n_minus_1(measure):
    n = 4
    G = tensap.IntegrationRule.gauss(measure, n)
    for k in range(2 * n):
        exact = _exact_moment(measure, k)
        npt.assert_allclose(G.integrate(lambda x: x**k), exact, rtol=1e-10, atol=1e-12)


def _exact_moment(measure, k):
    x, w = (_gauss_legendre(30, *measure.support(), mass=measure.mass())
            if not isinstance(measure, tensap.NormalRandomVariable)
            else (1 + 2 * _gauss_hermite(30)[0], _gauss_hermite(30)[1]))
    return np.sum(w * x**k)


def test_gauss_legendre_composite():
    knots = [0.0, 0.5, 2.0, 3.0]
    G = tensap.IntegrationRule.gauss_legendre_composite(knots, 2)
    assert G.points.size == 6 and G.ndim() == 1
    # exact for piecewise polynomials of degree 3 on the intervals
    npt.assert_allclose(G.integrate(lambda x: x**3), 3.0**4 / 4)
    f = lambda x: np.abs(x - 2.0) ** 3  # noqa: E731
    npt.assert_allclose(G.integrate(f), (2.0**4 + 1.0) / 4)
    G = tensap.IntegrationRule.gauss_legendre_composite(knots, [1, 2, 3])
    assert G.points.size == 6


def test_tensorized_rule():
    G = tensap.LebesgueMeasure(0, 1).gauss_integration_rule(3).tensorize(2)
    assert isinstance(G, tensap.FullTensorProductIntegrationRule) and G.ndim() == 2
    npt.assert_allclose(G.integrate(lambda x: x[:, 0] ** 2 * x[:, 1] ** 5), 1 / 18)
    npt.assert_allclose(np.sum(G.weights_on_grid()), 1)


def test_full_tensor_product_rule():
    G = tensap.FullTensorProductIntegrationRule(
        [np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])],
        [np.array([0.5, 0.5]), np.array([1.0, 1.0, 1.0])],
    )
    rule = G.integration_rule()
    assert rule.points.shape == (6, 2)
    npt.assert_allclose(G.integrate(lambda x: x[:, 0] + x[:, 1]), 0.5 * 3 + 3.0)
    with pytest.raises(ValueError):
        tensap.FullTensorProductIntegrationRule(np.zeros((2, 2)), [np.ones(2)])
    with pytest.raises(AssertionError):
        tensap.FullTensorProductIntegrationRule([np.zeros(2)], [np.ones(2), np.ones(2)])


def test_composite_tensor_product_rule():
    G = tensap.FullTensorProductIntegrationRule.gauss_legendre_composite(
        [np.linspace(0, 1, 3), np.linspace(0, 2, 3)], 2
    )
    npt.assert_allclose(G.integrate(lambda x: x[:, 0] ** 3 * x[:, 1]), 1 / 4 * 2)


def test_shifted_hermite_moments_with_another_normal_measure():
    # NormalRandomVariable.shift must be the affine map bias + scaling * X
    p = tensap.NormalRandomVariable(1, 2).orthonormal_polynomials()
    x, w = _gauss_hermite(30)
    P = p.polyval([0, 1, 2], 2 + 3 * x)
    expected = [np.sum(w * P[:, 1] * P[:, 2]), np.sum(w * P[:, 1] ** 2)]
    m = tensap.NormalRandomVariable(2, 3)
    npt.assert_allclose(p.moment([[1, 2], [1, 1]], m), expected, atol=1e-12)
