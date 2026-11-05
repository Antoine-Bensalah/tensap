"""
Unit tests of the measures and random variables of tensap.

Distributions are checked against scipy.stats, moments against closed forms,
and the discrete and product measures on small examples written by hand.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats as st
import tensap

X = np.array([-0.5, 0.0, 1.0, 2.5])
P = np.array([0.1, 0.5, 0.9])

CONTINUOUS = {
    "uniform": (lambda: tensap.UniformRandomVariable(-1, 3), st.uniform(-1, 4)),
    "normal": (lambda: tensap.NormalRandomVariable(1, 2), st.norm(1, 2)),
}


@pytest.fixture(params=list(CONTINUOUS))
def continuous(request):
    constructor, reference = CONTINUOUS[request.param]
    return constructor(), reference


@pytest.fixture
def discrete():
    return tensap.DiscreteRandomVariable(np.array([0.0, 1.0, 3.0]), np.array([0.2, 0.5, 0.3]))


# --- continuous random variables --------------------------------------------------


def test_distribution_functions(continuous):
    X_, ref = continuous
    npt.assert_allclose(X_.pdf(X), ref.pdf(X))
    npt.assert_allclose(X_.cdf(X), ref.cdf(X))
    npt.assert_allclose(X_.icdf(P), ref.ppf(P))
    npt.assert_allclose(X_.likelihood(X), np.sum(np.log(ref.pdf(X))))


def test_statistics(continuous):
    X_, ref = continuous
    npt.assert_allclose([X_.mean(), X_.variance(), X_.std()], [ref.mean(), ref.var(), ref.std()])
    npt.assert_allclose(X_.random_variable_statistics(), [ref.mean(), ref.var()])
    npt.assert_allclose(X_.moment([1, 2, 3]), [ref.moment(k) for k in (1, 2, 3)])
    assert X_.number_of_parameters() == 2


def test_supports(continuous):
    X_, ref = continuous
    npt.assert_allclose(X_.support(), ref.support())
    npt.assert_allclose([X_.min(), X_.max()], ref.support())
    a, b = X_.truncated_support()
    assert ref.cdf(a) < 1e-15 and ref.sf(b) < 1e-15
    if np.isfinite(ref.support()[0]):
        npt.assert_allclose([a, b], ref.support())


def test_random_samples(continuous):
    X_, ref = continuous
    x = X_.random(2000)
    assert x.shape == (2000,)
    assert st.kstest(x, ref.cdf).pvalue > 1e-3
    # latin hypercube: one point in each of the n intervals of probability 1/n
    u = np.sort(ref.cdf(X_.lhs_random(10, seed=1)))
    npt.assert_array_equal(np.floor(10 * u), np.arange(10))


def test_iso_probabilistic_grid(continuous):
    X_, ref = continuous
    g = X_.iso_probabilistic_grid(4)
    npt.assert_allclose(ref.cdf(g), [0, 0.25, 0.5, 0.75, 1], atol=1e-14)
    npt.assert_allclose(X_.iso_probabilistic_grid(2)[1], ref.median())
    assert X_.iso_probabilistic_grid(1).size == 2
    npt.assert_allclose(X_.iso_probabilistic_grid(0.5), X_.iso_probabilistic_grid(2))


def test_discretizations(continuous):
    X_, ref = continuous
    D = X_.discretize(4)
    npt.assert_allclose(ref.cdf(np.ravel(D.values)), [1 / 8, 3 / 8, 5 / 8, 7 / 8])
    npt.assert_allclose(D.probabilities, 0.25)
    D = X_.discretize_support(5)
    assert np.size(D.values) == 5
    npt.assert_allclose(np.sum(D.probabilities), 1)
    D = X_.discretize_support(np.array([0.0, 1.0, 2.0]))
    npt.assert_allclose(D.probabilities, np.diff(ref.cdf([-np.inf, 0.5, 1.5, np.inf])))


def test_shift_and_standard_variable(continuous):
    X_, ref = continuous
    Y = X_.shift(1, 2)  # 1 + 2 X
    npt.assert_allclose(Y.cdf(1 + 2 * X), ref.cdf(X))
    S = X_.get_standard_random_variable()
    npt.assert_allclose(S.mean(), 0, atol=1e-15)
    npt.assert_allclose(X_.transfer(S, X), S.icdf(ref.cdf(X)), atol=1e-12)


def test_gauss_integration_rule(continuous):
    X_, ref = continuous
    G = X_.gauss_integration_rule(3)
    npt.assert_allclose(G.integrate(lambda x: x**5), ref.moment(5), rtol=1e-12)


def test_equality(continuous):
    X_, _ = continuous
    assert X_ == X_.shift(0, 1) and not X_ != X_.shift(0, 1)
    assert X_ != X_.shift(1, 1)
    assert not X_ == tensap.LebesgueMeasure(-1, 3)
    assert "RandomVariable" in repr(X_)


def test_orthonormal_polynomials_of_the_standard_variables():
    assert isinstance(tensap.NormalRandomVariable().orthonormal_polynomials(),
                      tensap.HermitePolynomials)
    assert isinstance(tensap.UniformRandomVariable().orthonormal_polynomials(),
                      tensap.LegendrePolynomials)


# --- discrete random variables -----------------------------------------------------------


def test_discrete_random_variable(discrete):
    npt.assert_allclose(discrete.cdf(X), [0, 0.2, 0.7, 0.7])
    npt.assert_allclose(discrete.icdf(P), [0, 1, 3])
    npt.assert_allclose(discrete.moment([1, 2, 3]), [1.4, 3.2, 8.6])
    npt.assert_allclose([discrete.mean(), discrete.variance()], [[1.4], [1.24]])
    npt.assert_allclose(discrete.random_variable_statistics(), [[1.4], [1.24]])
    npt.assert_allclose([discrete.min(), discrete.max()], [0, 3])
    npt.assert_allclose(np.ravel(discrete.support()), [0, 3])
    assert discrete.is_discrete() and discrete.number_of_parameters() == 6
    G = discrete.integration_rule()
    npt.assert_allclose(G.integrate(lambda x: x**2), 3.2)
    x = discrete.random(1000)
    assert set(np.unique(x)) <= {0.0, 1.0, 3.0}
    npt.assert_allclose(np.mean(x == 1.0), 0.5, atol=0.06)


def test_discrete_random_variable_default_probabilities_and_equality(discrete):
    D = tensap.DiscreteRandomVariable(np.array([0.0, 1.0, 3.0]))
    npt.assert_allclose(D.probabilities, 1 / 3)
    assert discrete == tensap.DiscreteRandomVariable(
        np.array([0.0, 1.0, 3.0]), np.array([0.2, 0.5, 0.3]))
    assert discrete != D
    assert discrete != tensap.DiscreteRandomVariable(np.array([0.0, 1.0]))


def test_discrete_measure():
    M = tensap.DiscreteMeasure(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 2.0, 3.0]))
    assert M.mass() == 6 and M.ndim() == 1 and M.is_discrete()
    npt.assert_allclose(np.ravel(M.support()), [0, 2])
    npt.assert_allclose(np.ravel(M.truncated_support()), [0, 2])
    assert M == M
    G = M.integration_rule()
    npt.assert_allclose(G.integrate(lambda x: x), 8)
    assert set(np.ravel(M.random(20))) <= {0.0, 1.0, 2.0}


# --- empirical random variables ------------------------------------------------------------


def test_empirical_random_variable():
    sample = np.random.randn(40)
    E = tensap.EmpiricalRandomVariable(sample)
    h = E.bandwidth
    npt.assert_allclose(h, 3.5 * np.std(sample) * 40 ** (-1 / 3))
    # gaussian kernel density estimate
    kde = np.mean(st.norm.pdf((X[:, None] - sample) / h), 1) / h
    npt.assert_allclose(E.pdf(X), kde)
    npt.assert_allclose(E.icdf(E.cdf(X)), X, atol=1e-8)
    npt.assert_allclose(E.random_variable_statistics(), [np.mean(sample), np.var(sample)])
    # moments of the smoothed distribution
    npt.assert_allclose(E.moment([2]), np.mean(sample**2) + h**2)
    S = E.get_standard_random_variable()
    npt.assert_allclose([np.mean(S.sample), np.std(S.sample)], [0, 1], atol=1e-14)
    npt.assert_allclose(E.transfer(S, X), (X - np.mean(sample)) / np.std(sample))
    npt.assert_allclose(E.shift(1, 2).sample, 1 + 2 * sample)
    assert E == tensap.EmpiricalRandomVariable(sample) and E != S
    assert E.random(3).shape == (3,)
    a, b = E.truncated_support()
    assert a < sample.min() and b > sample.max()


# --- lebesgue and product measures ----------------------------------------------------------


def test_lebesgue_measure():
    L = tensap.LebesgueMeasure(0, 2)
    assert L.mass() == 2 and L.ndim() == 1
    npt.assert_array_equal(L.support(), [0, 2])
    npt.assert_array_equal(L.truncated_support(), [0, 2])
    assert L == tensap.LebesgueMeasure(0, 2) and L != tensap.LebesgueMeasure(0, 3)
    npt.assert_array_equal(L.shift(1, 2).support(), [1, 5])
    G = L.gauss_integration_rule(3)
    npt.assert_allclose(G.integrate(lambda x: x**5), 2**6 / 6)
    x = L.random(50)
    assert np.all((x >= 0) & (x <= 2))
    assert L.get_parameters() == (0, 2) and "LebesgueMeasure" in repr(L)
    assert isinstance(L.orthonormal_polynomials(), tensap.UnivariatePolynomials)


def test_product_measure():
    U = tensap.UniformRandomVariable(-1, 3)
    L = tensap.LebesgueMeasure(0, 2)
    M = tensap.ProductMeasure([U, L])
    assert M.mass() == 2 and M.ndim() == 2 and not M.is_discrete() and M == M
    npt.assert_array_equal(M.support(), [[-1, 3], [0, 2]])
    G = M.gauss_integration_rule([2, 3])
    npt.assert_allclose(G.integrate(lambda x: x[:, 0] ** 2 * x[:, 1] ** 3), 7 / 3 * 4)
    x = M.random(5)
    assert x.shape == (5, 2) and np.all((x[:, 1] >= 0) & (x[:, 1] <= 2))
    assert M.marginal([1]).ndim() == 1
    with pytest.raises(AssertionError):
        M.random_vector()
    assert tensap.ProductMeasure.duplicate(U, 3).ndim() == 3
    assert "ProductMeasure" in repr(M)


def test_product_of_random_variables_is_a_random_vector():
    M = tensap.ProductMeasure([tensap.UniformRandomVariable(), tensap.NormalRandomVariable()])
    V = M.random_vector()
    assert isinstance(V, tensap.RandomVector) and V.ndim() == 2


# --- random vectors --------------------------------------------------------------------------


@pytest.fixture
def vector():
    return tensap.RandomVector([tensap.UniformRandomVariable(-1, 3),
                                tensap.NormalRandomVariable(1, 2)])


def test_random_vector_distribution(vector):
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    u, n = st.uniform(-1, 4), st.norm(1, 2)
    npt.assert_allclose(vector.pdf(x), u.pdf(x[:, 0]) * n.pdf(x[:, 1]))
    npt.assert_allclose(vector.cdf(x), u.cdf(x[:, 0]) * n.cdf(x[:, 1]))
    npt.assert_allclose(vector.mean(), [1, 1])
    npt.assert_allclose(vector.variance(), [4 / 3, 4])
    npt.assert_allclose(vector.std(), np.sqrt([4 / 3, 4]))


def test_random_vector_properties(vector):
    assert vector.ndim() == 2 and vector.size == 2
    assert vector == vector and vector != vector.transpose([1, 0])
    npt.assert_array_equal(vector.support()[0], [-1, 3])
    assert np.all(np.isfinite(vector.truncated_support()[1]))
    assert vector.marginal(1).random_variables[0] == tensap.NormalRandomVariable(1, 2)
    assert vector.random(4).shape == (4, 2) and vector.lhs_random(4).shape == (4, 2)
    assert len(vector.orthonormal_polynomials()) == 2
    S = vector.get_standard_random_vector()
    npt.assert_allclose(S.mean(), [0, 0], atol=1e-15)
    assert tensap.RandomVector(tensap.UniformRandomVariable(), 3).ndim() == 3
    assert "RandomVector" in repr(vector)


def test_random_vector_transfer(vector):
    S = tensap.RandomVector([tensap.NormalRandomVariable()] * 2)
    y = vector.transfer(S, np.array([[0.0, 1.0]]))
    npt.assert_allclose(y, [[st.norm.ppf(0.25), 0.0]])


def test_random_vector_iso_probabilistic_grid(vector):
    g = vector.iso_probabilistic_grid(3)
    assert g.ndim() == 2


def test_random_multi_indices():
    V = tensap.random_multi_indices([2, 3])
    assert V.ndim() == 2
    x = V.random(50)
    assert np.all((x[:, 0] >= 0) & (x[:, 0] <= 1) & (x[:, 1] <= 2))
    V = tensap.random_multi_indices([np.array([5, 7]), np.array([1, 2])])
    assert set(np.ravel(V.random(20)[:, 0])) <= {5, 7}


def test_independent_copula():
    c = tensap.IndependentCopula()
    npt.assert_allclose(np.ravel(c.pdf(np.array([[0.2, 0.3]]))), 1)
    npt.assert_allclose(np.ravel(c.cdf(np.array([[0.2, 0.3]]))), 0.06)
