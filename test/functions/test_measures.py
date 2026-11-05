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


def test_shift_and_standard_variable(continuous):
    X_, ref = continuous
    Y = X_.shift(1, 2)  # 1 + 2 X
    npt.assert_allclose(Y.cdf(1 + 2 * X), ref.cdf(X))
    S = X_.get_standard_random_variable()
    npt.assert_allclose(S.mean(), 0, atol=1e-15)
    npt.assert_allclose(X_.transfer(S, X), S.icdf(ref.cdf(X)), atol=1e-12)


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


# --- random vectors --------------------------------------------------------------------------


@pytest.fixture
def vector():
    return tensap.RandomVector([tensap.UniformRandomVariable(-1, 3),
                                tensap.NormalRandomVariable(1, 2)])
