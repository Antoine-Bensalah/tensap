r"""
Measures and random variables
=============================

Uniform and normal random variables, Gauss quadrature, discretisation of
random variables, random vectors, Lebesgue and product measures.

The functions of tensap are defined on
:math:`\mathcal{X} = \mathcal{X}_1 \times \dots \times \mathcal{X}_d`
equipped with a product measure :math:`\mu = \mu_1 \otimes \dots \otimes
\mu_d`, often the law of a random vector :math:`X = (X_1, \dots, X_d)` with
independent components. This measure defines the space :math:`L^2_\mu`, the
orthonormal bases and the integration rules. Here we build univariate
measures, sample them, integrate with them, and combine them into product
measures. See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/functions/tutorial_measures.py``.
"""

# This file is part of tensap (tensor approximation package).

import tensap as tp
import numpy as np
import matplotlib.pyplot as plt

# Fixed seed, so that the example is reproducible.
np.random.seed(0)
LHS_RNG = np.random.default_rng(0)


# %%
# Uniform probability measure on (a,b)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.UniformRandomVariable` on :math:`(a, b) = (1, 3)` has the
# density :math:`1 / (b - a)` on its support. We plot its probability density
# function, its cumulative distribution function :math:`F` and the quantile
# function :math:`F^{-1}`, evaluate them at one point, and draw samples, with
# the pseudo-random generator or by latin hypercube sampling.

a = 1
b = 3
X = tp.UniformRandomVariable(a, b)

# support
print("support ", X.support())
print("truncated support ", X.truncated_support())

# plot
plt.figure()
plt.title("probability density function")
X.pdf_plot()

plt.figure()
plt.title("cumulative distribution function")
X.cdf_plot()

plt.figure()
plt.title(
    "inverse cumulative distribution function (quantile function)"
)
X.icdf_plot()

# probability density function
print("X.pdf(2) ", X.pdf([2]))
# cumulative distribution function
print("X.cdf(2) ", X.cdf([2]))
# inverse cumulative distribution function (quantile function)
print("X.icdf(.5) ", X.icdf([0.5]))

# pseudo-random generator
print("X.random(2)= \n", X.random(2))

# latin hypercube sampling
print(
    "X.lhs_random(2, seed=LHS_RNG) = \n", X.lhs_random(2, seed=LHS_RNG)
)

# %%
# Normal (Gaussian) probability measure N(mu, sigma^2)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The normal law :math:`\mathcal{N}(\mu, \sigma^2)`, here with
# :math:`\mu = 1` and :math:`\sigma = 3`, has the support
# :math:`\mathbb{R}`. For plots and discretisations, ``truncated_support``
# gives a bounded interval that carries almost all the probability, and its
# argument ``p`` sets this probability.
mu = 1
sigma = 3
X = tp.NormalRandomVariable(mu, sigma)

# support
print("support ", X.support())
print("truncated support ", X.truncated_support())
print(
    "truncated support with probability 1-2e-8",
    X.truncated_support(p=1 - 2e-8),
)


# plot
plt.figure()
plt.title("probability density function")
X.pdf_plot()

# %%
# Gauss quadrature
# ^^^^^^^^^^^^^^^^
# The Gauss rule of a measure, with points :math:`x_k` and weights
# :math:`w_k`, approximates :math:`\mathbb{E}(f(X)) = \int f \, d\mu` by
# :math:`\sum_k w_k f(x_k)`, exactly for the polynomials of degree at most
# :math:`2n - 1` with :math:`n` points. For
# :math:`X \sim \mathcal{N}(0, 2^2)` and 6 points, it gives
# :math:`\mathbb{E}(X^2) = 4` up to machine precision.
X = tp.NormalRandomVariable(0, 2)
G = X.gauss_integration_rule(6)


def fun(x):
    return x**2


v = np.sum(fun(G.points) * G.weights)
print("E(X^2) = ", v)
print("integration error = ", np.abs(v - X.variance()) / X.variance())


# %%
# Discretisation of a random variable
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``discretize(n)`` replaces :math:`X \sim \mathcal{N}(0, 1)` by a discrete
# random variable :math:`X_d` taking :math:`n` values with equal
# probability, the quantiles :math:`F^{-1}((2k - 1) / (2n))`,
# :math:`k = 1, \dots, n`. Its integration rule is the mean over these
# values, and :math:`\mathbb{E}(X_d^2)` approximates
# :math:`\mathbb{E}(X^2) = 1`,
# with a relative error of the order of :math:`10^{-3}` for
# :math:`n = 1000`.

X = tp.NormalRandomVariable(0, 1)

# returns a discrete random variable taking n values with equal
# probability
Xd = X.discretize(1000)
plt.figure()
Xd.pdf_plot()


def fun(x):
    return x**2


G = Xd.integration_rule()
v = G.integrate(fun)
print("E(Xd^2) = ", v)
print("integration error = ", np.abs(v - X.variance()) / X.variance())

# %%
# Discretisation of a random variable from a discretisation of its support
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``discretize_support`` takes the other route: the values of the discrete
# random variable are fixed (a uniform grid of the truncated support or of a
# given interval, or given points :math:`x_1 < \dots < x_n`), and
#
# .. math::
#
#    \mathbb{P}(X_d = x_k) = \mathbb{P}(v_k < X \le v_{k+1}),
#
# with :math:`v_k = (x_{k-1} + x_k) / 2` the midpoints, :math:`v_1 = -\infty`
# and :math:`v_{n+1} = +\infty`.

X = tp.NormalRandomVariable(0, 1)

# returns a discrete random variable taking n values from a uniform
# discretisation of a truncated support
Xd2 = X.discretize_support(50)
plt.figure(1)
plt.title("uniform discretization of the truncated support")
Xd2.pdf_plot()


Xd2 = X.discretize_support(50, [-8, 8])

plt.figure()
plt.title("uniform discretization of the provided support")
Xd2.pdf_plot()

# returns a discrete random variable taking n specified values
Xd3 = X.discretize_support(np.linspace(-6, 6, 100))
plt.figure()
plt.title("given discretization")
Xd3.pdf_plot()


# %%
# Random vector
# ^^^^^^^^^^^^^
# A :class:`~tensap.RandomVector` gathers random variables
# :math:`X = (X_1, \dots, X_d)`, here independent (``IndependentCopula``),
# so that its law is :math:`\mu = \mu_1 \otimes \dots \otimes \mu_d`. A
# sample is an array with one row per point and one column per variable. A
# single random variable can be duplicated to get :math:`d` independent
# copies.

X1 = tp.UniformRandomVariable(-1, 1)
X2 = tp.NormalRandomVariable(1, 1)
X = tp.RandomVector([X1, X2])
print("random vector X")
print(X)
print("X.random(3) = \n", X.random(3))
print(
    "X.lhs_random(3, seed=LHS_RNG) = \n", X.lhs_random(3, seed=LHS_RNG)
)

# duplicate a random variable
X = tp.RandomVector(X1, 5)
print("random vector X")
print(X)
print("X.random(3) = \n", X.random(3))

# %%
# Lebesgue Measure
# ^^^^^^^^^^^^^^^^
# Not every measure is a probability measure: the
# :class:`~tensap.LebesgueMeasure` on :math:`[2, 5]` has total mass 3. It is
# used, for instance, for orthonormal bases in the usual :math:`L^2` space.
L = tp.LebesgueMeasure(2, 5)

# %%
# Product Measure
# ^^^^^^^^^^^^^^^
# A :class:`~tensap.ProductMeasure` :math:`\mu_1 \otimes \mu_2 \otimes \mu_3`
# can mix random variables and Lebesgue measures. ``marginal`` extracts the
# product of some of the factors (numbered from 0), and ``duplicate`` builds
# the product of :math:`d` copies of one measure.
m1 = tp.UniformRandomVariable()
m2 = tp.NormalRandomVariable()
m3 = tp.LebesgueMeasure(-1, 1)
m = tp.ProductMeasure([m1, m2, m3])
print("product measure\n", m)

# get the marginals
print("marginal 1\n", m.marginal([1]))
print("marginal [1,2]\n", m.marginal([1, 2]))

# tensorize a random variable -> returns a random vector
m = tp.ProductMeasure.duplicate(m1, 3)
print("duplicate a random variable\n", m)

# tensorize a measure -> returns a product measure
m = tp.ProductMeasure.duplicate(m3, 3)
print("duplicate a Lebesgue measure\n", m)

# %%
