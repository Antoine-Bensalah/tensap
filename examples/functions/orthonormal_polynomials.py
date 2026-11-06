r"""
Orthonormal polynomials
=======================

Legendre and Hermite polynomials, and polynomials orthonormal with respect
to a discrete measure.

For a measure :math:`\mu` on :math:`\mathbb{R}`, the orthonormal polynomials
:math:`(p_n)_{n \ge 0}`, with :math:`p_n` of degree :math:`n`, satisfy
:math:`\int p_m \, p_n \, d\mu = \delta_{mn}`; they are computed from a
three-term recurrence. Each random variable or measure of tensap gives its
own family with ``orthonormal_polynomials``. For several measures, we plot
the basis :math:`\varphi_i = p_{i-1}`, :math:`i = 1, \dots, 6`, and check
that its Gram matrix is the identity. See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/functions/tutorial_orthonormal_polynomials.py``.
"""

# This file is part of tensap (tensor approximation package).

import tensap as tp
import numpy as np
import matplotlib.pyplot as plt

# Fixed seed, so that the example is reproducible.
np.random.seed(0)


# %%
# Two helpers: ``plot_poly`` plots the basis functions on an interval, and
# ``check_orthonormality`` computes the Gram matrix with an integration rule
# :math:`(x_k, w_k)_k` of the measure,
#
# .. math::
#
#    G(i, j) = \sum_k w_k \, \varphi_i(x_k) \, \varphi_j(x_k)
#    \approx \int \varphi_i \, \varphi_j \, d\mu,
#
# and prints the distance :math:`\|G - I\|` to the identity. With the Gauss
# rule with :math:`n + 1` points for polynomials of degree at most
# :math:`n`, the integrand has degree at most :math:`2n`, so the rule is
# exact and :math:`G` is the exact Gram matrix, up to rounding errors.
def plot_poly(phi, s):
    x = np.linspace(s[0], s[1], 100)
    plt.figure(1)
    plt.plot(x, phi.eval(x))
    plt.legend([i for i in range(deg + 1)])
    plt.title(type(phi.basis).__name__)


def check_orthonormality(phi, g):
    A = np.dot(np.diag(g.weights), phi.eval(g.points))
    G = A.T @ phi.eval(g.points)
    print(
        "Distance from the Gram to identity = ",
        np.linalg.norm(G - np.eye(phi.cardinal())),
    )


# %%
# Legendre polynomials, uniform probability measure on (-1,1)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the uniform measure :math:`\mu` on :math:`(-1, 1)`, of density
# :math:`1/2`, these are the Legendre polynomials, normalized so that
# :math:`\|p_n\|_{L^2_\mu} = 1`. ``polyval``, ``d_polyval`` and
# ``dn_polyval`` evaluate some of them and their derivatives at given
# points; a :class:`~tensap.PolynomialFunctionalBasis` then gathers the
# degrees 0 to 5 into a basis.

X = tp.UniformRandomVariable(-1, 1)
p = X.orthonormal_polynomials()
# evaluate the polynomials of degree 1 and 2 and their derivatives at
# points 0 and 1
print(p.polyval([1, 2], [0, 1]))
print(p.d_polyval([1, 2], [0, 1]))
print(p.dn_polyval(2, [1, 2], [0, 1]))

deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)

# %%
# Legendre polynomials, uniform probability measure on (1,3)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the uniform measure on :math:`(a, b) = (1, 3)`, the polynomials are
# obtained by the affine change of variable :math:`t = (2x - a - b)/(b - a)`
# from those on :math:`(-1, 1)`: the printed basis is a
# :class:`~tensap.ShiftedOrthonormalPolynomials`.

X = tp.UniformRandomVariable(1, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %%
# Legendre polynomials, Lebesgue measure on (-1,1)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Lebesgue measure on :math:`(-1, 1)` has mass 2 instead of 1: the
# polynomials are the same up to a factor :math:`1/\sqrt{2}`, so that they
# are orthonormal in the usual :math:`L^2(-1, 1)`.

X = tp.LebesgueMeasure(-1, 1)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %%
# Legendre polynomials, Lebesgue measure on (1,3)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same on :math:`(1, 3)`, again by an affine change of variable.

X = tp.LebesgueMeasure(1, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %%
# Hermite polynomials, normal distribution N(0,1)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For the standard normal measure, of density
# :math:`e^{-x^2/2} / \sqrt{2 \pi}` on :math:`\mathbb{R}`, these are the
# Hermite polynomials. The support is unbounded, so they are plotted on
# :math:`[-3, 3]`.

X = tp.NormalRandomVariable()
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [-3, 3]
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)

# %%
# Hermite polynomials, normal distribution N(2,3)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For :math:`\mathcal{N}(2, 3^2)`, the Hermite polynomials are shifted and
# scaled by :math:`t = (x - 2)/3`.

X = tp.NormalRandomVariable(2, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [-5, 9]
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %%
# Orthonormal polynomials with respect to a discrete measure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.DiscreteMeasure`
# :math:`\mu = \sum_{k=1}^{10} w_k \delta_{x_k}`, with random points and
# integer weights, gives :class:`~tensap.DiscretePolynomials`: the recurrence
# coefficients are computed with sums over the :math:`x_k`. Orthonormal
# polynomials exist up to degree 9, one less than the number of points. The
# points :math:`x_k` are shown on the horizontal axis, and the Gram matrix is
# computed with the measure itself as integration rule.

X = tp.DiscreteMeasure(
    np.random.random(10), np.random.randint(1, 10, 10)
)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)
plt.scatter(X.values[:, 0], np.zeros(X.values.shape[0]))

# check orthonormality
g = phi.measure.integration_rule()
check_orthonormality(phi, g)


# %%
# Orthonormal polynomials with respect to a discrete random variable
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same with a :class:`~tensap.DiscreteRandomVariable`, whose weights are
# probabilities summing to 1.

points = np.random.random(10)
probabilities = np.random.randint(1, 10, 10)
probabilities = probabilities / np.sum(probabilities)
X = tp.DiscreteRandomVariable(points, probabilities)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)
plt.scatter(X.values[:, 0], np.zeros(X.values.shape[0]))

# check orthonormality
g = phi.measure.integration_rule()
check_orthonormality(phi, g)


# %%
# Empirical polynomials associated with an empirical random variable
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# An :class:`~tensap.EmpiricalRandomVariable` is estimated from a sample
# :math:`x_1, \dots, x_N` by a kernel density: its law is a mixture of
# Gaussian laws centred at the sample points, with a common bandwidth. Here
# the sample is the sum of two uniform variables on :math:`(0, 1)`, with
# :math:`N = 100`. The orthonormality is checked with two rules: a Gauss
# rule with 10 points for the empirical measure, and a rule that sums Gauss
# rules of the Gaussian laws of the mixture.

sample = np.random.random(100) + np.random.random(100)
X = tp.EmpiricalRandomVariable(sample)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [0, 2]
plot_poly(phi, s)
plt.scatter(X.sample, np.zeros(X.sample.shape[0]))

# check orthonormality
# using gauss integration rule for the empirical measure
g = phi.measure.gauss_integration_rule(10)
check_orthonormality(phi, g)

# using an integration rule that exploits the gaussian mixture property,
# that is a sum of integration rules w.r.t. to gaussian distributions
g = phi.measure.integration_rule(deg + 1)
check_orthonormality(phi, g)

# %%
# In every case, the distance of the Gram matrix to the identity is at the
# level of machine precision.
