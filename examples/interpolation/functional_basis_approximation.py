"""
Interpolation, projection and least squares on a functional basis
=================================================================

Interpolation with Chebyshev and magic points, projection by quadrature,
interpolation on user-defined and radial bases, and least-squares
approximation.

A function :math:`f` of one variable is approximated on a basis,
:math:`f \\approx \\sum_{i \\in I} a(i)\\, \\varphi_i`, with coefficients
given by interpolation at points :math:`x_k`, by the projection
:math:`a(i) = \\int f \\varphi_i \\, d\\mu` onto an orthonormal basis of
:math:`L^2_\\mu` computed with a quadrature, or by least squares from
samples. See :ref:`theory-functions` and :ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_FunctionalBasis_interpolation_projection_approximation.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)


# %%
# Interpolation on a polynomial space using Chebyshev Points
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We interpolate :math:`f(x) = \cos(10 x)` on :math:`[-1, 1]` with the
# polynomials of degree at most 30, orthonormal for the uniform measure
# :math:`\mu` (``X.orthonormal_polynomials()``). The interpolation
# :math:`\mathcal{I} f = \sum_{i=1}^{31} a(i)\, \varphi_i` is defined by
# :math:`\mathcal{I} f(x_k) = f(x_k)` at the 31 Chebyshev points
# :math:`x_k`, which gives a linear system for the coefficients :math:`a`.
def FUN(x):
    # Function to approximate
    return np.cos(10 * x)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Interpolation basis and points
P = 30
H = tensap.PolynomialFunctionalBasis(
    X.orthonormal_polynomials(), range(P + 1)
)
X_CP = tensap.chebyshev_points(H.cardinal(), [-1, 1])

# Interpolation of the function
F = H.interpolate(FUN, X_CP)

# %%
# The figure compares :math:`f` and its interpolation, and ``test_error``
# returns the relative :math:`\ell^2` and :math:`\ell^\infty` errors on 100
# random points (the first one is printed). The two curves cannot be told
# apart, and the error is close to machine precision.
print("Interpolation on a polynomial space using Chebyshev Points")
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(("True function", "Interpolation"))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)


# %%
# Interpolation on a polynomial space using magic Points
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Same function and basis, but the points are now the magic points of the
# basis, selected by a greedy algorithm among 10000 random points of
# :math:`[-1, 1]` (see the example on magic points). They need no explicit
# formula, unlike the Chebyshev points, and the error is of the same order.
def FUN(x):
    # Function to approximate
    return np.cos(10 * x)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Interpolation basis and points
P = 30
H = tensap.PolynomialFunctionalBasis(
    X.orthonormal_polynomials(), range(P + 1)
)
X_MP = H.magic_points(X.random(10000))[0]

# Interpolation of the function
F = H.interpolate(FUN, X_MP)

# Displays and error
print("\nInterpolation on a polynomial space using magic points")
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(("True function", "Interpolation"))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)


# %%
# Projection on polynomial space through quadrature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For :math:`X` standard normal and an orthonormal basis of
# :math:`L^2_\mu`, the orthogonal projection of :math:`f` has the
# coefficients :math:`a(i) = \mathbb{E}(f(X) \varphi_i(X))`. They are
# computed with a Gauss quadrature of 5 points,
#
# .. math::
#
#    a(i) \approx \sum_{k=1}^{5} w_k\, f(x_k)\, \varphi_i(x_k),
#
# for :math:`f(x) = x^2 / 2` and the orthonormal polynomials of degree at
# most 3. The quadrature is exact for these products of polynomials, and
# :math:`f` is in the span of the basis: the error is at machine precision.
def FUN(x):
    # Function to approximate
    return x**2 / 2


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.NormalRandomVariable()

# Integration rule
I0 = X.gauss_integration_rule(5)

# Approximation basis
P = 3
H = tensap.PolynomialFunctionalBasis(
    X.orthonormal_polynomials(), range(P + 1)
)

# Computation of the projection
F = H.projection(FUN, I0)

# Displays and error
print("\nProjection on polynomial space through quadrature")
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)


# %%
# Interpolation on UserDefinedFunctionalBasis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Any family of functions can be a basis
# (:class:`~tensap.UserDefinedFunctionalBasis`). Here the Fourier basis
# :math:`1, \sqrt{2} \cos(2 \pi k x), \sqrt{2} \sin(2 \pi k x)`,
# :math:`1 \le k \le 15`, orthonormal for the uniform measure on
# :math:`[0, 1]`, is used to interpolate
# :math:`f(x) = \exp(-(x - 1/2)^2)` on its magic points.
def FUN(x):
    # Function to approximate
    return np.exp(-((x - 1 / 2) ** 2))


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(0, 1)

# Fourier Basis
N = 15
H = np.empty(2 * N + 1, dtype=object)
H[0] = lambda x: np.ones(np.size(x))
for i in np.arange(1, N + 1):
    H[2 * i - 1] = lambda x, i=i: np.sqrt(2) * np.cos(2 * np.pi * i * x)
    H[2 * i] = lambda x, i=i: np.sqrt(2) * np.sin(2 * np.pi * i * x)
H = tensap.UserDefinedFunctionalBasis(H, X)

# Display of the Fourier basis
plt.figure()
X_PLOT = np.linspace(0, 1, 100)
plt.plot(X_PLOT, H.eval(X_PLOT))
plt.title("Fourier basis")
plt.show()

# Computation of the interpolation
X_MP = H.magic_points(X.random(10000))[0]
F = H.interpolate(FUN, X_MP)

# %%
# The periodic extension of :math:`f` is continuous but has a kink at the
# integers, which slows down the convergence of its Fourier series: the
# error is much larger than with polynomials.
print("\nInterpolation on UserDefinedFunctionalBasis")
plt.figure()
X_PLOT = np.linspace(0, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(("True function", "Interpolation"))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)


# %%
# Interpolation with a radial basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Another user-defined basis: 20 Gaussian functions
# :math:`\varphi_i(x) = \exp(-(x - c_i)^2 / s^2)`, with centers :math:`c_i`
# uniformly spaced in :math:`[-1, 1]` and width :math:`s = 1/2`, to
# interpolate :math:`f(x) = \exp(-x^2)` on magic points.
def FUN(x):
    # Function to approximate
    return np.exp(-(x**2))


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Radial basis
N = 20
X_LIN = np.linspace(-1, 1, N)
S = 10 / N


def k(x, y, s=S):
    return np.exp(-((x - y) ** 2) / s**2)


H = np.empty(N, dtype=object)
for i in range(N):
    H[i] = lambda y, x=X_LIN[i], k=k: k(y, x)
H = tensap.UserDefinedFunctionalBasis(H, X)

# Computation of the interpolation
X_MP = H.magic_points(X.random(100000))[0]
F = H.interpolate(FUN, X_MP)

# Displays and error
print("\nInterpolation with a radial basis")
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(("True function", "Interpolation"))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)


# %%
# Projection on polynomial space through quadrature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same projection, with the basis given directly by
# :class:`~tensap.HermitePolynomials`. The approximation
# :math:`F` of :math:`f(x) = x^2 / 2` is then differentiated with
# ``eval_derivative``, and :math:`F'` and :math:`F''` are projected on the
# same basis.
def FUN(x):
    # Function to approximate
    return x**2 / 2


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.NormalRandomVariable()

# Integration rule
I0 = X.gauss_integration_rule(5)

# Approximation basis
P = 3
H = tensap.PolynomialFunctionalBasis(
    tensap.HermitePolynomials(), range(P + 1)
)

# Computation of the approximation
F = H.projection(FUN, I0)

# Derivative and second derivative of F through projection
DF = H.projection(lambda x, F=F: F.eval_derivative(1, x), I0)
DDF = H.projection(lambda x, DF=DF: DF.eval_derivative(1, x), I0)

# %%
# The figure shows :math:`F`, :math:`F'` and :math:`F''`, that is
# :math:`x^2 / 2`, :math:`x` and :math:`1`.
print("\nProjection on polynomial space through quadrature")
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print("Mean squared error = %2.5e" % ERR_L2)
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, F(X_PLOT))
plt.plot(X_PLOT, DF(X_PLOT))
plt.plot(X_PLOT, DDF(X_PLOT))
plt.legend(("F", "df", "ddf"))
plt.show()


# %%
# Least-squares approximation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Finally, from 100 samples :math:`(x_k, f(x_k))` of :math:`X` uniform on
# :math:`[0, 1]`, with :math:`f(x) = \exp(-(x - 1/2)^2)`, the coefficients
# on the polynomials of degree at most 10 minimize
# :math:`\frac{1}{100} \sum_k (f(x_k) - \sum_i a(i) \varphi_i(x_k))^2`.
# :class:`~tensap.LinearModelLearningSquareLoss` is used with an
# :math:`\ell^1` regularization, and the support of :math:`a` is selected
# by cross-validation along its solution path. The printed error is the
# relative error on 100 test samples.
def FUN(x):
    # Function to approximate
    return np.exp(-((x - 1 / 2) ** 2))


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(0, 1)

# Approximation basis: orthonormal polynomials of maximal degree P
P = 10
H = tensap.PolynomialFunctionalBasis(
    X.orthonormal_polynomials(), range(P + 1)
)

# Solver
SOLVER = tensap.LinearModelLearningSquareLoss()
SOLVER.regularization = True
SOLVER.regularization_type = "l1"
SOLVER.basis_adaptation = False
SOLVER.model_selection = True

# Training sample
X_TRAIN = X.random(100)
Y_TRAIN = FUN(X_TRAIN)

# Computation of the approximation
SOLVER.basis = H
SOLVER.training_data = [X_TRAIN, Y_TRAIN]
F, OUTPUT = SOLVER.solve()

# Displays and error
print("\nLeast-squares approximation")
plt.figure()
X_PLOT = np.linspace(0, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(("True function", "Interpolation"))
plt.show()
X_TEST = X.random(100)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Mean squared error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)
