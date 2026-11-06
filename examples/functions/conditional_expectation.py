r"""
Conditional expectations
========================

Approximate a function of random variables by interpolation, then compute
its conditional expectations.

Let :math:`X = (X_1, \dots, X_6)` be a standard Gaussian vector, with law
:math:`\mu = \mu_1 \otimes \dots \otimes \mu_6`, and
:math:`F(X) = X_1 + X_1 X_2 + X_1 X_3^2 + X_4^3 + X_5 + X_6`. We approximate
:math:`F` in a full tensor product basis of Hermite polynomials, then compute
the conditional expectations :math:`\mathbb{E}(F \mid X_\alpha)` for subsets
:math:`\alpha` of the variables directly on the coefficients.
See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/functions/tutorial_conditional_expectation.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Approximation of a function F(X)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For each variable, :math:`\varphi^{(\nu)}_1, \dots, \varphi^{(\nu)}_4` are
# the Hermite polynomials of degree 0 to 3, orthonormal in
# :math:`L^2_{\mu_\nu}`, and the approximation reads
#
# .. math::
#
#    F(x) \approx \sum_{i_1 = 1}^{4} \cdots \sum_{i_6 = 1}^{4}
#    a(i_1, \dots, i_6) \, \varphi^{(1)}_{i_1}(x_1) \cdots
#    \varphi^{(6)}_{i_6}(x_6).
#
# The tensor :math:`a` is obtained by interpolation on a product grid
# (``tensor_product_interpolation`` of a
# :class:`~tensap.FullTensorProductFunctionalBasis`): in each dimension, the
# interpolation points are selected among a sample of 1000 points of
# :math:`X_\nu`. Since :math:`F` is a polynomial of degree at most 3 in each
# variable, it belongs to the approximation space, and the relative error on
# a test sample is at the level of machine precision. In the code, the
# variables are numbered from 0: ``x0`` stands for :math:`x_1`.
DIM = 6
FUN = tensap.UserDefinedFunction(
    "x0 + x0*x1 + x0*x2**2 + x3**3 + x4 + x5", DIM
)
FUN.evaluationAtMultiplePoints = True

DEGREE = 3
X = tensap.RandomVector(tensap.NormalRandomVariable(0, 1), DIM)
P = tensap.PolynomialFunctionalBasis(
    tensap.HermitePolynomials(), range(DEGREE + 1)
)
BASES = tensap.FunctionalBases.duplicate(P, DIM)
GRIDS = X.random(1000)

G = tensap.FullTensorGrid(
    [np.reshape(GRIDS[:, i], [-1, 1]) for i in range(DIM)]
)
H = tensap.FullTensorProductFunctionalBasis(BASES)

F, OUTPUT = H.tensor_product_interpolation(FUN, G)

X_TEST = X.random(1000)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
ERR = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print("Mean squared error = %2.5e\n" % ERR)

# %%
# Conditional expectations
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For a subset :math:`\alpha` of :math:`\{1, \dots, 6\}` and its complement
# :math:`\alpha^c`, the conditional expectation integrates out the other
# variables,
#
# .. math::
#
#    \mathbb{E}(F \mid X_\alpha)(x_\alpha) = \int F(x_\alpha, x_{\alpha^c})
#    \, d\mu_{\alpha^c}(x_{\alpha^c}).
#
# Since :math:`\varphi^{(\nu)}_1 = 1` and the other basis functions have zero
# mean, it only keeps the coefficients :math:`a(i)` with :math:`i_\nu = 1`
# for :math:`\nu \in \alpha^c`: this is what ``conditional_expectation``
# does, and it returns a function of the variables :math:`x_\alpha` only.
# Each result is compared with the exact conditional expectation, here
# :math:`\mathbb{E}(F \mid X_1) = 2 X_1`, since :math:`\mathbb{E}(X_2) = 0`
# and :math:`\mathbb{E}(X_3^2) = 1`.
DIMS = 0
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction("2*x0", np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST) - F_CE(X_TEST)) / np.linalg.norm(
    FUN_CE(X_TEST)
)
print("Dim %s:          error = %2.5e\n" % (DIMS, ERR))

# %%
# The same with two or three variables. The reference function is written
# with its own arguments: ``x1`` is the second variable of :math:`\alpha`,
# not :math:`X_2`.

# E(F | X_1, X_2) = 2 X_1 + X_1 X_2
DIMS = [0, 1]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction("2*x0+x0*x1", np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST) - F_CE(X_TEST)) / np.linalg.norm(
    FUN_CE(X_TEST)
)
print("Dims %s:    error = %2.5e\n" % (DIMS, ERR))

# E(F | X_1, X_3) = X_1 + X_1 X_3^2
DIMS = [0, 2]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction("x0+x0*x1**2", np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST) - F_CE(X_TEST)) / np.linalg.norm(
    FUN_CE(X_TEST)
)
print("Dims %s:    error = %2.5e\n" % (DIMS, ERR))

# E(F | X_1, X_4) = 2 X_1 + X_4^3
DIMS = [0, 3]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction("2*x0+x1**3", np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST) - F_CE(X_TEST)) / np.linalg.norm(
    FUN_CE(X_TEST)
)
print("Dims %s:    error = %2.5e\n" % (DIMS, ERR))

# E(F | X_1, X_3, X_4) = X_1 + X_1 X_3^2 + X_4^3
DIMS = [0, 2, 3]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction("x0+x0*x1**2+x2**3", np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST) - F_CE(X_TEST)) / np.linalg.norm(
    FUN_CE(X_TEST)
)
print("Dims %s: error = %2.5e\n" % (DIMS, ERR))

# %%
# All the errors are at the level of machine precision: the conditional
# expectations of the interpolant are exact, as :math:`F` itself is.
