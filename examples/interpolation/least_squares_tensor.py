"""
Least squares in a sparse tensor product basis
==============================================

Least-squares approximation of a multivariate function in a sparse tensor
product functional basis.

From samples :math:`y_k = f(x_k)`, :math:`1 \\le k \\le n`, we solve
:math:`\\min_{a} \\frac1n \\sum_{k=1}^{n} \\big(y_k - \\sum_{i \\in \\Lambda}
a(i)\\, \\varphi_i(x_k)\\big)^2` for :math:`d = 2`, with fewer samples than
basis functions, and compare regularizations and selections of the basis
by their test errors. See :ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_leastsquares_Tensor.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Definitions
# ^^^^^^^^^^^
# The function to approximate is
#
# .. math::
#
#    f(x) = (1 + x_1)^2 + \frac{2 + x_1}{2 + x_2},
#
# with :math:`X = (X_1, X_2)` uniform on :math:`[-1.5, 2]^2`. In each
# dimension, the basis is made of the polynomials of degree at most 10,
# orthonormal for the law of :math:`X_\nu`.
D = 2
P = 10

FUN = tensap.UserDefinedFunction("(1+x0)**2+(2+x0)/(2+x1)", D)
FUN.evaluation_at_multiple_points = True
V = tensap.UniformRandomVariable(-1.5, 2)
X = tensap.RandomVector(V, D)
BASIS = tensap.PolynomialFunctionalBasis(
    V.orthonormal_polynomials(), range(P + 1)
)
BASES = tensap.FunctionalBases.duplicate(BASIS, D)

# %%
# Sparse tensor product functional basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The basis is indexed by the multi-indices of total degree at most 10,
# :math:`\Lambda = \{i : i_1 + i_2 \le 10\}`, that is 66 functions
# :math:`\varphi_i(x) = \varphi^{(1)}_{i_1}(x_1) \varphi^{(2)}_{i_2}(x_2)`.
IND = tensap.MultiIndices.with_bounded_norm(D, 1, P)
H = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

# %%
# Training and test samples, solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Only :math:`n = 50` training samples, fewer than :math:`\#\Lambda`, and a
# test sample of size 1000 to measure the relative error
# :math:`\|y - v(x)\|_2 / \|y\|_2`.
# :class:`~tensap.LinearModelLearningSquareLoss` minimizes the empirical
# risk with the square loss over the linear models on the basis ``H``.
X_TRAIN = X.random(50)
Y_TRAIN = FUN(X_TRAIN)
X_TEST = X.random(1000)
Y_TEST = FUN(X_TEST)

SOLVER = tensap.LinearModelLearningSquareLoss()
SOLVER.basis = H
SOLVER.training_data = [X_TRAIN, Y_TRAIN]

# %%
# Standard least squares
# ^^^^^^^^^^^^^^^^^^^^^^
# With fewer samples than unknowns, the ordinary least-squares problem has
# many solutions, and the one returned fits the data but generalizes badly:
# the test error is large.
SOLVER.regularization = False
SOLVER.basis_adaptation = False

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Standard least-squares")
print("Test error = %2.5e" % ERR)

# %%
# Regularized least squares
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# An :math:`\ell^2` penalization,
# :math:`\min_a \sum_k (y_k - v(x_k))^2 + \lambda \|a\|_2^2`
# (``alpha`` is :math:`\lambda`), does not help here: it shrinks the
# coefficients but keeps them all.
SOLVER.regularization = True
SOLVER.regularization_type = "l2"
SOLVER.regularization_options = {"alpha": 0.1}

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Least-squares with l2 regularization")
print("Test error = %2.5e" % ERR)

# %%
# A penalization by :math:`\lambda \|a\|_1` instead gives sparse
# solutions. Its solution path, for decreasing values of :math:`\lambda`
# down to ``alpha``, gives a sequence of supports :math:`K_\lambda \subset
# \Lambda`; with ``model_selection``, a least-squares approximation is
# computed on each support and the one with the smallest cross-validation
# error estimate is kept. The error drops by an order of magnitude.
SOLVER.regularization = True
SOLVER.regularization_type = "l1"
SOLVER.regularization_options = {"alpha": 0.1}
SOLVER.model_selection = True

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Least-squares with l1 regularization")
print("Test error = %2.5e" % ERR)

# %%
# With ``alpha`` set to 0, the whole :math:`\ell^1` solution path is
# explored before the selection of the support, which improves the error
# again.
SOLVER.regularization = True
SOLVER.regularization_type = "l1"
SOLVER.regularization_options = {"alpha": 0.0}
SOLVER.model_selection = True
SOLVER.basis_adaptation = False

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print(
    "Least-squares with l1 regularization and selection of optimal "
    "pattern"
)
print("Test error = %2.5e" % ERR)

# %%
# Least squares with basis adaptation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Without regularization, ``basis_adaptation`` computes the least-squares
# approximations on a sequence of nested subspaces, by default spanned by
# the first :math:`m` basis functions in their natural ordering, and selects
# :math:`m` by cross-validation.
SOLVER.regularization = False
SOLVER.model_selection = True
SOLVER.basis_adaptation = True
SOLVER.model_selection = True

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Standard least-squares with basis adaptation")
print("Test error = %2.5e" % ERR)

# %%
# The sequence of subspaces can also be given by
# ``SOLVER.basis_adaptation_path``, a boolean matrix whose columns select
# the basis functions of each subspace. Here the subspaces are the
# polynomials of total degree at most :math:`m`, for :math:`m = 0, \dots,
# 10`. The printed error is the same as with the default ordering, which
# already sorts the multi-indices by total degree.
B = np.zeros((H.cardinal(), P + 1), dtype=bool)
for i in range(P + 1):
    B[:, i] = np.sum(IND.array, axis=1) <= i

SOLVER.basis_adaptation_path = B
F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print(
    "Standard least-squares with basis adaptation (specific sequence "
    "of "
    "spaces)"
)
print("Test error = %2.5e" % ERR)
