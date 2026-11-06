"""
Interpolation in a sparse tensor product basis
==============================================

Interpolation of a multivariate function on magic grids, in a sparse tensor
product functional basis.

We interpolate a function :math:`f` of :math:`d = 3` variables in a sparse
basis, :math:`I_\\Lambda f = \\sum_{i \\in \\Lambda} a(i)\\, \\varphi_i` with
:math:`I_\\Lambda f(x_k) = f(x_k)` at :math:`\\#\\Lambda` points, and compare
three choices of points by the error on a test sample. See
:ref:`theory-functions` and :ref:`theory-multi-indices`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_interpolation_Tensor.py``.
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
# The function to interpolate is
#
# .. math::
#
#    f(x) = 1 + x_1 + \frac{2 + x_1}{2 + x_2} + 0.04\,(x_3 - x_3^3),
#
# with :math:`X = (X_1, X_2, X_3)` uniform on :math:`[-1, 1]^3` (the code
# numbers the variables from 0). In each dimension, the basis
# :math:`\varphi^{(\nu)}_{i_\nu}`, :math:`0 \le i_\nu \le 10`, is made of the
# Legendre polynomials, orthonormal for the uniform measure.
D = 3
P = 10

F = tensap.UserDefinedFunction("1+x0+(2+x0)/(2+x1)+0.04*(x2-x2**3)", D)
F.evaluation_at_multiple_points = True
V = tensap.UniformRandomVariable(-1, 1)
X = tensap.RandomVector(V, D)
BASIS = tensap.PolynomialFunctionalBasis(
    tensap.LegendrePolynomials(), range(P + 1)
)
BASES = tensap.FunctionalBases.duplicate(BASIS, D)

# %%
# Sparse tensor product functional basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Instead of the full tensor product basis, with :math:`11^3` functions, we
# keep the products :math:`\varphi_i(x) = \varphi^{(1)}_{i_1}(x_1)
# \varphi^{(2)}_{i_2}(x_2) \varphi^{(3)}_{i_3}(x_3)` of total degree at most
# 10, that is :math:`\Lambda = \{i : i_1 + i_2 + i_3 \le 10\}`, the
# :math:`\ell^1` ball given by :class:`~tensap.MultiIndices`.
IND = tensap.MultiIndices.with_bounded_norm(D, 1, P)
H = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

# %%
# Interpolation on a magic grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The interpolation points are selected among 1000 random points of
# :math:`[-1, 1]^3` by ``magic_points``: a greedy algorithm (empirical
# interpolation) adds, at each step, the point where the interpolation error
# of the basis functions on the points already selected is the largest.
# Then ``interpolate`` solves the linear system
# :math:`\sum_{i \in \Lambda} a(i)\, \varphi_i(x_k) = f(x_k)`. The printed
# error is the relative :math:`\ell^2` error on 1000 new random points.
G, _, _ = H.magic_points(X.random(1000))
IF = H.interpolate(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST) - IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)

# %%
# Interpolation on a structured magic grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A cheaper selection, with more structure: magic points
# :math:`x^{(\nu)}_1, x^{(\nu)}_2, \dots` are selected for each univariate
# basis, and the grid is the sparse grid
# :math:`\{(x^{(1)}_{i_1}, x^{(2)}_{i_2}, x^{(3)}_{i_3}) : i \in \Lambda\}`
# (:class:`~tensap.SparseTensorGrid`), with as many points as basis
# functions.
GRIDS = X.random(1000)
G, _ = BASES.magic_points([GRIDS[:, i] for i in range(GRIDS.shape[1])])
G = tensap.SparseTensorGrid(G, H.indices)
IF = H.interpolate(F, G.array())

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST) - IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)

# %%
# Interpolation on a structured magic grid (alternative)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``tensor_product_interpolation`` does the same construction in one call,
# from the univariate magic points. The three grids give errors of the same
# order: the structured grids are as good as the one selected in
# :math:`[-1, 1]^3`.
GRIDS = X.random(1000)
G, _ = BASES.magic_points([GRIDS[:, i] for i in range(GRIDS.shape[1])])
IF, OUTPUT = H.tensor_product_interpolation(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST) - IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)
