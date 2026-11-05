"""
Interpolation in a sparse tensor product basis
==============================================

Interpolation of a multivariate function on magic grids, in a sparse tensor
product functional basis.

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
IND = tensap.MultiIndices.with_bounded_norm(D, 1, P)
H = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

# %%
# Interpolation on a magic grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
G, _, _ = H.magic_points(X.random(1000))
IF = H.interpolate(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST) - IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)

# %%
# Interpolation on a structured magic grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
GRIDS = X.random(1000)
G, _ = BASES.magic_points([GRIDS[:, i] for i in range(GRIDS.shape[1])])
IF, OUTPUT = H.tensor_product_interpolation(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST) - IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)
