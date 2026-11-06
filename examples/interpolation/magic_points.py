"""
Magic points
============

Magic points of univariate and tensor product functional bases, and
interpolation of a function on them.

For a basis :math:`\\{\\varphi_i\\}_{i \\in I}`, the magic points
:math:`x_1, \\dots, x_{\\#I}` are chosen greedily among candidate points and
define the interpolation :math:`\\mathcal{I} f = \\sum_{i \\in I} a(i)\\,
\\varphi_i`, :math:`\\mathcal{I} f(x_k) = f(x_k)`. We look at them for
univariate, full and sparse tensor product polynomial bases. See
:ref:`theory-functions` and :ref:`theory-multi-indices`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_magic_points.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Magic points associated with a Functional Basis (1D)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``magic_points`` evaluates the basis on the candidate points and runs a
# greedy algorithm (empirical interpolation): at each step, it adds the
# candidate where the interpolation error of the basis functions on the
# points already chosen is the largest. For the Legendre polynomials of
# degree at most 20 on :math:`[-1, 1]`, 21 magic points are selected among
# 200 uniformly spaced points (black). The
# selected points (red circles) are denser near the endpoints of the
# interval, as the Chebyshev points are.
P1 = tensap.PolynomialFunctionalBasis(
    tensap.LegendrePolynomials(), range(21)
)
DOM = P1.domain()
G1 = np.linspace(DOM[0], DOM[1], 200)
MAGIC_POINTS, IND, OUTPUT = P1.magic_points(G1)

plt.figure()
plt.plot(G1, np.zeros(G1.shape), "k.", markersize=1)
plt.plot(
    MAGIC_POINTS, np.zeros(MAGIC_POINTS.shape), "ro", fillstyle="none"
)
plt.show()

# %%
# Magic points associated with a FullTensorProductFunctionalBasis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In dimension :math:`d = 2`, for the full tensor product basis
# :math:`\varphi_i(x) = \varphi^{(1)}_{i_1}(x_1) \varphi^{(2)}_{i_2}(x_2)`,
# :math:`0 \le i_1, i_2 \le 20`, the candidates are the points of the
# tensor product grid of two uniform grids of 30 points
# (:class:`~tensap.FullTensorGrid`), and :math:`21^2` magic points are
# selected among them.
D = 2
P1 = tensap.PolynomialFunctionalBasis(
    tensap.LegendrePolynomials(), range(21)
)
BASES = tensap.FunctionalBases.duplicate(P1, D)
P = tensap.FullTensorProductFunctionalBasis(BASES)
G1 = np.linspace(DOM[0], DOM[1], 30)
G = tensap.FullTensorGrid(G1, D).array()
MAGIC_POINTS, IND, OUTPUT = P.magic_points(G)

if D == 2:
    plt.figure()
    plt.plot(G[:, 0], G[:, 1], "k.", markersize=1)
    plt.plot(
        MAGIC_POINTS[:, 0], MAGIC_POINTS[:, 1], "ro", fillstyle="none"
    )
    plt.show()

# %%
# Magic points associated with a SparseTensorProductFunctionalBasis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A sparse basis is indexed by an anisotropic set of multi-indices, here
#
# .. math::
#
#    \Lambda = \{i \in \mathbb{N}_0^2 : i_1 + 2 i_2 \le 20\},
#
# a weighted :math:`\ell^1` ball which allows higher degrees in
# :math:`x_1` than in :math:`x_2`. The candidates are either the tensor
# product of univariate uniform grids, or, as below
# (``TENSORIZATION_OF_MAGIC_POINTS = True``), the tensor product of the 21
# univariate magic points. The magic points of the sparse basis are then a
# subset of this grid, with :math:`\#\Lambda` points.
TENSORIZATION_OF_MAGIC_POINTS = True
D = 2
P1 = tensap.PolynomialFunctionalBasis(
    tensap.LegendrePolynomials(), range(21)
)
BASES = tensap.FunctionalBases.duplicate(P1, D)
W = [1, 2]  # Weights for the anisotropic sparsity
IND = tensap.MultiIndices.with_bounded_weighted_norm(
    2, 1, P1.cardinal() - 1, W
)
P = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

if not TENSORIZATION_OF_MAGIC_POINTS:
    G1 = np.linspace(DOM[0], DOM[1], 100)
    G = tensap.FullTensorGrid(G1, D).array()
else:
    G1 = np.linspace(DOM[0], DOM[1], 1000)
    M1 = P1.magic_points(G1)[0]
    G = tensap.FullTensorGrid(np.ravel(M1), D).array()

MAGIC_POINTS, IND, OUTPUT = P.magic_points(G)

if D == 2:
    plt.figure()
    plt.plot(G[:, 0], G[:, 1], "k.", markersize=1)
    plt.plot(
        MAGIC_POINTS[:, 0], MAGIC_POINTS[:, 1], "ro", fillstyle="none"
    )
    plt.show()

# %%
# Interpolation of a function using magic points
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The function :math:`f(x) = \cos(x_1) + x_2^6 + x_1^2 x_2` is interpolated
# in the sparse basis on its magic points. The polynomial part of :math:`f`
# is in the span of the basis, and :math:`\cos(x_1)` is well approximated
# by polynomials of degree 20: the relative error on a test sample is close
# to machine precision.
D = 2


def F(x):
    return np.cos(x[:, 0]) + x[:, 1] ** 6 + x[:, 0] ** 2 * x[:, 1]


IF = P.interpolate(F, MAGIC_POINTS)

X = tensap.RandomVector(tensap.UniformRandomVariable(), D)
N_TEST = 1000
X_TEST = X.random(N_TEST)
ERR = np.linalg.norm(IF(X_TEST) - F(X_TEST)) / np.linalg.norm(F(X_TEST))
print("Test error = %2.5e" % ERR)
