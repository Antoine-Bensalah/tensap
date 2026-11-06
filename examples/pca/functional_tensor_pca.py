"""
Principal component analysis of multivariate functions
======================================================

Approximation of a multivariate function in Tucker, tree-based and tensor-
train formats by functional principal component analysis.

With bases :math:`\\varphi^{(\\nu)}` of the variables of
:math:`X = (X_1, \\dots, X_d)` and their interpolation points
:math:`x^{(\\nu)}_{i_\\nu}`, a function :math:`f` is identified with the
tensor :math:`u(i_1, \\dots, i_d) = f(x^{(1)}_{i_1}, \\dots, x^{(d)}_{i_d})`.
The principal component analysis of :math:`u` gives the
:math:`\\alpha`-principal subspaces of :math:`f` and an interpolation of
:math:`f` in Tucker, tree-based or tensor-train format, from a small number
of evaluations. See :ref:`theory-learning` and :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_PCA_FunctionalTensorPrincipalComponentAnalysis.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Choice of the function to approximate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Several functions are available: the Henon-Heiles potential, an
# anisotropic function, the sine of a sum, a composition of functions along
# a tree, the borehole function and a tensorized function, each with its
# random vector :math:`X` and its bases, polynomials orthonormal for the
# law of :math:`X_\nu`. ``CHOICE = 2`` selects the anisotropic function
#
# .. math::
#
#    f(x) = \frac{1}{(10 + 2 x_1 + x_3 + 2 x_4 - x_5)^2}, \qquad
#    x \in [-1, 1]^6,
#
# with :math:`X` uniform and the polynomials of degree at most 13: the tensor
# :math:`u` of the values of :math:`f` on the grid has :math:`14^6` entries.
# It does not depend on :math:`x_2` and :math:`x_6`.
CHOICE = 2
if CHOICE == 1:
    print("Henon-Heiles")
    D = 5
    FUN, X = tensap.multivariate_functions_benchmark("henon_heiles", D)

    DEGREE = 4
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]

elif CHOICE == 2:
    print("Anisotropic function")
    D = 6
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    def FUN(x):
        return (
            1
            / (10 + 2 * x[:, 0] + x[:, 2] + 2 * x[:, 3] - x[:, 4]) ** 2
        )

    DEGREE = 13
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]

elif CHOICE == 3:
    print("Sinus of a sum")
    D = 10
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    def FUN(x):
        return np.sin(np.sum(x[:, :10], 1))

    DEGREE = 17
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]

elif CHOICE == 4:
    print("Composition of functions")
    D = 6
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    TREE = tensap.DimensionTree.balanced(D)

    def FUN(x1, x2):
        return 1 / (1 + x1**2 + x2**2)

    FUN = tensap.CompositionalModelFunction(TREE, FUN, X)

    DEGREE = 4
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]

elif CHOICE == 5:
    print("Borehole function")
    FUN, X = tensap.multivariate_functions_benchmark("borehole")
    D = X.size

    DEGREE = 14
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]

elif CHOICE == 6:
    print("Tensorized function")
    R = 11  # Resolution
    B = 2  # Scaling factor
    D = R + 1

    X = tensap.UniformRandomVariable(0, 1)
    Y = tensap.UniformRandomVariable(0, 1)

    def IFUN(x):
        return 1 / (1 + x)

    T = tensap.Tensorizer(B, R, 1, X, Y)
    FUN = T.tensorize(IFUN)
    FUN.fun.evaluation_at_multiple_points = True
    DEGREE = 1
    H = tensap.PolynomialFunctionalBasis(
        Y.orthonormal_polynomials(), range(DEGREE + 1)
    )
    BASES = T.tensorized_function_functional_bases(H)
    X = tensap.RandomVector(BASES.measure)

else:
    raise ValueError("Bad function choice.")

# %%
# HOPCA (PCA for each dimension, provides reduced spaces)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For each :math:`\nu`, the :math:`\{\nu\}`-principal components of
# :math:`u` are estimated from the values of :math:`f` on
# :math:`\{x^{(\nu)}_{i_\nu}\}_{i_\nu} \times \{z^1, \dots, z^N\}`, for
# :math:`N` random points :math:`z^k` of the grid in the other variables,
# and interpolated on the basis :math:`\varphi^{(\nu)}`: this gives the
# principal subspaces of functions of :math:`x_\nu`, as
# :class:`~tensap.SubFunctionalBasis`. Their dimensions, the ranks, are 1
# for :math:`x_2` and :math:`x_6`.
print("--- Higher order PCA ---")
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.bases = BASES
FPCA.pca_sampling_factor = 1
FPCA.tol = 1e-10
SUB_BASES, OUTPUTS = FPCA.hopca(FUN)
print(
    "Number of evaluations = \n%s"
    % [x["number_of_evaluations"] for x in OUTPUTS]
)
print("Ranks {1, ..., d} = \n%s" % [x.cardinal() for x in SUB_BASES])

# %%
# Approximation in Tucker Format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Tucker approximation interpolates :math:`f` on the tensor product of
# the principal subspaces of all the variables. The lines ``alpha = ...``
# give, for each node of the Tucker tree (numbered from 1, the root being
# 1), the rank and the number of evaluations of :math:`f`. With ranks drawn
# at random, too small for some variables, the error is large; with the
# precision :math:`10^{-10}`, the ranks are adapted and the error is small,
# for a number of evaluations far below the :math:`14^6` entries of
# :math:`u`.
print("\n--- Approximation in Tucker format ---")
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 1
FPCA.pca_adaptive_sampling = True
FPCA.bases = BASES

print("\nPrescribed ranks")
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 5, D)
F, OUTPUT = FPCA.tucker_approximation(FUN)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Prescribed ranks = %s" % FPCA.max_rank)
print("Ranks = %s" % F.tensor.ranks[F.tensor.tree.dim2ind - 1])
X_TEST = X.random(1e4)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed tolerance")
FPCA.tol = 1e-10
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tucker_approximation(FUN)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Ranks = %s" % F.tensor.ranks[F.tensor.tree.dim2ind - 1])
X_TEST = X.random(1e4)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# Approximation in tree based format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# On a balanced dimension tree, the :math:`\alpha`-principal subspaces are
# computed from the leaves to the root, the subspace of an interior node
# being searched in the tensor product of the subspaces of its children.
# The result is a function in tree-based format, a
# :class:`~tensap.FunctionalTensor` whose tensor of coefficients is a
# tree-based tensor. Here ``pca_sampling_factor = 20`` takes 20 times more
# random samples :math:`z^k` than the default factor 1, for a more reliable
# estimation of the subspaces, at the price of more evaluations.
print("\n--- Approximation in tree based format ---")
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 20
FPCA.bases = BASES
TREE = tensap.DimensionTree.balanced(D)

print("\nPrescribed ranks")
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 9, TREE.nb_nodes)
FPCA.max_rank[TREE.root - 1] = 1
F, OUTPUT = FPCA.tree_based_approximation(FUN, TREE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Prescribed ranks = %s" % FPCA.max_rank)
print("Ranks = %s" % F.tensor.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed tolerance")
FPCA.tol = 1e-10
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tree_based_approximation(FUN, TREE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Ranks = %s" % F.tensor.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# Approximation in Tensor Train format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same for a linear tree, which gives the tensor-train format, with
# prescribed TT-ranks, then with the precision :math:`10^{-4}`, for which
# the printed error is of the same order.
print("\n--- Approximation in tensor train format ---")
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.projection_type = "interpolation"
FPCA.pca_sampling_factor = 3
FPCA.pca_adaptive_sampling = True
FPCA.bases = BASES

print("\nPrescribed ranks")
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 9, D - 1)
F, OUTPUT = FPCA.tt_approximation(FUN)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Prescribed ranks = %s" % FPCA.max_rank)
TT_RANKS = np.flip(F.tensor.ranks[F.tensor.is_active_node])
print("TT-ranks = %s" % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed tolerance")
FPCA.tol = 1e-4
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tt_approximation(FUN)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
TT_RANKS = np.flip(F.tensor.ranks[F.tensor.is_active_node])
print("TT-ranks = %s" % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)
