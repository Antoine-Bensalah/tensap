"""
Principal component analysis of tensors
=======================================

Approximation of a tensor in Tucker, tree-based and tensor-train formats by
higher-order principal component analysis, and comparison of subsampling
schemes.

A tensor :math:`u` of order :math:`d = 5` is only known through a function
returning its entries :math:`u(i_1, \\dots, i_d)`. For a set of dimensions
:math:`\\alpha`, its :math:`\\alpha`-principal components, the dominant left
singular vectors of the matricization :math:`M_\\alpha(u)`, are estimated
from a few entries, and give approximations of :math:`u` with prescribed
:math:`\\alpha`-ranks or precision. See :ref:`theory-learning` and
:ref:`theory-truncation`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_PCA_TensorPrincipalComponentAnalysis.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Creating a function providing the entries of a tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The tensor has :math:`d = 5` dimensions of size :math:`N_\nu = 10`, and
# entries
#
# .. math::
#
#    u(i_1, \dots, i_5) = \cos(t_1) + \frac{1}{1 + t_2^2 + t_3^4} + t_3,
#    \qquad t_\nu = \frac{i_\nu - 1}{10},
#
# for :math:`i_\nu \in I_\nu = \{1, \dots, 10\}` (the code numbers the
# indices from 0). It does not depend on :math:`i_4` and :math:`i_5`. The
# ``fun`` below evaluates :math:`u` at a batch of multi-indices, and ``X``
# draws random multi-indices, used to measure the errors.
D = 5
N = 10
SIZE = np.full(D, N)


def fun(i):
    return (
        np.cos(i[:, 0] / SIZE[0])
        + 1 / (1 + (i[:, 1] / SIZE[1]) ** 2 + (i[:, 2] / SIZE[2]) ** 4)
        + i[:, 2] / SIZE[2]
    )


X = tensap.random_multi_indices(SIZE)
TOL = 1e-8

# %%
# HOPCA (PCA for each dimension, provides reduced spaces)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The higher-order principal component analysis treats each
# :math:`\alpha = \{\nu\}`. The entries of :math:`u` are evaluated on
# :math:`I_\nu \times \{j^1, \dots, j^N\}`, for :math:`N` random
# multi-indices :math:`j^k` of the other dimensions: this gives an
# :math:`N_\nu \times N` matrix, a sample of the columns of
# :math:`M_\nu(u)`, whose truncated SVD gives the principal components.
# With ``pca_sampling_factor = 1``, :math:`N` is :math:`N_\nu` for a
# prescribed precision ``tol`` and the rank for a prescribed rank
# ``max_rank``. For the precision :math:`10^{-8}`, the estimated ranks are 1
# in the dimensions 4 and 5, on which :math:`u` does not depend.
print("--- Higher order PCA ---")
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print("\nPrescribed tolerance")
TPCA.tol = TOL
SUB_BASES, OUTPUTS = TPCA.hopca(fun, SIZE)
print(
    "Number of evaluations = \n%s"
    % [x["number_of_evaluations"] for x in OUTPUTS]
)
print("Ranks {1, ..., d} = \n%s" % [x.shape[1] for x in SUB_BASES])

print("\nPrescribed ranks")
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 5, D)
SUB_BASES, OUTPUTS = TPCA.hopca(fun, SIZE)
print(
    "Number of evaluations = \n%s"
    % [x["number_of_evaluations"] for x in OUTPUTS]
)
print("Ranks {1, ..., d} = \n%s" % [x.shape[1] for x in SUB_BASES])

# %%
# Approximation in Tucker Format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With the principal subspaces :math:`U^{(\nu)}` of all the dimensions,
# ``tucker_approximation`` interpolates :math:`u` on
# :math:`U^{(1)} \otimes \dots \otimes U^{(d)}`: in each subspace, magic
# indices are selected, and the core is computed from the entries of
# :math:`u` on their product grid. The lines ``alpha = ...`` give, for each
# node of the Tucker tree (numbered from 1, the root being 1), the rank and
# the number of evaluations, and the last line the evaluations used for the
# interpolation. The error is the relative error on :math:`10^4` random
# entries.
print("\n--- Approximation in Tucker format ---")
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print("\nPrescribed ranks")
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 5, D)
F, OUTPUT = TPCA.tucker_approximation(fun, SIZE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Prescribed ranks = %s" % TPCA.max_rank)
print("Ranks = %s" % F.ranks[F.tree.dim2ind - 1])
X_TEST = X.random(1e4)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed tolerance")
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tucker_approximation(fun, SIZE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Ranks = %s" % F.ranks[F.tree.dim2ind - 1])
X_TEST = X.random(1e4)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# Approximation in tree based format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For a dimension tree :math:`T`, here a balanced tree, the
# :math:`\alpha`-principal subspaces are computed from the leaves to the
# root: for an interior node :math:`\alpha`, the subspace is searched in
# the tensor product of the subspaces of its children, from evaluations on
# the product of their magic indices and on random indices of
# :math:`\alpha^c`. The result is a tree-based tensor with
# :math:`\alpha`-ranks :math:`r_\alpha`, :math:`\alpha \in T`, and
# ``max_rank`` is set to 1 at the root.
print("\n--- Approximation in tree based format ---")
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False
TREE = tensap.DimensionTree.balanced(D)

print("\nPrescribed ranks")
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 9, TREE.nb_nodes)
TPCA.max_rank[TREE.root - 1] = 1
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Prescribed ranks = %s" % TPCA.max_rank)
print("Ranks = %s" % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed tolerance")
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Ranks = %s" % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# With ``pca_adaptive_sampling``, the random indices :math:`j^k` are added
# one at a time, until their number exceeds ``pca_sampling_factor`` times
# the estimated rank. The ranks are
# almost the same as above, for far fewer evaluations of :math:`u`.
print("\nPrescribed tolerance (adaptive sampling)")
TPCA.tol = TOL
TPCA.pca_adaptive_sampling = True
TPCA.pca_sampling_factor = 1.2
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Ranks = %s" % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# Approximation in Tensor Train format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The tensor-train format is the tree-based format for a linear tree, with
# the :math:`d - 1` ranks :math:`r_{\{1, \dots, \nu\}}`, the TT-ranks.
# ``tt_approximation`` builds this tree and estimates the principal
# subspaces of its nodes as above.
print("\n--- Approximation in tensor train format ---")
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print("\nPrescribed tolerance")
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
TT_RANKS = np.flip(F.ranks[F.is_active_node])
print("TT-ranks = %s" % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("\nPrescribed ranks")
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 9, D - 1)
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)

print("Number of evaluations = %i" % OUTPUT["number_of_evaluations"])
print("Storage = %i" % F.storage())
print("Prescribed ranks = %s" % TPCA.max_rank)
TT_RANKS = np.flip(F.ranks[F.is_active_node])
print("TT-ranks = %s" % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# %%
# Comparison of different subsampling schemes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the subspace of each leaf, the interpolation indices are either magic
# indices (``"eim"``, empirical interpolation) or drawn at random
# (``"random"``). For the same prescribed TT-ranks, both choices give
# errors of the same order here.
X_TEST = X.random(1e3)
Y_TEST = fun(X_TEST)

TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 3
TPCA.pca_adaptive_sampling = False
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 9, D - 1)
TPCA.display = False

print("random:")
TPCA.subsampling = "random"
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)
F_X_TEST = F.eval_at_indices(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

print("eim:")
TPCA.subsampling = "eim"
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)
F_X_TEST = F.eval_at_indices(X_TEST)
print(
    "Error = %2.5e"
    % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
)

# print("maxvol:")
# TPCA.subsampling = "maxvol"
# TPCA.subsampling_tol = 1.
# F, OUTPUT = TPCA.tt_approximation(fun, SIZE)
# F_X_TEST = F.eval_at_indices(X_TEST)
# print(
#     "Error = %2.5e"
#     % (np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST))
# )
