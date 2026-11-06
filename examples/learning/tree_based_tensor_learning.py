"""
Learning in tree-based tensor format
====================================

Learn a function from samples in tree-based tensor format, with rank and
tree adaptation.

From samples :math:`(x_k, y_k)_{1 \\le k \\le n}` of :math:`Y = u(X)`, we
solve the least-squares problem :math:`\\min_{v} \\frac1n \\sum_{k=1}^{n}
(y_k - v(x_k))^2` over functions :math:`v` in a tree-based tensor format,
with a dimension tree :math:`T` and ranks :math:`(r_\\alpha)_{\\alpha \\in T}`
that are both adapted. The function is the borehole function of
:math:`d = 8` variables, learned from :math:`n = 100` samples only.
See :ref:`theory-tensor-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_tensor_learning_TreeBasedTensorLearning.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time
import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Function to approximate
# ^^^^^^^^^^^^^^^^^^^^^^^
# A function of :math:`d = 5` normal variables, or (the default) the
# borehole function of :math:`d = 8` independent normal and uniform
# variables, with ``X`` the :class:`~tensap.RandomVector`
# :math:`X = (X_1, \dots, X_d)` of law :math:`\mu`.
CHOICE = 2
if CHOICE == 1:
    ORDER = 5
    X = tensap.RandomVector(tensap.NormalRandomVariable(), ORDER)

    def fun(x):
        return 1 / (10 + x[:, 0] + 0.5 * x[:, 1]) ** 2

elif CHOICE == 2:
    fun, X = tensap.multivariate_functions_benchmark("borehole")
    ORDER = X.size
else:
    raise ValueError("Bad function choice.")

# %%
# Approximation basis
# ^^^^^^^^^^^^^^^^^^^
# Polynomials of degree at most 8 in each variable, orthonormal for the law
# :math:`\mu_\nu` of :math:`X_\nu` (or canonical monomials, with
# ``ORTHONORMAL_BASES = False``). The function is sought as
#
# .. math::
#
#    v(x) = \sum_{i_1 \in I_1} \cdots \sum_{i_d \in I_d}
#    a(i_1, \dots, i_d)\, \varphi^{(1)}_{i_1}(x_1) \cdots
#    \varphi^{(d)}_{i_d}(x_d),
#
# with :math:`I_\nu = \{1, \dots, 9\}` and a tensor of coefficients
# :math:`a` in tree-based format.
DEGREE = 8
ORTHONORMAL_BASES = True
if ORTHONORMAL_BASES:
    BASES = [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]
else:
    BASES = [
        tensap.PolynomialFunctionalBasis(
            tensap.CanonicalPolynomials(), range(DEGREE + 1)
        )
        for x in X.random_variables
    ]
BASES = tensap.FunctionalBases(BASES)

# %%
# Training and test samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# :math:`n = 100` training samples, fewer than the :math:`9^8` coefficients
# of the full tensor, and a test sample to measure the error.
NUM_TRAIN = 100
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = fun(X_TRAIN)

NUM_TEST = 10000
X_TEST = X.random(NUM_TEST)
Y_TEST = fun(X_TEST)

# %%
# Tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The format is given by the dimension tree :math:`T` and its active nodes,
# the nodes :math:`\alpha` whose rank :math:`r_\alpha` is bounded:
#
# 1. a random tree, all nodes active;
# 2. the tensor-train format;
# 3. the tensor train Tucker format, a linear binary tree with active leaves
#    (the default);
# 4. a balanced binary tree.
CHOICE = 3
if CHOICE == 1:
    print("Random tree with active nodes")
    ARITY = [2, 4]
    TREE = tensap.DimensionTree.random(ORDER, ARITY)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(
        TREE, IS_ACTIVE_NODE, tensap.SquareLossFunction()
    )
elif CHOICE == 2:
    print("Tensor-train format")
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train(
        ORDER, tensap.SquareLossFunction()
    )
elif CHOICE == 3:
    print("Tensor Train Tucker")
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        ORDER, tensap.SquareLossFunction()
    )
elif CHOICE == 4:
    print("Binary tree")
    TREE = tensap.DimensionTree.balanced(ORDER)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(
        TREE, IS_ACTIVE_NODE, tensap.SquareLossFunction()
    )
else:
    raise NotImplementedError("Not implemented.")

# %%
# Random shuffling of the dimensions associated to the leaves
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The variables are assigned to the leaves in a random order, so that the
# tree adaptation below has a bad initial tree to correct.
RANDOMIZE = True
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

# %%
# Learning in tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.TreeBasedTensorLearning` with the
# :class:`~tensap.SquareLossFunction`. For fixed tree and ranks, the
# parametrization is multilinear: with all the tensors :math:`C^\beta`,
# :math:`\beta \ne \alpha`, fixed, :math:`v` is a linear model in
# :math:`C^\alpha`, learned by least squares with a selection of the basis
# functions (``basis_adaptation``). The alternating minimization goes
# through the nodes. Then
#
# - ``rank_adaptation``: the ranks of the nodes with the largest estimated
#   truncation errors are increased, ``theta`` controlling how many nodes are
#   enriched at each iteration;
# - ``tree_adaptation``: permutations of nodes that reduce the storage of
#   the current approximation;
# - ``model_selection``: the iterate with the smallest test error is kept.
#
# The bases are evaluated once on the training points (``bases_eval``), so
# the training data only contain the outputs.
SOLVER.bases = BASES
SOLVER.bases_eval = BASES.eval(X_TRAIN)
SOLVER.training_data = [None, Y_TRAIN]

SOLVER.tolerance["on_stagnation"] = 1e-6
SOLVER.tolerance["on_error"] = 1e-6

SOLVER.initialization_type = "canonical"

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options["max_iterations"] = 10
SOLVER.rank_adaptation_options["theta"] = 0.8
SOLVER.rank_adaptation_options["early_stopping"] = True
SOLVER.rank_adaptation_options["early_stopping_factor"] = 10

SOLVER.tree_adaptation = True
SOLVER.tree_adaptation_options["max_iterations"] = 1e2
# SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

SOLVER.alternating_minimization_parameters["stagnation"] = 1e-10
SOLVER.alternating_minimization_parameters["max_iterations"] = 5

SOLVER.display = True
SOLVER.alternating_minimization_parameters["display"] = False

SOLVER.model_selection = True
SOLVER.model_selection_options["type"] = "test_error"

# DMRG (inner rank adaptation)
DMRG = False
if DMRG:
    # Rank of the initialization
    SOLVER.rank = 3
    # Type of rank adaptation: can be 'dmrg' to perform classical DMRG
    # (using a truncation of the factor to adapt the rank) or
    # 'dmrg_low_rank' to learn the factor using a rank-adaptive learning
    # algorithm
    SOLVER.rank_adaptation_options["type"] = "dmrg_low_rank"
    # Maximum alpha-rank for all the nodes of the tree
    SOLVER.rank_adaptation_options["max_rank"] = 10
    # If True: perform an alternating minimization with fixed tree and
    # ranks after the rank adaptation using DMRG
    SOLVER.rank_adaptation_options["post_alternating_minimization"] = (
        False
    )
    # Model selection type when type is 'dmrg_low_rank': can be
    # 'cv_error' to use a cross-validation estimator of the error or
    # 'test_error' to use the error on a test sample
    SOLVER.rank_adaptation_options["model_selection_type"] = "cv_error"

T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1 - T0)

# %%
# Displays
# ^^^^^^^^
# The ranks of the selected approximation, its leave-one-out
# cross-validation error and its test error, then the adapted tree: active
# nodes, variables at the leaves, and representation ranks. With only 100
# samples, the selected approximation has ranks at most 2 and a relative
# test error of a few :math:`10^{-4}`.
TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print("Ranks: %s" % F.tensor.ranks)
print("Loo error = %2.5e" % OUTPUT["error"])
print("Test error = %2.5e" % TEST_ERROR)

F.tensor.plot(title="Active nodes")
F.tensor.tree.plot_dims(title="Dimensions associated to the leaf nodes")
F.tensor.tree.plot_with_labels_at_nodes(
    F.tensor.representation_rank, title="Representation ranks"
)
