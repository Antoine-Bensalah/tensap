"""
Tensor completion
=================

Recover a tensor from some of its entries, in tree-based tensor format.

A tensor :math:`u \\in \\mathbb{R}^{I_1 \\times \\cdots \\times I_d}` is known
only through its entries :math:`u(i)`, :math:`i \\in \\Omega`, for a random
set :math:`\\Omega` of 30% of the indices. We solve
:math:`\\min_{v} \\sum_{i \\in \\Omega} (u(i) - v(i))^2` over tensors
:math:`v` in tree-based format, of order :math:`d = 5`, and look at the error
on all the entries. See :ref:`theory-tensor-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_tensor_learning_tensor_completion.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

from enum import Enum
from time import time
import numpy as np
import tensap

np.random.seed(0)

# %%
# Generation of the tensor to recover
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A random :class:`~tensap.TreeBasedTensor` of size
# :math:`5 \times 8 \times 4 \times 3 \times 6`, for a linear dimension tree
# :math:`T` and given ranks :math:`(r_\alpha)_{\alpha \in T}`: the tensor to
# recover has an exact low-rank structure.
sz = [5, 8, 4, 3, 6]  # Size of the tensor
ORDER = len(sz)  # Order of the tensor
T = tensap.DimensionTree.linear(ORDER)
ranks = [1, 3, 3, 4, 3, 6, 3, 2, 4]
np.random.seed(0)
FTBT = tensap.TreeBasedTensor.randn(T, ranks, sz)
F = FTBT.full()

# %%
# Training and test samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# The training sample is the set :math:`\Omega` of known entries, drawn
# without replacement. The test sample is made of all the entries, so that
# the test error is the relative error on the whole tensor.
p = 0.3  # Proportion of known entries of the tensor

N = np.prod(sz)
NUM_TRAIN = np.int64(np.round(p * N))
loc_TRAIN = np.random.choice(
    N, NUM_TRAIN, replace=False
)  # Random selection of n entries
indices_TRAIN = tensap.MultiIndices.ind2sub(sz, loc_TRAIN)
Y_TRAIN = F.eval_at_indices(indices_TRAIN.array)

NUM_TEST = N
loc_TEST = range(N)
indices_TEST = tensap.MultiIndices.ind2sub(sz, loc_TEST)
Y_TEST = F.eval_at_indices(indices_TEST.array)
print(
    "Tensor to recover with ",
    p * 100,
    "% of its entries known (",
    NUM_TRAIN,
    " entries):",
)
print(FTBT)

# %%
# Features: indicator vectors of the indices
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Completion is a learning problem on a discrete set: an entry is
# :math:`v(i) = (v, e^{(1)}_{i_1} \otimes \cdots \otimes e^{(d)}_{i_d})`,
# with :math:`e^{(\nu)}_{i_\nu}` the canonical basis vectors of
# :math:`\mathbb{R}^{N_\nu}`. The features of dimension :math:`\nu` are
# therefore, for each sample, the indicator vector of its index
# :math:`i_\nu`: one row per sample, with a single 1.
FEATURES_TRAIN = []
for i in range(ORDER):
    M = np.zeros((NUM_TRAIN, sz[i]))
    np.put_along_axis(M, indices_TRAIN.array[:, i : i + 1], 1.0, axis=1)
    FEATURES_TRAIN.append(M)

FEATURES_TEST = []
for i in range(ORDER):
    M = np.zeros((NUM_TEST, sz[i]))
    np.put_along_axis(M, indices_TEST.array[:, i : i + 1], 1.0, axis=1)
    FEATURES_TEST.append(M)


# %%
# Tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The format of the approximation: a random tree, the tensor-train format,
# the tensor train Tucker format (the default) or a balanced binary tree.


class TensorFormat(str, Enum):
    RANDON_TREE_AND_ACTIVE_NODES = "Random tree and active nodes"
    TENSOR_TRAIN = "Tensor-Train"
    HIERARCHICAL_TENSOR_TRAIN = "Hierarchial Tensor-Train"
    BINARY_TREE = "Binary tree"


CHOICE = TensorFormat.HIERARCHICAL_TENSOR_TRAIN
if CHOICE == TensorFormat.RANDON_TREE_AND_ACTIVE_NODES:
    print("Random tree with active nodes")
    ARITY = [2, 4]
    TREE = tensap.DimensionTree.random(ORDER, ARITY)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(
        TREE, IS_ACTIVE_NODE, tensap.SquareLossFunction()
    )
elif CHOICE == TensorFormat.TENSOR_TRAIN:
    print("Tensor-train format")
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train(
        ORDER, tensap.SquareLossFunction()
    )
elif CHOICE == TensorFormat.HIERARCHICAL_TENSOR_TRAIN:
    print("Tensor Train Tucker")
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        ORDER, tensap.SquareLossFunction()
    )
elif CHOICE == TensorFormat.BINARY_TREE:
    print("Binary tree")
    TREE = tensap.DimensionTree.balanced(ORDER)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(
        TREE, IS_ACTIVE_NODE, tensap.SquareLossFunction()
    )
else:
    raise NotImplementedError("Not implemented.")

TREE = SOLVER.tree
IS_ACTIVE_NODE = SOLVER.is_active_node

# %%
# Random shuffling of the dimensions associated to the leaves
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Off here: the variables keep their natural order along the tree.
RANDOMIZE = False
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

# %%
# Initial guess: known entries in a rank-1 tree-based tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A natural starting point: the tensor equal to :math:`u` on :math:`\Omega`
# and to 0 elsewhere, truncated to rank 1 by the hierarchical SVD
# (``Truncator.hsvd``). It is only built here: the solver below starts from
# its own ``"canonical"`` initialization.
guess = np.zeros(np.prod(sz))
guess[loc_TRAIN] = Y_TRAIN
guess = guess.reshape(sz, order="F")
guess = tensap.FullTensor(guess, order=ORDER, shape=sz)
tr = tensap.Truncator(tolerance=0, max_rank=1)
guess = tr.hsvd(guess, SOLVER.tree, SOLVER.is_active_node)

# %%
# Learning in tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.TreeBasedTensorLearning` with the square loss, the
# indicator features as evaluated bases, and rank adaptation from rank 1:
# at each iteration, the ranks of the nodes with the largest estimated
# truncation errors are increased, and the tensors of the network are
# updated by alternating least squares. The tree is kept fixed.
SOLVER.bases_eval = FEATURES_TRAIN
SOLVER.training_data = [None, Y_TRAIN]

SOLVER.tolerance["on_stagnation"] = 1e-8
SOLVER.tolerance["on_error"] = 1e-8

SOLVER.initialization_type = "canonical"

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [None, Y_TEST]
SOLVER.bases_eval_test = FEATURES_TEST

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options["max_iterations"] = 20
SOLVER.rank_adaptation_options["theta"] = 0.8
SOLVER.rank_adaptation_options["early_stopping"] = False
SOLVER.rank_adaptation_options["early_stopping_factor"] = 10

SOLVER.tree_adaptation = False
SOLVER.tree_adaptation_options["max_iterations"] = 1e2
# SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

SOLVER.alternating_minimization_parameters["stagnation"] = 1e-10
SOLVER.alternating_minimization_parameters["max_iterations"] = 50

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
Fpred, OUTPUT = SOLVER.solve()
T1 = time()
print(T1 - T0)

# ESTIMATED TENSOR
Fpred = Fpred.tensor

# %%
# Displays
# ^^^^^^^^
# The ranks found, to compare with the ranks of :math:`u`, the
# leave-one-out error on the known entries, and the error on all the
# entries of the tensor. The rank adaptation finds the ranks of :math:`u`,
# and the tensor is recovered from 30% of its entries up to a relative
# error of the order of :math:`10^{-10}`.
TEST_ERROR = SOLVER.loss_function.test_error(
    Fpred.eval_at_indices(indices_TEST.array), [None, Y_TEST]
)
print("Ranks: %s" % Fpred.ranks)
print("Loo error = %2.5e" % OUTPUT["error"])
print("Test error = %2.5e" % TEST_ERROR)

Fpred.plot(title="Active nodes")
Fpred.tree.plot_dims(title="Dimensions associated to the leaf nodes")
Fpred.tree.plot_with_labels_at_nodes(
    Fpred.representation_rank, title="Representation ranks"
)
