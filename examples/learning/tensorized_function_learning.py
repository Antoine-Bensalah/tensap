"""
Learning a tensorized function
==============================

Learn a tensorized function in tree-based tensor format, with model
selection by a slope heuristic instead of a test sample.

A function :math:`f` on :math:`[0, 1)` is identified with a function of
:math:`L + 1` variables, :math:`f(x) = f_L(i_1, \\dots, i_L, y)` with
:math:`x = b^{-L}(i + y)`, and :math:`f_L` is learned from :math:`n = 200`
samples by least squares, :math:`\\min_{v} \\frac1n \\sum_{k=1}^{n}
(f(x_k) - v(x_k))^2`, with :math:`v` in tree-based tensor format. Here
:math:`b = 5` and :math:`L = 4`. See :ref:`theory-qtt` and
:ref:`theory-tensor-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_tensor_learning_tensorized_function_learning.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Function to approximate: identification of a function f(x) with a function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :class:`~tensap.Tensorizer` with base :math:`b = 5` and resolution
# :math:`L = 4` writes :math:`x \in [0, 1)` as :math:`x = b^{-L}(i + y)`,
# with :math:`i = [i_1 \dots i_L]_b` the first :math:`L` digits of
# :math:`x` in base :math:`b` and :math:`y \in [0, 1)`. ``tensorize``
# returns :math:`f_L`, a function of :math:`d = L + 1 = 5` variables. The
# default :math:`f` oscillates, with an amplitude decreasing with :math:`x`.
R = 4  # Resolution
B = 5  # Scaling factor
ORDER = R + 1

X = tensap.UniformRandomVariable(0, 1)
Y = tensap.UniformRandomVariable(0, 1)

CHOICE = 1
if CHOICE == 1:

    def FUN(x):
        return (
            np.sin(10 * np.pi * (2 * x + 0.5)) / (4 * x + 1)
            + (2 * x - 0.5) ** 4
        )

elif CHOICE == 2:

    def FUN(x):
        return (
            np.sin(4 * np.pi * x) + 0.2 * np.cos(16 * np.pi * x)
        ) ** (x < 0.5) + (2 * x - 1) * (x >= 0.5)

else:
    raise ValueError("Bad function choice.")

T = tensap.Tensorizer(B, R, 1, X, Y)
TENSORIZED_FUN = T.tensorize(FUN)
TENSORIZED_FUN.fun.evaluation_at_multiple_points = True

# %%
# Approximation basis
# ^^^^^^^^^^^^^^^^^^^
# For each digit :math:`i_\nu`, a basis of all the functions on
# :math:`\{0, \dots, b - 1\}`, and for :math:`y`, the polynomials of degree
# at most 5, orthonormal on :math:`[0, 1)`: the approximation is piecewise
# polynomial on the :math:`b^L` intervals of the grid.
DEGREE = 5
H = tensap.PolynomialFunctionalBasis(
    Y.orthonormal_polynomials(), range(DEGREE + 1)
)
BASES = T.tensorized_function_functional_bases(H)

# %%
# Training and test samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Samples of :math:`x`, uniform on :math:`[0, 1)`, mapped by ``T.map`` to
# their variables :math:`(i_1, \dots, i_L, y)`.
NUM_TRAIN = 200
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = FUN(X_TRAIN)
X_TRAIN = T.map(
    X_TRAIN
)  # Identification of X_TRAIN with (i_1,...,i_d,y)

NUM_TEST = 1000
X_TEST = X.random(NUM_TEST)
Y_TEST = FUN(X_TEST)
X_TEST = T.map(X_TEST)

# %%
# Tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The format of :math:`v`: a random tree, the tensor-train format, the
# tensor train Tucker format (the default) or a balanced binary tree.
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
# The digits are assigned to the leaves in a random order; the tree
# adaptation then looks for a better ordering.
RANDOMIZE = True
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

# %%
# Learning in tree-based tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.TreeBasedTensorLearning` with the square loss, rank and
# tree adaptation. The solver is not stopped early: it returns the whole
# sequence of iterates :math:`v_1, v_2, \dots` of increasing ranks, among
# which a model is selected, here by the test error. Selection matters: with
# 200 samples, the late iterates of large ranks fit the training data better
# and better while their test error grows.
SOLVER.bases = BASES
SOLVER.bases_eval = BASES.eval(X_TRAIN)
SOLVER.training_data = [None, Y_TRAIN]

# Let the solver go to the maximum number of iterations, to perform
# model selection a posteriori using the test error and a slope
# heuristic
SOLVER.tolerance["on_stagnation"] = 0
SOLVER.tolerance["on_error"] = 0
SOLVER.rank_adaptation_options["early_stopping"] = False

SOLVER.initialization_type = "canonical"

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options["max_iterations"] = 50
SOLVER.rank_adaptation_options["theta"] = 0.8

SOLVER.tree_adaptation = True
SOLVER.tree_adaptation_options["max_iterations"] = 100
# SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

SOLVER.alternating_minimization_parameters["stagnation"] = 1e-10
SOLVER.alternating_minimization_parameters["max_iterations"] = 50

SOLVER.display = True
SOLVER.alternating_minimization_parameters["display"] = False

SOLVER.model_selection = True
SOLVER.model_selection_options["type"] = "test_error"

T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1 - T0)

# %%
# Displays
# ^^^^^^^^
# The ranks and errors of the selected model, its tree, and the
# approximation against the function on :math:`[0, 1]`.
TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print("\nRanks: %s" % F.tensor.ranks)
print("Loo error = %2.5e" % OUTPUT["error"])
print("Test error = %2.5e" % TEST_ERROR)

F.tensor.plot(title="Active nodes")
F.tensor.tree.plot_dims(title="Dimensions associated to the leaf nodes")
F.tensor.tree.plot_with_labels_at_nodes(
    F.tensor.representation_rank, title="Representation ranks"
)

plt.figure()
x_lin = np.linspace(0, 1, 1000)
plt.plot(x_lin, FUN(x_lin), x_lin, F(T.map(x_lin)))
plt.legend(("True function", "Approximation"))
plt.show()

# %%
# Model selection using a slope heuristic instead of the test error
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Without a test sample, a model is chosen among the iterates :math:`v_m`
# by minimizing a penalized empirical risk,
#
# .. math::
#
#    \widehat m(\lambda) \in \arg\min_m \widehat{\mathcal{R}}_n(v_m)
#    + \lambda\, \frac{C(v_m)}{n},
#
# with :math:`C(v_m)` the storage of :math:`v_m`. The slope heuristic of
# :class:`~tensap.ModelSelection` finds the value of :math:`\lambda` at
# which the complexity of :math:`\widehat m(\lambda)` drops the most, and
# takes :math:`\widehat\lambda` equal to ``gap_factor`` times this value.
SEL = tensap.ModelSelection()
# Complexity of each model: standard complexity based on the method
# storage
SEL.data["complexity"] = SEL.complexity(
    OUTPUT["iterates"], fun="storage", c_type="standard"
)
# Empirical risk associated with each model
SEL.data["empirical_risk"] = [
    SOLVER.loss_function.risk_estimation(x, (X_TRAIN, Y_TRAIN))
    for x in OUTPUT["iterates"]
]
# Shape of the penalization function
SEL.pen_shape = lambda x: x / NUM_TRAIN
# Gap factor used in the slope heuristic
SEL.gap_factor = 2

LAMBDA_HAT, M_HAT, LAMBDA_PATH, M_PATH = SEL.slope_heuristic()

# %%
# The two selected models, with their test errors: the slope heuristic only
# uses the training sample.

print(
    "Model selected using a test sample:         %i, test error = "
    "%2.5e."
    % (
        np.ravel(OUTPUT["selected_model_number"])[0],
        OUTPUT["test_error"],
    )
)
print(
    "Model selected using the slope heuristic:   %i, test error = "
    "%2.5e."
    % (
        np.ravel(M_HAT)[0],
        OUTPUT["test_error_iterations"][np.ravel(M_HAT)[0]],
    )
)
