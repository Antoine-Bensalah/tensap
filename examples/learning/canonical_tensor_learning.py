"""
Learning in canonical tensor format
===================================

Learn a function from samples in canonical tensor format.

Given samples :math:`(x_k, y_k)_{1 \\le k \\le n}` of :math:`Y = u(X)`, with
:math:`X = (X_1, \\dots, X_d)` a random vector, we solve the least-squares
problem :math:`\\min_{v \\in \\mathcal{R}_r} \\frac1n \\sum_{k=1}^{n}
(y_k - v(x_k))^2` over the functions of canonical rank at most :math:`r`, for
increasing :math:`r`, on the borehole function in dimension :math:`d = 8`.
See :ref:`theory-tensor-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_tensor_learning_CanonicalTensorLearning.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import tensap
import numpy as np

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Function to approximate
# ^^^^^^^^^^^^^^^^^^^^^^^
# Two choices: a function of :math:`d = 5` normal variables that depends on
# two of them only, or (the default) the borehole function, which models
# the flow of water through a borehole as a function of :math:`d = 8`
# independent normal and uniform variables. ``X`` is the
# :class:`~tensap.RandomVector` that gives the measure :math:`\mu`.
CHOICE = 2
if CHOICE == 1:
    ORDER = 5
    X = tensap.RandomVector(tensap.NormalRandomVariable(), ORDER)

    def fun(x):
        return 1 / (10 + x[:, 0] + 0.5 * x[:, 1]) ** 2

elif CHOICE == 2:
    fun, X = tensap.multivariate_functions_benchmark("borehole")
    ORDER = X.size

# %%
# Approximation basis
# ^^^^^^^^^^^^^^^^^^^
# In each dimension :math:`\nu`, the polynomials
# :math:`\varphi^{(\nu)}_i`, :math:`i \in I_\nu = \{1, \dots, 9\}`
# (degree at most 8), orthonormal for the law :math:`\mu_\nu` of
# :math:`X_\nu`. A function of canonical rank :math:`r` then reads
#
# .. math::
#
#    v(x) = \sum_{k=1}^{r} v^{(1)}_k(x_1) \cdots v^{(d)}_k(x_d),
#    \qquad v^{(\nu)}_k(x_\nu) = \sum_{i \in I_\nu}
#    a^{(\nu)}_k(i)\, \varphi^{(\nu)}_i(x_\nu).
DEGREE = 8
BASES = [
    tensap.PolynomialFunctionalBasis(
        X_TRAIN.orthonormal_polynomials(), range(DEGREE + 1)
    )
    for X_TRAIN in X.random_variables
]
BASES = tensap.FunctionalBases(BASES)

# %%
# Training and test samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# :math:`n = 1000` samples to learn from, and a larger independent sample to
# estimate the error of the approximation.
NUM_TRAIN = 1000
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = fun(X_TRAIN)

NUM_TEST = 10000
X_TEST = X.random(NUM_TEST)
Y_TEST = fun(X_TEST)

# %%
# Learning in canonical tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.CanonicalTensorLearning` with the
# :class:`~tensap.SquareLossFunction`. For a fixed rank, the problem is
# solved by alternating minimization: with all the functions
# :math:`v^{(\eta)}_k`, :math:`\eta \ne \nu`, fixed, :math:`v` is a linear
# model in the coefficients :math:`(a^{(\nu)}_k)_{k}`, learned by least
# squares (here with a selection of the basis functions, ``basis_adaptation``).
# With ``rank_adaptation``, the rank is then increased by one, up to 20
# times, and ``model_selection`` keeps the iterate with the smallest test
# error.
SOLVER = tensap.CanonicalTensorLearning(
    ORDER, tensap.SquareLossFunction()
)
SOLVER.rank_adaptation = True
SOLVER.initialization_type = "mean"
SOLVER.tolerance["on_error"] = 1e-6
SOLVER.alternating_minimization_parameters["stagnation"] = 1e-8
SOLVER.alternating_minimization_parameters["max_iterations"] = 100
SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.bases = BASES
SOLVER.training_data = [X_TRAIN, Y_TRAIN]
SOLVER.display = True
SOLVER.alternating_minimization_parameters["display"] = False
SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
SOLVER.alternating_minimization_parameters["one_by_one_factor"] = False
SOLVER.alternating_minimization_parameters["inner_loops"] = 2
SOLVER.alternating_minimization_parameters["random"] = False
SOLVER.rank_adaptation_options["max_iterations"] = 20
SOLVER.model_selection = True
SOLVER.model_selection_options["type"] = "test_error"

F, OUTPUT = SOLVER.solve()

# %%
# The selected approximation: its canonical rank, and its relative error on
# the test sample. The test error stops decreasing after a few ranks, and
# the selected approximation has rank 4, with a relative test error of the
# order of :math:`10^{-5}`.

TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print(
    "\nCanonical rank = %i, test error = %2.5e"
    % (len(F.tensor.core.data), TEST_ERROR)
)
