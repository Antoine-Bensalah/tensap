"""
Learning in canonical tensor format
===================================

Learn a function from samples in canonical tensor format.

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
NUM_TRAIN = 1000
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = fun(X_TRAIN)

NUM_TEST = 10000
X_TEST = X.random(NUM_TEST)
Y_TEST = fun(X_TEST)

# %%
# Learning in canonical tensor format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SOLVER = tensap.CanonicalTensorLearning(ORDER, tensap.SquareLossFunction())
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

TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print(
    "\nCanonical rank = %i, test error = %2.5e" % (len(F.tensor.core.data), TEST_ERROR)
)
