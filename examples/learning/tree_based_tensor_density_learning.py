"""
Density estimation in tree-based tensor format
==============================================

Estimate a multivariate probability density from samples in tree-based
tensor format, by least squares.

From samples :math:`x_1, \\dots, x_n` of a random vector
:math:`X = (X_1, \\dots, X_d)` of density :math:`u` with respect to a
reference measure :math:`\\mu`, we minimize the empirical :math:`L^2`
contrast :math:`\\|v\\|_{L^2_\\mu}^2 - \\frac2n \\sum_{k=1}^{n} v(x_k)` over
functions :math:`v` in tree-based tensor format, here for a correlated
gaussian vector in dimension :math:`d = 6`. See :ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_tensor_learning_TreeBasedTensorDensityLearning.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time
import numpy as np
from scipy.stats import multivariate_normal
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Function to approximate: multivariate gaussian distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A centred gaussian vector of covariance :math:`S`, whose components are
# not independent. The reference measure :math:`\mu` is the uniform
# probability measure on the box :math:`\prod_\nu [-5 S_{\nu\nu},
# 5 S_{\nu\nu}]`, and ``U`` is the density with respect to :math:`\mu`: the
# gaussian density times the volume of the box.
ORDER = 6
# Covariance matrix
S = np.array(
    [
        [2, 0, 0.5, 1, 0, 0.5],
        [0, 1, 0, 0, 0.5, 0],
        [0.5, 0, 2, 0, 0, 1],
        [1, 0, 0, 3, 0, 0],
        [0, 0.5, 0, 0, 1, 0],
        [0.5, 0, 1, 0, 0, 2],
    ]
)
# Reference measure
REF = tensap.RandomVector(
    [tensap.UniformRandomVariable(-x, x) for x in 5 * np.diag(S)]
)


# Density, defined with respect to the reference measure
def U(x):
    return multivariate_normal.pdf(x, np.zeros(ORDER), S) * np.prod(
        2 * 5 * np.diag(S)
    )


# %%
# Training and test samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Samples of :math:`X` to learn from and to estimate the risk, and a sample
# of the reference measure :math:`\mu` to estimate the :math:`L^2_\mu` error
# against the true density at the end.
NUM_TRAIN = 1e5  # Training sample size
X_TRAIN = multivariate_normal.rvs(np.zeros(ORDER), S, int(NUM_TRAIN))

NUM_TEST = 1e4  # Test sample size
X_TEST = multivariate_normal.rvs(np.zeros(ORDER), S, int(NUM_TEST))

NUM_REF = 1e4  # Size of the sample used to compute the L2 error
XI = REF.random(NUM_REF)

# %%
# Approximation basis
# ^^^^^^^^^^^^^^^^^^^
# Polynomials :math:`\varphi^{(\nu)}_i` of degree at most 20, orthonormal
# for the uniform measure :math:`\mu_\nu`. With an orthonormal basis,
# :math:`\|v\|_{L^2_\mu}` is the canonical norm of the tensor of
# coefficients of :math:`v`, which makes the contrast cheap to evaluate.
DEGREE = 20
# Orthonormal bases in each dimension, with respect to the reference
# measure
BASES = [
    tensap.PolynomialFunctionalBasis(
        x.orthonormal_polynomials(), range(DEGREE + 1)
    )
    for x in REF.random_variables
]
BASES = tensap.FunctionalBases(BASES)

# %%
# Tree-based tensor learning parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.TreeBasedTensorLearning` in tensor train Tucker format,
# with the :class:`~tensap.DensityL2LossFunction`
# :math:`\ell(v, x) = \|v\|_{L^2_\mu}^2 - 2 v(x)`. Its risk is
# :math:`\mathbb{E}(\ell(v, X)) = \|v - u\|_{L^2_\mu}^2 - \|u\|_{L^2_\mu}^2`,
# minimized by :math:`v = u`. As in regression, the minimization alternates
# over the tensors of the network (each step is a linear problem), with rank
# and tree adaptation; the leaves are first assigned to the variables in a
# random order. The risk is negative, hence the tolerances below.
SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
    ORDER, tensap.DensityL2LossFunction()
)

RANDOMIZE = True
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

SOLVER.bases = BASES
SOLVER.bases_eval = BASES.eval(X_TRAIN)
SOLVER.training_data = X_TRAIN

SOLVER.tolerance["on_stagnation"] = 1e-6
# In density estimation, the error is the risk, which is negative
SOLVER.tolerance["on_error"] = -np.inf

SOLVER.initialization_type = "random"

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = X_TEST
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options["max_iterations"] = 10
SOLVER.rank_adaptation_options["theta"] = 0.8
SOLVER.rank_adaptation_options["early_stopping"] = True
# early_stopping_factor < 1 because we the risk is negative
SOLVER.rank_adaptation_options["early_stopping_factor"] = 0.1

SOLVER.tree_adaptation = True
# For the tree adaptation in density estimation, a tolerance must be
# provided
SOLVER.tree_adaptation_options["tolerance"] = 1e-6

SOLVER.alternating_minimization_parameters["stagnation"] = 1e-6
SOLVER.alternating_minimization_parameters["max_iterations"] = 50

SOLVER.display = True
SOLVER.alternating_minimization_parameters["display"] = False

SOLVER.model_selection = True
SOLVER.model_selection_options["type"] = "test_error"

# %%
# Density estimation
# ^^^^^^^^^^^^^^^^^^
# After the solve, the risk of the selected approximation is estimated by
# leave-one-out and on the test sample, and the relative :math:`L^2_\mu`
# error against the true density on the sample of :math:`\mu`. The plots
# show the adapted tree, the variables at its leaves, and its ranks.
T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1 - T0)

L2_ERR = np.linalg.norm(U(XI) - F(XI)) / np.linalg.norm(U(XI))
print("Risk leave-one-out estimation =       %2.5e" % OUTPUT["error"])
print(
    "Risk estimation using a test sample = %2.5e" % OUTPUT["test_error"]
)
print("L2 relative error estimation =         %2.5e" % L2_ERR)

F.tensor.plot(title="Active nodes")
F.tensor.tree.plot_dims(title="Dimensions associated to the leaf nodes")
F.tensor.tree.plot_with_labels_at_nodes(
    F.tensor.representation_rank, title="Representation ranks"
)
