"""
Principal component analysis of a tensorized function
=====================================================

Identify a multivariate function with a tensorized function, then
approximate it with a principal component analysis based algorithm.

Each variable :math:`x_\\nu \\in [0, 1)` is written
:math:`x_\\nu = b^{-L}(i_\\nu + y_\\nu)`, with the digits
:math:`(i_{\\nu,1}, \\dots, i_{\\nu,L})` of :math:`i_\\nu` in base
:math:`b`, so that a function of :math:`d = 4` variables becomes a
function of :math:`(L + 1) d` variables, approximated in tree-based format
by principal component analysis. See :ref:`theory-qtt` and
:ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_TensorizedFunction_Learning_PCA.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)


# %%
# Tensorization of a function of four variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The function is
#
# .. math::
#
#    f(x) = \frac{1}{1 + x_1 + 2 x_2 + 3 x_3 + 4 x_4}, \qquad
#    x \in [0, 1)^4.
#
# A :class:`~tensap.Tensorizer` with base :math:`b = 2` and resolution
# :math:`L = 13` identifies it with a function :math:`g` of
# :math:`(L + 1) d = 56` variables. With ``ordering_type = 2`` (the
# resolution-wise ordering), the digits of the same level are grouped:
#
# .. math::
#
#    f(x) = g(i_{1,1}, \dots, i_{4,1}, \dots, i_{1,L}, \dots, i_{4,L},
#    y_1, \dots, y_4).

DIM = 4
L = 13  # Resolution
B = 2  # Scaling factor

T = tensap.Tensorizer(B, L, DIM)
T.ordering_type = 2
FUN = tensap.UserDefinedFunction("1/(1+x0+2*x1+3*x2+4*x3)", DIM)
FUN.evaluation_at_multiple_points = True
TENSORIZED_FUN = T.tensorize(FUN)

# %%
# The bases used for :math:`g` are, for each digit, all the
# functions on :math:`\{0, 1\}`, and, for each :math:`y_\nu \in [0, 1)`,
# the polynomials of degree at most 2.
DEGREE = 2
BASES = T.tensorized_function_functional_bases(DEGREE)


# %%
# Learning with PCA based algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.FunctionalTensorPrincipalComponentAnalysis` builds a
# tree-based approximation of :math:`g`, on a balanced dimension tree over
# the 56 variables, with a relative precision :math:`10^{-8}` for the
# estimation of each :math:`\alpha`-principal subspace and no bound on the
# :math:`\alpha`-ranks. The result, a :class:`~tensap.TensorizedFunction`,
# is evaluated as a function of :math:`x` at 1000 random points of
# :math:`[0, 1)^4`, where the printed relative error is small.
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 20
FPCA.bases = BASES
FPCA.display = False
TREE = tensap.DimensionTree.balanced(TENSORIZED_FUN.fun.dim)
FPCA.tol = 1e-8
FPCA.max_rank = np.inf
FUN_TB, OUTPUT = FPCA.tree_based_approximation(TENSORIZED_FUN, TREE)
TENSORIZED_FUN_TB = tensap.TensorizedFunction(FUN_TB, T)
X_TEST = T.X.random(1000)
F_X_TEST = TENSORIZED_FUN_TB(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print("Mean squared error = %2.5e" % ERR_L2)
