r"""
Tensorized functions
====================

Identify a function of one or two variables on (0, 1) with a function of
more variables, by b-adic tensorization.

Writing :math:`x \in [0, 1)` as :math:`x = b^{-L}(i + y)`, with
:math:`i = [i_1 \dots i_L]_b` the first :math:`L` digits of :math:`x` in base
:math:`b` and :math:`y \in [0, 1)`, identifies :math:`f` with a function of
:math:`L + 1` variables, :math:`f(x) = f_L(i_1, \dots, i_L, y)`. We
interpolate :math:`f_L` in a tensor product basis, then truncate its tensor of
coefficients in tensor train format, and look at the ranks, the storage and
the error for decreasing tolerances. See :ref:`theory-qtt`.

This example comes from the script
``test/non_regression_scripts/functions/tutorial_TensorizedFunction.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Identification of a function f(x) on (0,1) with a function g(i1, ...,id , y)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.Tensorizer` with base :math:`b = 3` and resolution
# :math:`L = 10` implements the map :math:`x \mapsto (i_1, \dots, i_L, y)`,
# and ``tensorize`` turns :math:`f(x) = \sqrt{x}` into the function
# :math:`f_L` of :math:`L + 1` variables.

L = 10  # Resolution
B = 3  # Scaling factor

T = tensap.Tensorizer(B, L, 1)


def FUN(x):
    return np.sqrt(x)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True

TENSORIZED_FUN = T.tensorize(FUN)
TENSORIZED_FUN.fun.evaluation_at_multiple_points = True

# %%
# ``tensorized_function_functional_bases`` gives a basis for each variable of
# :math:`f_L`: a basis of all the functions of a digit
# :math:`i_k \in \{0, \dots, b - 1\}`, and the polynomials of degree at most
# 3 in :math:`y`. The interpolant in the full tensor product basis,
#
# .. math::
#
#    f_L(i_1, \dots, i_L, y) \approx \sum_{j_1, \dots, j_{L+1}}
#    a(j_1, \dots, j_{L+1}) \, \varphi^{(1)}_{j_1}(i_1) \cdots
#    \varphi^{(L)}_{j_L}(i_L) \, \varphi^{(L+1)}_{j_{L+1}}(y),
#
# is a piecewise cubic interpolant of :math:`f` on the :math:`b^L` intervals
# of length :math:`b^{-L}`. Its tensor of coefficients :math:`a` is stored in
# full, with :math:`3^{10} \times 4` entries, and its relative error on a
# random sample is at the level of machine precision. A
# :class:`~tensap.TensorizedFunction` composes it with the map of the
# tensorizer, to evaluate it at points :math:`x`.

# Interpolation of the function in the tensor product feature space
DEGREE = 3
BASES = T.tensorized_function_functional_bases(DEGREE)
H = tensap.FullTensorProductFunctionalBasis(BASES)
FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)
X_TEST = T.X.random(100)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print("Mean squared error for the interpolation = %2.5e" % ERR_L2)

# %%
# The tensor :math:`a` is then truncated in tensor train format by
# ``Truncator.ttsvd``, for the tolerances :math:`\epsilon = 10^{-1}, \dots,
# 10^{-8}`. The result is a :class:`~tensap.TreeBasedTensor` with a linear
# dimension tree; in the printed ranks, the zeros are those of its inactive
# leaves. For each tolerance, we print the ranks, the storage and the
# relative error of the truncated function on a random sample.

# Truncation in tensor train format
TR = tensap.Truncator()
tens = TENSORIZED_FUN_INTERP.fun.tensor
for k in range(1, 9):
    TR.tolerance = 10 ** (-k)
    print("Tolerance =%s" % TR.tolerance)
    TENSORIZED_FUN_TT = TENSORIZED_FUN_INTERP
    TENSORIZED_FUN_TT.fun.tensor = TR.ttsvd(tens)
    print(
        "Representation ranks = %s"
        % TENSORIZED_FUN_TT.fun.tensor.representation_rank
    )
    print("Complexity = %s" % TENSORIZED_FUN_TT.fun.tensor.storage())
    X_TEST = T.X.random(1000)
    F_X_TEST = TENSORIZED_FUN_TT(X_TEST)
    Y_TEST = FUN(X_TEST)
    ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
    print("Mean squared error = %2.5e" % ERR_L2)

# %%
# The error decreases with the tolerance and stays below it, while the ranks
# grow slowly: even for :math:`\epsilon = 10^{-8}`, the storage is less than
# a thousand entries, instead of the :math:`3^{10} \times 4` entries of the
# full tensor.

# %%
# Identification of a bivariate function f(x1, x2) on (0,1)^2 with a function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In dimension 2, each variable is tensorized with its own digits,
# :math:`x_1 = b^{-L}(i + y_1)` and :math:`x_2 = b^{-L}(j + y_2)`, so that
# :math:`f(x_1, x_2)` becomes a function
# :math:`g(i_1, j_1, \dots, i_L, j_L, y_1, y_2)` of :math:`2L + 2`
# variables. Here :math:`b = 2`, :math:`L = 8`, and
# :math:`f(x) = 1 / (1 + x_1 + x_2)`. With ``ordering_type = 2`` the digits
# are ordered resolution-wise, :math:`i_1, j_1, i_2, j_2, \dots`, so that the
# digits of the same scale are neighbours in the tensor train. The same
# steps follow: interpolation with polynomials of degree 2 in
# :math:`y_1, y_2`, then truncation.

DIM = 2
L = 8  # Resolution
B = 2  # Scaling factor

T = tensap.Tensorizer(B, L, DIM)
T.ordering_type = 2
FUN = tensap.UserDefinedFunction("1/(1+x0+x1)", DIM)
FUN.evaluation_at_multiple_points = True
TENSORIZED_FUN = T.tensorize(FUN)

# Interpolation of the function in the tensor product feature space
DEGREE = 2
BASES = T.tensorized_function_functional_bases(DEGREE)

H = tensap.FullTensorProductFunctionalBasis(BASES)
FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)
X_TEST = T.X.random(100)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print("Mean squared error for the interpolation = %2.5e" % ERR_L2)

# Truncation in tensor train format
TR = tensap.Truncator()
tens = TENSORIZED_FUN_INTERP.fun.tensor
for k in range(1, 9):
    TR.tolerance = 10 ** (-k)
    print("Tolerance =%s" % TR.tolerance)
    TENSORIZED_FUN_TT = TENSORIZED_FUN_INTERP
    TENSORIZED_FUN_TT.fun.tensor = TR.ttsvd(tens)
    print(
        "Representation ranks = %s"
        % TENSORIZED_FUN_TT.fun.tensor.representation_rank
    )
    print("Complexity = %s" % TENSORIZED_FUN_TT.fun.tensor.storage())
    X_TEST = T.X.random(1000)
    F_X_TEST = TENSORIZED_FUN_TT(X_TEST)
    Y_TEST = FUN(X_TEST)
    ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
    print("Mean squared error = %2.5e" % ERR_L2)

# %%
# The behaviour is the same: the ranks stay small, at most 5 here, and the
# error follows the tolerance.
