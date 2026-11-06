"""
Quantized tensor trains
=======================

Represent a vector of :math:`2^L` values, or a function on a grid of
:math:`2^L` points, as a tensor train of order :math:`L` with modes of
size 2 (QTT), and build matrices in the same format.

The index :math:`i = \\sum_{k=1}^{L} i_k 2^{L-k}` of a grid point is
written with its binary digits :math:`i_k \\in \\{0, 1\\}`, most
significant first, and the vector
:math:`v(i)` becomes the tensor :math:`v(i_1, \\dots, i_L)`. Many smooth
functions and structured matrices have QTT ranks independent of
:math:`L`: the storage grows like :math:`L`, the logarithm of the number
of points. We look at these ranks for a few functions and for the second
difference matrix. See :ref:`theory-qtt`.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import tensap

# %%
# Functions on a grid
# ^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.Tensorizer` with base 2 and :math:`L` digits describes
# the grid of :math:`2^L` points :math:`x_i = 2^{-L} i` of :math:`[0, 1)`
# and the order of the digits. :func:`~tensap.quantize_function` samples a
# function :math:`f` on this grid and compresses the tensor
# :math:`v(i_1, \dots, i_L) = f(x_i)` by a TT-SVD at the given precision.
#
# Some ranks are known exactly. Since
# :math:`x = \sum_k i_k 2^{-k}`, an exponential factorizes digit by digit,
#
# .. math::
#
#    e^{a x} = \prod_{k=1}^{L} e^{a\, 2^{-k} i_k},
#
# an elementary tensor of rank 1, and a sine, a sum of two complex
# exponentials, has rank 2. A function with a localized feature, here a
# steep variation near :math:`x = 0`, needs larger ranks, which grow slowly
# with :math:`L`.
functions = {
    "exp(x)": lambda x: np.exp(x[:, 0]),
    "sin(10 x)": lambda x: np.sin(10 * x[:, 0]),
    "1 / (x + 0.01)": lambda x: 1 / (x[:, 0] + 0.01),
}
for L in (10, 20):
    tensorizer = tensap.Tensorizer(2, L, 1)
    for name, fun in functions.items():
        v = tensap.quantize_function(fun, tensorizer, tolerance=1e-10)
        print(
            f"L = {L}: {name:15s} max QTT rank {max(v.ranks)}, "
            f"storage {v.storage()} for {2**L} values"
        )

# %%
# Going from :math:`L = 10` to :math:`L = 20` multiplies the number of values
# by about :math:`10^3`, while the storage of each function roughly doubles,
# like :math:`L`; the rank of the last function only increases by one.

# %%
# Back to the values
# ^^^^^^^^^^^^^^^^^^
# :func:`~tensap.dequantize` maps a QTT tensor back to the vector of its
# :math:`2^L` values on the grid, which is only reasonable for a small
# :math:`L`. Here :math:`f(x) = \sin(10 x)\, e^{-x}` with :math:`L = 8`,
# compressed at the default precision, and plotted against the grid points
# given by :func:`~tensap.grid_points`.
tensorizer = tensap.Tensorizer(2, 8, 1)
v = tensap.quantize_function(
    lambda x: np.sin(10 * x[:, 0]) * np.exp(-x[:, 0]), tensorizer
)
x = tensap.grid_points(tensorizer)
plt.figure()
plt.plot(
    x, tensap.dequantize(v, tensorizer), label="from the QTT tensor"
)
plt.xlabel("x")
plt.legend()
plt.show()

# %%
# Matrices
# ^^^^^^^^
# A matrix :math:`A \in \mathbb{R}^{2^L \times 2^L}` acting on such vectors
# is tensorized with the same digits for its rows and its columns,
# :math:`A(i, j) = A(i_1, j_1, \dots, i_L, j_L)`: it is a
# :class:`~tensap.TTOperator` with cores of size :math:`2 \times 2`. The
# QTT primitives build the usual 1D operators directly in this format,
# without forming the matrix: the shift :math:`(S v)(i) = v(i + 1)`
# (:func:`~tensap.qtt_shift`), of QTT rank 2, banded Toeplitz matrices
# :math:`\sum_c t_c S^c` (:func:`~tensap.qtt_banded`), corner blocks.
# The second difference matrix :math:`\mathrm{tridiag}(1, -2, 1)` has QTT
# ranks at most 3 whatever the number of points, up to more than
# :math:`10^9` points here.
for L in (10, 20, 30):
    D2 = tensap.qtt_banded({-1: 1.0, 0: -2.0, 1: 1.0}, L)
    print(
        f"L = {L}: {2**L} x {2**L} matrix, QTT ranks at most {max(D2.ranks)}"
    )

# %%
# On a small grid we can check the QTT matrix against its dense version
# (:func:`~tensap.dequantize_operator`): the product computed core by core,
# ``D2 @ v``, agrees with the dense matrix-vector product up to rounding
# errors.
L = 8
D2 = tensap.qtt_banded({-1: 1.0, 0: -2.0, 1: 1.0}, L)
dense = tensap.dequantize_operator(D2)
error = np.abs(
    tensap.dequantize(D2 @ v) - dense @ tensap.dequantize(v)
).max()
print(f"D2 @ v against the dense matrix: max difference {error:.1e}")
