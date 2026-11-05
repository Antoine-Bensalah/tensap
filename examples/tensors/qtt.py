"""
Quantized tensor trains
=======================

Represent a vector of :math:`2^L` values, or a function on a grid of
:math:`2^L` points, as a tensor train of order :math:`L` with modes of
size 2 (QTT), and build matrices in the same format.

The index :math:`i = \\sum_{k=1}^{L} i_k 2^{L-k}` of a grid point is
written with its binary digits :math:`(i_1, \\dots, i_L)`, and the vector
:math:`v(i)` becomes the tensor :math:`v(i_1, \\dots, i_L)`. Many smooth
functions and structured matrices have QTT ranks independent of
:math:`L`: the storage grows like :math:`L`, the logarithm of the number
of points.
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
# the grid of :math:`2^L` points of :math:`[0, 1)` and the order of the
# digits. An exponential has QTT rank 1, a sine rank 2; a function with a
# localized feature needs larger ranks, which grow slowly with
# :math:`L`.
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
# Back to the values
# ^^^^^^^^^^^^^^^^^^
# :func:`~tensap.dequantize` gives the array of the values on the grid
# (only for a small :math:`L`: it has :math:`2^L` entries).
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
# A matrix acting on such vectors is a :class:`~tensap.TTOperator` with
# cores of size :math:`2 \times 2`. The QTT primitives build the usual
# 1D operators directly in this format: the shift
# (:func:`~tensap.qtt_shift`), banded Toeplitz matrices
# (:func:`~tensap.qtt_banded`), corner blocks. The second difference
# matrix has QTT rank 3 whatever the number of points.
for L in (10, 20, 30):
    D2 = tensap.qtt_banded({-1: 1.0, 0: -2.0, 1: 1.0}, L)
    print(
        f"L = {L}: {2**L} x {2**L} matrix, QTT ranks at most {max(D2.ranks)}"
    )

# On a small grid, the operator applied to a QTT vector agrees with the
# dense matrix.
L = 8
D2 = tensap.qtt_banded({-1: 1.0, 0: -2.0, 1: 1.0}, L)
dense = tensap.dequantize_operator(D2)
error = np.abs(
    tensap.dequantize(D2 @ v) - dense @ tensap.dequantize(v)
).max()
print(f"D2 @ v against the dense matrix: max difference {error:.1e}")
