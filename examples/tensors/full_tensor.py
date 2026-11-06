"""
Full tensors
============

Create full tensors, perform operations on them (products,
orthogonalization, singular values) and convert them to sparse tensors.

A tensor of order :math:`d` is an element :math:`u \\in \\mathbb{R}^{I_1
\\times \\dots \\times I_d}`, :math:`I_\\nu = \\{1, \\dots, N_\\nu\\}`, given
by the array of its entries :math:`u(i_1, \\dots, i_d)`. A
:class:`~tensap.FullTensor` stores all the :math:`\\prod_\\nu N_\\nu`
entries; we compute products of such tensors, and look at the
matricizations :math:`M_\\alpha(u)` and their singular values, from which
all the low-rank formats are built. See :ref:`theory-full-tensors`.

This example comes from the script
``test/non_regression_scripts/tensor_algebra/tutorial_FullTensor.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import tensap

np.random.seed(0)

# %%
# Creation of random FullTensor with prescribed shapes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We start with two random tensors, :math:`u` of order 4 and size
# :math:`2 \times 3 \times 4 \times 5`, and :math:`v` of order 6. The first
# one has i.i.d. entries drawn according to the uniform distribution on
# (0, 1).

TENSOR_1 = tensap.FullTensor.rand([2, 3, 4, 5])
print("TENSOR_1 = tensap.FullTensor.rand([2, 3, 4, 5]) =")
print(TENSOR_1)

# The second one, of size 2 x 3 x 4 x 4 x 3 x 2, has i.i.d. entries drawn
# according to the standard gaussian distribution.
TENSOR_2 = tensap.FullTensor.randn([2, 3, 4, 4, 3, 2])
print("\nTENSOR_2 = tensap.FullTensor.rand([2, 3, 4, 4, 3, 2]) =")
print(TENSOR_2)

# The canonical (Frobenius) norm, the number of stored entries
# (2 x 3 x 4 x 5 = 120) and the number of non-zero entries.
print("\nNorm of TENSOR_1 = %f" % TENSOR_1.norm())
print("Number of entries of TENSOR_1 = %i" % TENSOR_1.storage())
print(
    "Number of non-zero entries of TENSOR_1 = %i"
    % TENSOR_1.sparse_storage()
)

# %%
# Operations on FullTensor
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The contracted product sums over pairs of dimensions of the same size.
# Here the dimensions 1 and 3 of :math:`u` are paired with the dimensions 6
# and 4 of :math:`v` (``[0, 2]`` and ``[5, 3]`` in the code, which numbers
# the dimensions from 0):
#
# .. math::
#
#    w(i_2, i_4, j_1, j_2, j_3, j_5) = \sum_{k = 1}^{2} \sum_{l = 1}^{4}
#    u(k, i_2, l, i_4)\, v(j_1, j_2, j_3, l, j_5, k).
#
# The result ``TENSOR_3`` is of order 4 + 6 - 4 = 6, and of size
# 3 x 5 x 2 x 3 x 4 x 3.

TENSOR_3 = TENSOR_1.tensordot(TENSOR_2, [0, 2], [5, 3])
print("\nTENSOR_3 = TENSOR_1.tensordot(TENSOR_2, [0, 2], [5, 3]) =")
print(TENSOR_3)

# Often only a diagonal of a product is needed, and it is much cheaper to
# compute it directly. With the same contraction, we keep only the entries
# for which the index of dimension 2 of u equals the index of dimension 5
# of v:
#     w(k, i_4, j_1, j_2, j_3)
#         = sum_{a, b} u(a, k, b, i_4) v(j_1, j_2, j_3, b, k, a).
# The resulting tensor TENSOR_4 is of order 4 + 6 - 4 - 1 = 5, and of size
# 3 x 5 x 2 x 3 x 4.
TENSOR_4 = TENSOR_1.tensordot_eval_diag(TENSOR_2, [0, 2], [5, 3], 1, 4)
print(
    "\nTENSOR_4 = TENSOR_1.tensordot_eval_diag(TENSOR_2, [0, 2], "
    + "[5, 3], 1, 4) ="
)
print(TENSOR_4)

# The outer product of u and v, of order 10, with evaluation of the diagonal
# in the dimensions 3 of u and 4 of v. The resulting tensor TENSOR_5, of
# order 9, is such that
#     w(i_1, i_2, k, i_4, j_1, j_2, j_3, j_5, j_6)
#         = u(i_1, i_2, k, i_4) v(j_1, j_2, j_3, k, j_5, j_6).
TENSOR_5 = TENSOR_1.outer_product_eval_diag(TENSOR_2, 2, 3)
print("\nTENSOR_5 = TENSOR_1.outer_product_eval_diag(TENSOR_2, 2, 3) =")
print(TENSOR_5)

# The same with several diagonals, in the dimensions 2 and 3 of u and 5 and
# 4 of v. The resulting tensor TENSOR_6 is such that
#     w(i_1, k, l, i_4, j_1, j_2, j_3, j_6)
#         = u(i_1, k, l, i_4) v(j_1, j_2, j_3, l, k, j_6).
TENSOR_6 = TENSOR_1.outer_product_eval_diag(
    TENSOR_2, [1, 2], [4, 3], True
)
print(
    "\nTENSOR_6 = TENSOR_1.outer_product_eval_diag(TENSOR_2, [1, 2], "
    + "[4, 3], True) ="
)
print(TENSOR_6)

# %%
# Orthogonalization of a FullTensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For a set of dimensions :math:`\alpha`, the :math:`\alpha`-matricization
# :math:`M_\alpha(u)` is the matrix with the indices :math:`i_\alpha` as
# rows and the other indices as columns. ``orth(alpha)`` returns a tensor
# of the same shape as ``TENSOR_4`` whose matricization for
# :math:`\alpha = \{3\}` (``2`` in the code) has orthonormal rows,
# :math:`M_\alpha M_\alpha^{\mathsf T} = I`: its slices for the two values
# of :math:`i_3` are orthonormal.
TENSOR_7 = TENSOR_4.orth(2)[0]
print("\nTENSOR_7 = TENSOR_4.orth(2) =")
print(TENSOR_7)

# We check the orthogonality by computing the product of the
# {3}-matricization of TENSOR_7 with its transpose: the printed matrix is the
# 2 x 2 identity, up to rounding errors.
MAT_TENSOR_7 = TENSOR_7.matricize(2)
P = MAT_TENSOR_7.tensordot(MAT_TENSOR_7, 1, 1)
print("\nChecking the orthogonality condition on TENSOR_7:")
print(P.data)

# %%
# Singular values of a FullTensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :math:`\nu`-singular values :math:`\sigma^{\{\nu\}}_k` of a tensor are
# the singular values of its matricization :math:`M_{\{\nu\}}(u)`. The
# number of non-zero ones is the :math:`\nu`-rank, and their decay tells
# how well :math:`u` can be compressed in the dimension :math:`\nu`
# (:ref:`theory-truncation`). ``singular_values`` returns them for each of
# the 6 dimensions of ``TENSOR_3``, plotted below (titles with dimensions
# numbered from 0).
SIN_VAL = TENSOR_3.singular_values()

ORDER = TENSOR_3.order
SIZE_2 = int(np.floor(np.sqrt(ORDER)))
SIZE_1 = int(np.ceil(ORDER / SIZE_2))
for nb_plot in np.arange(1, ORDER + 1):
    plt.subplot(SIZE_1, SIZE_2, nb_plot)
    plt.bar(range(SIN_VAL[nb_plot - 1].size), SIN_VAL[nb_plot - 1])
    plt.title("%i-singular values" % (nb_plot - 1))
plt.show()

# %%
# Conversion of a FullTensor into a SparseTensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.SparseTensor` stores only the non-zero entries and their
# indices. TENSOR_8 is a tensor of order 3 and shape (3, 4, 2) containing
# only 0's and 1's, drawn with probability 2/3 for 0 and 1/3 for 1. It is
# converted into a SparseTensor with the method ``sparse()``, whose storage
# is the number of non-zero entries.
TENSOR_8 = tensap.FullTensor(
    np.random.choice([0, 1], size=24, p=[2 / 3, 1 / 3]).reshape(
        (3, 4, 2)
    )
)
print(f"\nTENSOR_8 has a storage complexity of {TENSOR_8.storage()}")
print(
    f"It has a sparse storage complexity of {TENSOR_8.sparse_storage()}"
)
print(
    "Its sparse conversion has a storage complexity of "
    f"{TENSOR_8.sparse().storage()}"
)

# %%
# Evaluation at indices
# ^^^^^^^^^^^^^^^^^^^^^
# ``eval_at_indices`` returns the entries at a list of multi-indices, given
# in the code from 0: ``[1, 2, 3]`` is the entry :math:`u(2, 3, 4)`. Given
# a subset of dimensions, it fixes only these indices: ``[1, 2]`` in the
# dimensions ``[0, 1]`` gives the 4 entries :math:`u(2, 3, i_3)`, returned
# as a tensor of shape (1, 4).
x = tensap.FullTensor.rand([4, 4, 4])
print(
    "[x(1,2,3) x(2,3,2)] = ", x.eval_at_indices([[1, 2, 3], [2, 3, 2]])
)
print("x(1,2,:) = ", x.eval_at_indices([1, 2], [0, 1]))

# %%
# Sum along given dimensions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``reduce_sum`` sums the entries over the indices of some dimensions, for
# instance over :math:`i_2` and :math:`i_4`:
# :math:`w(i_1, 1, i_3, 1) = \sum_{i_2, i_4} u(i_1, i_2, i_3, i_4)`. The
# summed dimensions are kept with size 1, and ``squeeze`` removes them.
x = tensap.FullTensor.rand([2, 3, 4, 5])
print("Initial tensor x")
print(x)
print("Sum of x over all dimensions")
print(x.reduce_sum())
print("Sum of x along dimension 0")
x1 = x.reduce_sum(0)
print(x1)
print("Sum of x along dimensions 1 and 3")
x2 = x.reduce_sum((1, 3))
print(x2)
print("Squeeze the resulting tensor (removing modes with size 1)")
print(x2.squeeze())
