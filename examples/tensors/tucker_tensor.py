"""
Tucker tensors with structured cores
====================================

Build tensors and operators in Tucker format, with a full, a diagonal
(canonical format) or a tree-based core, compute with them, and truncate
them.

A Tucker tensor of order :math:`d` is a core :math:`C` contracted with one
frame per dimension: :math:`u = \\sum_{k} C(k_1, \\dots, k_d)\\,
v^{(1)}_{k_1} \\otimes \\dots \\otimes v^{(d)}_{k_d}`, where the frame
:math:`(v^{(\\nu)}_k)_{1 \\le k \\le R_\\nu}` of dimension :math:`\\nu` is
a set of vectors (:class:`~tensap.FrameVector`) or of matrices
(:class:`~tensap.FrameOperator`). Its storage
:math:`\\sum_\\nu N_\\nu R_\\nu + \\prod_\\nu R_\\nu` is small when the
representation rank :math:`(R_1, \\dots, R_d)` is. We compute with such
tensors and with the discrete Laplacian in this format, and recompress them.
See :ref:`theory-tucker-canonical`.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import scipy.sparse as sp
import tensap

np.random.seed(0)

# %%
# A Tucker tensor with a full core
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A tensor ``x`` of order 3 in :math:`\mathbb{R}^{10 \times 10 \times 10}`,
# with random frames of 4, 3 and 2 vectors in the three dimensions and a
# full core of shape (4, 3, 2): a :class:`~tensap.TuckerTensorVector` of
# representation rank (4, 3, 2), stored with
# :math:`10 (4 + 3 + 2) + 4 \cdot 3 \cdot 2` numbers.
x = tensap.TuckerTensorVector.randn([4, 3, 2], [10, 10, 10])
print(
    f"core: {type(x.core).__name__}, representation rank "
    f"{x.representation_rank()}, storage {x.storage()} "
    f"(the full tensor has {10**3} entries)"
)

# Vector space operations and inner products work without forming the full
# tensor. In a sum the frames are concatenated, so that the representation
# ranks add up, here :math:`(4, 3, 2) + (2, 2, 2)`. An inner product only
# needs the small Gram matrices of the frames,
#
# .. math::
#
#    (u, u') = \sum_{k, k'} C(k)\, C'(k') \prod_{\nu=1}^{d}
#    \big(v^{(\nu)}_{k_\nu}, v'^{(\nu)}_{k'_\nu}\big),
#
# and agrees with the inner product of the full arrays.
y = tensap.TuckerTensorVector.randn([2, 2, 2], [10, 10, 10])
z = x + 2 * y
print(f"rank of x + 2 y: {z.representation_rank()}")
error = abs(z.dot(x) - np.sum(z.numpy() * x.numpy())) / abs(z.dot(x))
print(f"(z, x) against the full arrays: relative error {error:.1e}")

# %%
# Canonical format: a diagonal core
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A canonical tensor :math:`\sum_{k=1}^{R} C(k)\, v^{(1)}_k \otimes \dots
# \otimes v^{(d)}_k` is a Tucker tensor with a diagonal core
# (:class:`~tensap.DiagonalTensor`): only the terms :math:`k_1 = \dots =
# k_d` remain. The ``ones`` constructor with ranks 1 gives the tensor of
# ones, :math:`1 \otimes 1 \otimes 1`, with a diagonal core.
ones = tensap.TuckerTensorVector.ones([1, 1, 1], [10, 10, 10])
print(
    f"core of ones: {type(ones.core).__name__}, "
    f"max entry error {np.abs(ones.numpy() - 1).max():.1e}"
)

# %%
# Operators: the discrete Laplacian
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Operators are Tucker tensors too, with frames of matrices.
# :class:`~tensap.LaplacianLikeOperatorFactory` builds the finite difference
# Laplacian on :math:`(0, 1)^d`,
#
# .. math::
#
#    A = \sum_{\nu=1}^{d} I \otimes \dots \otimes K \otimes \dots
#    \otimes I, \qquad K = (n + 1)^2\, \mathrm{tridiag}(-1, 2, -1),
#
# from the 1D matrices :math:`K` and :math:`I`. There are three choices.
# In canonical format, each of the :math:`d` terms has its own matrices,
# with a diagonal core. With the frames :math:`\{K, I\}` in every
# dimension, the core selects the terms, which contain :math:`K` exactly
# once: it is a full tensor of shape :math:`2 \times \dots \times 2`, or
# the same core stored in tree-based format.
n, d = 10, 3
K = (n + 1) ** 2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
factory = tensap.LaplacianLikeOperatorFactory(K, sp.identity(n), d)
for A in (
    factory.make_canonical_tensor(),
    factory.make_tucker_tensor(),
    factory.make_tree_based_tensor(),
):
    print(
        f"{type(A.core).__name__:16s} core, representation rank "
        f"{A.representation_rank()}"
    )

# Applying the operator to a Tucker tensor gives a Tucker tensor whose frames
# are the products :math:`A^{(\nu)}_j v^{(\nu)}_k` of the frames, so that
# the representation ranks are multiplied, here by 2.
Ax = A @ x
print(
    f"A @ x: {type(Ax).__name__}, representation rank "
    f"{Ax.representation_rank()}"
)

# %%
# Truncation
# ^^^^^^^^^^
# :class:`~tensap.Truncator` truncates a Tucker tensor at a relative
# precision :math:`\varepsilon`, :math:`\|u - u_r\| \le \varepsilon \|u\|`:
# by an HOSVD of the core for a full core, by the hierarchical SVD of the
# tree-based tensor for a tree-based core. Here :math:`y + y + y` is the
# tensor :math:`3y`, of Tucker rank at most (2, 2, 2), but stored with
# redundant terms and representation rank (6, 6, 6); the truncation finds
# back the rank (2, 2, 2), with an error at the level of rounding errors.
redundant = y + y + y
truncator = tensap.Truncator(tolerance=1e-10)
compressed = truncator.truncate(redundant)
reference = 3 * y.numpy()
error = np.linalg.norm(compressed.numpy() - reference) / np.linalg.norm(
    reference
)
print(
    f"rank {redundant.representation_rank()} -> "
    f"{compressed.representation_rank()}, relative error {error:.1e}"
)
