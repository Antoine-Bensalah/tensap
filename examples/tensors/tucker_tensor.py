"""
Tucker tensors with structured cores
====================================

Build tensors and operators in Tucker format, with a full, a diagonal
(canonical format) or a tree-based core, compute with them, and truncate
them.

A Tucker tensor of order :math:`d` is a core :math:`C` contracted with one
frame per dimension: :math:`x = \\sum_{k} C(k_1, \\dots, k_d)\\,
w^{(1)}_{k_1} \\otimes \\dots \\otimes w^{(d)}_{k_d}`, where the frame
:math:`(w^{(\\nu)}_k)_k` of dimension :math:`\\nu` is a set of vectors
(:class:`~tensap.FrameVector`) or of matrices
(:class:`~tensap.FrameOperator`).
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
# Random frames of 4, 3 and 2 vectors of size 10 in the three dimensions,
# and a full core of shape (4, 3, 2).
x = tensap.TuckerTensorVector.randn([4, 3, 2], [10, 10, 10])
print(
    f"core: {type(x.core).__name__}, representation rank "
    f"{x.representation_rank()}, storage {x.storage()} "
    f"(the full tensor has {10**3} entries)"
)

# Vector space operations and inner products work without forming the full
# tensor; the representation ranks add up in a sum.
y = tensap.TuckerTensorVector.randn([2, 2, 2], [10, 10, 10])
z = x + 2 * y
print(f"rank of x + 2 y: {z.representation_rank()}")
error = abs(z.dot(x) - np.sum(z.numpy() * x.numpy())) / abs(z.dot(x))
print(f"(z, x) against the full arrays: relative error {error:.1e}")

# %%
# Canonical format: a diagonal core
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A canonical tensor :math:`\sum_k \sigma_k w^{(1)}_k \otimes \dots
# \otimes w^{(d)}_k` is a Tucker tensor with a diagonal core: the
# ``ones`` constructor with ranks 1 gives the tensor of ones.
ones = tensap.TuckerTensorVector.ones([1, 1, 1], [10, 10, 10])
print(
    f"core of ones: {type(ones.core).__name__}, "
    f"max entry error {np.abs(ones.numpy() - 1).max():.1e}"
)

# %%
# Operators: the discrete Laplacian
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.LaplacianLikeOperatorFactory` builds
# :math:`A = \sum_{\nu} I \otimes \dots \otimes K \otimes \dots \otimes I`
# from the 1D matrices :math:`K` and :math:`I`, with frames
# :math:`\{K, I\}` in every dimension and a core that selects the terms:
# diagonal (canonical format, :math:`d` terms), full, or tree-based.
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
# are the products of the frames.
Ax = A @ x
print(
    f"A @ x: {type(Ax).__name__}, representation rank "
    f"{Ax.representation_rank()}"
)

# %%
# Truncation
# ^^^^^^^^^^
# :class:`~tensap.Truncator` truncates a Tucker tensor at a relative
# precision: by an HOSVD of the core for a full core, by the hierarchical
# SVD of the tree-based tensor for a tree-based core. Here a tensor of rank
# (2, 2, 2) stored with redundant terms is recompressed.
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
