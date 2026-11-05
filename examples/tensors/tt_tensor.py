"""
Tensor trains
=============

Build tensors and operators in tensor-train (TT) format, round them at a
given precision, and compute entrywise products at a given precision.

A tensor train of order :math:`d` is a chain of cores
:math:`x(i_1, \\dots, i_d) = G_1(i_1) G_2(i_2) \\cdots G_d(i_d)`, where
:math:`G_k(i_k)` is a matrix of size :math:`r_{k-1} \\times r_k`
(:math:`r_0 = r_d = 1`); the :math:`r_k` are the TT ranks.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import scipy.sparse as sp
import tensap

np.random.seed(0)

# %%
# From a full tensor
# ^^^^^^^^^^^^^^^^^^
# The TT-SVD of the values of a smooth function on a grid of
# :math:`10^5` points: the TT ranks are small, the storage is a small
# fraction of the :math:`10^5` values.
x = np.linspace(0, 1, 10)
grid = np.meshgrid(*[x] * 5, indexing="ij")
values = 1 / (1 + sum(grid))
u = tensap.TTTensor.from_full(values, tolerance=1e-8)
error = np.linalg.norm(u.numpy() - values) / np.linalg.norm(values)
print(f"TT ranks {u.ranks}, storage {u.storage()}, error {error:.1e}")

# %%
# Rounding
# ^^^^^^^^
# Sums and products increase the ranks; :class:`~tensap.Truncator`
# rounds a TT tensor at a relative precision (orthogonalization, then a
# sweep of truncated SVDs).
v = u + u + 0.5 * u
truncator = tensap.Truncator(tolerance=1e-8)
w = truncator.truncate(v)
error = np.linalg.norm(w.numpy() - 2.5 * values) / np.linalg.norm(
    2.5 * values
)
print(
    f"ranks of u + u + u / 2: {v.ranks} -> {w.ranks}, error {error:.1e}"
)

# %%
# Entrywise product at a given precision
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The exact entrywise product of two TT tensors has the products of their
# ranks. :meth:`~tensap.Truncator.hadamard_product` computes it directly at the
# precision of the truncator, by a fit of the product (two-site DMRG),
# without forming those ranks.
exact = u.hadamard_product(u)
product = truncator.hadamard_product(u, u)
error = np.linalg.norm(product.numpy() - values**2) / np.linalg.norm(
    values**2
)
print(
    f"exact product: ranks {exact.ranks}; at precision 1e-8: ranks "
    f"{product.ranks}, error {error:.1e}"
)

# %%
# Operators
# ^^^^^^^^^
# A :class:`~tensap.TTOperator` has cores with two physical indices. The
# discrete Laplacian
# :math:`\sum_\nu I \otimes \dots \otimes K \otimes \dots \otimes I` is a
# TT operator of ranks 2; applying it multiplies the ranks.
n = 10
K = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
factory = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), 5)
A = factory.make_tt_operator()
Au = A @ u
print(f"A: TT ranks {A.ranks}; A @ u: TT ranks {Au.ranks}")
print(f"rounded: TT ranks {truncator.truncate(Au).ranks}")
