"""
Tensor trains
=============

Build tensors and operators in tensor-train (TT) format, round them at a
given precision, and compute entrywise products at a given precision.

A tensor train of order :math:`d` is a chain of cores
:math:`u(i_1, \\dots, i_d) = C^1(i_1)\\, C^2(i_2) \\cdots C^d(i_d)`, where
:math:`C^\\nu(i_\\nu)` is a matrix of size :math:`r_{\\nu-1} \\times r_\\nu`
(:math:`r_0 = r_d = 1`); the :math:`r_\\nu` are the TT ranks, and the
storage :math:`\\sum_\\nu r_{\\nu-1} N_\\nu r_\\nu` is linear in :math:`d`.
We follow a tensor of order 5 through compression, sums, products and the
application of an operator, and watch its ranks. See
:ref:`theory-tensor-trains`.
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
# The tensor is :math:`u(i_1, \dots, i_5) = f(x_{i_1}, \dots, x_{i_5})`, the
# values of the smooth function :math:`f(x) = 1 / (1 + x_1 + \dots + x_5)`
# on a grid of 10 points per dimension, :math:`10^5` values in all.
# :meth:`~tensap.TTTensor.from_full` computes its TT-SVD: truncated SVDs of
# the matricizations :math:`M_{\{1, \dots, \nu\}}(u)`, one after the
# other, at the relative precision :math:`10^{-8}`. The TT ranks are small,
# and the storage is a small fraction of the :math:`10^5` values.
x = np.linspace(0, 1, 10)
grid = np.meshgrid(*[x] * 5, indexing="ij")
values = 1 / (1 + sum(grid))
u = tensap.TTTensor.from_full(values, tolerance=1e-8)
error = np.linalg.norm(u.numpy() - values) / np.linalg.norm(values)
print(f"TT ranks {u.ranks}, storage {u.storage()}, error {error:.1e}")

# %%
# Rounding
# ^^^^^^^^
# The TT ranks of a sum are the sums of the ranks: the cores of
# :math:`u + v` are block-diagonal combinations of those of :math:`u` and
# :math:`v`. Here :math:`u + u + u/2 = 2.5\, u` has three times the ranks of
# :math:`u`, although it has the same matricizations up to a factor.
# :class:`~tensap.Truncator` rounds a TT tensor at a relative precision:
# an orthogonalization by QR factorizations, then a sweep of truncated
# SVDs of the cores. It recovers the ranks of :math:`u`.
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
# The entrywise (Hadamard) product :math:`(u \odot u)(i) = u(i)^2` is a
# TT tensor whose cores are the Kronecker products of the cores, so that its
# exact ranks are the products :math:`r_\nu^2` of the ranks.
# :meth:`~tensap.Truncator.hadamard_product` computes it directly at the
# precision of the truncator, by a fit of the product (two-site DMRG),
# without forming those ranks. The rounded product, the values of
# :math:`f^2`, has ranks much smaller than the exact ones.
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
# A :class:`~tensap.TTOperator` has cores with two physical indices,
# :math:`A^{(\nu)}(i_\nu, j_\nu)` a matrix of size
# :math:`r_{\nu-1} \times r_\nu`. The discrete Laplacian
#
# .. math::
#
#    A = \sum_{\nu=1}^{d} I \otimes \dots \otimes K \otimes \dots
#    \otimes I,
#
# with :math:`K = \mathrm{tridiag}(-1, 2, -1)` in the dimension :math:`\nu`,
# is a sum of :math:`d` elementary operators but a TT operator of ranks 2:
# its cores carry a two-state index, "the :math:`K` has been placed or not".
# Applying it to :math:`u` multiplies the ranks of :math:`u` by those of
# :math:`A`, and a rounding at the precision of the truncator reduces them.
n = 10
K = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
factory = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), 5)
A = factory.make_tt_operator()
Au = A @ u
print(f"A: TT ranks {A.ranks}; A @ u: TT ranks {Au.ranks}")
print(f"rounded: TT ranks {truncator.truncate(Au).ranks}")
