"""
Truncation of tensors
=====================

Truncate a tensor in Tucker, tree-based and tensor-train formats, and
optimize the dimension tree.

Given a tensor :math:`u` and a relative precision :math:`\\varepsilon`, the
truncation computes a tensor :math:`u_r` of small ranks with
:math:`\\|u - u_r\\| \\le \\varepsilon \\|u\\|`, by truncated SVDs of
matricizations :math:`M_\\alpha(u)`. Here :math:`u` is a full tensor of
order 6, the values of a function on a grid, and we compare the ranks and
storage of its truncations in Tucker, tree-based and tensor-train formats,
and for different orderings of the dimensions. See :ref:`theory-truncation`.

This example comes from the script
``test/non_regression_scripts/tensor_algebra/tutorial_tensor_truncation.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Definitions
# ^^^^^^^^^^^
# The tensor is :math:`u(i_1, \dots, i_6) = f(x_{i_1}, \dots, x_{i_6})`, the
# values of
#
# .. math::
#
#    f(x) = \frac{1}{3 + x_1 + x_3} + x_2 + \cos(x_4 + x_5)
#
# on the grid of :math:`8^6` points with :math:`8` equispaced points
# :math:`x_i` in :math:`[0, 1]` per dimension. The function is a sum of
# terms each involving a few variables, and it does not depend on
# :math:`x_6`: its ranks are small, and depend on how the variables are
# grouped.
ORDER = 6
NK = 8


def fun(inputs):
    """
    Example function used for the tutorial.

    Parameters
    ----------
    inputs : numpy.ndarray
        The inputs points at which the function is to be evaluated.

    Returns
    -------
    numpy.ndarray
        The evaluations of the function at the input points.

    """
    return (
        1 / (3 + inputs[:, 0] + inputs[:, 2])
        + inputs[:, 1]
        + np.cos(inputs[:, 3] + inputs[:, 4])
    )


GRID = tensap.FullTensorGrid(np.linspace(0, 1, NK), ORDER)
ARRAY = GRID.array()

U = fun(ARRAY)
U = tensap.FullTensor(U, ORDER, np.full(ORDER, NK))

# %%
# Higher Order SVD - truncation in Tucker format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The higher-order SVD keeps, in each dimension :math:`\nu`, the first
# :math:`r_\nu` left singular vectors of :math:`M_{\{\nu\}}(u)`, with
# :math:`r_\nu` chosen for the precision :math:`\varepsilon / \sqrt{d}`,
# and projects :math:`u` onto their tensor product. The result is a Tucker
# tensor; the printed dimensions of the spaces are its Tucker ranks
# :math:`r_\nu`. The rank is 1 in the last dimension, since :math:`f` does
# not depend on :math:`x_6`, and 2 in the second one, where :math:`f` is
# affine.
TR = tensap.Truncator()
TR.tolerance = 1e-8
UR = TR.hosvd(U)
print("Tucker format - Compression at tolerance %2.5e" % TR.tolerance)
print("Error = %2.5e" % ((UR.full() - U).norm() / U.norm()))
print("Storage = %d" % UR.storage())
print("Dimension of spaces = %s\n" % UR.tensors[0].shape)

# %%
# Tree-based format
# ^^^^^^^^^^^^^^^^^
# For a dimension tree :math:`T`, the hierarchical SVD ``hsvd`` truncates the
# :math:`\alpha`-matricizations for all the nodes :math:`\alpha \in T`, from
# the leaves to the root, and gives a tensor in the tree-based format with
# ranks :math:`(r_\alpha)_{\alpha \in T}`. The tree is drawn at random. At
# the precision :math:`10^{-15}`, close to machine precision, almost nothing
# is removed and the ranks are large.
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TR = tensap.Truncator()
TR.tolerance = 1e-15
UR = TR.hsvd(U, TREE)

print(
    "Tree-based format - Compression at tolerance %2.5e" % TR.tolerance
)
print("Error = %2.5e" % ((UR.full() - U).norm() / U.norm()))
print("Storage = %d" % UR.storage())
print("Ranks = %s\n" % UR.ranks)

# Rounding this tree-based tensor at the precision :math:`10^{-6}` works on
# its small tensors only, without going back to the full tensor, and gives
# much smaller ranks and storage.
TR.tolerance = 1e-6
URR = TR.hsvd(UR, TREE)
print("Rounding at tolerance %2.5e" % TR.tolerance)
print("Error = %2.5e" % ((URR.full() - U).norm() / U.norm()))
print("Storage = %d" % URR.storage())
print("Ranks = %s\n" % URR.ranks)


# %%
# Truncation in tensor-train format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The TT-SVD truncates the matricizations :math:`M_{\{1, \dots, \nu\}}(u)`
# one after the other. In tensap, the result is a
# :class:`~tensap.TreeBasedTensor` on a linear tree whose leaves, except
# the first one, are inactive: the printed ranks contain a 0 for each
# inactive leaf, and the others are the TT ranks.
TR = tensap.Truncator()
TR.tolerance = 1e-8
UR = TR.ttsvd(U)
print(UR)

SIN_VAL = UR.singular_values()

print(
    "Tensor Train format - Compression at tolerance %2.5e"
    % TR.tolerance
)
print("Error = %2.5e" % ((UR.full() - U).norm() / U.norm()))
print("Storage = %d" % UR.storage())
print("TT-rank = %s\n" % UR.ranks)

# %%
# Truncation in tensor-train format after permutation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The TT ranks depend on the ordering of the dimensions. We permute the
# dimensions of :math:`u` at random: variables that interact in :math:`f`,
# such as :math:`x_1` and :math:`x_3`, can end up far apart in the chain,
# which makes the ranks between them grow. The truncation is here at the
# smaller precision :math:`10^{-10}`, with ranks capped at 200.
PERM = np.random.permutation(ORDER)
U_PERM = U.transpose(PERM)

TR = tensap.Truncator()
TR.tolerance = 1e-10
TR.max_rank = 200

UR = TR.ttsvd(U_PERM)
print("With permutation...")
print("\tOrdering = %s" % PERM)
print("\tError = %2.5e" % ((UR.full() - U_PERM).norm() / U_PERM.norm()))
print("\tStorage = %d" % UR.storage())
print("\tRanks = %s\n" % UR.ranks)

# %%
# Tensor-train tree optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``optimize_leaves_permutations`` looks for a better ordering of the
# dimensions in the same linear tree: it tries random permutations of the
# leaves, keeps those that reduce the storage, and represents the tensor in
# the new tree at the given precision. The tree stays linear; the plots show
# the dimensions at its leaves before and after.
print("Tree optimization (leaves permutation)...")
UR_PERM = UR.optimize_leaves_permutations(1e-10, 10)
print("\tInitial storage = %d" % UR.storage())
print("\tError = %2.5e" % ((UR_PERM.full() - U_PERM).norm() / U.norm()))
print("\tFinal storage = %d" % UR_PERM.storage())
print("\tRanks = %s" % UR_PERM.ranks)

UR.tree.plot_dims(title="Dimensions before tree optimization")
UR_PERM.tree.plot_dims(title="Dimensions before after optimization")

# %%
# Global tree optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^
# ``optimize_dimension_tree`` also permutes interior nodes, so that the
# tree itself can change; here it reaches a smaller storage than the
# optimization of the leaves alone.
print("Tree optimization...")
UR_PERM = UR.optimize_dimension_tree(1e-10, 100)
print("\tInitial storage = %d" % UR.storage())
print("\tError = %2.5e" % ((UR_PERM.full() - U_PERM).norm() / U.norm()))
print("\tFinal storage = %d" % UR_PERM.storage())
print("\tRanks = %s" % UR_PERM.ranks)

UR.tree.plot_dims(title="Dimensions before tree optimization")
UR_PERM.tree.plot_dims(title="Dimensions before after optimization")
