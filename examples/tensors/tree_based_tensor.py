"""
Tree-based tensors
==================

Generate tensors in tree-based format, truncate them with a prescribed rank
or precision, use the tensor-train format and perform algebraic operations.

For a dimension tree :math:`T` over :math:`D = \\{1, \\dots, d\\}` and ranks
:math:`(r_\\alpha)_{\\alpha \\in T}`, the tree-based format is the set of
tensors :math:`\\{u : \\operatorname{rank}_\\alpha(u) \\le r_\\alpha,\\;
\\alpha \\in T\\}`, represented by one small tensor :math:`C^\\alpha` per node
(a tree tensor network). Its storage is linear in :math:`d` for bounded
ranks; the Tucker, hierarchical Tucker and tensor-train formats are
particular trees. See :ref:`theory-tree-based-tensors`.

This example comes from the script
``test/non_regression_scripts/tensor_algebra/tutorial_TreeBasedTensor.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Linear dimension tree
# ^^^^^^^^^^^^^^^^^^^^^
# The trees first (see also the example on dimension trees): a linear tree,
# with interior nodes :math:`\{1, \dots, \nu\}`, ...
ORDER = 5
TREE = tensap.DimensionTree.linear(ORDER)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# Random dimension tree
# ^^^^^^^^^^^^^^^^^^^^^
# ... and a random tree over 10 dimensions, whose interior nodes have 2 or 3
# children.
ORDER = 10
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# TreeBasedTensor: random generation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :meth:`~tensap.TreeBasedTensor.rand` draws random tensors :math:`C^\alpha`
# for the nodes of the tree, with random ranks (between 1 and 5) and sizes.
# A node is *active* if it carries a tensor; an inactive leaf has none, and
# the index of its parent is directly the physical index :math:`i_\nu`. By
# default all the nodes are active. The plots show the active nodes, the
# storage of each tensor :math:`C^\alpha`, and the representation ranks
# :math:`r_\alpha`.
TENSOR = tensap.TreeBasedTensor.rand(TREE)
TENSOR.plot(title="Active nodes")
TENSOR.plot(
    [x.storage() for x in TENSOR.tensors],
    title="Tensors' storage complexity",
)
TENSOR.plot(TENSOR.representation_rank, title="Representation ranks")

# %%
# Truncation of a TreeBasedTensor with prescribed rank
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The hierarchical SVD ``hsvd`` computes the :math:`\alpha`-singular values
# of :math:`u` node by node, on the small tensors of the representation, and
# truncates each :math:`\alpha`-rank. With ``tolerance=0`` and ``max_rank``
# set, all the ranks are bounded by the same prescribed value. With this
# seed, the drawn tensor already has all its ranks below the prescribed one:
# the truncation is exact, and the printed relative error
# :math:`\|u - u_r\| / \|u\|` is at the level of rounding errors.
TRUNCATOR = tensap.Truncator(
    tolerance=0, max_rank=np.random.randint(1, 5)
)
TENSOR = tensap.TreeBasedTensor.rand(TREE)
TENSOR_TRUNCATED = TRUNCATOR.hsvd(TENSOR)
print(
    "Prescribed rank, error = %2.5e\n"
    % ((TENSOR - TENSOR_TRUNCATED).norm() / TENSOR.norm())
)

# %%
# Truncation of a TreeBasedTensor with prescribed relative precision
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With a tolerance :math:`\varepsilon`, the ranks are chosen so that
# :math:`\|u - u_r\| \le \varepsilon \|u\|`. The truncation can go from the
# root to the leaves or from the leaves to the root (the private option
# ``_hsvd_type``); in both cases the printed error is below the tolerance.
ORDER = 10
TREE = tensap.DimensionTree.random(ORDER, 3)
TENSOR = tensap.TreeBasedTensor.rand(TREE)

TRUNCATOR = tensap.Truncator(tolerance=1e-8)
TRUNCATOR._hsvd_type = 1  # Root to leaves truncation
TENSOR_TRUNCATED_1 = TRUNCATOR.hsvd(TENSOR)
ERR_1 = (TENSOR - TENSOR_TRUNCATED_1).norm() / TENSOR.norm()
print(
    "Root to leaves: prescribed tolerance = %2.5e, error = %2.5e\n"
    % (TRUNCATOR.tolerance, ERR_1)
)

TRUNCATOR._hsvd_type = 2  # Leaves to root truncation
TENSOR_TRUNCATED_2 = TRUNCATOR.hsvd(TENSOR)
ERR_2 = (TENSOR - TENSOR_TRUNCATED_2).norm() / TENSOR.norm()
print(
    "Leaves to root: prescribed tolerance = %2.5e, error = %2.5e\n"
    % (TRUNCATOR.tolerance, ERR_2)
)

TENSOR_TRUNCATED_2.plot(
    TENSOR_TRUNCATED_2.representation_rank, title="Representation ranks"
)

# %%
# Truncation of a FullTensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``hsvd`` also takes a full tensor, with a tree and a choice of active nodes,
# and returns its truncation in the tree-based format. A tensor with i.i.d.
# gaussian entries has no low-rank structure: at the coarse precision
# :math:`5 \cdot 10^{-1}` the ranks are reduced, but at :math:`10^{-2}` they
# are those of an exact representation, and the error is at the level of
# rounding errors.
ORDER = 5
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)

SIZE = np.random.randint(8, 11, ORDER)
ACTIVE_NODES = tensap.TreeBasedTensor._random_is_active_node(TREE)

A = tensap.FullTensor.randn(SIZE)
TRUNCATOR = tensap.Truncator(tolerance=5e-1)
A_TRUNCATED = TRUNCATOR.hsvd(A, TREE, ACTIVE_NODES)
ERR = (A - A_TRUNCATED.full()).norm() / A.norm()
print("Tolerance = %2.5e, error = %2.5e\n" % (TRUNCATOR.tolerance, ERR))
A_TRUNCATED.plot(
    A_TRUNCATED.representation_rank, title="Representation ranks"
)

TRUNCATOR.tolerance = 1e-2
A_TRUNCATED = TRUNCATOR.hsvd(A, TREE, ACTIVE_NODES)
ERR = (A - A_TRUNCATED.full()).norm() / A.norm()
print("Tolerance = %2.5e, error = %2.5e\n" % (TRUNCATOR.tolerance, ERR))
A_TRUNCATED.plot(
    A_TRUNCATED.representation_rank, title="Representation ranks"
)

# %%
# Tensor train format
# ^^^^^^^^^^^^^^^^^^^
# The tensor-train format is the tree-based format of a linear tree in which
# only the interior nodes and the first leaf are active, so that
#
# .. math::
#
#    u(i_1, \dots, i_d) = C^1(i_1)\, C^2(i_2) \cdots C^d(i_d),
#
# with :math:`C^\nu(i_\nu)` a matrix of size :math:`r_{\nu-1} \times
# r_\nu`. Here with modes of size 3 and random TT ranks 1 or 2.
ORDER = 10
TREE = tensap.DimensionTree.linear(ORDER)
RANKS = np.hstack((1, np.random.randint(1, 3, TREE.nb_nodes - 1)))

SIZE = np.full(ORDER, 3)
ACTIVE_NODES = np.logical_not(TREE.is_leaf)
ACTIVE_NODES[TREE.dim2ind[0] - 1] = True
TENSOR = tensap.TreeBasedTensor.rand(TREE, RANKS, SIZE, ACTIVE_NODES)

TENSOR.plot(title="Nodes indices")
TENSOR.tree.plot_dims(title="Nodes dimensions")

# The same on a linear tree built on a permutation of the dimensions: the
# chain takes the dimensions in the order of the permutation.
RPERM = np.random.permutation(range(ORDER))
TREE = tensap.DimensionTree.linear(RPERM)
TENSOR = tensap.TreeBasedTensor.rand(TREE, RANKS, SIZE, ACTIVE_NODES)

TENSOR.plot(title="Nodes indices")
TENSOR.tree.plot_dims(title="Nodes dimensions")

# %%
# Algebraic operations
# ^^^^^^^^^^^^^^^^^^^^
# Two tree-based tensors on the same tree can be added and multiplied
# without forming the full tensors. The representation ranks add up in a
# sum, :math:`r_\alpha(u + v) = r_\alpha(u) + r_\alpha(v)`, and multiply in
# the Hadamard (entrywise) product, :math:`r_\alpha(u \odot v) =
# r_\alpha(u)\, r_\alpha(v)`, as the printed ranks show: a truncation is
# needed afterwards to keep them small. ``reduce_sum`` sums over the indices
# of some dimensions (from 0 in the code) and removes them.
ORDER = 8
TREE = tensap.DimensionTree.linear(ORDER)
SZ = np.random.randint(3, 10, ORDER)
T1 = tensap.TreeBasedTensor.rand(TREE, shape=SZ)
T2 = tensap.TreeBasedTensor.rand(TREE, shape=SZ)
print("ranks of T1   :", T1.ranks)
print("ranks of T2   :", T2.ranks)
print("\nAddition of tensors:\n--------------------")
Tplus = T1 + T2
print("ranks of T1+T2:", Tplus.ranks)
print("\nSubstraction of tensors:\n----------------------")
Tminus = T1 - T2
print("ranks of T1-T2:", Tminus.ranks)
print("\nHadamard product of tensors:\n-------------------------")
Ttimes = T1 * T2
print("ranks of T1*T2:", Ttimes.ranks)
print("\nNorm of tensors:\n--------------------")
print("norm of T1", T1.norm())
print("norm of T1*T2", Ttimes.norm())

print("\nSum of T1 along dimensions:\n-------------------------")
print("dimensions of T1        : ", T1.shape)
print("dimensions of T1.reduce_sum(0) : ", T1.reduce_sum(0).shape)
print(
    "dimensions of T1.reduce_sum([2,5,7]) : ",
    T1.reduce_sum([2, 5, 7]).shape,
)
print("sum of all entries = ", T1.reduce_sum())


# %%
# Changing root node of a tree based tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``change_root`` represents the tensor on the tree with another root (see the
# example on dimension trees). Going to the root 9 directly, or through the
# root 2, gives the same tensor: the printed relative difference is at the
# level of rounding errors.

ORDER = 5
TREE = tensap.DimensionTree.balanced(ORDER)
RANKS = np.hstack((1, np.random.randint(2, 3, TREE.nb_nodes - 1)))
SZ = np.random.randint(2, 4, ORDER)
T1 = tensap.TreeBasedTensor.rand(TREE, ranks=RANKS, shape=SZ)
T1.plot(title="Nodes indices with root 1")
num2 = 2
T2 = T1.change_root(num2)
T2.plot(title="Nodes indices with root " + str(num2))
num3 = 9
T3 = T1.change_root(num3)
T3.plot(title="Nodes indices with root" + str(num3))
T3bis = T2.change_root(num3)
T3bis.plot(
    title="Nodes indices with root "
    + str(num3)
    + " from format with root "
    + str(num2)
)
print((T3 - T3bis).norm() / (T3.norm() + T3bis.norm()) * 2)


# %%
# Activate and inactivate nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Inactivating a leaf merges its tensor into the tensor of its parent, whose
# index becomes the physical index; activating it again gives the leaf the
# identity matrix as tensor. Neither changes the tensor :math:`u`, only its
# representation, as the printed differences check.
ORDER = 5
TREE = tensap.DimensionTree.balanced(ORDER)
T1 = tensap.TreeBasedTensor.rand(TREE)
T1.plot(title="Initial tensor format")
T2 = T1.inactivate_nodes([9, 5])
print(
    "Check T1-T2 = 0 : ", (T2 - T1).norm() / (T1.norm() + T2.norm()) * 2
)
T2.plot(title="Inactivate nodes [9,5]")
T3 = T2.activate_nodes([9, 5])
T3.plot(title="Reactivate nodes [9,5]")
print(
    "Check T1-T3 = 0 : ", (T3 - T1).norm() / (T1.norm() + T3.norm()) * 2
)


# %%
# Extraction of a subtree and corresponding tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The tensors :math:`C^\beta` of the nodes :math:`\beta` below a node
# :math:`\alpha` represent :math:`r_\alpha` tensors over the dimensions in
# :math:`\alpha`, which span a subspace of :math:`\mathbb{R}^{I_\alpha}`
# containing the minimal subspace :math:`U^{\min}_\alpha(u)`. With the
# subtree of :math:`\alpha`, they give a tree-based tensor of rank
# :math:`r_\alpha` at its root, that is a basis of this subspace.
ORDER = 5
TREE = tensap.DimensionTree.balanced(ORDER)
T = tensap.TreeBasedTensor.rand(TREE)
subTREE, nodes = TREE.sub_dimension_tree(2)
subT = tensap.TreeBasedTensor(T.tensors[nodes - 1], subTREE)

# %%
# Evaluation at indices
# ^^^^^^^^^^^^^^^^^^^^^
# An entry :math:`u(i_1, \dots, i_d)` is computed by contracting the network
# with the physical indices fixed. Fixing only some of them (indices and
# dimensions from 0 in the code) gives a tree-based tensor of lower order;
# for several partial multi-indices, a first dimension numbers them.
ORDER = 5
SIZE = [10] * ORDER
TREE = tensap.DimensionTree.balanced(ORDER)
T = tensap.TreeBasedTensor.rand(TREE, shape=SIZE)

print("evaluation at 3 entries drawn at random")
Ind = tensap.random_multi_indices(SIZE)
print(T.eval_at_indices(Ind.random(3)))
print("partial evaluation - squeezing the tensor")
print(T.eval_at_indices([1, 2], [0, 1]))
print(T.eval_at_indices([1, 2], [0, 4]))
print("partial evaluation at multiple indices")
print(T.eval_at_indices([[1, 2], [2, 3]], [1, 2]))
