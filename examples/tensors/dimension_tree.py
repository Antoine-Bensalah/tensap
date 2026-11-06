"""
Dimension trees
===============

Create linear, random and balanced dimension trees, extract subtrees, change
the root and plot them.

A dimension tree :math:`T` over :math:`D = \\{1, \\dots, d\\}` is a
collection of subsets :math:`\\alpha \\subset D`, with root :math:`D`, in
which every node with :math:`\\#\\alpha \\ge 2` is partitioned by its
children and the leaves are the singletons :math:`\\{\\nu\\}`. It is the
skeleton of a tree-based tensor, whose ranks :math:`(r_\\alpha)_{\\alpha \\in
T}` bound the :math:`\\alpha`-ranks of the tensor. Here we only build and
look at trees. See :ref:`theory-tree-based-tensors`.

This example comes from the script
``test/non_regression_scripts/tensor_algebra/tutorial_DimensionTree.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import tensap
import numpy as np

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Linear dimension tree
# ^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.DimensionTree` numbers its nodes from 1 (the root) and
# stores for each node :math:`\alpha` its set of dimensions. The linear tree
# is the tree of the tensor-train format: its interior nodes are
# :math:`\{1, \dots, \nu\}`, so that each one splits off a single
# dimension. The first plot shows the node numbers, the second one the
# dimension :math:`\nu` of each leaf :math:`\{\nu\}` (numbered from 0, as in
# the code); the set :math:`\alpha` of a node is that of the leaves below it.
ORDER = 5
TREE = tensap.DimensionTree.linear(ORDER)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# Random dimension tree
# ^^^^^^^^^^^^^^^^^^^^^
# A random tree over 10 dimensions, in which the number of children of each
# interior node is drawn in ``ARITY_INTERVAL``, here 2 or 3.
ORDER = 10
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# Balanced dimension tree
# ^^^^^^^^^^^^^^^^^^^^^^^
# In a balanced binary tree each interior node is split into two halves, so
# that the depth grows like :math:`\log_2 d`: this is the tree of the
# hierarchical Tucker format. The ascendants of a node are the nodes on its
# path to the root, its descendants the nodes of the subtree below it.
ORDER = 10
TREE = tensap.DimensionTree.balanced(ORDER)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions at the leaves")

TREE.plot(
    colored_nodes=TREE.ascendants(4),
    node_color="b",
    title="Ascendants of node 4",
)

TREE.plot(
    colored_nodes=TREE.descendants(4),
    node_color="b",
    title="Descendants of node 4",
)

# %%
# Extraction of a subtree
# ^^^^^^^^^^^^^^^^^^^^^^^
# The descendants of a node :math:`\alpha` form a dimension tree over
# :math:`\alpha`. ``sub_dimension_tree`` returns this tree and the numbers of
# its nodes in the original tree: this is how the part of a tree-based tensor
# below a node is extracted.
[SUB_TREE, NOD] = TREE.sub_dimension_tree(4)
TREE.plot(
    colored_nodes=NOD, node_color="b", title="Sub-tree to extract"
)
SUB_TREE.plot(title="Sub-tree")

# %%
# Plot with and without the same level at the same height
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The level of a node is its distance to the root,
# :math:`\operatorname{level}(D) = 0`. Here it is written at each node, with
# the nodes of the same level drawn or not at the same height.
ORDER = 8
TREE = tensap.DimensionTree.random(ORDER)

TREE.plot_options["level_alignment"] = False
TREE.plot_with_labels_at_nodes(TREE.level, title="No level alignment")
TREE.plot_options["level_alignment"] = True
TREE.plot_with_labels_at_nodes(TREE.level, title="Level alignment")

# %%
# Change the root of a tree
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Forgetting the root, a dimension tree is a graph whose leaves are the
# dimensions. Choosing another node as the root gives another dimension tree
# over the same dimensions, in which the nodes between the old and the new
# root get new sets :math:`\alpha`. When the new root is a leaf, a new
# node is created for its dimension.
ORDER = 8
TREE = tensap.DimensionTree.balanced(ORDER)
TREE.plot(title="Nodes indices before changing the root")
TREE.plot_dims(title="Nodes dimensions before changing the root")
NEWTREE, modified_nodes = TREE.change_root(5)
NEWTREE.plot(title="Nodes indices after changing the root to node 5")
NEWTREE.plot_dims(
    title="Nodes dimensions after changing the root to node 5"
)
NEWTREE, modified_nodes = TREE.change_root(11)
NEWTREE.plot(
    title="Nodes indices after changing the root to leaf node 11"
)
NEWTREE.plot_dims(
    title="Nodes dimensions after changing the root to leaf node 11"
)


# %%
# Add a child to a node
# ^^^^^^^^^^^^^^^^^^^^^
# ``add_child`` returns a new tree with one more node. Added to an interior
# node, the child is a new leaf with a new dimension :math:`d + 1`; added to a
# leaf, the leaf becomes an interior node and the child carries its
# dimension.
ORDER = 8
TREE = tensap.DimensionTree.balanced(ORDER)
TREE.plot(title="Nodes indices before changing the root")
NEWTREE = TREE.add_child(5)
NEWTREE.plot(title="Nodes indices after adding a child to node 5")
NEWTREE = TREE.add_child(11)
NEWTREE.plot(title="Nodes indices after adding a child to leaf node 11")
NEWTREE = TREE.add_child(1)
NEWTREE.plot(title="Nodes indices after adding a child to root node 1")
