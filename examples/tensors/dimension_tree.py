"""
Dimension trees
===============

Create linear, random and balanced dimension trees, extract subtrees, change
the root and plot them.

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
ORDER = 5
TREE = tensap.DimensionTree.linear(ORDER)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# Random dimension tree
# ^^^^^^^^^^^^^^^^^^^^^
ORDER = 10
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TREE.plot(title="Nodes indices")
TREE.plot_dims(title="Nodes dimensions")

# %%
# Balanced dimension tree
# ^^^^^^^^^^^^^^^^^^^^^^^
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
[SUB_TREE, NOD] = TREE.sub_dimension_tree(4)
TREE.plot(
    colored_nodes=NOD, node_color="b", title="Sub-tree to extract"
)
SUB_TREE.plot(title="Sub-tree")

# %%
# Plot with and without the same level at the same height
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ORDER = 8
TREE = tensap.DimensionTree.random(ORDER)

TREE.plot_options["level_alignment"] = False
TREE.plot_with_labels_at_nodes(TREE.level, title="No level alignment")
TREE.plot_options["level_alignment"] = True
TREE.plot_with_labels_at_nodes(TREE.level, title="Level alignment")

# %%
# Change the root of a tree
# ^^^^^^^^^^^^^^^^^^^^^^^^^
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
ORDER = 8
TREE = tensap.DimensionTree.balanced(ORDER)
TREE.plot(title="Nodes indices before changing the root")
NEWTREE = TREE.add_child(5)
NEWTREE.plot(title="Nodes indices after adding a child to node 5")
NEWTREE = TREE.add_child(11)
NEWTREE.plot(title="Nodes indices after adding a child to leaf node 11")
NEWTREE = TREE.add_child(1)
NEWTREE.plot(title="Nodes indices after adding a child to root node 1")
