"""
Multi-indices
=============

Sets of multi-indices: product sets, sets with bounded p-norm, operations,
margins and downward closed sets.

This example comes from the script
``test/non_regression_scripts/tools/tutorial_MultiIndices.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import tensap
import numpy as np

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Product set
# ^^^^^^^^^^^
I0 = tensap.MultiIndices.product_set([[1, 2, 3], [1, 2]])
print("I.array = \n%s\n" % I0.array)

# %%
# Set of indices with bounded p-norm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
m = 6
p = 1 / 2
d = 2
Ip = tensap.MultiIndices.with_bounded_norm(d, p, m)
print("Ip.array = \n%s\n" % Ip.array)

# %%
# Set of indices bounded by an multi-index
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
p = [3, 2]
Ip = tensap.MultiIndices.bounded_by(p)
print("Ip.array = \n%s\n" % Ip.array)

# %%
# Product of sets
# ^^^^^^^^^^^^^^^
I1 = tensap.MultiIndices.with_bounded_norm(2, 1, 2)
I2 = tensap.MultiIndices.with_bounded_norm(2, 1 / 2, 6)
Iprod = I1.product(I2)
print("I1.array = \n%s\n" % I1.array)
print("I2.array = \n%s\n" % I2.array)
print("(I1.product(I2)).array = \n%s\n" % Iprod.array)

# %%
# Operations on indices
# ^^^^^^^^^^^^^^^^^^^^^
I1 = tensap.MultiIndices.bounded_by([1, 3])
I2 = tensap.MultiIndices.bounded_by([3, 1])

# Adding two sets
I0 = I1.add_indices(I2)
print("I1.add_indices(I2).array = \n%s\n" % I0.array)

# Subtracting a set
I0 = I1.remove_indices(I2)
print("I1.remove_indices(I2).array = \n%s\n" % I0.array)

# Adding (or subtracting) an integer
I0 = I0 + 2
print("(I+2).array = \n%s\n" % I0.array)

# %%
# Maximal elements, margin, reduced margin
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I1 = tensap.MultiIndices.bounded_by([1, 3])
I2 = tensap.MultiIndices.bounded_by([3, 1])
I0 = I1.add_indices(I2)
print("I.array = I1.add_indices(I2).array = \n%s\n" % I0.array)
I_marg = I0.get_margin()
print("I.get_margin().array = \n%s\n" % I_marg.array)
I_red = I0.get_reduced_margin()
print("I.get_reduced_margin().array = \n%s\n" % I_red.array)
I_max = I0.get_maximal_indices()
print("I.get_maximal_indices().array = \n%s\n" % I_max.array)

# %%
# Check whether a set is downward closed
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dim = 2
I0 = tensap.MultiIndices.with_bounded_norm(dim, 1, 4)
print("I.is_downward_closed() = %s\n" % I0.is_downward_closed())

J = I0.add_indices(tensap.MultiIndices([2, 4]))
print("J.is_downward_closed() = %s\n" % J.is_downward_closed())

# %%
# When indices represent subindices of an nd array of shape sz
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Obtaining the position of multi-indices in the nd array
I0 = tensap.MultiIndices.bounded_by([3, 4, 2], 1)
e = I0.sub2ind([5, 5, 5])
# Creating a multi-index set associated with entries of a multi-array
I0 = tensap.MultiIndices.ind2sub([5, 5, 5], e)
print("I.array = \n%s\n" % I0.array)

# %%
# Reduced margin of a sum of two tensap.MultiIndices
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I1 = tensap.MultiIndices.bounded_by([3, 5, 7, 4, 3])
I2 = tensap.MultiIndices.bounded_by([7, 5, 3, 4, 2])
I0 = I1.add_indices(I2)
I_marg = I0.get_reduced_margin()
print(
    "I1.add_indices(I2).get_reduced_margin().array = \n%s\n"
    % I_marg.array
)
