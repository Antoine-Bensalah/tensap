r"""
Multi-indices
=============

Sets of multi-indices: product sets, sets with bounded p-norm, operations,
margins and downward closed sets.

A set of multi-indices :math:`\Lambda \subset \mathbb{N}_0^d` indexes the
products :math:`\varphi^{(1)}_{i_1}(x_1) \cdots \varphi^{(d)}_{i_d}(x_d)`,
:math:`i = (i_1, \dots, i_d) \in \Lambda`, of a sparse tensor product basis,
or the points of a sparse grid. We build the usual sets with
:class:`~tensap.MultiIndices`, combine them, and compute their margins, which
drive the adaptive enrichment of sparse polynomial bases.
See :ref:`theory-multi-indices`.

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
# The product set :math:`\Lambda = I_1 \times I_2` of
# :math:`I_1 = \{1, 2, 3\}` and :math:`I_2 = \{1, 2\}`, which would index a
# full tensor product basis. The ``array`` attribute has one multi-index per
# row.
I0 = tensap.MultiIndices.product_set([[1, 2, 3], [1, 2]])
print("I.array = \n%s\n" % I0.array)

# %%
# Set of indices with bounded p-norm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :math:`\ell^p` ball
#
# .. math::
#
#    \Lambda = \{i \in \mathbb{N}_0^d : \|i\|_{\ell^p} \le m\},
#    \qquad \|i\|_{\ell^p} = \Big(\sum_{\nu=1}^{d} i_\nu^p\Big)^{1/p}.
#
# For :math:`p = 1` it gives the polynomials of total degree at most
# :math:`m`. Here :math:`p = 1/2` and :math:`m = 6` in dimension 2: a
# hyperbolic cross, with high degrees in one variable only and few mixed
# terms.
m = 6
p = 1 / 2
d = 2
Ip = tensap.MultiIndices.with_bounded_norm(d, p, m)
print("Ip.array = \n%s\n" % Ip.array)

# %%
# Set of indices bounded by an multi-index
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The box :math:`\Lambda = \{i : i_\nu \le m_\nu,\ 1 \le \nu \le d\}`, here
# with :math:`m = (3, 2)`.
p = [3, 2]
Ip = tensap.MultiIndices.bounded_by(p)
print("Ip.array = \n%s\n" % Ip.array)

# %%
# Product of sets
# ^^^^^^^^^^^^^^^
# For :math:`\Lambda_1 \subset \mathbb{N}_0^{d_1}` and
# :math:`\Lambda_2 \subset \mathbb{N}_0^{d_2}`, ``product`` returns
# :math:`\Lambda_1 \times \Lambda_2 \subset \mathbb{N}_0^{d_1 + d_2}`, the
# concatenations :math:`(i, j)` of the multi-indices. Here two sets of
# dimension 2 give a set of dimension 4.
I1 = tensap.MultiIndices.with_bounded_norm(2, 1, 2)
I2 = tensap.MultiIndices.with_bounded_norm(2, 1 / 2, 6)
Iprod = I1.product(I2)
print("I1.array = \n%s\n" % I1.array)
print("I2.array = \n%s\n" % I2.array)
print("(I1.product(I2)).array = \n%s\n" % Iprod.array)

# %%
# Operations on indices
# ^^^^^^^^^^^^^^^^^^^^^
# Sets of multi-indices support the union (``add_indices``), the difference
# (``remove_indices``), and the shift of all the indices by an integer.
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
# Let :math:`\Lambda` be the union of the boxes bounded by :math:`(1, 3)` and
# :math:`(3, 1)`, a downward closed set shaped like an L. With
# :math:`e_\nu` the :math:`\nu`-th unit vector, its *margin* is the set of
# the :math:`i \notin \Lambda` such that :math:`i - e_\nu \in \Lambda` for
# some :math:`\nu`, and its *reduced margin* the set of the
# :math:`i \notin \Lambda` such that :math:`i - e_\nu \in \Lambda` for every
# :math:`\nu` with :math:`i_\nu > 0`. Adding to :math:`\Lambda` an element of
# the reduced margin keeps it downward closed. The maximal elements are the
# :math:`i \in \Lambda` such that no :math:`i + e_\nu` is in :math:`\Lambda`.
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
# A set :math:`\Lambda` is downward closed if :math:`i \in \Lambda` and
# :math:`j \le i` componentwise imply :math:`j \in \Lambda`. The set
# :math:`\{i : i_1 + i_2 \le 4\}` is; adding :math:`(2, 4)` alone breaks the
# property, since for instance :math:`(2, 3)` is not in the set.
dim = 2
I0 = tensap.MultiIndices.with_bounded_norm(dim, 1, 4)
print("I.is_downward_closed() = %s\n" % I0.is_downward_closed())

J = I0.add_indices(tensap.MultiIndices([2, 4]))
print("J.is_downward_closed() = %s\n" % J.is_downward_closed())

# %%
# When indices represent subindices of an nd array of shape sz
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Multi-indices can also be the positions :math:`(i_1, i_2, i_3)` of entries
# in an array of shape :math:`5 \times 5 \times 5`. ``sub2ind`` converts them
# to positions in the flattened array (in column-major order, as
# ``np.ravel_multi_index`` with ``order="F"``), and ``ind2sub`` goes back.
# The box here starts at 1 (second argument of ``bounded_by``).

# Obtaining the position of multi-indices in the nd array
I0 = tensap.MultiIndices.bounded_by([3, 4, 2], 1)
e = I0.sub2ind([5, 5, 5])
# Creating a multi-index set associated with entries of a multi-array
I0 = tensap.MultiIndices.ind2sub([5, 5, 5], e)
print("I.array = \n%s\n" % I0.array)

# %%
# Reduced margin of a sum of two tensap.MultiIndices
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same computation in dimension 5, for the union of two boxes: these are
# the candidates an adaptive algorithm would consider to enrich the set.
I1 = tensap.MultiIndices.bounded_by([3, 5, 7, 4, 3])
I2 = tensap.MultiIndices.bounded_by([7, 5, 3, 4, 2])
I0 = I1.add_indices(I2)
I_marg = I0.get_reduced_margin()
print(
    "I1.add_indices(I2).get_reduced_margin().array = \n%s\n"
    % I_marg.array
)
