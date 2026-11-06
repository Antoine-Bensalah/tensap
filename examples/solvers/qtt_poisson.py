r"""
Poisson equation in QTT format
==============================

Discretize the Poisson equation on a rectangle by cell-centred finite
volumes in QTT format, and solve it with the alternating TT solvers DMRG
and AMEn, compared with a sparse direct solver.

We solve :math:`-\Delta u = f` on :math:`\Omega = (0, 2) \times (0, 1)`
with :math:`u = 0` on the boundary. On a grid of :math:`2^{L_1} \times
2^{L_2}` cells, the discrete system :math:`-\Delta_h u = b`, with
:math:`\Delta_h = D^{(1)} \otimes I + I \otimes D^{(2)}`, is written in
QTT format: each cell index is split into its binary digits, and the
grid function becomes a tensor train of order :math:`L_1 + L_2` with
modes of size 2. We look at the accuracy, the time and the QTT ranks of
the solution. See :ref:`theory-finite-volumes` and :ref:`theory-qtt`.

This example comes from the script
``test/non_regression_scripts/solvers/tutorial_qtt_tt_solvers.py``, on
smaller grids.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import tensap
from tensap.tensorized_pde_discretization import Dirichlet
from tensap.tensorized_pde_discretization.finite_volumes import (
    Axis,
    Grid,
)

np.random.seed(0)

# %%
# The grid and the operator
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# The rectangle :math:`(0, 2) \times (0, 1)` with :math:`2^8 \times
# 2^7` cells and homogeneous Dirichlet conditions. An
# :class:`~tensap.tensorized_pde_discretization.finite_volumes.Axis`
# carries the length, the number of binary digits and the conditions at
# its two ends; the
# :class:`~tensap.tensorized_pde_discretization.finite_volumes.Grid`
# assembles the axes in the layout of a :class:`~tensap.Tensorizer`. The
# discrete Laplacian is a :class:`~tensap.TTOperator` of small ranks.
#
# On each axis, of length :math:`a_\nu` and with the cell size
# :math:`h_\nu = a_\nu / 2^{L_\nu}`, the scheme is the three-point
# difference
#
# .. math::
#
#    (D^{(\nu)} u)(i) = \frac{u(i-1) - 2 u(i) + u(i+1)}{h_\nu^2},
#
# with a ghost cell :math:`u(0) = -u(1)` (and likewise at the other end)
# for the Dirichlet condition. The printed QTT ranks of the operator stay
# small, whatever the number of cells.
grid = Grid(
    [
        Axis(2.0, 8, Dirichlet(), Dirichlet()),
        Axis(1.0, 7, Dirichlet(), Dirichlet()),
    ]
)
A = -1.0 * grid.laplacian()
print(
    f"{grid.shape[0]} x {grid.shape[1]} cells; operator: QTT ranks "
    f"{A.ranks}, storage {A.storage()}"
)

# %%
# The right-hand side
# ^^^^^^^^^^^^^^^^^^^
# A Gaussian source
#
# .. math::
#
#    f(x, y) = \exp\big(-20\, ((x - 0.6)^2 + (y - 0.4)^2)\big),
#
# evaluated at the cell centres and quantized: ``grid.quantize`` reshapes
# the array of values into a tensor of order :math:`L_1 + L_2` and
# compresses it in TT format. A smooth function has small QTT ranks.
xc, yc = np.meshgrid(
    *[axis.cell_centers() for axis in grid.axes], indexing="ij"
)
f = np.exp(-20 * ((xc - 0.6) ** 2 + (yc - 0.4) ** 2))
b = grid.quantize(f)
print(f"right-hand side: QTT ranks {b.ranks}")

# %%
# Reference solution
# ^^^^^^^^^^^^^^^^^^
# The same finite volume system, assembled as a sparse matrix and solved
# by a sparse direct solver.
start = time()
u_ref = spla.spsolve(-grid.laplacian_matrix().tocsc(), f.ravel())
print(f"sparse direct solver: {time() - start:.2f} s")

# %%
# DMRG and AMEn
# ^^^^^^^^^^^^^
# :class:`~tensap.TTTensorALSLinearSolver` with adapted ranks: DMRG
# updates two cores at a time, AMEn one core at a time with an enrichment
# by the residual. Each local step minimizes the energy
# :math:`J(v) = \frac12 (A v, v) - (b, v)` over one core (or two) of the
# solution, the others being fixed: a small linear system. The truncation
# precision sets the accuracy; the solvers stop when the iterate
# stagnates. The output compares the time with the sparse direct solver,
# and the relative error with its solution.
for algorithm in ("dmrg", "amen"):
    start = time()
    solver = tensap.TTTensorALSLinearSolver(
        A,
        b,
        algorithm=algorithm,
        truncator=tensap.Truncator(tolerance=1e-10, max_rank=100),
        tolerance=1e-12,
        stagnation=1e-9,
        max_iterations=30,
        display=False,
    )
    u, output = solver.solve()
    elapsed = time() - start
    error = np.linalg.norm(
        grid.dequantize(u).ravel() - u_ref
    ) / np.linalg.norm(u_ref)
    print(
        f"{algorithm.upper():5s}: {elapsed:.2f} s, {output['iter']} sweeps, "
        f"error {error:.1e}, max QTT rank {max(u.ranks)}"
    )

# %%
# The solution
# ^^^^^^^^^^^^
# ``grid.dequantize`` gives back the array of the values at the cell
# centres, for the plot.
plt.figure()
plt.pcolormesh(xc, yc, grid.dequantize(u), shading="auto")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title("Solution by AMEn")
plt.show()

# %%
# A grid of a million cells
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# On the square with :math:`2^{10} \times 2^{10}` cells, the QTT ranks of
# the solution stay small: AMEn stores it with a few thousand numbers.
# The solver never forms an array of the size of the grid; only the
# quantization of the right-hand side does here, by a TT-SVD of its
# values, for simplicity.
grid = Grid(
    [
        Axis(1.0, 10, Dirichlet(), Dirichlet()),
        Axis(1.0, 10, Dirichlet(), Dirichlet()),
    ]
)
xc, yc = np.meshgrid(
    *[axis.cell_centers() for axis in grid.axes], indexing="ij"
)
b = grid.quantize(np.exp(-20 * ((xc - 0.6) ** 2 + (yc - 0.4) ** 2)))
start = time()
u, output = tensap.TTTensorALSLinearSolver(
    -1.0 * grid.laplacian(),
    b,
    algorithm="amen",
    truncator=tensap.Truncator(tolerance=1e-8, max_rank=100),
    tolerance=1e-12,
    stagnation=1e-7,
    max_iterations=30,
    display=False,
).solve()
print(
    f"{grid.shape[0] * grid.shape[1]} cells: AMEn {time() - start:.1f} s, "
    f"{output['iter']} sweeps, max QTT rank {max(u.ranks)}, "
    f"storage {u.storage()}"
)
