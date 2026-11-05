"""
Poisson equation in QTT format
==============================

Discretize the Poisson equation on a rectangle by cell-centred finite
volumes in QTT format, and solve it with the alternating TT solvers DMRG
and AMEn, compared with a sparse direct solver.

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
# A Gaussian source, evaluated at the cell centres and quantized.
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
# by the residual. The truncation precision sets the accuracy; the
# solvers stop when the iterate stagnates.
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
