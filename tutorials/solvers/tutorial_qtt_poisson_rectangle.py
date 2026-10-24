# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: Poisson equation on a rectangle in QTT format, by cell-centred
finite volumes.

No MATLAB counterpart: ApproximationToolbox has no quantized tensor trains.

Problem
-------
On the rectangle (0, Lx) x (0, Ly), with homogeneous Dirichlet conditions,

    - Delta u = f,   u = 0 on the boundary,

discretized by cell-centred finite volumes: Nx x Ny cells, Nx = 2^lx and
Ny = 2^ly, of size hx = Lx / Nx and hy = Ly / Ny, the unknowns being the
values at the cell centres. At a wall, the missing neighbour is a ghost value
u_ghost = -u_0, so that the mean of the ghost and of the first cell (the value
at the wall) is 0. The discrete solution is an array of shape (Nx, Ny), stored
in QTT format: a TTTensor of order lx + ly with dimensions of size 2.

Steps
-----
1. the grid: tensorized_pde_discretization.finite_volumes.Grid, built from two
   Axis objects; it carries the Tensorizer (layout and geometry of the cells)
   and assembles the Laplacian as a TTOperator;
2. a Gaussian right-hand side, solved by the truncated conjugate gradient and
   compared with scipy.sparse.linalg.spsolve, for the two orderings of the
   digits (blocks: all the digits of x, then those of y; interleaved: one digit
   of x, one digit of y, ...);
3. the order of convergence on a manufactured solution.
"""

# %% Imports
import time

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tensap
from tensap.tensorized_pde_discretization import Dirichlet
from tensap.tensorized_pde_discretization.finite_volumes import Axis, Grid

np.random.seed(0)

Lx, Ly = 2.0, 1.0
truncator = tensap.Truncator(tolerance=1e-10)


def make_grid(lx, ly, ordering="blocks"):
    """The rectangle with 2^lx x 2^ly cells and homogeneous Dirichlet walls."""
    return Grid([Axis(Lx, lx, Dirichlet(), Dirichlet()),
                 Axis(Ly, ly, Dirichlet(), Dirichlet())], ordering)


def solve(grid, f, tolerance=1e-8):
    """- Delta u = f by truncated PCG (no preconditioner)."""
    A = -1.0 * grid.laplacian()
    b = grid.quantize(f, tolerance=1e-10)
    solver = tensap.TruncatedPCG(A, b, truncator, tolerance=tolerance,
                                 max_iterations=1000, stagnation=1e-14, display=False)
    u, output = solver.solve()
    return u, output, A, b


# %% 1. Grid and operator
lx, ly = 6, 5
grid = make_grid(lx, ly)
x = grid.cell_centers()  # shape (Nx, Ny, 2)
print(f"{grid.shape[0]} x {grid.shape[1]} cells of size {grid.axes[0].h} x {grid.axes[1].h}; "
      f"first centre {x[0, 0]}, last centre {x[-1, -1]}")
A = -1.0 * grid.laplacian()
print(f"Operator: QTT ranks {A.ranks}, storage {A.storage()}")

# the QTT operator is the sparse matrix: same product with a random vector
v = np.random.randn(*grid.shape)
assert np.allclose(grid.dequantize(A @ grid.quantize(v)).ravel(),
                   -(grid.laplacian_matrix() @ v.ravel()))


# %% 2. Gaussian right-hand side, both orderings of the digits
def gaussian(p, center=(0.7, 0.4), width=0.05):
    return np.exp(-((p[:, 0] - center[0]) ** 2 + (p[:, 1] - center[1]) ** 2)
                  / (2 * width**2))


u_reference = spla.spsolve(-grid.laplacian_matrix().tocsc(),
                           gaussian(x.reshape(-1, 2))).reshape(grid.shape)
solutions = {}
for ordering in ["blocks", "interleaved"]:
    g = make_grid(lx, ly, ordering)
    start = time.perf_counter()
    u, output, A_o, b_o = solve(g, gaussian)
    u_dense = g.dequantize(u)
    error = np.linalg.norm(u_dense - u_reference) / np.linalg.norm(u_reference)
    print(f"{ordering:>11}: operator ranks {max(A_o.ranks)}, right-hand side ranks "
          f"{max(b_o.ranks)}, PCG {output['iter']} iterations "
          f"({time.perf_counter() - start:.1f} s), error w.r.t. spsolve {error:.1e}, "
          f"solution ranks {max(u.ranks)}, storage {u.storage()} for {u_dense.size} unknowns")
    assert error < 1e-6, error
    solutions[ordering] = u_dense
# the two orderings represent the same discrete solution
assert np.allclose(solutions["blocks"], solutions["interleaved"], atol=1e-8)


# %% 3. Order of convergence on a manufactured solution
# u = sin(pi x / Lx) sin(2 pi y / Ly), f = (pi^2 / Lx^2 + 4 pi^2 / Ly^2) u
def exact(p):
    return np.sin(np.pi * p[:, 0] / Lx) * np.sin(2 * np.pi * p[:, 1] / Ly)


k2 = np.pi**2 / Lx**2 + 4 * np.pi**2 / Ly**2
errors = []
for level in [3, 4, 5]:
    g = make_grid(level + 1, level)
    u, _, _, _ = solve(g, lambda p: k2 * exact(p), tolerance=1e-10)
    u_exact = exact(g.cell_centers().reshape(-1, 2))
    errors.append(np.max(np.abs(g.dequantize(u).ravel() - u_exact)))
    print(f"{g.shape[0]} x {g.shape[1]} cells: max error {errors[-1]:.2e}, "
          f"QTT ranks of u {u.ranks}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print(f"Error ratios when the cells are halved: {np.round(ratios, 2)} (order 2: 4)")
assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios

# %% Plot of the solution for the Gaussian right-hand side
fig, ax = plt.subplots()
contour = ax.contourf(x[:, :, 0], x[:, :, 1], solutions["blocks"], levels=30)
fig.colorbar(contour)
ax.set_aspect("equal")
ax.set_title("QTT solution of -Laplace(u) = Gaussian, u = 0 on the boundary")
plt.show()
