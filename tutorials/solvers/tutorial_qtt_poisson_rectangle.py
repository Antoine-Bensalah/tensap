# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: Poisson equation on a rectangle in QTT format, by cell-centred
finite volumes, with Dirichlet and Neumann conditions.

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
3. the order of convergence on a manufactured solution;
4. non-homogeneous Dirichlet conditions: the data g enter the boundary source
   s (Delta u ~ Delta_h u + s), built in QTT face by face from e_0 (or
   e_{N-1}) of the normal axis and the data on the face: order 2 on a harmonic
   function, then a "lid" problem (u = 1 on the top wall, 0 elsewhere);
5. Neumann conditions du/dn = g (outward normal) on the vertical walls,
   closed by the centred flux at the face (ghost u_ghost = u_0 + h g): order 2
   on the same harmonic function.
"""

# %% Imports
import time

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tensap
from tensap.tensorized_pde_discretization import Dirichlet, Neumann
from tensap.tensorized_pde_discretization.finite_volumes import Axis, Grid

np.random.seed(0)

Lx, Ly = 2.0, 1.0


def make_grid(lx, ly, ordering="blocks"):
    """The rectangle with 2^lx x 2^ly cells and homogeneous Dirichlet walls."""
    return Grid([Axis(Lx, lx, Dirichlet(), Dirichlet()),
                 Axis(Ly, ly, Dirichlet(), Dirichlet())], ordering)


def solve(grid, f, tolerance=1e-8):
    """- Delta u = f by truncated PCG (no preconditioner): - Delta_h u = f + s,
    s the boundary source of the data of the conditions of the grid."""
    A = -1.0 * grid.laplacian()
    b = grid.quantize(f, tolerance=1e-10)
    source = grid.boundary_source()
    if source.norm() > 0:  # homogeneous conditions: no source (and no extra rank)
        b = b + source
    # the truncation must be finer than the requested residual, otherwise the
    # truncation errors keep the residual above it and PCG stagnates
    truncator = tensap.Truncator(tolerance=1e-2 * tolerance)
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

# %% 4. Non-homogeneous Dirichlet conditions: order 2 on a harmonic function
# u = exp(x) cos(y), - Delta u = 0, data g = u on the four walls (functions of
# the coordinate along the wall). The error is measured in the discrete L2
# norm: the maximum error is pre-asymptotic near the walls (ghost closure, local
# truncation error O(1) in the first cells); its ratios tend to 4 more slowly
# (3.4, 3.7, 3.8, 3.9 from 16 x 8 to 512 x 256 cells).


def harmonic(p):
    return np.exp(p[:, 0]) * np.cos(p[:, 1])


def harmonic_grid(lx, ly):
    walls_x = [Dirichlet(lambda y: harmonic(np.column_stack((np.zeros(len(y)), y[:, 0])))),
               Dirichlet(lambda y: harmonic(np.column_stack((np.full(len(y), Lx), y[:, 0]))))]
    walls_y = [Dirichlet(lambda x: harmonic(np.column_stack((x[:, 0], np.zeros(len(x)))))),
               Dirichlet(lambda x: harmonic(np.column_stack((x[:, 0], np.full(len(x), Ly)))))]
    return Grid([Axis(Lx, lx, *walls_x), Axis(Ly, ly, *walls_y)])


errors = []
for level in [3, 4, 5]:
    g = harmonic_grid(level + 1, level)
    u, _, _, b = solve(g, lambda p: np.zeros(p.shape[0]), tolerance=1e-10)
    u_exact = harmonic(g.cell_centers().reshape(-1, 2))
    errors.append(np.sqrt(np.mean((g.dequantize(u).ravel() - u_exact) ** 2)))
    print(f"{g.shape[0]} x {g.shape[1]} cells: source ranks {max(b.ranks)}, "
          f"L2 error {errors[-1]:.2e}, QTT ranks of u {max(u.ranks)}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print(f"Non-homogeneous Dirichlet, error ratios: {np.round(ratios, 2)} (order 2: 4)")
assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios

# %% 4 (continued). A lid: u = 1 on the top wall, u = 0 on the others
lid = Grid([Axis(Lx, lx - 1, Dirichlet(), Dirichlet()),
            Axis(Ly, ly - 1, Dirichlet(), Dirichlet(1.0))])
u_lid, output, _, b_lid = solve(lid, lambda p: np.zeros(p.shape[0]))
u_lid_reference = spla.spsolve(-lid.laplacian_matrix().tocsc(),
                               lid.boundary_source_array().ravel()).reshape(lid.shape)
error = np.linalg.norm(lid.dequantize(u_lid) - u_lid_reference) / np.linalg.norm(u_lid_reference)
print(f"Lid: source ranks {max(b_lid.ranks)} (a constant on one wall), PCG "
      f"{output['iter']} iterations, error w.r.t. spsolve {error:.1e}, "
      f"solution ranks {max(u_lid.ranks)}")
assert error < 1e-6, error
# the discrete maximum principle: 0 <= u <= 1
assert np.all(lid.dequantize(u_lid) > -1e-8) and np.all(lid.dequantize(u_lid) < 1 + 1e-8)

# %% 5. Neumann conditions on the vertical walls, Dirichlet on the others
# u = exp(x) cos(y): du/dn = -du/dx = -cos(y) at x = 0, du/dx = e^Lx cos(y) at
# x = Lx


def neumann_grid(lx, ly):
    walls_x = [Neumann(lambda y: -np.cos(y[:, 0])),
               Neumann(lambda y: np.exp(Lx) * np.cos(y[:, 0]))]
    walls_y = [Dirichlet(lambda x: harmonic(np.column_stack((x[:, 0], np.zeros(len(x)))))),
               Dirichlet(lambda x: harmonic(np.column_stack((x[:, 0], np.full(len(x), Ly)))))]
    return Grid([Axis(Lx, lx, *walls_x), Axis(Ly, ly, *walls_y)])


errors = []
for level in [3, 4, 5]:
    g = neumann_grid(level + 1, level)
    u, output, _, b = solve(g, lambda p: np.zeros(p.shape[0]), tolerance=1e-10)
    u_exact = harmonic(g.cell_centers().reshape(-1, 2))
    errors.append(np.sqrt(np.mean((g.dequantize(u).ravel() - u_exact) ** 2)))
    print(f"{g.shape[0]} x {g.shape[1]} cells: source ranks {max(b.ranks)}, PCG "
          f"{output['iter']} iterations, L2 error {errors[-1]:.2e}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print(f"Neumann and Dirichlet, error ratios: {np.round(ratios, 2)} (order 2: 4)")
assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios

# %% Plot of the solution for the Gaussian right-hand side
fig, ax = plt.subplots()
contour = ax.contourf(x[:, :, 0], x[:, :, 1], solutions["blocks"], levels=30)
fig.colorbar(contour)
ax.set_aspect("equal")
ax.set_title("QTT solution of -Laplace(u) = Gaussian, u = 0 on the boundary")
plt.show()
