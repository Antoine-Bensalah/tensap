# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: alternating TT solvers (ALS, DMRG, AMEn) for the Poisson equation
in QTT format, compared with the truncated conjugate gradient.

ApproximationToolbox has ALS and DMRG (TTTensorALSLinearSolver) but no QTT
and no AMEn.

Problem
-------
- Delta u = f on a box, discretized by cell-centred finite volumes
(tensorized_pde_discretization.finite_volumes, see
tutorial_qtt_poisson_rectangle.py), with a Gaussian right-hand side; the
discrete solution is a TTTensor with dimensions of size 2.

Steps
-----
1. a small rectangle, Dirichlet conditions: truncated PCG, DMRG and AMEn
   against scipy.sparse.linalg.spsolve;
2. a large grid (2^10 x 2^10 cells, 10^6 unknowns): DMRG and AMEn only (PCG
   needs a number of iterations growing like the number of cells per axis);
3. Neumann conditions on every wall: singular problem, regularized by the
   rank-one term (sigma / N) 1 1^T, solved by AMEn;
4. a cube, 2^6 cells per axis, both orderings of the digits.
"""

# %% Imports
import time

import numpy as np
import scipy.sparse.linalg as spla
import tensap
from tensap.tensorized_pde_discretization import Dirichlet, Neumann
from tensap.tensorized_pde_discretization.finite_volumes import Axis, Grid

np.random.seed(0)


def gaussian(grid):
    centres = np.meshgrid(*[axis.cell_centers() for axis in grid.axes], indexing="ij")
    return np.exp(-20 * sum((c - 0.5 + 0.1 * k) ** 2 for k, c in enumerate(centres)))


def reference(matrix, f, direct=True):
    """Solution of the same discrete problem by scipy: sparse direct solver,
    or conjugate gradient in 3D, where the fill-in of the direct solver makes
    it very slow (about 30 minutes for 64^3 cells)."""
    if direct:
        return spla.spsolve(matrix.tocsc(), f.ravel())
    u, info = spla.cg(matrix.tocsr(), f.ravel(), rtol=1e-13, atol=0.0, maxiter=10000)
    assert info == 0
    return u


def solve(name, A, b, tolerance=1e-10):
    """Solve A u = b with one of the solvers; returns u, the time and the
    number of iterations (sweeps for the alternating solvers)."""
    start = time.time()
    if name == "pcg":
        # truncation 100 times finer than the requested residual (see the
        # theory notes on the truncated PCG)
        solver = tensap.TruncatedPCG(A, b, tensap.Truncator(tolerance=tolerance),
                                     tolerance=100 * tolerance, max_iterations=1000,
                                     display=False)
    else:
        # the residual of a badly conditioned problem stays at about kappa
        # times the truncation precision: stop on the stagnation of the iterate
        truncator = tensap.Truncator(tolerance=tolerance, max_rank=100)
        solver = tensap.TTTensorALSLinearSolver(A, b, algorithm=name, truncator=truncator,
                                                tolerance=1e-12, stagnation=1e-9,
                                                max_iterations=30, display=False)
    u, output = solver.solve()
    return u, time.time() - start, output["iter"]


def report(grid, name, u, elapsed, iterations, u_ref):
    error = np.linalg.norm(grid.dequantize(u).ravel() - u_ref) / np.linalg.norm(u_ref)
    print(f"  {name:5s} {elapsed:7.2f} s  {iterations:5d} iterations  "
          f"error {error:.1e}  max rank {max(u.ranks)}")
    return error


# %% 1. A small rectangle: PCG, DMRG and AMEn
grid = Grid([Axis(2.0, 6, Dirichlet(), Dirichlet()), Axis(1.0, 5, Dirichlet(), Dirichlet())])
A = -1.0 * grid.laplacian()
f = gaussian(grid)
b = grid.quantize(f)
u_ref = reference(-grid.laplacian_matrix(), f)
print(f"1. {grid.shape[0]} x {grid.shape[1]} cells, Dirichlet")
for name in ("pcg", "dmrg", "amen"):
    u, elapsed, iterations = solve(name, A, b)
    error = report(grid, name, u, elapsed, iterations, u_ref)
    assert error < (1e-5 if name == "pcg" else 1e-7)

# %% 2. A large grid: 10^6 unknowns
grid = Grid([Axis(1.0, 10, Dirichlet(), Dirichlet()), Axis(1.0, 10, Dirichlet(), Dirichlet())])
A = -1.0 * grid.laplacian()
f = gaussian(grid)
b = grid.quantize(f)
u_ref = reference(-grid.laplacian_matrix(), f)
print(f"2. {grid.shape[0]} x {grid.shape[1]} cells, Dirichlet")
for name in ("dmrg", "amen"):
    u, elapsed, iterations = solve(name, A, b)
    error = report(grid, name, u, elapsed, iterations, u_ref)
    assert error < 1e-7
    print(f"     storage {u.storage()} numbers instead of {grid.shape[0] * grid.shape[1]}")

# %% 3. Neumann conditions on every wall: a singular problem
grid = Grid([Axis(1.0, 6, Neumann(), Neumann()), Axis(1.0, 6, Neumann(), Neumann())])
A = -1.0 * grid.laplacian() + grid.constant_mode_regularization()
f = gaussian(grid)
f = f - f.mean()  # compatible data: the solution of zero mean
b = grid.quantize(f)
sigma = min(axis.smallest_nonzero_eigenvalue() for axis in grid.axes)
N = f.size
A_matrix = -grid.laplacian_matrix().toarray() + sigma / N * np.ones((N, N))
u_ref = np.linalg.solve(A_matrix, f.ravel())
print(f"3. {grid.shape[0]} x {grid.shape[1]} cells, Neumann everywhere (regularized)")
u, elapsed, iterations = solve("amen", A, b)
assert report(grid, "amen", u, elapsed, iterations, u_ref) < 1e-6
assert abs(grid.dequantize(u).mean()) < 1e-8

# %% 4. A cube: both orderings of the digits
for ordering in ("blocks", "interleaved"):
    grid = Grid([Axis(1.0, 6, Dirichlet(), Dirichlet()) for _ in range(3)], ordering=ordering)
    A = -1.0 * grid.laplacian()
    f = gaussian(grid)
    b = grid.quantize(f)
    u_ref = reference(-grid.laplacian_matrix(), f, direct=False)
    print(f"4. cube {grid.shape}, {ordering} ordering, b of max rank {max(b.ranks)}")
    for name in ("dmrg", "amen"):
        u, elapsed, iterations = solve(name, A, b)
        assert report(grid, name, u, elapsed, iterations, u_ref) < 1e-7
