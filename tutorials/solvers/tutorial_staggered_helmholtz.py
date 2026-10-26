# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: discrete Helmholtz-Hodge decomposition on a staggered (MAC) grid in
QTT format (plan P9, level 1).

No MATLAB counterpart.

Problem
-------
Any velocity w on the faces of a staggered grid splits uniquely into

    w = u + G phi,   D u = 0,

u the divergence-free part (on the faces), phi a potential (at the cell
centres), G the gradient (cells -> faces) and D the divergence (faces ->
cells). Applying D: D G phi = D w, that is L_p phi = D w, and u = w - G phi.
On the staggered grid D G = L_p exactly, so D u = 0 up to the precision of
the linear solver; and G = -D^T, so (u, G phi) = -(D u, phi) = 0: u is the
orthogonal projection of w on the divergence-free fields, and
||w||^2 = ||u||^2 + ||G phi||^2. This is the projection step of the Chorin
method used later for the Navier-Stokes equations.

Test with an exact solution: u is the discrete curl of a stream function psi
at the corners (u_x = d psi / dy, u_y = - d psi / dx, by differences), so that
D u = 0 exactly; phi is given at the cells, and w = u + G phi. The
decomposition of w must give back u and phi (up to a constant).

Steps
-----
1. periodic domain (0, 2 pi)^2, psi = sin x sin y, phi = cos 2x cos y;
2. box (0, 1)^2 with walls, psi = sin^2(pi x) sin^2(pi y) (zero normal
   velocity on the walls), phi = cos(pi x) cos(2 pi y);
3. on a small grid, the same decomposition with dense matrices (numpy), as a
   reference for the QTT computation.

The pressure Laplacian L_p (Neumann or periodic) is singular; it is
regularized by the rank-one term (sigma / N) 1 1^T and the potential of zero
mean is computed by AMEn (TTTensorALSLinearSolver).
"""

# %% Imports
import time

import numpy as np
import tensap
from tensap.tensorized_pde_discretization.finite_volumes import StaggeredAxis, StaggeredGrid


def decompose(grid, psi, phi):
    """Build w = curl psi + G phi, decompose it, and report the errors."""
    Psi, Phi = grid.quantize(psi), grid.quantize(phi)
    u_exact = [curl @ Psi for curl in grid.curl()]
    w = [uk + G @ Phi for uk, G in zip(u_exact, grid.gradient())]
    start = time.time()
    u, potential = grid.project(w)
    elapsed = time.time() - start

    divergence = grid.divergence([None if a.periodic else tensap.tensorized_pde_discretization
                                  .Dirichlet() for a in grid.axes])
    div_u = sum(grid.dequantize(D @ uk) for D, uk in zip(divergence, u))
    error_u = max(np.abs(grid.dequantize(a) - grid.dequantize(b)).max()
                  for a, b in zip(u, u_exact))
    error_phi = np.abs(grid.dequantize(potential) - (phi - phi.mean())).max()
    gradient = [G @ potential for G in grid.gradient()]
    energy = sum(a.dot(a) for a in w)
    split = sum(a.dot(a) for a in u) + sum(a.dot(a) for a in gradient)
    print(f"  {grid.shape[0]} x {grid.shape[1]} cells: max |D u| {np.abs(div_u).max():.1e}, "
          f"error on u {error_u:.1e}, on phi {error_phi:.1e}, "
          f"|w|^2 - |u|^2 - |G phi|^2 = {abs(energy - split) / energy:.1e} (relative), "
          f"{elapsed:.2f} s")
    print(f"  QTT ranks: w {[max(a.ranks) for a in w]}, u {[max(a.ranks) for a in u]}, "
          f"phi {max(potential.ranks)}")
    # the divergence of the error is of the order of error / h: compare h |D u|
    return grid.axes[0].h * np.abs(div_u).max(), error_u, error_phi


# %% 1. Periodic domain
print("1. periodic domain (0, 2 pi)^2")
for levels in (6, 10):
    grid = StaggeredGrid([StaggeredAxis(2 * np.pi, levels, periodic=True)] * 2)
    x, y = grid.positions("corner")
    xc, yc = grid.positions("p")
    results = decompose(grid, np.sin(x) * np.sin(y), np.cos(2 * xc) * np.cos(yc))
    assert max(results) < 1e-8

# %% 2. Box with walls
print("2. box (0, 1)^2 with walls")
for levels in (6, 10):
    grid = StaggeredGrid([StaggeredAxis(1.0, levels)] * 2)
    x, y = grid.positions("corner")
    xc, yc = grid.positions("p")
    psi = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
    results = decompose(grid, psi, np.cos(np.pi * xc) * np.cos(2 * np.pi * yc))
    assert max(results) < 1e-8

# %% 3. Dense reference on a small grid
# The same decomposition with the dense matrices of D and G: least-squares
# solution of D G phi = D w (singular, consistent), then u = w - G phi.
print("3. dense reference, 16 x 16 cells with walls")
grid = StaggeredGrid([StaggeredAxis(1.0, 4)] * 2)


def matrix(operator):
    full = operator.full().numpy()
    d = operator.order
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    n = int(np.prod([s[0] for s in operator.shape]))
    return full.transpose(perm).reshape(n, n)


G = [matrix(Gk) for Gk in grid.gradient()]
D = [matrix(Dk) for Dk in grid.divergence([tensap.tensorized_pde_discretization.Dirichlet()] * 2)]
x, y = grid.positions("corner")
xc, yc = grid.positions("p")
psi = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
phi = np.cos(np.pi * xc) * np.cos(2 * np.pi * yc)
curl = [matrix(c) for c in grid.curl()]
w = [c @ psi.ravel() + Gk @ phi.ravel() for c, Gk in zip(curl, G)]
L = sum(Dk @ Gk for Dk, Gk in zip(D, G))
phi_dense = np.linalg.lstsq(L, sum(Dk @ wk for Dk, wk in zip(D, w)), rcond=None)[0]
phi_dense -= phi_dense.mean()
u_dense = [wk - Gk @ phi_dense for wk, Gk in zip(w, G)]
u_qtt, phi_qtt = grid.project([grid.quantize(wk.reshape(grid.shape)) for wk in w])
difference = max(np.abs(grid.dequantize(a).ravel() - b).max() for a, b in zip(u_qtt, u_dense))
print(f"  dense against QTT: u {difference:.1e}, "
      f"phi {np.abs(grid.dequantize(phi_qtt).ravel() - phi_dense).max():.1e}")
assert difference < 1e-8
