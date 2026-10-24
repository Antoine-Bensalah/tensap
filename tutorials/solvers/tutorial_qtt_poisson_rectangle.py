# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: Poisson equation on a rectangle in QTT format.

No MATLAB counterpart: ApproximationToolbox has no quantized tensor trains.

Problem
-------
On the rectangle (0, Lx) x (0, Ly), with homogeneous Dirichlet conditions,

    - Delta u = f,   u = 0 on the boundary,

discretized by second order finite differences on the Nx x Ny interior nodes
x_i = (i + 1) hx, y_j = (j + 1) hy, hx = Lx / (Nx + 1), hy = Ly / (Ny + 1),
with Nx = 2^lx and Ny = 2^ly. The boundary nodes carry u = 0 and are not
unknowns. The discrete solution is an array of shape (Nx, Ny), stored in QTT
format: a TTTensor of order lx + ly with dimensions of size 2 (the digits of
i, then those of j).

Steps
-----
1. the geometry of the grid is carried by a Tensorizer, with one number of
   digits per axis: the node (i, j) is the image of the cell (i, j) of the
   dyadic grid of [0, 1]^2 (y0 = 0) by the uniform laws X_k on [h_k, L_k];
2. the operator A = (1 / hx^2) T_x (x) I + (1 / hy^2) I (x) T_y, with
   T = tridiag(-1, 2, -1), is assembled from the QTT primitives (banded
   matrices, identities, tensor products of the chains);
3. a Gaussian right-hand side, quantized on the grid, and a solve by the
   truncated conjugate gradient, compared with scipy.sparse.linalg.spsolve;
4. the order of convergence on a manufactured solution.
"""

# %% Imports
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tensap

np.random.seed(0)

Lx, Ly = 2.0, 1.0


# %% Geometry, operator and dense reference for a given grid
def tensorizer(lx, ly):
    """Interior nodes of the rectangle, 2^lx x 2^ly, as a Tensorizer."""
    hx, hy = Lx / (2**lx + 1), Ly / (2**ly + 1)
    X = tensap.RandomVector([tensap.UniformRandomVariable(hx, Lx),
                             tensap.UniformRandomVariable(hy, Ly)])
    return tensap.Tensorizer(2, [lx, ly], 2, X=X), hx, hy


def qtt_minus_laplacian(lx, ly, hx, hy):
    """- Delta_h with homogeneous Dirichlet conditions, as a TTOperator.

    Built from the QTT primitives, waiting for the finite difference
    operators of tensap (P4c): T = tridiag(-1, 2, -1) has QTT rank 3, and the
    tensor products of the chains assemble the 2D operator (block ordering:
    digits of x, then of y).
    """
    def T(level):
        return tensap.qtt_banded({-1: -1.0, 0: 2.0, 1: -1.0}, level)
    A = (T(lx).tensor_product(tensap.qtt_identity(ly)) * (1 / hx**2)
         + tensap.qtt_identity(lx).tensor_product(T(ly)) * (1 / hy**2))
    return tensap.Truncator(tolerance=1e-14).truncate(A)


def sparse_minus_laplacian(lx, ly, hx, hy):
    """The same matrix with scipy.sparse, unknowns u[i, j] in C order."""
    def T(n):
        return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    Nx, Ny = 2**lx, 2**ly
    return (sp.kron(T(Nx), sp.identity(Ny)) / hx**2
            + sp.kron(sp.identity(Nx), T(Ny)) / hy**2).tocsc()


# %% 1-2. Grid and operator
lx, ly = 6, 5
t, hx, hy = tensorizer(lx, ly)
points = tensap.grid_points(t)  # shape (Nx, Ny, 2)
print(f"Grid {2**lx} x {2**ly}: first node {points[0, 0]}, last node {points[-1, -1]}")
A = qtt_minus_laplacian(lx, ly, hx, hy)
A_sparse = sparse_minus_laplacian(lx, ly, hx, hy)
print(f"Operator: QTT ranks {A.ranks}, storage {A.storage()}")

# the QTT operator is the sparse matrix: same product with a random vector
v = np.random.randn(2**lx, 2**ly)
Av = tensap.dequantize(A @ tensap.quantize(v, t), t)
assert np.allclose(Av.ravel(), A_sparse @ v.ravel())


# %% 3. Gaussian right-hand side, solve by truncated PCG
def gaussian(x, center=(0.7, 0.4), width=0.05):
    return np.exp(-((x[:, 0] - center[0]) ** 2 + (x[:, 1] - center[1]) ** 2)
                  / (2 * width**2))


b = tensap.quantize_function(gaussian, t, tolerance=1e-10)
print(f"Right-hand side: QTT ranks {b.ranks}")

truncator = tensap.Truncator(tolerance=1e-10)
solver = tensap.TruncatedPCG(A, b, truncator, tolerance=1e-8, max_iterations=1000,
                             stagnation=1e-14, display=False)
start = time.perf_counter()
u, output = solver.solve()
print(f"TruncatedPCG: {output['iter']} iterations, residual {output['error']:.1e}, "
      f"{time.perf_counter() - start:.1f} s")

u_dense = tensap.dequantize(u, t)
u_reference = spla.spsolve(A_sparse, gaussian(points.reshape(-1, 2))).reshape(u_dense.shape)
error = np.linalg.norm(u_dense - u_reference) / np.linalg.norm(u_reference)
print(f"Relative error with respect to spsolve: {error:.1e}")
print(f"Solution: QTT ranks {u.ranks}, storage {u.storage()} "
      f"for {u_dense.size} unknowns")
assert error < 1e-6, error


# %% 4. Order of convergence on a manufactured solution
# u = sin(pi x / Lx) sin(2 pi y / Ly), f = (pi^2 / Lx^2 + 4 pi^2 / Ly^2) u
def exact(x):
    return np.sin(np.pi * x[:, 0] / Lx) * np.sin(2 * np.pi * x[:, 1] / Ly)


k2 = np.pi**2 / Lx**2 + 4 * np.pi**2 / Ly**2
errors = []
for level in [3, 4, 5]:
    t_l, hx_l, hy_l = tensorizer(level + 1, level)
    A_l = qtt_minus_laplacian(level + 1, level, hx_l, hy_l)
    b_l = tensap.quantize_function(lambda x: k2 * exact(x), t_l)
    s = tensap.TruncatedPCG(A_l, b_l, truncator, tolerance=1e-10, max_iterations=1000,
                            stagnation=1e-14, display=False)
    u_l, _ = s.solve()
    u_exact = exact(tensap.grid_points(t_l).reshape(-1, 2))
    errors.append(np.max(np.abs(tensap.dequantize(u_l, t_l).ravel() - u_exact)))
    print(f"Grid {2**(level + 1)} x {2**level}: max error {errors[-1]:.2e}, "
          f"QTT ranks of u {u_l.ranks}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print(f"Error ratios when the grid is refined: {np.round(ratios, 2)} (order 2: about 4)")
assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios

# %% Plot of the solution for the Gaussian right-hand side
fig, ax = plt.subplots()
contour = ax.contourf(points[:, :, 0], points[:, :, 1], u_dense, levels=30)
fig.colorbar(contour)
ax.set_aspect("equal")
ax.set_title("QTT solution of -Laplace(u) = Gaussian, u = 0 on the boundary")
plt.show()
