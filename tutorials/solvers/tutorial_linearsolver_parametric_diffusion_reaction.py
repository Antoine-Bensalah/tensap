# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: a parametric diffusion-reaction equation solved in canonical format.
Python counterpart of the MATLAB tutorial
tutorial_LinearSolvers_parametricDiffusionReactionEq.m.

Problem
-------
For Q values xi_1, ..., xi_Q of a reaction coefficient xi in [0, 50], solve

    - Delta u + xi u = 1 on (0, 1)^2,   u = 0 on the boundary,

by finite differences with N x N interior points. All the parameters are
solved at once: the solution is a tensor u in R^{N^2} x R^Q, with
u(:, q) the solution for xi_q, and

    A u = f,   A = Ax x I_Q + Bx x diag(xi_1, ..., xi_Q),   f = fx x 1_Q,

with Ax the 2D Laplacian, Bx the identity (mass) and fx = 1. A has a
diagonal core of rank 2 (canonical format), f is of rank one.

The tutorial compares

1. truncated Richardson iterations, preconditioned by the inverse of the
   operator for the mean value of xi (P = (Ax + xi_mean Bx)^{-1} x I_Q);
2. the greedy solver with rank-one corrections;
3. the ALS solver in Tucker format, from a greedy approximation of rank 2;

and checks each solution against a direct solve for every xi_q.
"""

# %% Imports
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tensap

np.random.seed(0)

# %% Spatial dimension: 2D finite difference Laplacian
L = 1.0
N = 10  # interior points per direction
N2 = N**2
h = L / (N + 1)
T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), dtype=float)
Ax = (1 / h**2) * (sp.kron(sp.identity(N), T) + sp.kron(T, sp.identity(N)))
Ax = Ax.toarray()  # MATLAB gallery('poisson', N) / h^2
Bx = np.eye(N2)
fx = np.ones(N2)

# %% Parametric dimension: Q random values of xi in [xi_min, xi_max]
Q = 200
xi_min, xi_max = 0.0, 50.0
Xi = xi_min + (xi_max - xi_min) * np.random.rand(Q)
Axi = np.eye(Q)
Bxi = np.diag(Xi)
fxi = np.ones(Q)

# %% Operator and right-hand side in canonical format (diagonal cores)
# A = Ax x Axi + Bx x Bxi: frames {Ax, Bx} and {Axi, Bxi}, core (1, 1)
d = 2
A = tensap.TuckerTensorOperator(
    tensap.DiagonalTensor(np.array([1.0, 1.0]), d),
    tensap.TFrameOperator([
        tensap.FrameOperator(np.stack([Ax, Bx])),
        tensap.FrameOperator(np.stack([Axi, Bxi])),
    ]),
)
f = tensap.TuckerTensorVector(
    tensap.DiagonalTensor(np.array([1.0]), d),
    tensap.TFrameVector([
        tensap.FrameVector(fx.reshape(1, N2, 1)),
        tensap.FrameVector(fxi.reshape(1, Q, 1)),
    ]),
)

# %% Reference: one direct solve per value of xi
u_ref = np.stack([np.linalg.solve(Ax + xi * Bx, fx) for xi in Xi], axis=1)


def relative_error(u):
    """Relative error over all the parameters, in Frobenius norm."""
    return np.linalg.norm(u.numpy() - u_ref) / np.linalg.norm(u_ref)


# %% Preconditioner: inverse of the operator for the mean value of xi
# (MATLAB: ImplicitMatrix.luInverse; a dense inverse at this size)
xi_mean = (xi_min + xi_max) / 2
P_xi = np.linalg.inv(Ax + xi_mean * Bx)
P = tensap.TuckerTensorOperator(
    tensap.DiagonalTensor(np.array([1.0]), d),
    tensap.TFrameOperator([
        tensap.FrameOperator(P_xi[None]),
        tensap.FrameOperator(np.eye(Q)[None]),
    ]),
)

# %% 1. Truncated Richardson iterations, preconditioned
# At each iteration, x + P (f - A x) has a diagonal core; it is truncated by
# a truncated SVD of order 2, which gives again a diagonal core.
print("TRUNCATED RICHARDSON ITERATIONS")
truncator = tensap.Truncator(tolerance=1e-10)
s = tensap.TruncatedRichardsonIterations(A, f, truncator, P=P, max_iterations=100)
u_richardson, output = s.solve()
error_richardson = relative_error(u_richardson)
print(f"Rank {u_richardson.representation_rank()}, "
      f"relative error w.r.t. the direct solves: {error_richardson:.2e}")

# %% 2. Greedy rank-one approximation
print("GREEDY RANK ONE APPROXIMATION")
s = tensap.GreedyLinearSolver(A, f, max_iterations=20, tolerance=1e-13, stagnation=1e-13)
u_greedy, output = s.solve()
error_greedy = relative_error(u_greedy)
print(f"Rank {u_greedy.representation_rank()}, "
      f"relative error w.r.t. the direct solves: {error_greedy:.2e}")

# %% 3. ALS in Tucker format, from a greedy approximation of rank 2
print("ALS IN TUCKER FORMAT")
s = tensap.GreedyLinearSolver(A, f, max_iterations=2, display=False)
u0, _ = s.solve()
s = tensap.TuckerTensorALSLinearSolver(A, f, max_iterations=10, x0=u0)
u_als, output = s.solve()
error_als = relative_error(u_als)
print(f"Rank {u_als.representation_rank()}, "
      f"relative error w.r.t. the direct solves: {error_als:.2e}")

# %% Self-check against the direct solves
# Richardson and the greedy solver stop on the relative residual (1e-6 and
# 1e-13); the ALS solution is of rank (2, 2) only.
assert error_richardson < 1e-5, error_richardson
assert error_greedy < 1e-10, error_greedy
assert error_als < 1e-2, error_als

# %% Solution for one value of xi, from the greedy approximation
q = np.random.randint(Q)
e_q = np.zeros(Q)
e_q[q] = 1.0
u_q = u_greedy.tensor_vector_product(e_q, [1]).numpy()
print(f"xi = {Xi[q]:.2f}: relative error "
      f"{np.linalg.norm(u_q - u_ref[:, q]) / np.linalg.norm(u_ref[:, q]):.2e}")

# add the boundary values u = 0
u_plot = np.zeros((N + 2, N + 2))
u_plot[1:-1, 1:-1] = u_q.reshape(N, N)
X, Y = np.meshgrid(np.linspace(0, L, N + 2), np.linspace(0, L, N + 2))
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, u_plot)
ax.set_title(f"Solution for xi = {Xi[q]:.2f}")
plt.show()
