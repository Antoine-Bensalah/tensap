# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: solving a high-dimensional Poisson equation with a truncated
Richardson iteration, using a Laplacian-like operator in Tucker format.

Problem
-------
We want to solve, on the hypercube (0, L)^d discretized with a standard
1D finite-difference scheme (n interior points per direction), the
discrete Poisson equation

    A x = b

where A is the order-d discrete Laplacian

    A = sum_{mu=1}^{d} I x ... x K x ... x I

(K acting on dimension mu, the identity I on all the other dimensions),
and b is a constant right-hand side (b = 1 everywhere).

Storing x, A and b as plain numpy arrays would require n^d numbers, which
quickly becomes intractable as d grows: this is the curse of dimensionality.
Instead, we represent A, b and every iterate x in Tucker format
(tensap.TuckerTensorOperator / tensap.TuckerTensorVector): each tensor is a
small core contracted with d "frames" (one small factor per dimension), so
the storage grows only linearly with d.

The linear system is solved with a Richardson iteration

    x_{k+1} = x_k + rho * (b - A @ x_k)

Because tensor operations (+, @, ...) tend to increase the Tucker
representation rank at every step, the iterate is truncated (compressed
back to a low rank, up to a prescribed tolerance) after each update: this is
what tensap.Truncator and tensap.TruncatedRichardsonIterations do.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensap

# %% 1D discretization
# We discretize the 1D Laplacian operator -d^2/dx^2 on (0, L) with a
# standard 3-point finite-difference scheme.
L = 1.0
n = 12
h = L / (n + 1)

K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
I = sp.identity(n)

print(f"1D discretization: n = {n} interior points, mesh size h = {h:.4f}")

# %% Construction of the operator and the right-hand side in Tucker format
# The order d of the problem is the number of spatial dimensions.
d = 3

factory = tensap.LaplacianLikeOperatorFactory(K, I, d)
A = factory.make_tucker_tensor()
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

print(f"\nProblem of order d = {d} (n^d = {n**d} unknowns if stored densely)")
print(f"Operator A: {A!r}")
print(f"Right-hand side b: {b!r}")

# %% Choice of the Richardson step size
# Richardson iterations x_{k+1} = x_k + rho*(b - A@x_k) converge if and only
# if 0 < rho < 2 / lambda_max(A). The 1D matrix K has eigenvalues bounded by
# 4/h^2, and A is a sum of d such contributions, so lambda_max(A) <= 4*d/h^2.
# We take rho = 1 / lambda_max(A) as a safe, non-marginal choice.
step_size = h**2 / (4 * d)
print(f"\nRichardson step size rho = {step_size:.3e}")

# %% Truncated Richardson iterations
# At every iteration, the current iterate is compressed back to a low Tucker
# rank (up to a relative tolerance), which keeps the computational cost
# under control as the iterations proceed.
truncator = tensap.Truncator(tolerance=1e-4)

solver = tensap.TruncatedRichardsonIterations(
    A,
    b,
    truncator,
    step_size=step_size,
    max_iterations=1000,
    tolerance=1e-3,
    stagnation=1e-8,
    display=False,
)

print("\nRunning truncated Richardson iterations (no preconditioner)...")
print(
    "Note: without a preconditioner, convergence is slow (the condition "
    "number of the discrete Laplacian grows like n^2); this tutorial keeps "
    "n small to remain fast, and favors clarity over performance."
)
x, output = solver.solve()

print(f"\nStopped after {output['iter']} iterations (flag = {output['flag']}, "
      f"0 = tolerance reached, 1 = max iterations reached, 2 = stagnation).")
print(f"Final relative error ||b - A@x|| / ||b|| = {output['error']:.3e}")
print(f"Final Tucker representation ranks of x: {x.representation_rank()}")

# %% Plot of the convergence history
plt.figure()
plt.semilogy(range(1, len(output["errvec"]) + 1), output["errvec"], marker="o")
plt.xlabel("Iteration")
plt.ylabel("Relative residual error ||b - A@x|| / ||b||")
plt.title("Truncated Richardson iterations for the high-dimensional Poisson equation")
plt.grid(True, which="both")
plt.show()

# %% Sanity check against a direct dense solve
# For such a small problem (n^d unknowns), we can also assemble the dense
# operator and compare: this validates the Tucker-format solution.
K_dense = K.toarray()
I_dense = np.eye(n)
A_dense = (
    np.kron(np.kron(K_dense, I_dense), I_dense)
    + np.kron(np.kron(I_dense, K_dense), I_dense)
    + np.kron(np.kron(I_dense, I_dense), K_dense)
)
b_dense = np.ones(n**d)
x_direct = np.linalg.solve(A_dense, b_dense)

x_tensor = x.full().numpy().ravel()
relative_error = np.linalg.norm(x_tensor - x_direct) / np.linalg.norm(x_direct)
print(
    f"\nRelative difference with the dense direct solve: {relative_error:.3e} "
    "(computed here only because the problem is small enough to be stored "
    "densely; this would be impossible for large d)"
)

# %%
