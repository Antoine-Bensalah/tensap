# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: solving a high-dimensional Poisson equation with low-rank tensor
solvers. Python counterpart of the MATLAB tutorial
tutorial_LinearSolvers_highDimPoissonEq.m.

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

Part 1 - Truncated Richardson iterations (small problem)
--------------------------------------------------------
The linear system is solved with a Richardson iteration

    x_{k+1} = x_k + rho * (b - A @ x_k)

Because tensor operations (+, @, ...) tend to increase the Tucker
representation rank at every step, the iterate is truncated (compressed
back to a low rank, up to a prescribed tolerance) after each update: this is
what tensap.Truncator and tensap.TruncatedRichardsonIterations do.

Part 2 - Greedy solver with rank-one corrections (large problem)
----------------------------------------------------------------
The solution is built by successive rank-one corrections

    x_m = x_{m-1} + lambda_m w_m^(1) x ... x w_m^(d),

each correction being computed by alternating least squares on the current
residual b - A @ x_{m-1} (tensap.GreedyLinearSolver with the default
tensap.RankOneALSLinearSolver). The operator is taken in canonical format
(diagonal core), so that A, x and the residual stay in canonical format and
no truncation is needed.

Part 3 - Greedy solver with updates by alternating least squares
----------------------------------------------------------------
The pure greedy algorithm never revisits a correction. After each correction,
x_m can be improved by a Galerkin update (tensap.TuckerTensorALSLinearSolver):
of its coefficients (update_core), or of all its rank-one factors, one
dimension at a time (update_frames_by_als). The same solver can also
alternate both updates on its own, starting from the greedy solution.

Part 4 - Updates with a TT core
-------------------------------
The iterate is now a Tucker tensor whose core is a tensor train
(TuckerTensorVector with a TTTensor core): after each correction, A and b are
projected on the frames of x, which gives a Galerkin problem on the core
alone, a TTOperator of size R x R per dimension, solved by
tensap.TTTensorALSLinearSolver from the current core: by ALS (the TT ranks of
the core follow the number of corrections) or by DMRG (ranks adapted by
truncated SVDs). The large local systems (R^3 unknowns for ALS) are solved
without assembling their matrix, by the conjugate gradient with a block
Jacobi preconditioner.

Part 5 - Updates with a tree-based core
---------------------------------------
The core is a tree-based tensor on a balanced dimension tree, updated node by
node. The local problem of a node is built by the tree contraction (the
subtrees of its children and the rest of the tree are contracted into small
interfaces), never forming the full core (30^6 entries); the local systems
of more than 2000 unknowns are solved by the conjugate gradient without
assembling their matrix, preconditioned by the inverse of the closest
Kronecker sum (exact here: the Laplacian is a Kronecker sum).
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensap

# %% Part 1 - 1D discretization
# We discretize the 1D Laplacian operator -d^2/dx^2 on (0, L) with a
# standard 3-point finite-difference scheme.
L = 1.0
n = 12
h = L / (n + 1)

K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
Id = sp.identity(n)

print(f"1D discretization: n = {n} interior points, mesh size h = {h:.4f}")

# %% Construction of the operator and the right-hand side in Tucker format
# The order d of the problem is the number of spatial dimensions.
d = 3

factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
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
# acceptance: the true residual, recomputed (observed 1.0e-3, flag 0)
true_residual = (b - A @ x).norm() / b.norm()
assert output["flag"] == 0 and true_residual < 1.1e-3, (output["flag"], true_residual)

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


# %% Part 2 - 1D discretization of a larger problem
# The greedy solver never forms a full core: we can take the sizes of the
# MATLAB tutorial, n^d = 10^12 unknowns.
n = 100
d = 6
h = L / (n + 1)

K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
Id = sp.identity(n)

print(f"\nPart 2: n = {n} interior points, d = {d}, n^d = {float(n)**d:.0e} unknowns")

# %% Operator in canonical format, right-hand side of rank one
# Canonical format: A = sum_k A^(1)_k x ... x A^(d)_k with A^(mu)_k = K if
# k == mu, I otherwise, i.e. a Tucker tensor with a diagonal core of ones
# and rank d in every dimension.
factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
A_can = factory.make_canonical_tensor()

ones_frames = tensap.TFrameVector(
    [tensap.FrameVector(np.ones((1, n, 1))) for _ in range(d)]
)
b = tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0]), d), ones_frames)

print(f"Operator A: {A_can!r}")
print(f"Right-hand side b: {b!r}")

# %% Greedy solver
# At iteration m, a rank-one correction c_m is computed by alternating least
# squares: all the factors but one are fixed, and the Galerkin condition gives
# a linear system of size n for the remaining factor. Then x_m = x_{m-1} + c_m.
np.random.seed(0)
solver = tensap.GreedyLinearSolver(A_can, b, max_iterations=30, display=True)
x_can, output_can = solver.solve()

print(f"\nStopped after {output_can['iter']} iterations (flag = {output_can['flag']}).")
print(f"Final relative error ||b - A@x|| / ||b|| = {output_can['error']:.3e}")
# acceptance (observed 4.0e-2 after 30 corrections, flag 1: the pure greedy
# solver is slow, the updates below are the remedy)
assert output_can["error"] < 0.1, output_can["error"]
print(f"Canonical rank of x: {x_can.representation_rank()[0]}")
print(f"Storage of x: {x_can.storage()} numbers instead of n^d = {float(n)**d:.0e}")

# %% Convergence of the greedy solver
plt.figure()
plt.semilogy(
    range(1, len(output_can["errvec"]) + 1), output_can["errvec"], marker="o"
)
plt.xlabel("Greedy iteration (rank of x)")
plt.ylabel("Relative residual error ||b - A@x|| / ||b||")
plt.title("Greedy rank-one corrections for the high-dimensional Poisson equation")
plt.grid(True, which="both")
plt.show()


# %% Part 3 - Greedy solver with update of the core
# After each correction, the coefficients sigma of
# x_m = sum_i sigma_i w_i^(1) x ... x w_i^(d) are recomputed by the Galerkin
# system G sigma = g, with G(j, i) = <A x_i, x_j> and g(j) = <b, x_j>.
updater = tensap.TuckerTensorALSLinearSolver(A_can, b, display=False)

np.random.seed(0)
solver = tensap.GreedyLinearSolver(
    A_can, b, max_iterations=30, update=updater.update_core, display=False
)
x_core, output_core = solver.solve()
print(f"\nGreedy + core update: relative error {output_core['error']:.3e}")

# %% Greedy solver with update of the frames
# After each correction, all the factors w_i^(mu) of a dimension mu are
# recomputed at once (Galerkin system of size n * rank), for mu = 1, ..., d,
# in at most 3 sweeps.
updater.max_iterations_frames = 3

np.random.seed(0)
solver = tensap.GreedyLinearSolver(
    A_can, b, max_iterations=30, update=updater.update_frames_by_als, display=False
)
x_frames, output_frames = solver.solve()
print(f"Greedy + frames update: relative error {output_frames['error']:.3e}")

# %% Alternating least squares from the pure greedy solution
# Starting from the rank-30 solution of Part 2, alternate core and frames
# updates; the rank stays 30.
solver = tensap.TuckerTensorALSLinearSolver(
    A_can, b, x0=x_can, max_iterations=5, max_iterations_frames=2, display=True
)
x_als, output_als = solver.solve()
print(f"ALS from the greedy solution: relative error {output_als['error']:.3e}")
# acceptance (observed: core 2.1e-2, frames 3.1e-4, ALS 3.8e-3)
for name, result, threshold in (("core", output_core, 0.05), ("frames", output_frames, 1e-3),
                                ("ALS", output_als, 1e-2)):
    assert result["error"] < min(threshold, output_can["error"]), (name, result["error"])

# %% Comparison of the greedy variants
plt.figure()
for output, label in (
    (output_can, "pure greedy"),
    (output_core, "greedy + core update"),
    (output_frames, "greedy + frames update"),
):
    plt.semilogy(range(1, len(output["errvec"]) + 1), output["errvec"], label=label)
plt.xlabel("Greedy iteration (rank of x)")
plt.ylabel("Relative residual error ||b - A@x|| / ||b||")
plt.title("Greedy rank-one corrections, with and without updates")
plt.legend()
plt.grid(True, which="both")
plt.show()


# %% Part 4 - Updates with a TT core
# The operator in Tucker format with a TT core (frames {K, I}, TT core of
# ranks 2) and the right-hand side with its rank-one core seen as a TT tensor.
# The greedy corrections are computed with the canonical operator; the
# updates use the TT one (same operator, other format).
from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor  # noqa: E402

A_tt = factory.make_tt_tucker_tensor()
b_tt = tensap.TuckerTensorVector(as_tt_tensor(b.core), b.tensor_frame)
outputs_tt = {}
for algorithm, iterations in (("als", 5), ("dmrg", 8)):
    updater = tensap.TuckerTensorALSLinearSolver(
        A_tt, b_tt, max_iterations_core=iterations, core_algorithm=algorithm,
        display=False
    )
    np.random.seed(0)
    solver = tensap.GreedyLinearSolver(
        A_can, b, max_iterations=30, update=updater.update_core, display=False
    )
    x_tt, outputs_tt[algorithm] = solver.solve()
    print(f"Greedy + TT core update by {algorithm.upper()}: relative error "
          f"{outputs_tt[algorithm]['error']:.3e} after {outputs_tt[algorithm]['iter']} "
          f"iterations, TT ranks of the core {x_tt.core.ranks}, storage {x_tt.storage()}")
    # acceptance (observed 1.4e-5 for ALS and DMRG)
    assert outputs_tt[algorithm]["error"] < min(1e-4, output_can["error"])

# %% Comparison with the updates of Part 3
plt.figure()
for output, label in (
    (output_can, "pure greedy"),
    (output_core, "greedy + core update (canonical)"),
    (outputs_tt["als"], "greedy + TT core update (ALS)"),
    (outputs_tt["dmrg"], "greedy + TT core update (DMRG)"),
):
    plt.semilogy(range(1, len(output["errvec"]) + 1), output["errvec"], label=label)
plt.xlabel("Greedy iteration")
plt.ylabel("Relative residual error ||b - A@x|| / ||b||")
plt.title("Greedy corrections with updates of a canonical or TT core")
plt.legend()
plt.grid(True, which="both")
plt.show()


# %% Part 5 - Updates with a tree-based core
# Operator with a tree-based core on the balanced tree (frames {K, I}),
# right-hand side with its rank-one core as a tree-based tensor. As in
# MATLAB, the greedy solver uses this operator for the corrections too, with
# 30 iterations and tolerances of 1e-10.
A_tree = factory.make_tree_based_tensor()
b_tree = tensap.TuckerTensorVector(b.core.tree_based_tensor(A_tree.core.tree), b.tensor_frame)
updater = tensap.TuckerTensorALSLinearSolver(A_tree, b_tree, max_iterations_core=5,
                                             display=False)
np.random.seed(0)
solver = tensap.GreedyLinearSolver(
    A_tree, b_tree, max_iterations=30, tolerance=1e-10, stagnation=1e-10,
    update=updater.update_core, display=False
)
x_tree, output_tree = solver.solve()
print(f"Greedy + tree-based core update: relative error {output_tree['error']:.3e} after "
      f"{output_tree['iter']} iterations, tree ranks {x_tree.core.ranks[x_tree.core.ranks > 0]},"
      f" storage {x_tree.storage()}")
assert output_tree["error"] < 1e-6

# %% Comparison of the core updates
plt.figure()
for output, label in (
    (output_can, "pure greedy"),
    (output_core, "canonical core update"),
    (outputs_tt["als"], "TT core update (ALS)"),
    (outputs_tt["dmrg"], "TT core update (DMRG)"),
    (output_tree, "tree-based core update"),
):
    plt.semilogy(range(1, len(output["errvec"]) + 1), output["errvec"], label=label)
plt.xlabel("Greedy iteration")
plt.ylabel("Relative residual error ||b - A@x|| / ||b||")
plt.title("Greedy corrections with updates of the core, d = 6")
plt.legend()
plt.grid(True, which="both")
plt.show()
