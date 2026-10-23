# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: solvers for a 3D Poisson equation in Tucker format. Python
counterpart of the MATLAB tutorial tutorial_LinearSolvers_tuckerPoissonEq.m.

Problem
-------
On (0, 1)^d, d = 3, with n = 50 interior points per direction, solve the
finite difference Poisson equation

    A x = b,   A = sum_{mu=1}^{d} I x ... x K x ... x I,   b = 1 x ... x 1,

with A in Tucker format (rank 2 per dimension, full 2 x ... x 2 core).

The tutorial compares several solvers:

1. a preconditioner P ~ A^{-1}, sum of rank-one operators computed by a greedy
   algorithm (tensap.GreedyInverseApproximator);
2. truncated Richardson iterations preconditioned by P;
3. truncated preconditioned conjugate gradient (tensap.TruncatedPCG);
4. a rank-one approximation of the solution by alternating least squares;
5. the greedy solver with rank-one corrections, in three variants.
"""

# %% Imports
import numpy as np
import scipy.sparse as sp
import tensap

np.random.seed(0)

# %% 1D discretization
L = 1.0
n = 50
h = L / (n + 1)
K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
Id = sp.identity(n)
f = np.ones(n)
d = 3

# %% Operator and right-hand side, built by hand
# Frames {K, I} in every dimension; the core selects K in exactly one
# dimension: C(1, 2, 2) = C(2, 1, 2) = C(2, 2, 1) = 1 (MATLAB indices).
frames = tensap.TFrameOperator(
    [tensap.FrameOperator(np.stack([K.toarray(), Id.toarray()])) for _ in range(d)]
)
core = np.zeros((2, 2, 2))
core[0, 1, 1] = 1
core[1, 0, 1] = 1
core[1, 1, 0] = 1
A = tensap.TuckerTensorOperator(tensap.FullTensor(core), frames)

b_frames = tensap.TFrameVector(
    [tensap.FrameVector(f.reshape(1, n, 1)) for _ in range(d)]
)
b = tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0]), d), b_frames)

# %% The same operator and right-hand side, with the factory
factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
A = factory.make_tucker_tensor()
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

T = tensap.Truncator()

# %% Preconditioner
# P = sum_k W_k, each W_k = a v^(1) x v^(2) x v^(3) minimizing
# 1/2 <P A, P> - <I, P> with the previous terms fixed (alternating least
# squares on the matrices v^(mu)).
print("PRECONDITIONER")
precond = tensap.GreedyInverseApproximator(A, max_iterations=8)
P, output = precond.solve()

# %% Truncated Richardson iterations, preconditioned
print("TRUNCATED RICHARDSON ITERATIONS")
s = tensap.TruncatedRichardsonIterations(A, b, T, P=P)
u_richardson, output = s.solve()

# %% Truncated PCG, preconditioned
print("TRUNCATED PCG")
s = tensap.TruncatedPCG(A, b, T, P=P)
u_pcg, output = s.solve()

# %% Rank-one approximation
print("RANK ONE APPROXIMATION")
s = tensap.RankOneALSLinearSolver(A, b)
u_r1, output = s.solve()

# %% Greedy rank-one approximation
print("GREEDY RANK ONE APPROXIMATION")
s = tensap.GreedyLinearSolver(A, b)
u_greedy, output = s.solve()

# %% Greedy rank-one approximation, residual minimization
# The corrections solve the normal equations A^T A x = A^T b.
print("GREEDY RANK ONE APPROXIMATION, RESIDUAL MINIMIZATION")
local_solver = tensap.RankOneALSLinearSolver(A.T @ A, A.T @ b, display=False)
s = tensap.GreedyLinearSolver(A, b, minimize_residual=True, local_solver=local_solver)
u_greedy, output = s.solve()

# %% Greedy rank-one approximation, residual computed every 5 iterations
print("GREEDY RANK ONE APPROXIMATION, COMPUTATION OF THE RESIDUAL EVERY 5 ITERATIONS")
s = tensap.GreedyLinearSolver(A, b, check_residual=5)
u_5, output = s.solve()

# %%
