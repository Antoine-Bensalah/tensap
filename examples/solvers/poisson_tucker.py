"""
Poisson equation in Tucker format
=================================

Solve the discrete Poisson equation on the cube with the iterative solvers
of tensap: truncated Richardson iterations and truncated preconditioned
conjugate gradient, then the greedy solver with rank-one corrections.

This example comes from the script
``test/non_regression_scripts/solvers/tutorial_linearsolver_tucker_poisson.py``.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensap

np.random.seed(0)

# %%
# The problem
# ^^^^^^^^^^^
# On :math:`(0, 1)^3`, with :math:`n` interior points per direction, the
# finite difference Poisson equation :math:`A x = b`, with
# :math:`A = K \otimes I \otimes I + I \otimes K \otimes I
# + I \otimes I \otimes K` and :math:`b = 1 \otimes 1 \otimes 1`. The
# operator is a Tucker tensor with frames :math:`\{K, I\}`, the
# right-hand side a Tucker tensor of rank one.
n, d = 50, 3
h = 1 / (n + 1)
K = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / h**2
factory = tensap.LaplacianLikeOperatorFactory(K, sp.identity(n), d)
A = factory.make_tucker_tensor()
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

# %%
# Reference solution
# ^^^^^^^^^^^^^^^^^^
# The Laplacian is diagonalized by the eigenvectors :math:`Q` of
# :math:`K`, which gives the solution with :math:`n^3` divisions.
eigenvalues, Q = np.linalg.eigh(K.toarray())
f_hat = Q.T @ np.ones(n)
denominator = eigenvalues[:, None, None] + eigenvalues[None, :, None]
denominator = denominator + eigenvalues[None, None, :]
u_hat = np.einsum("i,j,k->ijk", f_hat, f_hat, f_hat) / denominator
u_reference = np.einsum(
    "ai,bj,ck,ijk->abc", Q, Q, Q, u_hat, optimize=True
)


def error(x):
    """Relative error of x against the reference solution."""
    return np.linalg.norm(x.numpy() - u_reference) / np.linalg.norm(
        u_reference
    )


# %%
# A preconditioner in Tucker format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.GreedyInverseApproximator` builds an approximate
# inverse :math:`P = \sum_k W_k` of :math:`A`, each term :math:`W_k` being
# a Kronecker product of matrices computed by alternating least squares.
P, _ = tensap.GreedyInverseApproximator(
    A, max_iterations=8, display=False
).solve()

# %%
# Truncated iterations
# ^^^^^^^^^^^^^^^^^^^^
# Each iteration increases the ranks of the iterate, which is truncated by
# a :class:`~tensap.Truncator` (relative precision :math:`10^{-6}` by
# default). The solvers stop when the relative residual is below their
# tolerance.
truncator = tensap.Truncator()
outputs = {}
for name, solver in [
    ("Richardson", tensap.TruncatedRichardsonIterations),
    ("PCG", tensap.TruncatedPCG),
]:
    x, outputs[name] = solver(
        A, b, truncator, P=P, display=False
    ).solve()
    print(
        f"{name:10s}: {outputs[name]['iter']} iterations, error "
        f"{error(x):.1e}, rank {x.representation_rank()}"
    )

# %%
# Greedy rank-one corrections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~tensap.GreedyLinearSolver` builds the solution as a sum of
# rank-one tensors, each one computed by alternating least squares
# (:class:`~tensap.RankOneALSLinearSolver`) on the current residual. No
# truncation is needed: the iterate is in canonical format.
x, outputs["greedy"] = tensap.GreedyLinearSolver(
    A, b, max_iterations=20, display=False
).solve()
print(
    f"greedy    : {outputs['greedy']['iter']} corrections, error "
    f"{error(x):.1e}"
)

# %%
# Convergence
# ^^^^^^^^^^^
plt.figure()
for name, output in outputs.items():
    errors = output["errvec"]
    plt.semilogy(range(1, len(errors) + 1), errors, "o-", label=name)
plt.xlabel("Iteration")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()
