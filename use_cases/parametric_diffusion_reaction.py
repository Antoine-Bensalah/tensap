"""
Diffusion and reaction with an uncertain reaction rate
======================================================

A species diffuses in a square domain, is produced uniformly and consumed
by a reaction whose rate is uncertain. How does the concentration depend
on the rate, and what is the distribution of the mean concentration?
Instead of solving the problem once per value of the rate, we solve it
for all the values at once, as a single linear system whose solution is a
low-rank tensor, with the solvers of tensap.

This use case comes from the script
``test/non_regression_scripts/solvers/tutorial_linearsolver_parametric_diffusion_reaction.py``,
counterpart of a tutorial of the MATLAB toolbox ApproximationToolbox.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tensap

np.random.seed(0)

# %%
# The model
# ---------
#
# The concentration :math:`u` solves, on :math:`D = (0, 1)^2`,
#
# .. math::
#
#    -\Delta u + \xi u = 1 \quad \text{in } D, \qquad
#    u = 0 \quad \text{on } \partial D,
#
# where the reaction rate :math:`\xi` is uncertain, uniformly distributed
# on :math:`[0, 50]`. The finite difference discretization with
# :math:`N \times N` interior points gives, for each value of
# :math:`\xi`, the system :math:`(A_x + \xi B_x) u(\xi) = f_x`, with
# :math:`A_x` the 2D Laplacian, :math:`B_x` the identity and :math:`f_x`
# the vector of ones.
N = 30
h = 1 / (N + 1)
T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)) / h**2
Ax = (sp.kron(sp.identity(N), T) + sp.kron(T, sp.identity(N))).tocsc()
Bx = sp.identity(N**2, format="csc")
fx = np.ones(N**2)

# %%
# We study the problem on a sample of :math:`Q = 300` values of the rate.
Q = 300
xi = 50 * np.random.rand(Q)

# %%
# One solve per value of the rate
# -------------------------------
#
# The reference: :math:`Q` sparse direct solves. This is affordable here,
# and gives the error of the low-rank approximations below.
start = time()
u_ref = np.stack([spla.spsolve(Ax + x * Bx, fx) for x in xi], axis=1)
print(f"{Q} direct solves: {time() - start:.2f} s")

# %%
# All the values at once
# ----------------------
#
# The solutions for all the values of :math:`\xi` form a matrix
# :math:`U = [u(\xi_1), \dots, u(\xi_Q)]`, a tensor of order 2 in
# :math:`\mathbb{R}^{N^2} \otimes \mathbb{R}^{Q}`. It solves a single
# linear system,
#
# .. math::
#
#    \mathcal{A} U = F, \qquad
#    \mathcal{A} = A_x \otimes I_Q + B_x \otimes \operatorname{diag}(\xi),
#    \qquad F = f_x \otimes 1_Q,
#
# whose operator is in canonical format (a Tucker tensor with a diagonal
# core) with two terms, and whose right-hand side is of rank one. If
# :math:`u(\xi)` depends smoothly on :math:`\xi`, :math:`U` is close to a
# matrix of small rank: a few functions of space and a few functions of
# the rate suffice.
d = 2
A = tensap.TuckerTensorOperator(
    tensap.DiagonalTensor(np.array([1.0, 1.0]), d),
    tensap.TFrameOperator(
        [
            tensap.FrameOperator(
                np.stack([Ax.toarray(), Bx.toarray()])
            ),
            tensap.FrameOperator(np.stack([np.eye(Q), np.diag(xi)])),
        ]
    ),
)
F = tensap.TuckerTensorVector(
    tensap.DiagonalTensor(np.array([1.0]), d),
    tensap.TFrameVector(
        [
            tensap.FrameVector(fx.reshape(1, -1, 1)),
            tensap.FrameVector(np.ones((1, Q, 1))),
        ]
    ),
)

# %%
# Greedy rank-one corrections
# ---------------------------
#
# :class:`~tensap.GreedyLinearSolver` builds :math:`U` as a sum of
# rank-one terms :math:`\sum_m v_m \otimes w_m`, a function of space
# times a function of the rate, each one computed by alternating least
# squares on the current residual. After each correction, the iterate is
# improved by :class:`~tensap.TuckerTensorALSLinearSolver`, in two
# ways:
#
# - update of the coefficients of the terms only (``update_core``), a
#   Galerkin problem of size :math:`m`;
# - one alternating sweep: the coefficients, then all the functions of
#   space, then all the functions of the rate.
#
# Both stop when the relative residual is below :math:`10^{-7}`.
updater = tensap.TuckerTensorALSLinearSolver(
    A, F, max_iterations=1, max_iterations_frames=1, display=False
)
variants = {
    "update of the coefficients": updater.update_core,
    "alternating sweep": lambda x: updater.solve(x)[0],
}
outputs, solutions = {}, {}
for name, update in variants.items():
    np.random.seed(0)
    start = time()
    solutions[name], outputs[name] = tensap.GreedyLinearSolver(
        A,
        F,
        update=update,
        max_iterations=15,
        tolerance=1e-7,
        stagnation=1e-12,
        display=False,
    ).solve()
    error = np.linalg.norm(
        solutions[name].numpy() - u_ref
    ) / np.linalg.norm(u_ref)
    print(
        f"{name}: {time() - start:.2f} s, "
        f"rank {solutions[name].representation_rank()[0]}, "
        f"relative error {error:.1e}"
    )
U = solutions["alternating sweep"]

# %%
# How many terms are needed?
# --------------------------
#
# The singular values of the reference matrix decay exponentially: the
# best approximation of rank :math:`m` has an error of the order of the
# :math:`(m+1)`-th singular value, and five terms reach
# :math:`10^{-7}`. With the alternating sweeps, the greedy solver
# follows this decay; with the update of the coefficients only, the
# functions computed by the first corrections are never corrected, and
# the convergence is much slower.
singular_values = np.linalg.svd(u_ref, compute_uv=False)
plt.figure()
plt.semilogy(
    range(1, 16),
    singular_values[1:16] / singular_values[0],
    "k--",
    label="best approximation (singular values)",
)
for name, output in outputs.items():
    errors = output["errvec"]
    plt.semilogy(range(1, len(errors) + 1), errors, "o-", label=name)
plt.xlabel("Rank")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()

# %%
# The functions of space and of the rate
# --------------------------------------
#
# The first terms of the approximation: the functions of space
# :math:`v_m`, reshaped on the grid, and the functions of the rate
# :math:`w_m`, at the sampled values.
frames = U.orth().tensor_frame.frames
space, rate = frames[0].data[:, :, 0], frames[1].data[:, :, 0]
order = np.argsort(xi)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for m in range(3):
    axes[m].imshow(space[m].reshape(N, N), origin="lower")
    axes[m].set_title(f"space function {m + 1}")
    axes[m].axis("off")
    axes[3].plot(xi[order], rate[m][order], label=f"{m + 1}")
axes[3].set_xlabel(r"$\xi$")
axes[3].set_title("rate functions")
axes[3].legend()
plt.tight_layout()
plt.show()

# %%
# The mean concentration
# ----------------------
#
# The quantity of interest is the mean concentration
# :math:`q(\xi) = \frac{1}{|D|} \int_D u(\xi)`, approximated by the mean
# of the values at the grid points. On the low-rank approximation,
# :math:`q(\xi_k) = \sum_m \bar{v}_m\, w_m(\xi_k)`, with :math:`\bar{v}_m`
# the mean of the function of space :math:`v_m`: the product of the
# tensor by a vector in the first dimension, which costs one inner
# product per term.
q = np.ravel(U.tensor_vector_product(np.ones(N**2) / N**2, [0]).numpy())
q_ref = np.mean(u_ref, axis=0)
print(
    f"mean concentration: max relative error "
    f"{np.max(np.abs(q - q_ref) / q_ref):.1e}"
)

# %%
# The mean concentration is far more accurate than the concentration
# itself. Summed over the values of the rate, it is, up to a factor, the
# inner product :math:`(F, U)` of the solution with the right-hand
# side; the updates are Galerkin projections of a symmetric positive
# definite problem, and the error on this inner product is the square of
# the error in the energy norm. Here the error is of the same order for
# each value of the rate.
print(
    f"mean {q.mean():.4f}, standard deviation {q.std():.4f}, "
    f"range [{q.min():.4f}, {q.max():.4f}]"
)
plt.figure()
plt.hist(q, bins=30, density=True)
plt.xlabel("mean concentration")
plt.ylabel("density")
plt.show()

# %%
# Discussion
# ----------
#
# - The solution depends smoothly on the rate: five rank-one terms
#   represent the :math:`Q` solutions to a relative accuracy
#   :math:`10^{-7}`, with :math:`m (N^2 + Q)` numbers instead of
#   :math:`N^2 Q`.
# - The greedy solver is close to the best approximation only if the
#   terms already computed are corrected: the alternating sweeps cost
#   more per correction, but need three times fewer terms.
# - Corrections beyond the rank needed are redundant: in canonical format
#   the terms cannot be orthonormalized, and once they become numerically
#   linearly dependent the alternating sweeps cannot update the functions
#   any more (tensap then skips the update with a warning). The tolerance
#   of the greedy solver should match the accuracy that is needed; a
#   Tucker tensor with a structured core does not have this limitation.
# - The solvers need operations on the factors only: products by
#   :math:`A_x`, :math:`B_x` and diagonal matrices, and small Galerkin
#   problems. Here the direct solves are as fast, because the problem is
#   small; with a large spatial discretization or many values of the
#   parameter, the low-rank solve wins, since its cost grows like
#   :math:`N^2 + Q` and not like :math:`N^2 Q`.
# - With several uncertain parameters, the solution becomes a tensor of
#   higher order, and the same solvers apply in Tucker or tensor-train
#   format.
