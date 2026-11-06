"""
Poisson equation in high dimension
==================================

The Poisson equation on the hypercube :math:`(0, 1)^d`, discretized with
50 points per direction, has :math:`50^d` unknowns: :math:`10^{17}` in
dimension 10, :math:`10^{34}` in dimension 20. No computer can store such
a vector. Its solution, however, is close to a tensor of low rank, and the
solvers of tensap compute it with a storage that grows linearly with the
dimension. We study how far we can go, how accurate the result is, and
which tensor format suits it best.

This use case comes from the script
``test/non_regression_scripts/solvers/tutorial_linearsolver_high_dim_poisson.py``,
counterpart of a tutorial of the MATLAB toolbox ApproximationToolbox.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
import tensap
from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor

# %%
# The model
# ---------
#
# The equation :math:`-\Delta u = 1` on :math:`(0, 1)^d` with
# :math:`u = 0` on the boundary, discretized by finite differences with
# :math:`n` interior points per direction, gives the system
#
# .. math::
#
#    A x = b, \qquad A = \sum_{\nu=1}^{d} I \otimes \dots \otimes K
#    \otimes \dots \otimes I, \qquad b = 1 \otimes \dots \otimes 1,
#
# with :math:`K` the 1D second difference matrix (:math:`K` in position
# :math:`\nu`). :math:`A` is a sum of :math:`d` Kronecker products: a
# tensor of canonical rank :math:`d`.
n = 50
h = 1 / (n + 1)
K = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / h**2


def problem(d):
    """The operator in canonical format and the right-hand side."""
    factory = tensap.LaplacianLikeOperatorFactory(K, sp.identity(n), d)
    A = factory.make_canonical_tensor()
    b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)
    return factory, A, b


# %%
# A reference value in any dimension
# ----------------------------------
#
# The residual :math:`\|b - A x\| / \|b\|` measures how well an
# approximation solves the system, but not its error. We check the
# approximations on a quantity of interest that is known in any
# dimension: the mean value of the discrete solution
# :math:`\bar{x} = (b, x) / n^d`. With the eigendecomposition
# :math:`K = Q \Lambda Q^T` and :math:`\hat{f} = Q^T 1`,
#
# .. math::
#
#    \bar{x} = \frac{1}{n^d} \sum_{k_1, \dots, k_d}
#    \frac{\hat{f}_{k_1}^2 \cdots \hat{f}_{k_d}^2}
#         {\lambda_{k_1} + \dots + \lambda_{k_d}}
#    = \frac{1}{n^d} \int_0^\infty \Big( \sum_k \hat{f}_k^2
#      e^{-t \lambda_k} \Big)^d \, dt,
#
# using :math:`1 / \lambda = \int_0^\infty e^{-t \lambda} dt`: a 1D
# integral instead of a sum of :math:`n^d` terms.
eigenvalues, Q = np.linalg.eigh(K.toarray())
f_hat2 = (Q.T @ np.ones(n)) ** 2


def mean_reference(d):
    def integrand(t):
        return np.sum(f_hat2 * np.exp(-t * eigenvalues)) ** d

    value, _ = quad(integrand, 0, np.inf, limit=500, epsrel=1e-12)
    return value / n**d


# %%
# Greedy rank-one corrections
# ---------------------------
#
# :class:`~tensap.GreedyLinearSolver` builds :math:`x` as a sum of
# rank-one tensors :math:`w^{(1)}_m \otimes \dots \otimes w^{(d)}_m`. Each
# correction is computed by alternating least squares on the current
# residual: all the factors but one are fixed, and the remaining one
# solves a system of size :math:`n`. After each correction, all the
# factors are updated by one alternating sweep
# (``update_frames_by_als`` of :class:`~tensap.TuckerTensorALSLinearSolver`).
# The cost of an iteration grows linearly with :math:`d`.
results = {}
for d in (5, 10, 20):
    factory, A, b = problem(d)
    updater = tensap.TuckerTensorALSLinearSolver(
        A, b, max_iterations_frames=1, display=False
    )
    np.random.seed(0)
    start = time()
    x, output = tensap.GreedyLinearSolver(
        A,
        b,
        update=updater.update_frames_by_als,
        max_iterations=15,
        display=False,
    ).solve()
    elapsed = time() - start
    mean = x.dot(b) / n**d
    error = abs(mean / mean_reference(d) - 1)
    results[d] = output
    print(
        f"d = {d:2d}: {float(n) ** d:.0e} unknowns, {elapsed:5.1f} s, "
        f"storage {x.storage()}, residual {output['error']:.1e}, "
        f"error on the mean {error:.1e}"
    )

# %%
# With 15 terms, the residual is of the order of :math:`10^{-3}` in every
# dimension, and the mean value is correct to five to seven digits: it is
# the inner product :math:`(b, x)`, whose error, for a Galerkin
# approximation of a symmetric positive definite problem, is the square of
# the error in the energy norm. The storage, :math:`15\, n\, d` numbers,
# is linear in :math:`d`.
plt.figure()
for d, output in results.items():
    errors = output["errvec"]
    plt.semilogy(
        range(1, len(errors) + 1), errors, "o-", label=f"d = {d}"
    )
plt.xlabel("Number of terms")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()

# %%
# Which format for the solution?
# ------------------------------
#
# A sum of rank-one terms (canonical format) is the most compact format,
# but its convergence with the number of terms is slow: the best
# canonical approximations are hard to find. The greedy solver can also
# keep the frames :math:`\{w^{(\nu)}_1, \dots, w^{(\nu)}_m\}` of the
# corrections and combine them through a richer core: the iterate is then
# a Tucker tensor whose core is a tensor train (TT) or a tree-based
# tensor, and the update solves the Galerkin problem on the whole tensor
# product space spanned by the frames, with the core in that format. We
# compare the three formats in dimension 6, with 12 corrections.
d = 6
factory, A, b = problem(d)

# %%
# The same operator and right-hand side are needed with the core in the
# format of the update: a TT core of ranks 2 for :math:`A`
# (:meth:`~tensap.LaplacianLikeOperatorFactory.make_tt_tucker_tensor`),
# or a tree-based core on a balanced dimension tree.
A_tt = factory.make_tt_tucker_tensor()
b_tt = tensap.TuckerTensorVector(as_tt_tensor(b.core), b.tensor_frame)
A_tree = factory.make_tree_based_tensor()
b_tree = tensap.TuckerTensorVector(
    b.core.tree_based_tensor(A_tree.core.tree), b.tensor_frame
)

variants = {
    "canonical": (
        A,
        b,
        tensap.TuckerTensorALSLinearSolver(
            A, b, max_iterations_frames=1, display=False
        ).update_frames_by_als,
    ),
    "TT core": (
        A,
        b,
        tensap.TuckerTensorALSLinearSolver(
            A_tt,
            b_tt,
            core_algorithm="dmrg",
            max_iterations_core=3,
            display=False,
        ).update_core,
    ),
    "tree-based core": (
        A_tree,
        b_tree,
        tensap.TuckerTensorALSLinearSolver(
            A_tree, b_tree, max_iterations_core=3, display=False
        ).update_core,
    ),
}
outputs = {}
reference = mean_reference(d)
for name, (operator, rhs, update) in variants.items():
    np.random.seed(0)
    start = time()
    x, outputs[name] = tensap.GreedyLinearSolver(
        operator,
        rhs,
        update=update,
        max_iterations=12,
        tolerance=1e-10,
        stagnation=1e-12,
        display=False,
    ).solve()
    error = abs(x.dot(rhs) / n**d / reference - 1)
    print(
        f"{name:16s}: {time() - start:4.1f} s, storage {x.storage():6d}, "
        f"residual {outputs[name]['error']:.1e}, error on the mean "
        f"{error:.1e}"
    )

# %%
# The TT and tree-based cores need a few times more storage than the
# canonical format, but reach a residual five orders of magnitude
# smaller with the same frames: the Galerkin problem on the tensor
# product of the frames is solved, instead of the problem on the
# :math:`m` rank-one terms.
plt.figure()
for name, output in outputs.items():
    errors = output["errvec"]
    plt.semilogy(range(1, len(errors) + 1), errors, "o-", label=name)
plt.xlabel("Number of corrections")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()

# %%
# Discussion
# ----------
#
# - Low-rank formats make the Poisson equation tractable in dimensions
#   where the discrete solution is far beyond the memory of any
#   computer: storage and cost grow linearly with :math:`d`.
# - The residual alone does not tell the error; a quantity of interest
#   known in closed form, here the mean value, confirms the accuracy.
# - The canonical format is the most compact and the cheapest per
#   iteration. When accuracy matters, a Tucker tensor with a TT or
#   tree-based core, updated by Galerkin projections on the frames of the
#   corrections, converges much faster for a moderate increase of the
#   storage.
# - The same solvers apply to any operator given as a sum of Kronecker
#   products, for instance anisotropic or parametric diffusion.
