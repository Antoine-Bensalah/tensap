r"""
Greedy solver with updates
==========================

Improve the iterates of the greedy solver by Galerkin updates of their
core, or of their core and their frames, with
:class:`~tensap.TuckerTensorALSLinearSolver`.

We solve :math:`A u = b`, with
:math:`A = \sum_{\nu=1}^{d} I \otimes \dots \otimes K \otimes \dots
\otimes I` the discrete Laplacian in dimension :math:`d = 4`, by a sum of
rank-one corrections :math:`u^m = \sum_{k \le m} c_k\, w^{(1)}_k \otimes
\dots \otimes w^{(4)}_k`. We compare the residual reached at a given rank
with and without updates of the previous terms. See
:ref:`theory-greedy` and :ref:`theory-galerkin-updates`.

This example comes from the script
``test/non_regression_scripts/solvers/tutorial_linearsolver_update_poisson_solution.py``.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensap

# %%
# The problem
# ^^^^^^^^^^^
# The finite difference Poisson equation on :math:`(0, 1)^4` with
# :math:`n = 30` interior points per direction, the operator in
# canonical format (a diagonal core) and the right-hand side
# :math:`b = 1 \otimes \dots \otimes 1`. Here
# :math:`K = h^{-2}\, \mathrm{tridiag}(-1, 2, -1)` is the 1D Laplacian
# of size :math:`n`, with :math:`h = 1/(n+1)`, so that :math:`A` has
# :math:`n^4 \approx 8 \times 10^5` rows but is stored through the two
# matrices :math:`K` and :math:`I` of size :math:`n`.
n, d = 30, 4
h = 1 / (n + 1)
K = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / h**2
factory = tensap.LaplacianLikeOperatorFactory(K, sp.identity(n), d)
A = factory.make_canonical_tensor()
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

# %%
# Three variants of the greedy solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# At step :math:`m`, the greedy solver computes by alternating least
# squares a rank-one tensor :math:`w_m` minimizing the energy
#
# .. math::
#
#    J(u^{m-1} + w), \qquad J(v) = \tfrac12 (A v, v) - (b, v),
#
# which is the best rank-one improvement of the current iterate. The new
# iterate :math:`u^m = u^{m-1} + w_m` of rank :math:`m` can be kept as it
# is (pure greedy algorithm), or improved, without changing its rank:
#
# - ``update_core``: the coefficients of the :math:`m` rank-one terms solve
#   the Galerkin problem on the space they span, a system of size
#   :math:`m`;
# - one alternating sweep (``solve`` with one iteration): update of the
#   core, then of the frames, one dimension at a time.
#
# Each update is a Galerkin projection, so the energy can only decrease.
# The updates are passed to :class:`~tensap.GreedyLinearSolver` as a
# function of the iterate; the same random seed gives the same first
# correction to the three variants.
updater = tensap.TuckerTensorALSLinearSolver(
    A, b, max_iterations=1, max_iterations_frames=1, display=False
)
variants = {
    "pure greedy": None,
    "update of the core": updater.update_core,
    "core and frames": lambda x: updater.solve(x)[0],
}
outputs = {}
for name, update in variants.items():
    np.random.seed(0)
    solver = tensap.GreedyLinearSolver(
        A, b, update=update, max_iterations=15, display=False
    )
    x, outputs[name] = solver.solve()
    print(f"{name:20s}: relative residual {outputs[name]['error']:.1e}")

# %%
# Convergence
# ^^^^^^^^^^^
# The updates cost a Galerkin solve per correction, but reach a given
# residual with fewer terms. The pure greedy terms are never revised, so
# the first ones, computed with little information, stay in the sum; the
# update of the core recomputes their coefficients, and the update of the
# frames also corrects their factors, which gives the lowest residual of
# the three at the final rank.
plt.figure()
for name, output in outputs.items():
    errors = output["errvec"]
    plt.semilogy(range(1, len(errors) + 1), errors, "o-", label=name)
plt.xlabel("Rank of the iterate")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()
