"""
Alternating solvers in tensor-train format
==========================================

Solve a linear system whose operator is a :class:`~tensap.TTOperator`
and whose right-hand side is a :class:`~tensap.TTTensor` by alternating
sweeps over the cores of the solution: ALS with fixed ranks, DMRG and
AMEn with adapted ranks.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import tensap

np.random.seed(0)

# %%
# The problem
# ^^^^^^^^^^^
# The finite difference Poisson equation on :math:`(0, 1)^6` with
# :math:`n = 16` interior points per direction (:math:`1.7 \times 10^7`
# unknowns): the Laplacian is a TT operator of ranks 2, the right-hand
# side :math:`b = 1 \otimes \dots \otimes 1` a TT tensor of rank 1.
n, d = 16, 6
h = 1 / (n + 1)
K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
factory = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d)
A = factory.make_tt_operator()
b = tensap.TTTensor.ones([n] * d)

# %%
# The solvers
# ^^^^^^^^^^^
# Each sweep updates the cores of :math:`x` one after the other (two at a
# time for DMRG), the other ones being orthogonal: the updated core solves
# the Galerkin projection of the system, which minimizes the energy
# :math:`\frac12 (A x, x) - (b, x)` for a symmetric positive definite
# :math:`A`.
#
# - ALS keeps the ranks of the initial guess (here 4);
# - DMRG adapts the rank of each bond by a truncated SVD of the updated
#   pair of cores;
# - AMEn updates one core at a time and enriches it with a few directions
#   of the residual.
truncator = tensap.Truncator(tolerance=1e-6, max_rank=20)
outputs = {}
for algorithm in ("als", "dmrg", "amen"):
    np.random.seed(0)
    solver = tensap.TTTensorALSLinearSolver(
        A,
        b,
        algorithm=algorithm,
        truncator=truncator,
        ranks=4,
        tolerance=1e-6,
        max_iterations=10,
        display=False,
    )
    x, outputs[algorithm] = solver.solve()
    print(
        f"{algorithm.upper():5s}: {outputs[algorithm]['iter']} sweeps, "
        f"relative residual {outputs[algorithm]['error']:.1e}, "
        f"TT ranks {x.ranks}"
    )

# %%
# Convergence
# ^^^^^^^^^^^
# With fixed ranks, ALS stalls at the accuracy that these ranks allow;
# DMRG and AMEn increase the ranks where the residual requires it.
plt.figure()
for algorithm, output in outputs.items():
    errors = output["errvec"]
    plt.semilogy(
        range(1, len(errors) + 1), errors, "o-", label=algorithm.upper()
    )
plt.xlabel("Sweep")
plt.ylabel("Relative residual")
plt.legend()
plt.grid(True, which="both")
plt.show()
