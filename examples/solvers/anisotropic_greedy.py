"""
Greedy construction of anisotropic tensor subspaces
===================================================

Solve an anisotropic diffusion equation with
:class:`~tensap.AnisotropicGreedySpaceLinearSolver`: the solution is
sought in Tucker format with a TT core, on subspaces
:math:`U_1 \\otimes \\dots \\otimes U_d` enriched greedily, only in the
dimensions where the approximation needs it.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import tensap

np.random.seed(0)

# %%
# An anisotropic problem
# ^^^^^^^^^^^^^^^^^^^^^^
# The operator :math:`A = \sum_\nu a_\nu\, I \otimes \dots \otimes K
# \otimes \dots \otimes I` on :math:`(0, 1)^4`, with a diffusion
# coefficient :math:`a_\nu` per direction: 1 in the first direction,
# :math:`10^{-2}` in the others. It is a Tucker operator with frames
# :math:`\{a_\nu K, I\}` and the TT core of the Laplacian.
n, d = 20, 4
weights = [1.0, 0.01, 0.01, 0.01]
K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) * (n + 1) ** 2
frames = tensap.TFrameOperator(
    [
        tensap.FrameOperator(np.stack([w * K, np.eye(n)]))
        for w in weights
    ]
)
factory = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d)
A = tensap.TuckerTensorOperator(
    factory.make_tt_tucker_tensor().core, frames
)
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

# %%
# Greedy enrichment of the subspaces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# At each iteration, a rank-one correction
# :math:`c_1 \otimes \dots \otimes c_d` of the residual is computed; the
# space :math:`U_\nu` is enriched by :math:`c_\nu` only if the smallest
# :math:`\nu`-singular value of the new iterate is at least ``theta``
# times the largest of these values over the dimensions. Then the TT core
# solves the Galerkin problem on the subspaces (by DMRG).
solver = tensap.AnisotropicGreedySpaceLinearSolver(
    A, b, theta=0.8, max_iterations=10, tolerance=1e-6, display=False
)
x, output = solver.solve()
for k, (enriched, dims) in enumerate(
    zip(output["enriched"], output["dims"]), start=1
):
    print(
        f"iteration {k:2d}: enriched dimensions {enriched}, "
        f"dimensions of the spaces {dims}, "
        f"residual {output['errvec'][k - 1]:.1e}"
    )

# %%
# Comparison with the greedy solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The spaces are larger in the directions of small diffusion, where the
# solution has boundary layers. For about the same storage, the pure
# greedy solver, which adds one vector per correction in every direction,
# stays three orders of magnitude above in residual.
np.random.seed(0)
y, greedy = tensap.GreedyLinearSolver(
    A, b, max_iterations=10, display=False
).solve()
print(
    f"anisotropic: residual {output['error']:.1e}, storage {x.storage()}"
)
print(
    f"greedy     : residual {greedy['error']:.1e}, storage {y.storage()}"
)
