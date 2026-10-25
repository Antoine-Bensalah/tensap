# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: updates of the greedy iterates for a 4D Poisson equation, with a
canonical, Tucker, tree-based or TT core. Python counterpart of the MATLAB
tutorial tutorial_LinearSolvers_updatePoissonSolution.m; the TT core part is
an addition (Matlab uses it inside AnisotropicGreedySpaceLinearSolver).

Problem
-------
On (0, 1)^d, d = 4, with n = 50 interior points per direction, the finite
difference Poisson equation A x = b, A = sum_mu I x ... x K x ... x I and
b = 1 x ... x 1.

Steps
-----
After each rank-one correction of the greedy solver, the iterate is updated
by a Galerkin problem (TuckerTensorALSLinearSolver):

1. canonical format (diagonal core): update of the core, or of the core and
   the frames;
2. Tucker format (full core of size 10^4 after 10 corrections, solved by the
   conjugate gradient without assembling the matrix);
3. tree-based core (balanced tree), updated node by node;
4. TT core, the Galerkin problem on the core being solved by
   TTTensorALSLinearSolver (ALS, then DMRG).
"""

# %% Imports
import time

import numpy as np
import scipy.sparse as sp
import tensap
from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor


def residual(A, b, x):
    return (A @ x - b).norm() / b.norm()


def greedy(A, b, update=None, **kwargs):
    np.random.seed(0)
    start = time.time()
    solver = tensap.GreedyLinearSolver(A, b, update=update, display=False, **kwargs)
    x, _ = solver.solve()
    return x, time.time() - start


# %% 1D discretization
L = 1.0
n = 50
h = L / (n + 1)
K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
Id = sp.identity(n)
d = 4
factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
ones = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

# %% 1. Updates: canonical case
A = factory.make_canonical_tensor()
b = ones
u0, t0 = greedy(A, b)
su = tensap.TuckerTensorALSLinearSolver(A, b, display=False)
u1, t1 = greedy(A, b, update=su.update_core)
su.max_iterations = 1
su.max_iterations_frames = 1
u2, t2 = greedy(A, b, update=lambda x: su.solve(x)[0])
print("CANONICAL")
print(f"  pure greedy          residual {residual(A, b, u0):.2e}  ({t0:.1f} s)")
print(f"  update of the core   residual {residual(A, b, u1):.2e}  ({t1:.1f} s)")
print(f"  core and frames      residual {residual(A, b, u2):.2e}  ({t2:.1f} s)")
assert residual(A, b, u2) < residual(A, b, u1) < residual(A, b, u0)

# %% 2. Updates: Tucker case (full core)
A = factory.make_tucker_tensor()
b = tensap.TuckerTensorVector(tensap.FullTensor(ones.core.full().numpy()), ones.tensor_frame)
u3, t3 = greedy(A, b, max_iterations=10)
su = tensap.TuckerTensorALSLinearSolver(A, b, display=False)
u4, t4 = greedy(A, b, update=su.update_core, max_iterations=10)
su.max_iterations = 1
su.max_iterations_frames = 1
u5, t5 = greedy(A, b, update=lambda x: su.solve(x)[0], max_iterations=10)
print("TUCKER (full core)")
print(f"  pure greedy          residual {residual(A, b, u3):.2e}  ({t3:.1f} s)")
print(f"  update of the core   residual {residual(A, b, u4):.2e}  ({t4:.1f} s), "
      f"core of size {np.prod(u4.representation_rank())}")
print(f"  core and frames      residual {residual(A, b, u5):.2e}  ({t5:.1f} s)")
assert residual(A, b, u4) < residual(A, b, u3)

# %% 3. Updates: tree-based core
Ah = factory.make_tree_based_tensor()
b = tensap.TuckerTensorVector(ones.core.tree_based_tensor(Ah.core.tree), ones.tensor_frame)
su = tensap.TuckerTensorALSLinearSolver(Ah, b, max_iterations_core=3, display=False)
uh, th = greedy(Ah, b, update=su.update_core, max_iterations=10)
print("TREE-BASED CORE")
print(f"  update of the core   residual {residual(Ah, b, uh):.2e}  ({th:.1f} s), "
      f"tree ranks {uh.core.ranks}")
assert residual(Ah, b, uh) < residual(A, b, u3)

# %% 4. Updates: TT core
At = factory.make_tt_tucker_tensor()
b = tensap.TuckerTensorVector(as_tt_tensor(ones.core), ones.tensor_frame)
for algorithm in ("als", "dmrg"):
    su = tensap.TuckerTensorALSLinearSolver(At, b, core_algorithm=algorithm,
                                            max_iterations_core=3, display=False)
    ut, tt = greedy(At, b, update=su.update_core, max_iterations=10)
    print(f"TT CORE ({algorithm})")
    print(f"  update of the core   residual {residual(At, b, ut):.2e}  ({tt:.1f} s), "
          f"TT ranks of the core {ut.core.ranks}")
    assert residual(At, b, ut) < residual(A, b, u3)
