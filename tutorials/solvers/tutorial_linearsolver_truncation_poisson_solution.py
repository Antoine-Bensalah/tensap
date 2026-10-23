# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: truncation of the solution of a Poisson equation in several
formats. Python counterpart of the MATLAB tutorial
tutorial_LinearSolvers_truncationPoissonSolution.m (canonical and tree-based
parts; the TT part needs the TT formats, not ported yet).

Problem
-------
On (0, 1)^d, d = 10, with n = 50 interior points per direction, the finite
difference Poisson equation

    A u = b,   A = sum_{mu=1}^{d} I x ... x K x ... x I,   b = 1 x ... x 1,

is solved in canonical format by the greedy algorithm with rank-one
corrections (50 corrections, so a canonical rank of 50). The solution is
then truncated at the relative precision 1e-4:

1. in canonical format: recompression by a greedy algorithm with rank-one
   corrections and updates of the factors, applied to I y = u (diagonal
   core). The best approximation in canonical format is a hard problem: the
   requested precision is not reached with fewer terms than u, and the
   truncator warns about it;
2. in tree-based format: the diagonal core is converted to a tree-based
   tensor on a balanced tree, the frames are absorbed into the leaves and the
   tree-based tensor is truncated by hsvd.
"""

# %% Imports
import warnings

import numpy as np
import scipy.sparse as sp
import tensap

np.random.seed(0)

d = 10

# %% 1D discretization
L = 1.0
n = 50
h = L / (n + 1)
K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
Id = sp.identity(n)

# %% Right-hand side and operator in canonical format
# b has a diagonal core, as TuckerLikeTensor.ones in MATLAB: with a full core
# of shape (1, ..., 1), the residual b - A u would convert the diagonal core of
# A u (rank up to 50 d) to a full core with (50 d)^d entries.
b = tensap.TuckerTensorVector(
    tensap.DiagonalTensor(np.array([1.0]), d),
    tensap.TFrameVector([tensap.FrameVector(np.ones((1, n, 1))) for _ in range(d)]),
)
factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
A = factory.make_canonical_tensor()

# %% Greedy solver: a canonical approximation of rank 50
s = tensap.GreedyLinearSolver(A, b, max_iterations=50)
u, output = s.solve()
print(f"Greedy solution: ranks {u.representation_rank()}, storage {u.storage()}")

t = tensap.Truncator(tolerance=1e-4)


def report(name, v):
    error = (v - u).norm() / u.norm()
    print(f"{name}: relative error {error:.2e}, storage {v.storage()} "
          f"({v.storage() / u.storage():.1%} of the greedy solution)")
    return error


# %% 1. Truncation in canonical format
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    u_can = t.truncate(u)
for warning in caught:
    print(f"Warning: {warning.message}")
error_can = report(f"Canonical, ranks {u_can.representation_rank()}", u_can)

# %% 2. Truncation in tree-based format
tree = tensap.DimensionTree.balanced(d)
u_tree = tensap.TuckerTensorVector(u.core.tree_based_tensor(tree), u.tensor_frame)
u_tree_t = t.truncate(u_tree)
error_tree = report("Tree-based, ranks "
                    f"{u_tree_t.tree_based_tensor().ranks.tolist()}", u_tree_t)

# %% Self-check
# The tree-based truncation reaches the requested precision; the canonical one
# stays within a factor 10 of it (with as many terms as u).
assert error_tree <= t.tolerance, error_tree
assert error_can <= 10 * t.tolerance, error_can
