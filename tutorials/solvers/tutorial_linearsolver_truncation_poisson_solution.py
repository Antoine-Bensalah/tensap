# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: truncation of the solution of a Poisson equation in several
formats. Python counterpart of the MATLAB tutorial
tutorial_LinearSolvers_truncationPoissonSolution.m.

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
2. in Tucker format with a TT core: the diagonal core is converted to a
   TTTensor (exactly, ranks 50), then rounded, and the frames are truncated
   by SVD;
3. in tree-based format: the diagonal core is converted to a tree-based
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
# b is of rank one, with a diagonal core (as TuckerLikeTensor.ones in MATLAB)
b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)
factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
A = factory.make_canonical_tensor()

# %% Greedy solver: a canonical approximation of rank 50
s = tensap.GreedyLinearSolver(A, b, max_iterations=50)
u, output = s.solve()
print(f"Greedy solution: ranks {u.representation_rank()}, storage {u.storage()}")

t = tensap.Truncator(tolerance=1e-4)


def pde_residual(v):
    """||b - A v|| / ||b||: how well v solves the PDE (not only how well it
    approximates u)."""
    return (b - A @ v).norm() / b.norm()


residuals = {"greedy": pde_residual(u)}
print(f"Greedy solution: PDE residual {residuals['greedy']:.2e}")


def report(name, v):
    error = (v - u).norm() / u.norm()
    residuals[name] = pde_residual(v)
    print(f"{name}: relative error {error:.2e}, PDE residual {residuals[name]:.2e}, "
          f"storage {v.storage()} ({v.storage() / u.storage():.1%} of the greedy solution)")
    return error


# %% 1. Truncation in canonical format
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    u_can = t.truncate(u)
for warning in caught:
    print(f"Warning: {warning.message}")
error_can = report("canonical", u_can)
print(f"  ranks {u_can.representation_rank()}")

# %% 2. Truncation in Tucker format with a TT core
u_tt = tensap.TuckerTensorVector(tensap.TTTensor.from_diagonal_tensor(u.core), u.tensor_frame)
u_tt_t = t.truncate(u_tt)
error_tt = report("Tucker-TT", u_tt_t)
print(f"  Tucker ranks {u_tt_t.representation_rank()}, TT ranks {u_tt_t.core.ranks}")

# %% 3. Truncation in tree-based format
tree = tensap.DimensionTree.balanced(d)
u_tree = tensap.TuckerTensorVector(u.core.tree_based_tensor(tree), u.tensor_frame)
u_tree_t = t.truncate(u_tree)
error_tree = report("tree-based", u_tree_t)
print(f"  ranks {u_tree_t.tree_based_tensor().ranks.tolist()}")

# %% Self-check
# The Tucker-TT and tree-based truncations reach the requested precision; the
# canonical one stays within a factor 10 of it (with as many terms as u).
assert error_tt <= t.tolerance, error_tt
assert error_tree <= t.tolerance, error_tree
assert error_can <= 10 * t.tolerance, error_can
# The truncations approximate the greedy tensor u, not the solution of the
# PDE: u itself (50 rank-one corrections in dimension 10) has a PDE residual of
# about 7e-2, and the truncations must not degrade it (observed: equal to 3
# digits). An excellent compression of a poor solution is still a poor solution.
assert residuals["greedy"] < 0.1, residuals["greedy"]
for name in ("canonical", "Tucker-TT", "tree-based"):
    assert abs(residuals[name] - residuals["greedy"]) < 1e-2 * residuals["greedy"], name
