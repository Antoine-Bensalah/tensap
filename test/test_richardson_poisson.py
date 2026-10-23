# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Test module test_richardson_poisson.

Reproduces (a slice of) the MATLAB tutorial
tutorial_LinearSolvers_tuckerPoissonEq.m: solve a d-dimensional discrete
Poisson equation, with the operator represented in Tucker format, using
truncated Richardson iterations (no preconditioner).
"""

import numpy as np
import scipy.sparse as sp
import tensap


def test_truncated_richardson_poisson_3d():
    L = 1.0
    n = 10
    h = L / (n + 1)

    K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
    I = sp.identity(n)

    d = 3
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_tucker_tensor()
    b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)

    truncator = tensap.Truncator(tolerance=1e-4)

    # Richardson iterations converge for step_size in (0, 2 / largest eigenvalue of A).
    # The largest eigenvalue of the 1D Laplacian K is bounded by 4/h^2, so the
    # d-dimensional operator (sum of d such terms) has largest eigenvalue
    # bounded by 4*d/h^2; a safe (non-marginal) choice is its inverse.
    step_size = h**2 / (4 * d)

    solver = tensap.TruncatedRichardsonIterations(
        A,
        b,
        truncator,
        step_size=step_size,
        max_iterations=2000,
        tolerance=1e-2,
        stagnation=1e-8,
        display=False,
    )
    x, output = solver.solve()

    assert output["flag"] == 0
    assert output["error"] < 1e-2

    # Cross-check against a direct dense solve of the same (small) linear system.
    K_dense = K.toarray()
    I_dense = np.eye(n)
    A_dense = (
        np.kron(np.kron(K_dense, I_dense), I_dense)
        + np.kron(np.kron(I_dense, K_dense), I_dense)
        + np.kron(np.kron(I_dense, I_dense), K_dense)
    )
    b_dense = np.ones(n**d)
    x_direct = np.linalg.solve(A_dense, b_dense)

    x_tensor = x.full().numpy().ravel()
    relative_error = np.linalg.norm(x_tensor - x_direct) / np.linalg.norm(x_direct)
    assert relative_error < 2e-2
