# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Test module test_greedy_poisson.

Reproduces the first part of the MATLAB tutorial
tutorial_LinearSolvers_highDimPoissonEq.m: solve a d-dimensional discrete
Poisson equation, with the operator in canonical format, using the greedy
solver with rank-one corrections computed by alternating least squares.
"""

import numpy as np
import scipy.sparse as sp
import tensap


def _poisson_1d(n, L=1.0):
    h = L / (n + 1)
    K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
    I = sp.identity(n)
    return K, I


def _rank_one_ones(n, d):
    """The vector 1 x ... x 1 as a rank-one TuckerTensorVector."""
    frames = tensap.TFrameVector(
        [tensap.FrameVector(np.ones((1, n, 1))) for _ in range(d)]
    )
    return tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0]), d), frames)


def _dense_laplacian(K, d):
    K = K.toarray()
    n = K.shape[0]
    A = np.zeros((n**d, n**d))
    for mu in range(d):
        term = np.ones((1, 1))
        for nu in range(d):
            term = np.kron(term, K if nu == mu else np.eye(n))
        A += term
    return A


def test_canonical_laplacian_equals_tucker_laplacian():
    K, I = _poisson_1d(5)
    factory = tensap.LaplacianLikeOperatorFactory(K, I, 3)

    A_can = factory.make_canonical_tensor()
    A_tucker = factory.make_tucker_tensor()

    assert isinstance(A_can.core, tensap.DiagonalTensor)
    assert A_can.representation_rank() == (3, 3, 3)
    assert np.allclose(A_can.full().numpy(), A_tucker.full().numpy())


def test_diagonal_core_arithmetic():
    n, d = 5, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_canonical_tensor()
    b = _rank_one_ones(n, d)

    r = b - A @ b
    r_dense = np.ones(n**d) - _dense_laplacian(K, d) @ np.ones(n**d)

    assert isinstance(r.core, tensap.DiagonalTensor)
    assert np.allclose(r.full().numpy().ravel(), r_dense)
    assert np.isclose(r.norm(), np.linalg.norm(r_dense))
    assert np.isclose(b.dot(r), r_dense.sum())
    assert np.allclose((2.0 * r).full().numpy().ravel(), 2.0 * r_dense)
    assert np.allclose((r / 2.0).full().numpy().ravel(), r_dense / 2.0)


def test_rank_one_als_exact_rank_one_solution():
    # A = K x K x K and b = 1 x 1 x 1: the solution is exactly of rank one
    np.random.seed(0)
    n, d = 8, 3
    K, _ = _poisson_1d(n)
    K_dense = K.toarray()
    frames = tensap.TFrameOperator([tensap.FrameOperator(K_dense[None]) for _ in range(d)])
    A = tensap.TuckerTensorOperator(tensap.DiagonalTensor(np.array([1.0]), d), frames)
    b = _rank_one_ones(n, d)

    solver = tensap.RankOneALSLinearSolver(
        A, b, stagnation=1e-12, max_iterations=50, display=False
    )
    x, output = solver.solve()

    x_direct = np.linalg.solve(
        np.kron(np.kron(K_dense, K_dense), K_dense), np.ones(n**d)
    )
    relative_error = np.linalg.norm(x.full().numpy().ravel() - x_direct) / np.linalg.norm(
        x_direct
    )
    assert x.representation_rank() == (1, 1, 1)
    assert relative_error < 1e-10


def test_greedy_poisson_3d():
    np.random.seed(0)
    n, d = 10, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_canonical_tensor()
    b = _rank_one_ones(n, d)

    solver = tensap.GreedyLinearSolver(
        A, b, max_iterations=40, tolerance=1e-4, stagnation=1e-12, display=False
    )
    x, output = solver.solve()

    assert output["flag"] == 0
    assert output["error"] < 1e-4
    assert len(output["errvec"]) == output["iter"]

    x_direct = np.linalg.solve(_dense_laplacian(K, d), np.ones(n**d))
    relative_error = np.linalg.norm(x.full().numpy().ravel() - x_direct) / np.linalg.norm(
        x_direct
    )
    assert relative_error < 1e-4


def test_greedy_with_initial_guess_and_update():
    np.random.seed(0)
    n, d = 6, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_canonical_tensor()
    b = _rank_one_ones(n, d)

    calls = []

    def update(x):
        calls.append(x.representation_rank())
        return x

    x0, _ = tensap.GreedyLinearSolver(A, b, max_iterations=2, display=False).solve()
    solver = tensap.GreedyLinearSolver(
        A, b, x0=x0, update=update, initial_update=True, max_iterations=3,
        tolerance=1e-14, stagnation=1e-14, display=False,
    )
    x, output = solver.solve()

    # initial update on x0, then one update per correction
    assert calls[0] == (2, 2, 2)
    assert len(calls) == 1 + output["iter"]
    assert x.representation_rank() == (2 + output["iter"],) * d


def test_greedy_minimize_residual_and_adjoint_options():
    n, d = 6, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_canonical_tensor()
    b = _rank_one_ones(n, d)

    for options in (
        {"minimize_residual": True},
        {"local_solver": tensap.RankOneALSLinearSolver(A, b, adjoint=True, display=False)},
    ):
        np.random.seed(1)
        _, output = tensap.GreedyLinearSolver(
            A, b, max_iterations=15, display=False, **options
        ).solve()
        assert output["error"] < 1e-3
