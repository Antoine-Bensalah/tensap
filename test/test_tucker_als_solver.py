# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Test module test_tucker_als_solver.

Galerkin updates of TuckerTensorALSLinearSolver (core and frames), alone and
as the update of the greedy solver, on a small d-dimensional discrete Poisson
equation. Reproduces the canonical part of the MATLAB tutorial
tutorial_LinearSolvers_highDimPoissonEq.m.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import tensap


def _poisson(n, d):
    h = 1.0 / (n + 1)
    K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
    Id = sp.identity(n)
    factory = tensap.LaplacianLikeOperatorFactory(K, Id, d)
    frames = tensap.TFrameVector(
        [tensap.FrameVector(np.ones((1, n, 1))) for _ in range(d)]
    )
    b = tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0]), d), frames)

    A_dense = np.zeros((n**d, n**d))
    for mu in range(d):
        term = np.ones((1, 1))
        for nu in range(d):
            term = np.kron(term, K.toarray() if nu == mu else np.eye(n))
        A_dense += term
    return factory, b, A_dense, np.ones(n**d)


def _greedy_iterate(A, b, iterations):
    np.random.seed(0)
    x, _ = tensap.GreedyLinearSolver(
        A, b, max_iterations=iterations, display=False
    ).solve()
    return x


def _energy(A_dense, b_dense, x):
    y = x.full().numpy().ravel()
    return 0.5 * y @ A_dense @ y - b_dense @ y


def test_update_core_is_galerkin_on_rank_one_terms():
    n, d = 8, 3
    factory, b, A_dense, b_dense = _poisson(n, d)
    A = factory.make_canonical_tensor()
    x = _greedy_iterate(A, b, 4)

    x = tensap.TuckerTensorALSLinearSolver(A, b, display=False).update_core(x)

    residual = b_dense - A_dense @ x.full().numpy().ravel()
    frames = [frame.data[:, :, 0] for frame in x.tensor_frame.frames]
    for i in range(x.core.data.size):
        term = np.kron(np.kron(frames[0][i], frames[1][i]), frames[2][i])
        assert abs(term @ residual) < 1e-10 * np.linalg.norm(residual)


def test_update_frame_is_galerkin_in_direction_mu():
    n, d = 8, 3
    factory, b, A_dense, b_dense = _poisson(n, d)
    A = factory.make_canonical_tensor()
    x = _greedy_iterate(A, b, 4)

    x = tensap.TuckerTensorALSLinearSolver(A, b, display=False).update_frame(x, 1)

    # tangent directions of dimension 1: w^(0)_i x e x w^(2)_i for all e
    residual = b_dense - A_dense @ x.full().numpy().ravel()
    frames = [frame.data[:, :, 0] for frame in x.tensor_frame.frames]
    for i in range(x.core.data.size):
        tangent = np.kron(
            np.kron(frames[0][i][:, None], np.eye(n)), frames[2][i][:, None]
        )
        assert np.linalg.norm(tangent.T @ residual) < 1e-9 * np.linalg.norm(residual)


def test_updates_decrease_energy():
    n, d = 8, 3
    factory, b, A_dense, b_dense = _poisson(n, d)
    A = factory.make_canonical_tensor()
    x = _greedy_iterate(A, b, 4)
    solver = tensap.TuckerTensorALSLinearSolver(A, b, display=False)

    x_core = solver.update_core(x)
    x_frames = solver.update_frames_by_als(x_core)

    energies = [_energy(A_dense, b_dense, y) for y in (x, x_core, x_frames)]
    assert energies[0] >= energies[1] >= energies[2]


def test_update_core_full_core_operator():
    n, d = 6, 3
    factory, _, A_dense, b_dense = _poisson(n, d)
    A_can = factory.make_canonical_tensor()
    b_can = _poisson(n, d)[1]
    x = _greedy_iterate(A_can, b_can, 2)

    A = factory.make_tucker_tensor()
    # the solver requires cores of the same type: ones() has a diagonal core
    b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)
    b = tensap.TuckerTensorVector(b.core.full(), b.tensor_frame)
    x_core = tensap.TuckerTensorALSLinearSolver(A, b, display=False).update_core(x)

    # Galerkin on the whole Tucker space spanned by the (orthonormalized) frames
    residual = b_dense - A_dense @ x_core.full().numpy().ravel()
    U = [f.data[:, :, 0].T for f in x_core.tensor_frame.frames]
    projected = np.kron(np.kron(U[0], U[1]), U[2]).T @ residual
    assert isinstance(x_core.core, tensap.FullTensor)
    assert np.linalg.norm(residual) > 1e-6 * np.linalg.norm(b_dense)
    assert np.linalg.norm(projected) < 1e-9 * np.linalg.norm(residual)


def test_greedy_with_updates_converges_faster():
    n, d = 10, 3
    factory, b, _, _ = _poisson(n, d)
    A = factory.make_canonical_tensor()
    updater = tensap.TuckerTensorALSLinearSolver(
        A, b, max_iterations_frames=3, display=False
    )

    errors = {}
    for name, update in (
        ("pure", None),
        ("frames", updater.update_frames_by_als),
    ):
        np.random.seed(0)
        _, output = tensap.GreedyLinearSolver(
            A, b, max_iterations=8, update=update, stagnation=1e-14, display=False
        ).solve()
        errors[name] = output["error"]

    assert errors["frames"] < errors["pure"]


def test_als_solve_from_greedy_iterate():
    n, d = 8, 3
    factory, b, _, _ = _poisson(n, d)
    A = factory.make_canonical_tensor()
    x0 = _greedy_iterate(A, b, 4)
    error0 = (b - A @ x0).norm() / b.norm()

    solver = tensap.TuckerTensorALSLinearSolver(
        A, b, x0=x0, max_iterations=3, max_iterations_frames=2, display=False
    )
    x, output = solver.solve()

    assert x.representation_rank() == x0.representation_rank()
    assert output["error"] < error0
    assert len(output["errvec"]) == output["iter"]


def _parametric_diffusion_reaction(N=10, Q=50, seed=0):
    """A_x x I + B_x x diag(xi) in canonical format: the solution has
    exponentially decaying singular values, so that greedy corrections beyond
    a few terms are redundant."""
    rng = np.random.default_rng(seed)
    h = 1 / (N + 1)
    T = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h**2
    Ax = np.kron(np.eye(N), T) + np.kron(T, np.eye(N))
    xi = 50 * rng.random(Q)
    A = tensap.TuckerTensorOperator(
        tensap.DiagonalTensor(np.ones(2), 2),
        tensap.TFrameOperator([
            tensap.FrameOperator(np.stack([Ax, np.eye(N * N)])),
            tensap.FrameOperator(np.stack([np.eye(Q), np.diag(xi)])),
        ]),
    )
    b = tensap.TuckerTensorVector(
        tensap.DiagonalTensor(np.ones(1), 2),
        tensap.TFrameVector([
            tensap.FrameVector(np.ones((1, N * N, 1))),
            tensap.FrameVector(np.ones((1, Q, 1))),
        ]),
    )
    return A, b


def test_greedy_with_als_updates_beyond_the_rank_needed():
    """Canonical format: the frame updates do not depend on the spread of the
    coefficients (the residual stays at its level after convergence), and when
    the frames become linearly dependent, the update is skipped with a warning
    instead of returning NaN."""
    A, b = _parametric_diffusion_reaction()
    updater = tensap.TuckerTensorALSLinearSolver(
        A, b, max_iterations=1, max_iterations_frames=1, display=False)
    np.random.seed(0)
    with pytest.warns(RuntimeWarning, match="not updated"):
        x, output = tensap.GreedyLinearSolver(
            A, b, update=lambda y: updater.solve(y)[0], max_iterations=14,
            tolerance=0.0, stagnation=0.0, display=False).solve()
    assert np.all(np.isfinite(x.numpy()))
    errors = np.array(output["errvec"])
    assert errors[4] < 1e-7  # converged at rank 5
    assert np.all(errors[4:] < 1e-7)  # and stays there
