# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Test module test_tucker_poisson_solvers.

Low-rank approximation of the inverse (GreedyInverseApproximator,
RankOneInverseApproximatorByALS), truncated PCG, the dense helpers lyapsym and
gspai, and the operator arithmetic they rely on. Reproduces the MATLAB
tutorial tutorial_LinearSolvers_tuckerPoissonEq.m on a small problem.
"""

import numpy as np
import scipy.sparse as sp
import tensap


def _poisson_1d(n):
    h = 1.0 / (n + 1)
    K = 1 / h**2 * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
    return K, sp.identity(n)


def _kron(matrices):
    out = np.ones((1, 1))
    for M in matrices:
        out = np.kron(out, M)
    return out


def _dense_operator(X, n, d):
    """Dense matrix of a TuckerTensorOperator with square factors of size n."""
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    return X.full().numpy().reshape([n, n] * d).transpose(perm).reshape(n**d, n**d)


def _dense_laplacian(K, d):
    n = K.shape[0]
    return sum(
        _kron([K.toarray() if nu == mu else np.eye(n) for nu in range(d)])
        for mu in range(d)
    )


def test_lyapsym():
    np.random.seed(0)
    n = 8
    M = np.random.randn(n, n)
    A = M @ M.T + n * np.eye(n)
    C = np.random.randn(n, n)
    C = C + C.T

    X = tensap.lyapsym(A, C)

    assert np.allclose(X @ A + A @ X, C)
    assert np.allclose(X, X.T)


def test_gspai_patterns():
    n = 8
    K, _ = _poisson_1d(n)
    K = K.toarray()
    B = np.eye(n)

    P_diag = tensap.gspai(K, B, "diag")
    assert P_diag.nnz == n
    P_tri = tensap.gspai(K, B, "tridiag")
    mask = np.abs(np.subtract.outer(range(n), range(n))) <= 1
    assert np.allclose(P_tri.toarray(), tensap.gspai(K, B, "imposed", mask).toarray())
    assert np.all(P_tri.toarray()[~mask] == 0)

    # with enough entries per row, the exact inverse is recovered
    for pattern, args in (("auto", (n, 1e-12)), ("power", (n, 1e-10))):
        P = tensap.gspai(K, B, pattern, *args)
        assert np.linalg.norm(P @ K - B) < 1e-8


def test_operator_products_with_mixed_cores():
    np.random.seed(0)
    n, d = 3, 3
    F = [np.random.randn(1, n, n) for _ in range(d)]
    W = tensap.TuckerTensorOperator(
        tensap.DiagonalTensor(np.array([2.0]), d),
        tensap.TFrameOperator([tensap.FrameOperator(f) for f in F]),
    )
    G = [np.random.randn(2, n, n) for _ in range(d)]
    C = np.random.randn(2, 2, 2)
    B = tensap.TuckerTensorOperator(
        tensap.FullTensor(C), tensap.TFrameOperator([tensap.FrameOperator(g) for g in G])
    )
    W_dense = 2 * _kron([f[0] for f in F])
    B_dense = sum(
        C[i, j, k] * _kron([G[0][i], G[1][j], G[2][k]])
        for i in range(2)
        for j in range(2)
        for k in range(2)
    )

    # full() of non-symmetric operators
    assert np.allclose(_dense_operator(W, n, d), W_dense)
    assert np.allclose(_dense_operator(B, n, d), B_dense)
    # products with a diagonal and a full core
    I = tensap.BaseTuckerTensor.eye([n] * d)
    for X, X_dense in (
        (W @ B, W_dense @ B_dense),
        (B @ W, B_dense @ W_dense),
        (I - W @ B, np.eye(n**d) - W_dense @ B_dense),
    ):
        assert np.allclose(_dense_operator(X, n, d), X_dense)
        assert np.isclose(X.norm(), np.linalg.norm(X_dense))


def test_orth_with_more_elements_than_the_dimension():
    np.random.seed(0)
    x = tensap.TuckerTensorVector.randn([7, 3, 2], [5, 5, 5])
    x_orth = x.orth()

    assert x_orth.representation_rank() == (5, 3, 2)
    assert np.allclose(x.full().numpy(), x_orth.full().numpy())


def test_rank_one_inverse_is_exact_for_a_rank_one_operator():
    np.random.seed(0)
    n, d = 8, 3
    K, _ = _poisson_1d(n)
    K = K.toarray()
    A = tensap.TuckerTensorOperator(
        tensap.DiagonalTensor(np.array([1.0]), d),
        tensap.TFrameOperator([tensap.FrameOperator(K[None]) for _ in range(d)]),
    )
    I = tensap.BaseTuckerTensor.eye([n] * d)

    W, output = tensap.RankOneInverseApproximatorByALS(
        A, I, max_iterations=20, stagnation=1e-12, display=False
    ).solve()

    K_inv = np.linalg.inv(K)
    inverse = _kron([K_inv] * d)
    assert output["flag"] == 2
    assert np.linalg.norm(_dense_operator(W, n, d) - inverse) < 1e-10 * np.linalg.norm(
        inverse
    )

    W_sym, _ = tensap.RankOneInverseApproximatorByALS(
        A, I, symmetric=True, display=False
    ).solve()
    for frame in W_sym.tensor_frame.frames:
        assert np.allclose(frame.data[0], frame.data[0].T)

    W_sparse, _ = tensap.RankOneInverseApproximatorByALS(
        A, I, sparse=True, sparse_options=("tridiag",), display=False
    ).solve()
    mask = np.abs(np.subtract.outer(range(n), range(n))) <= 1
    for frame in W_sparse.tensor_frame.frames:
        assert np.all(frame.data[0][~mask] == 0)


def test_greedy_inverse_approximator():
    np.random.seed(0)
    n, d = 8, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_tucker_tensor()

    P, output = tensap.GreedyInverseApproximator(A, max_iterations=6, display=False).solve()

    A_dense = _dense_laplacian(K, d)
    PA = _dense_operator(P, n, d) @ A_dense
    assert np.isclose(output["relres"], np.linalg.norm(np.eye(n**d) - PA) / np.sqrt(n**d))
    assert output["resvec"][-1] < output["resvec"][0]
    assert np.linalg.cond(PA) < 0.1 * np.linalg.cond(A_dense)

    np.random.seed(0)
    _, output = tensap.GreedyInverseApproximator(
        A, max_iterations=3, minimize_residual=True, display=False
    ).solve()
    assert output["resvec"][-1] < output["resvec"][0]


def test_truncated_pcg_poisson_3d():
    np.random.seed(0)
    n, d = 10, 3
    K, I = _poisson_1d(n)
    A = tensap.LaplacianLikeOperatorFactory(K, I, d).make_tucker_tensor()
    b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)
    truncator = tensap.Truncator()
    x_direct = np.linalg.solve(_dense_laplacian(K, d), np.ones(n**d))

    P, _ = tensap.GreedyInverseApproximator(A, max_iterations=8, display=False).solve()

    iterations = {}
    for name, precond in (("none", None), ("greedy", P)):
        x, output = tensap.TruncatedPCG(
            A, b, truncator, P=precond, max_iterations=100, display=False
        ).solve()
        relative_error = np.linalg.norm(x.full().numpy().ravel() - x_direct) / np.linalg.norm(
            x_direct
        )
        assert output["flag"] in (0, 2)
        assert relative_error < 1e-5
        iterations[name] = output["iter"]

    assert iterations["greedy"] < iterations["none"]
