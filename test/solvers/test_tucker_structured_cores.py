"""
Unit tests of TuckerTensorALSLinearSolver.update_core for structured cores:
TT cores (Galerkin problem on the core solved by TTTensorALSLinearSolver),
tree-based cores (node by node) and full cores solved matrix-free.

Oracles: dense Galerkin projections on the span of the frames, and the
energy J(v) = 1/2 (Av, v) - (b, v) of the dense system.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor


N, D = 4, 3


def _factory(n=N, d=D):
    K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) * (n + 1) ** 2
    return tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d)


def _ones(n=N, d=D):
    frames = tensap.TFrameVector([tensap.FrameVector(np.ones((1, n, 1))) for _ in range(d)])
    return tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0]), d), frames)


def _with_core(x, core):
    return tensap.TuckerTensorVector(core, x.tensor_frame)


def _dense_operator(n=N, d=D):
    K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) * (n + 1) ** 2
    A = np.zeros((n**d, n**d))
    for mu in range(d):
        term = np.ones((1, 1))
        for nu in range(d):
            term = np.kron(term, K if nu == mu else np.eye(n))
        A += term
    return A


def _frames_basis(x):
    """Matrix whose columns span the tensor product of the frames of x."""
    basis = np.ones((1, 1))
    for frame in x.tensor_frame.frames:
        basis = np.kron(basis, frame.data[:, :, 0].T)
    return basis


def _galerkin(A, b, basis):
    """Dense Galerkin solution on the range of basis."""
    q, _ = np.linalg.qr(basis)
    return q @ np.linalg.solve(q.T @ A @ q, q.T @ b)


def _random_tucker(ranks, core, n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    frames = tensap.TFrameVector(
        [tensap.FrameVector(rng.standard_normal((r, n, 1))) for r in ranks]
    )
    return tensap.TuckerTensorVector(core, frames)


# --- TT core ------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["als", "dmrg", "amen"])
def test_tt_core_update_is_galerkin_on_the_frames(algorithm):
    A = _factory().make_tt_tucker_tensor()
    b = _ones()
    b = _with_core(b, as_tt_tensor(b.core))
    np.random.seed(1)
    core = tensap.TTTensor.randn([3, 3, 3], [3, 3])
    x = _random_tucker([3, 3, 3], core)
    solver = tensap.TuckerTensorALSLinearSolver(
        A, b, core_algorithm=algorithm, max_iterations_core=20,
        stagnation_core=1e-14, display=False)
    y = solver.update_core(x)
    assert isinstance(y.core, tensap.TTTensor)
    expected = _galerkin(_dense_operator(), b.full().numpy().ravel(), _frames_basis(x))
    npt.assert_allclose(y.full().numpy().ravel(), expected, atol=1e-9)


def test_tt_core_from_a_diagonal_core_and_solve():
    A = _factory().make_tt_tucker_tensor()
    b = _ones()
    b = _with_core(b, as_tt_tensor(b.core))
    x = _random_tucker([2, 2, 2], tensap.DiagonalTensor(np.array([1.0, 0.5]), D))
    solver = tensap.TuckerTensorALSLinearSolver(
        A, b, core_algorithm="dmrg", max_iterations=3, display=False)
    A_dense, b_dense = _dense_operator(), b.full().numpy().ravel()

    def energy(v):
        v = v.full().numpy().ravel()
        return 0.5 * v @ A_dense @ v - b_dense @ v

    y = solver.update_core(x)
    assert isinstance(y.core, tensap.TTTensor)
    assert energy(y) <= energy(x) + 1e-12
    z, output = solver.solve(y)
    assert energy(z) <= energy(y) + 1e-12
    assert len(output["errvec"]) == output["iter"]


def test_tt_core_as_greedy_update():
    A = _factory().make_tt_tucker_tensor()
    b = _ones()
    b = _with_core(b, as_tt_tensor(b.core))
    A_dense, b_dense = _dense_operator(), b.full().numpy().ravel()
    np.random.seed(0)
    pure, _ = tensap.GreedyLinearSolver(A, b, max_iterations=4, display=False).solve()
    updater = tensap.TuckerTensorALSLinearSolver(A, b, display=False)
    np.random.seed(0)
    updated, _ = tensap.GreedyLinearSolver(A, b, max_iterations=4, display=False,
                                           update=updater.update_core).solve()

    def residual(v):
        return np.linalg.norm(b_dense - A_dense @ v.full().numpy().ravel())

    assert residual(updated) < residual(pure)


# --- tree-based core ----------------------------------------------------------------


def _tree_problem(d=4):
    factory = _factory(d=d)
    A = factory.make_tree_based_tensor()
    b = _ones(d=d)
    b = _with_core(b, b.core.tree_based_tensor(A.core.tree))
    return A, b, _dense_operator(d=d), b.full().numpy().ravel()


def test_tree_core_update_with_full_ranks_is_galerkin():
    A, b, A_dense, b_dense = _tree_problem()
    rng = np.random.default_rng(2)
    core = tensap.FullTensor(rng.standard_normal((2, 2, 2, 2)))
    core = tensap.Truncator(tolerance=0.0).hsvd(core, A.core.tree)
    x = _random_tucker([2, 2, 2, 2], core, d=4, seed=3)
    solver = tensap.TuckerTensorALSLinearSolver(A, b, max_iterations_core=5, display=False)
    y = solver.update_core(x)
    assert isinstance(y.core, tensap.TreeBasedTensor)
    expected = _galerkin(A_dense, b_dense, _frames_basis(x))
    npt.assert_allclose(y.full().numpy().ravel(), expected, atol=1e-8)


def test_tree_core_greedy_update_decreases_the_residual():
    A, b, A_dense, b_dense = _tree_problem()
    np.random.seed(0)
    pure, _ = tensap.GreedyLinearSolver(A, b, max_iterations=3, display=False).solve()
    updater = tensap.TuckerTensorALSLinearSolver(A, b, max_iterations_core=3, display=False)
    np.random.seed(0)
    updated, _ = tensap.GreedyLinearSolver(A, b, max_iterations=3, display=False,
                                           update=updater.update_core).solve()

    def residual(v):
        return np.linalg.norm(b_dense - A_dense @ v.full().numpy().ravel())

    assert isinstance(updated.core, tensap.TreeBasedTensor)
    assert residual(updated) < residual(pure)


# --- full core, matrix-free ---------------------------------------------------------


def test_full_core_matrix_free_matches_dense():
    A = _factory().make_tucker_tensor()
    b = _ones()
    b = _with_core(b, tensap.FullTensor(b.core.full().numpy()))
    rng = np.random.default_rng(4)
    x = _random_tucker([3, 3, 3], tensap.FullTensor(rng.standard_normal((3, 3, 3))))
    dense = tensap.TuckerTensorALSLinearSolver(A, b, display=False).update_core(x)
    free = tensap.TuckerTensorALSLinearSolver(A, b, dense_max_size=0,
                                              display=False).update_core(x)
    npt.assert_allclose(free.full().numpy(), dense.full().numpy(), atol=1e-9)
