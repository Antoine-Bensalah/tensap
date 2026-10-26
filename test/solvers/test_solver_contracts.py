"""Contracts shared by the iterative solvers (review R0, M6): an exact initial
guess or b = 0 is accepted before any division, convergence is tested before
stagnation, a non-positive curvature in PCG is a breakdown (flag 3)."""

import numpy as np
import pytest
import tensap


def _system(n=4):
    A = tensap.TTOperator.eye([n, n])
    rng = np.random.default_rng(0)
    b = tensap.TTTensor.randn([n, n], [2])
    return A, b, rng


SOLVERS = [
    lambda A, b, **kw: tensap.TruncatedPCG(A, b, tensap.Truncator(tolerance=1e-12), **kw),
    lambda A, b, **kw: tensap.TruncatedRichardsonIterations(
        A, b, tensap.Truncator(tolerance=1e-12), **kw),
]


@pytest.mark.parametrize("make", SOLVERS)
def test_zero_right_hand_side(make):
    A, b, _ = _system()
    x, output = make(A, b * 0.0, display=False).solve()
    assert x.norm() == 0 and output["flag"] == 0 and output["iter"] == 0
    assert output["error"] == 0 and output["errvec"] == []


@pytest.mark.parametrize("make", SOLVERS)
def test_exact_initial_guess(make):
    A, b, _ = _system()
    x, output = make(A, b, x0=b, display=False, tolerance=1e-10).solve()
    assert output["flag"] == 0 and output["iter"] == 0 and output["error"] < 1e-12


@pytest.mark.parametrize("make", SOLVERS)
def test_convergence_is_tested_before_stagnation(make):
    """Identity, x0 = 0: the first step is exact; its error is below the
    tolerance and its change is large -- flag 0 (Richardson returned 2 when
    started at the solution)."""
    A, b, _ = _system()
    x, output = make(A, b, display=False, tolerance=1e-10, stagnation=1e3).solve()
    assert output["flag"] == 0 and output["iter"] == 1


def test_pcg_breakdown_on_a_negative_operator():
    A, b, _ = _system()
    x, output = tensap.TruncatedPCG(A * -1.0, b, tensap.Truncator(tolerance=1e-12),
                                    display=False).solve()
    assert output["flag"] == 3 and output["iter"] == 1 and output["error"] == pytest.approx(1.0)


def test_kronecker_sum_preconditioner_is_rejected_when_not_positive():
    """M = diag(1, 1, 1, 100) = I x I + 99 e e^T x e e^T is SPD, its projection
    on the Kronecker sums is not (review M2): no preconditioner."""
    from tensap.solvers.linear_solvers import _kronecker_sum_preconditioner
    terms = np.stack([np.eye(2), np.diag([0.0, 1.0])], axis=1)  # (test, k, trial)
    node = np.array([[1.0, 0.0], [0.0, 99.0]])
    assert _kronecker_sum_preconditioner(node, [terms, terms], (2, 2)) is None
    # an exact Kronecker sum (I x T + T x I, T SPD) keeps its exact inverse
    T = np.array([[2.0, -1.0], [-1.0, 2.0]])
    terms = np.stack([np.eye(2), T], axis=1)
    node = np.array([[0.0, 1.0], [1.0, 0.0]])
    P = _kronecker_sum_preconditioner(node, [terms, terms], (2, 2))
    M = np.kron(np.eye(2), T) + np.kron(T, np.eye(2))
    np.testing.assert_allclose(P @ (M @ np.arange(4.0)), np.arange(4.0), atol=1e-12)


def test_local_failures_are_reported():
    """Matrix-free local CG limited to one iteration: the failures and the
    worst local residual reach the output of the outer solve (review A2)."""
    d, n = 4, 4
    A = tensap.TTOperator.eye([n] * d) * 2.0
    for k in range(d):
        cores = [np.eye(n).reshape(1, n, n, 1)] * d
        T = np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
        cores = list(cores)
        cores[k] = T.reshape(1, n, n, 1) * 0.5
        A = A + tensap.TTOperator(cores)
    b = tensap.TTTensor.randn([n] * d, [2] * (d - 1))
    x0 = tensap.TTTensor.randn([n] * d, [2] * (d - 1))
    _, output = tensap.TTTensorALSLinearSolver(
        A, b, x0=x0, dense_max_size=0, local_max_iterations=1, max_iterations=2,
        tolerance=0.0, display=False).solve()
    local = output["local"]
    assert local["solves"] > 0 and local["iterative"] == local["solves"]
    assert local["failures"] > 0 and local["worst_residual"] > 1e-6
    _, output = tensap.TTTensorALSLinearSolver(A, b, x0=x0, max_iterations=2, tolerance=0.0,
                                               display=False).solve()
    assert output["local"]["failures"] == 0 and output["local"]["iterative"] == 0
    _, output = tensap.TTTensorALSLinearSolver(
        A, b, x0=x0, dense_max_size=0, local_method="gmres", max_iterations=2,
        tolerance=0.0, display=False).solve()
    assert output["local"]["failures"] == 0 and output["local"]["worst_residual"] < 1e-8
    with pytest.raises(ValueError, match="local_method"):
        tensap.TTTensorALSLinearSolver(A, b, local_method="minres")


def test_tt_tensor_and_frame_own_their_data():
    """Review A1: neither an alias of the input nor a replacement of a core can
    invalidate the norm certified by the orthogonality center."""
    owner = np.ones((2, 3, 3))
    frame = tensap.FrameVector(owner[:, :, :1])
    owner[:] = 7.0
    assert np.all(frame.data == 1.0)
    core = np.ones((1, 2, 1))
    x = tensap.TTTensor([core, core]).orth()
    core[:] = 3.0
    assert x.norm() == pytest.approx(np.linalg.norm(x.full().numpy()))
    with pytest.raises(TypeError):
        x.cores[1] = x.cores[1] * 3.0
    with pytest.raises(AttributeError):
        x.orth_center = 0
    with pytest.raises(ValueError, match="orth_center"):
        tensap.TTTensor([core, core], orth_center=2)
    shared = tensap.TTTensor(x.cores)
    assert shared.cores[0] is x.cores[0]  # read-only cores are shared, not copied
