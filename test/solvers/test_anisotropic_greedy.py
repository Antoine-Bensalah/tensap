"""
Unit tests of tensap.AnisotropicGreedySpaceLinearSolver: greedy enrichment of
tensor subspaces with a TT core, on small Poisson-type problems.

Oracle: dense residuals of the same systems.
"""

import numpy as np
import pytest
import tensap


def _problem(n=8, d=3, weights=None):
    weights = np.ones(d) if weights is None else np.asarray(weights)
    K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) * (n + 1) ** 2
    frames = tensap.TFrameOperator(
        [tensap.FrameOperator(np.stack([w * K, np.eye(n)])) for w in weights]
    )
    core = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d).make_tt_tucker_tensor().core
    A = tensap.TuckerTensorOperator(core, frames)
    b = tensap.TuckerTensorVector.ones([1] * d, [n] * d)
    return A, b


def _residual(A, b, x):
    return (A @ x - b).norm() / b.norm()


def test_converges_faster_than_the_pure_greedy():
    A, b = _problem()
    np.random.seed(0)
    x, output = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, max_iterations=6, tolerance=0.0, stagnation=0.0, display=False).solve()
    np.random.seed(0)
    y, _ = tensap.GreedyLinearSolver(A, b, max_iterations=6, display=False).solve()
    assert isinstance(x.core, tensap.TTTensor)
    assert _residual(A, b, x) < 0.1 * _residual(A, b, y)
    assert output["iter"] == 6 and len(output["errvec"]) == 6
    assert output["errvec"][-1] < output["errvec"][0]


def test_theta_zero_enriches_every_dimension():
    A, b = _problem()
    np.random.seed(1)
    _, output = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, theta=0.0, max_iterations=4, tolerance=0.0, stagnation=0.0,
        display=False).solve()
    assert all(e == [0, 1, 2] for e in output["enriched"])
    assert output["dims"][-1] == (4, 4, 4)


def test_anisotropic_problem_gives_anisotropic_spaces():
    A, b = _problem(weights=[1.0, 0.01, 0.01])
    np.random.seed(2)
    _, output = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, theta=0.9, max_iterations=6, tolerance=0.0, stagnation=0.0,
        display=False).solve()
    assert any(len(e) < 3 for e in output["enriched"][1:])
    # fewer frame vectors than an isotropic enrichment (6 per dimension)
    assert sum(output["dims"][-1]) < 3 * 6


@pytest.mark.parametrize("algorithm", ["amen", "als"])
def test_core_algorithms_and_options(algorithm, capsys):
    A, b = _problem(d=3)
    np.random.seed(3)
    x0, _ = tensap.GreedyLinearSolver(A, b, max_iterations=1, display=False).solve()
    solver = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, x0=x0, core_algorithm=algorithm, check_residual=2, max_iterations=3,
        tolerance=0.0, stagnation=0.0, display=True)
    x, output = solver.solve()
    assert output["errvec"][0] is None and output["errvec"][-1] is not None
    assert "AnisotropicGreedy" in capsys.readouterr().out

    def energy(v):
        return 0.5 * (A @ v).dot(v) - b.dot(v)

    assert energy(x) < energy(x0)


def test_minimize_residual_and_stopping():
    A, b = _problem(n=6, d=3)
    np.random.seed(4)
    x, output = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, minimize_residual=True, max_iterations=20, tolerance=1e-3,
        display=False).solve()
    assert output["flag"] in (0, 2)
    _, output = tensap.AnisotropicGreedySpaceLinearSolver(
        A, b, max_iterations=20, stagnation=0.5, tolerance=0.0, display=False).solve()
    assert output["flag"] == 2
