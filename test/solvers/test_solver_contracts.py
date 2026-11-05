"""Contracts shared by the iterative solvers: an exact initial
guess or b = 0 is accepted before any division, convergence is tested before
stagnation, a non-positive curvature in PCG is a breakdown (flag 3)."""

import numpy as np
import pytest
import tensap


def _system(n=4):
    A = tensap.BaseTuckerTensor.eye([n, n])
    rng = np.random.default_rng(0)
    b = tensap.TuckerTensorVector.randn([2, 2], [n, n])
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
    # the residual norm of a Tucker difference is computed with a round-off
    # of the order of sqrt(eps)
    x, output = make(A, b, display=False, tolerance=1e-6, stagnation=1e3).solve()
    assert output["flag"] == 0 and output["iter"] == 1


def test_pcg_breakdown_on_a_negative_operator():
    A, b, _ = _system()
    x, output = tensap.TruncatedPCG(A * -1.0, b, tensap.Truncator(tolerance=1e-12),
                                    display=False).solve()
    assert output["flag"] == 3 and output["iter"] == 1 and output["error"] == pytest.approx(1.0)
