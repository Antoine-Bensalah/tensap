"""
Unit tests of the PyTorch benchmarks of Poincaré learning (skipped without
PyTorch).

The Jacobians computed by automatic differentiation are checked against
finite differences, and the benchmarks shared with the numpy module against
their numpy versions.
"""

import numpy as np
import numpy.testing as npt
import pytest

torch = pytest.importorskip("torch")

from tensap.poincare_learning.benchmarks.poincare_benchmarks import (  # noqa: E402
    build_benchmark,
)
from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import (  # noqa: E402
    build_benchmark_torch,
)

CASES = [
    ("borehole", {}),
    ("ishigami", {}),
    ("sin_of_asum", {"d": 4}),
    ("canonical_rank_2", {}),
    ("mixture", {}),
    ("field", {}),
    ("henon_heiles", {"d": 4}),
    ("sin_squared_norm", {"d": 4}),
    ("cos_squared_norm", {"d": 4}),
    ("sum_cos_sin_squared_norm", {"d": 4}),
    ("exp_mean_sin_exp_cos", {"d": 5}),
    ("cos_low_rank", {"d": 8}),
    ("schaffer", {"d": 4}),
    ("ackley", {"d": 4}),
    ("robot_arm", {"d": 8}),
    ("piston", {}),
    ("cos_tree", {"d": 8}),
    ("quartic_sin_collective", {"d": 5}),
]


def _evaluate(fun, x):
    return np.reshape(fun(x), (x.shape[0], -1))


@pytest.mark.parametrize("case, kwargs", CASES)
def test_jacobians_by_finite_differences(case, kwargs):
    fun, fun_jac, X = build_benchmark_torch(case, **kwargs)
    x = X.random(5)
    y = _evaluate(fun, x)
    assert np.all(np.isfinite(y))
    jac = np.reshape(fun_jac(x), (5, y.shape[1], x.shape[1]))
    h = 1e-6
    for j in range(x.shape[1]):
        e = np.zeros(x.shape[1])
        e[j] = h
        fd = (_evaluate(fun, x + e) - _evaluate(fun, x - e)) / (2 * h)
        npt.assert_allclose(jac[:, :, j], fd, rtol=1e-5, atol=1e-6 * (1 + np.abs(y).max()))


@pytest.mark.parametrize(
    "case, kwargs",
    [
        ("sin_squared_norm", {"d": 4, "c": 0.7, "R": np.diag([1.0, 2.0, 0.5, 1.0])}),
        ("cos_squared_norm", {"d": 4}),
        ("sum_cos_sin_squared_norm", {"d": 3, "R1": np.diag([1.0, 2.0, 3.0])}),
        ("exp_mean_sin_exp_cos", {"d": 5, "c": 2.0}),
        ("quartic_sin_collective", {"d": 4, "mat_lst": [np.eye(3), np.diag([1.0, 2.0, 3.0])]}),
    ],
)
def test_torch_and_numpy_benchmarks_agree(case, kwargs):
    fun, fun_jac, X = build_benchmark_torch(case, **kwargs)
    fun_np, fun_jac_np, _ = build_benchmark(case, **kwargs)
    x = X.random(6)
    npt.assert_allclose(_evaluate(fun, x), _evaluate(fun_np, x), rtol=1e-12)
    npt.assert_allclose(np.reshape(fun_jac(x), (6, -1)), np.reshape(fun_jac_np(x), (6, -1)),
                        rtol=1e-10, atol=1e-12)


def test_known_values():
    # Ishigami at (pi/2, pi/2, 1): 1 + a + b
    fun, _, _ = build_benchmark_torch("ishigami")
    npt.assert_allclose(fun(np.array([[np.pi / 2, np.pi / 2, 1.0]])), 1 + 7 + 0.1)
    # Ackley: global minimum 0 at the origin
    fun, _, _ = build_benchmark_torch("ackley", d=4)
    npt.assert_allclose(_evaluate(fun, np.zeros((1, 4))), 0, atol=1e-12)
    # sin of a sum, with default and given coefficients
    fun, _, _ = build_benchmark_torch("sin_of_asum", d=3)
    npt.assert_allclose(_evaluate(fun, np.array([[0.1, 0.2, 0.3]])), np.sin(0.6))
    fun, _, _ = build_benchmark_torch("sin_of_asum", d=2, c=torch.tensor([2.0, -1.0],
                                                                         dtype=torch.float64))
    npt.assert_allclose(_evaluate(fun, np.array([[0.5, 0.25]])), np.sin(0.75))


def test_piston_against_its_formula():
    # cycle time of a piston (Kenett and Zacks), with the variables mapped
    # from the standard variables to their ranges
    import tensap
    from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import (
        _get_shift_scaling,
    )

    ranges = [(30, 60), (0.005, 0.02), (0.002, 0.01), (1000, 5000), (90000, 110000),
              (290, 296), (340, 360)]
    shift, scaling = _get_shift_scaling(
        tensap.RandomVector([tensap.UniformRandomVariable(*r) for r in ranges]))
    fun, _, X = build_benchmark_torch("piston")
    z = X.random(10)
    M, S, V0, k, P0, Ta, T0 = (z * scaling + shift).T
    for value, (a, b) in zip((M, S, V0, k, P0, Ta, T0), ranges):
        assert np.all((value >= a) & (value <= b))
    A = P0 * S + 19.62 * M - k * V0 / S
    V = S / (2 * k) * (np.sqrt(A**2 + 4 * k * P0 * V0 / T0 * Ta) - A)
    C = 2 * np.pi * np.sqrt(M / (k + S**2 * P0 * V0 / T0 * Ta / V**2))
    npt.assert_allclose(_evaluate(fun, z).ravel(), C, rtol=1e-12)


def test_unknown_benchmark():
    with pytest.raises(NotImplementedError):
        build_benchmark_torch("other")
