"""
Unit tests of TuckerTensorVector.power and sqrt (entrywise powers).

Oracle: the same powers of the dense array.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _positive(n=6, d=3, seed=0):
    rng = np.random.default_rng(seed)
    frames = tensap.TFrameVector([
        tensap.FrameVector(np.stack([np.ones(n), 0.3 * rng.random(n)])[:, :, None])
        for _ in range(d)
    ])
    return tensap.TuckerTensorVector(tensap.DiagonalTensor(np.array([1.0, 0.5]), d), frames)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_nonnegative_integer_powers_are_exact(n):
    x = _positive()
    npt.assert_allclose(x.power(n).full().numpy(), x.full().numpy() ** n, atol=1e-14)


@pytest.mark.parametrize("n", [-1, -2])
def test_negative_powers(n):
    x = _positive()
    expected = x.full().numpy() ** n
    error = np.abs(x.power(n, tolerance=1e-8).full().numpy() - expected).max()
    assert error < 1e-4 * np.abs(expected).max()


def test_sqrt_and_power_one_half(capsys):
    x = _positive()
    expected = np.sqrt(x.full().numpy())
    y = x.sqrt(inverse_tolerance=1e-8, display=True)
    assert np.abs(y.full().numpy() - expected).max() < 1e-5
    assert "Newton" in capsys.readouterr().out
    z = x.power(0.5)  # Matlab defaults: inverse at 1e-2
    assert np.abs(z.full().numpy() - expected).max() < 1e-2


def test_sqrt_without_convergence_warns():
    x = _positive()
    with pytest.warns(UserWarning):
        x.sqrt(max_iterations=1, tolerance=1e-30)


def test_power_rejects_non_integers():
    with pytest.raises(ValueError):
        _positive().power(1.5)


def test_full_core_inverse():
    x = _positive()
    x = tensap.TuckerTensorVector(tensap.FullTensor(x.core.full().numpy()), x.tensor_frame)
    expected = 1 / x.full().numpy()
    error = np.abs(x.power(-1, tolerance=1e-8).full().numpy() - expected).max()
    assert error < 1e-4 * np.abs(expected).max()
