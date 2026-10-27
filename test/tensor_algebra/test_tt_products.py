"""Entrywise product rounded at a precision: Truncator.hadamard_product, the
same contract for every format (the exact product, truncated), and for two
TTTensors the DMRG fit tools.tt_products.hadamard_round (review R3)."""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap.tensor_algebra.tools.tt_products import hadamard_round


def _smooth_qtt(levels, tolerance, phase=0.0):
    t = tensap.Tensorizer(2, levels, 2)
    x = (np.arange(2 ** levels[0]) + 0.5) / 2 ** levels[0]
    y = (np.arange(2 ** levels[1]) + 0.5) / 2 ** levels[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.exp(-((X - 0.4) ** 2 + (Y - 0.6) ** 2) / 0.02) * (Y - 0.6) + np.sin(5 * X * Y + phase)
    return tensap.quantize(f, t, tolerance), f


@pytest.mark.parametrize("tolerance", [1e-4, 1e-8])
def test_tt_product_meets_the_tolerance(tolerance):
    x, f = _smooth_qtt([5, 4], tolerance)
    y, g = _smooth_qtt([5, 4], tolerance, phase=1.0)
    exact = x.hadamard_product(y)
    truncator = tensap.Truncator(tolerance=tolerance, max_rank=200)
    z = truncator.hadamard_product(x, y)
    assert isinstance(z, tensap.TTTensor)
    assert (z - exact).norm() <= 2 * tolerance * exact.norm()
    reference = truncator.truncate(exact)
    assert max(z.ranks) <= max(reference.ranks) + 2
    z2, info = hadamard_round(x, y, truncator, x0=reference)
    assert info["sweeps"] <= 3 and (z2 - exact).norm() <= 2 * tolerance * exact.norm()


def test_tt_product_rank_cap_and_special_cases():
    x, _ = _smooth_qtt([4, 4], 1e-10)
    capped = tensap.Truncator(tolerance=1e-12, max_rank=3).hadamard_product(x, x)
    assert max(capped.ranks) <= 3
    zero = x * 0.0
    z, info = hadamard_round(x, zero, tensap.Truncator(tolerance=1e-8))
    assert z.norm() == 0 and info["sweeps"] == 0
    line = tensap.TTTensor([np.arange(4.0).reshape(1, 4, 1)])
    z = tensap.Truncator(tolerance=1e-12).hadamard_product(line, line)
    npt.assert_allclose(z.full().numpy().ravel(), np.arange(4.0) ** 2)
    with pytest.raises(ValueError):
        hadamard_round(x, tensap.TTTensor([np.ones((1, 2, 1))] * 3), tensap.Truncator())


def test_the_other_formats_take_the_exact_product_truncated():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 5, 3)), rng.standard_normal((4, 5, 3))
    truncator = tensap.Truncator(tolerance=1e-12)
    full = truncator.hadamard_product(tensap.FullTensor(a), tensap.FullTensor(b))
    npt.assert_allclose(full.full().numpy(), a * b, atol=1e-10)
    ta, tb = tensap.FullTensor(a).tree_based_tensor(), tensap.FullTensor(b).tree_based_tensor()
    product = truncator.hadamard_product(ta, tb)
    assert isinstance(product, tensap.TreeBasedTensor)
    npt.assert_allclose(product.full().numpy(), a * b, atol=1e-9)
