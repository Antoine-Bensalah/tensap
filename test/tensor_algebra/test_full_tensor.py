"""
Unit tests of tensap.FullTensor.

Every operation is checked against the same computation done directly on the
numpy array (the oracle), on small tensors.
"""

import numpy as np
import numpy.testing as npt
import tensap

SHAPES = [(3,), (3, 4), (2, 3, 4)]


def _tensor(rng, shape):
    data = rng.standard_normal(shape)
    return tensap.FullTensor(data), data


# --- construction and properties ---------------------------------------------


def test_size_is_the_number_of_elements(rng):
    x, _ = _tensor(rng, (2, 3, 4))
    assert np.ndim(x.size) == 0 and x.size == 24


# --- arithmetic ---------------------------------------------------------------


# --- indexing ---------------------------------------------------------------------


def test_eval_at_indices_several_dimensions(rng):
    x, a = _tensor(rng, (3, 4, 5))
    indices = np.array([[0, 1], [2, 4]])  # pairs (i0, i2)
    s = x.eval_at_indices(indices, [0, 2])
    expected = np.stack([a[0, :, 1], a[2, :, 4]], axis=0)
    npt.assert_allclose(s.numpy(), expected)


# --- reshaping ----------------------------------------------------------------


# --- concatenation and reductions ---------------------------------------------


def test_reductions(rng):
    x, a = _tensor(rng, (2, 3, 4))
    npt.assert_allclose(x.reduce_sum(1).numpy(), a.sum(1, keepdims=True))
    npt.assert_allclose(x.reduce_mean([0, 2]).numpy(), a.mean((0, 2), keepdims=True))


# --- products and contractions ------------------------------------------------


def test_kron(rng):
    x, a = _tensor(rng, (2, 3))
    y, b = _tensor(rng, (4, 2))
    npt.assert_allclose(x.kron(y).numpy(), np.kron(a, b))
    x3, a3 = _tensor(rng, (2, 2, 2))
    y3, b3 = _tensor(rng, (3, 1, 2))
    expected = np.einsum("ijk,lmn->iljmkn", a3, b3).reshape(6, 2, 4)
    npt.assert_allclose(x3.kron(y3).numpy(), expected)
    # different orders: the missing dimensions of the second tensor have size 1
    z = x3.kron(x)
    assert tuple(z.shape) == (4, 6, 2)
    npt.assert_allclose(z.numpy(), np.einsum("ijk,lm->iljmk", a3, a).reshape(4, 6, 2))


# --- orthogonalization and decompositions -------------------------------------


# --- conversions ------------------------------------------------------------------


# --- remaining call variants ---------------------------------------------------


def test_eval_at_indices_accepts_transposed_indices(rng):
    x, a = _tensor(rng, (3, 4, 5))
    indices = np.array([[0, 2, 1], [1, 4, 3]])  # one column per evaluation
    s = x.eval_at_indices(indices, [0, 2])
    expected = np.stack([a[0, :, 1], a[2, :, 4], a[1, :, 3]], axis=0)
    npt.assert_allclose(s.numpy(), expected)
