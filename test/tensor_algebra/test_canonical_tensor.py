"""
Unit tests of tensap.CanonicalTensor.

A canonical tensor x = sum_k sigma_k v1_k x ... x vd_k is stored as its
factor matrices space[nu] (n_nu x r) and its diagonal core sigma. Every
operation is checked against the same computation on the dense array.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _dense(space, sigma):
    order = len(space)
    letters = "abcdefgh"[:order]
    expr = ",".join(f"{c}r" for c in letters) + ",r->" + letters
    return np.einsum(expr, *space, sigma)


def _canonical(rng, shape=(3, 4, 5), rank=2):
    space = [rng.standard_normal((n, rank)) for n in shape]
    sigma = rng.standard_normal(rank)
    return tensap.CanonicalTensor(space, sigma), _dense(space, sigma)


# --- construction and properties ---------------------------------------------


# --- arithmetic ---------------------------------------------------------------


def test_dot_and_norm(rng):
    x, a = _canonical(rng)
    y, b = _canonical(rng, rank=3)
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))
    with pytest.raises(NotImplementedError):
        x.norm(np.eye(3))


# --- contractions ---------------------------------------------------------------


# --- diagonal ---------------------------------------------------------------------


def test_eval_diag_does_not_modify_the_tensor(rng):
    x, a = _canonical(rng, (3, 3, 3))
    x.eval_diag()
    npt.assert_allclose(x.numpy(), a)


def test_eval_diag_on_some_dimensions(rng):
    x, a = _canonical(rng, (3, 4, 3))
    expected = np.stack([a[i, :, i] for i in range(3)])
    npt.assert_allclose(x.eval_diag([0, 2]).numpy(), expected)


# --- orthogonalization and conversions ----------------------------------------


# --- gradient of the diagonal evaluation ------------------------------------------


def test_sum_of_canonical_tensors(rng):
    x, a = _canonical(rng)
    y, b = _canonical(rng)
    npt.assert_allclose(sum([x, y]).numpy(), a + b)
