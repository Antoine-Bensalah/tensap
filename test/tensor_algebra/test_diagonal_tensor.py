"""
Unit tests of tensap.DiagonalTensor.

A diagonal tensor of order d and rank r has entries x(i, ..., i) = data[i].
Every operation is checked against the same computation on its dense array.
"""

import numpy as np
import numpy.testing as npt
import tensap


def _diag(rng, rank=3, order=3):
    data = rng.standard_normal(rank)
    dense = np.zeros((rank,) * order)
    for i in range(rank):
        dense[(i,) * order] = data[i]
    return tensap.DiagonalTensor(data, order), dense


# --- construction and properties ---------------------------------------------


# --- arithmetic ---------------------------------------------------------------


def test_multiplication_by_a_scalar(rng):
    x, a = _diag(rng)
    npt.assert_allclose((2.0 * x).numpy(), 2.0 * a)


# --- structure ---------------------------------------------------------------------


# --- contractions ---------------------------------------------------------------


def test_dot_with_rank_one_metric(rng):
    x, a = _diag(rng, 3, 2)
    y, b = _diag(rng, 4, 2)
    M = [rng.standard_normal((3, 4)) for _ in range(2)]
    npt.assert_allclose(
        x.dot_with_rank_one_metric(y, M), np.einsum("ij,kl,ik,jl->", a, b, *M)
    )


# --- diagonal and evaluations ----------------------------------------------------


def test_eval_diag_on_non_contiguous_dimensions(rng):
    x, _ = _diag(rng, 3, 3)
    npt.assert_allclose(x.eval_diag([0, 2]).numpy(), x.full().eval_diag([0, 2]).numpy())


# --- conversions ------------------------------------------------------------------
