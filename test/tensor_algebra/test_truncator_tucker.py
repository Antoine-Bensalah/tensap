"""
Unit tests of Truncator.truncate on tensap.BaseTuckerTensor with every core.

Order 2: compared with the truncated SVD of the dense matrix (Eckart-Young).
Order > 2: diagonal cores (greedy recompression) and tree-based cores (hsvd
of the tree with the frames absorbed in the leaves), checked on tensors of
known low rank stored with redundant terms, against the dense tensor.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap import (
    FrameVector, FrameOperator, TFrameVector, TFrameOperator,
    TuckerTensorVector, TuckerTensorOperator,
)


def _redundant_canonical(rng, shape, rank, copies=2):
    """Canonical tensor of rank `rank` stored with copies * rank terms."""
    frames = [rng.standard_normal((rank, n, 1)) for n in shape]
    weights = rng.standard_normal(rank)
    frames = TFrameVector([FrameVector(np.concatenate([f] * copies)) for f in frames])
    core = tensap.DiagonalTensor(np.concatenate([weights / copies] * copies), len(shape))
    return TuckerTensorVector(core, frames)


def _relative_error(y, x):
    return np.linalg.norm(y.numpy() - x.numpy()) / np.linalg.norm(x.numpy())


# --- order 2: truncated SVD -----------------------------------------------------------


@pytest.mark.parametrize("kind", ["diagonal", "full"])
@pytest.mark.parametrize("max_rank", [2, np.inf])
def test_order_two_is_the_truncated_svd(rng, kind, max_rank):
    x = _redundant_canonical(rng, [6, 5], 3)
    if kind == "full":
        x = TuckerTensorVector(tensap.FullTensor(x.core.full()), x.tensor_frame)
    y = tensap.Truncator(tolerance=1e-10, max_rank=max_rank).truncate(x)

    assert isinstance(y, TuckerTensorVector)
    assert isinstance(y.core, tensap.DiagonalTensor)
    u, s, vt = np.linalg.svd(x.numpy())
    r = min(3, max_rank)
    assert y.representation_rank() == (r, r)
    npt.assert_allclose(np.sort(np.abs(y.core.data))[::-1], s[:r])
    npt.assert_allclose(y.numpy(), (u[:, :r] * s[:r]) @ vt[:r], atol=1e-10)


def test_order_two_operator(rng):
    # the operator is truncated as its vectorization
    frames = TFrameOperator([FrameOperator(rng.standard_normal((2, 3, 4))),
                             FrameOperator(rng.standard_normal((2, 2, 2)))])
    frames = TFrameOperator([FrameOperator(np.concatenate([f.data, f.data]))
                             for f in frames.frames])
    A = TuckerTensorOperator(tensap.DiagonalTensor(np.ones(4), 2), frames)
    B = tensap.Truncator(tolerance=1e-10).truncate(A)
    assert isinstance(B, TuckerTensorOperator)
    assert B.shape == A.shape
    assert B.representation_rank() == (2, 2)
    npt.assert_allclose(B.numpy(), A.numpy(), atol=1e-10)


# --- diagonal core: greedy recompression -------------------------------------------------


def test_diagonal_core_greedy_recompression(rng):
    np.random.seed(0)  # random initializations of the rank-one ALS
    x = _redundant_canonical(rng, [5, 4, 6], 2, copies=3)
    y = tensap.Truncator(tolerance=1e-6).truncate(x)
    assert isinstance(y.core, tensap.DiagonalTensor)
    assert _relative_error(y, x) <= 1e-6
    assert max(y.representation_rank()) <= 5  # at most n_1 corrections


def test_diagonal_core_pure_greedy_option(rng):
    # pure greedy algorithm (MATLAB): less accurate than with updates
    x = _redundant_canonical(rng, [5, 4, 6], 4)
    errors = []
    for greedy_updates in [False, True]:
        np.random.seed(0)
        truncator = tensap.Truncator(tolerance=1e-12, max_rank=3,
                                     greedy_updates=greedy_updates)
        with pytest.warns(UserWarning, match="not accurate enough"):
            y = truncator.truncate(x)
        assert isinstance(y.core, tensap.DiagonalTensor)
        errors.append(_relative_error(y, x))
    assert errors[1] <= errors[0]
    assert "greedy_updates = True" in repr(truncator)


def test_diagonal_core_not_accurate_enough_warns(rng):
    np.random.seed(0)
    x = _redundant_canonical(rng, [5, 4, 6], 3)
    with pytest.warns(UserWarning, match="not accurate enough"):
        y = tensap.Truncator(tolerance=1e-8, max_rank=1).truncate(x)
    assert y.representation_rank() == (1, 1, 1)


def test_diagonal_core_zero_tensor(rng):
    x = _redundant_canonical(rng, [3, 3, 3], 2) * 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = tensap.Truncator().truncate(x)
    assert y.norm() == 0


# --- tree-based core: hsvd --------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [1e-10, 1e-2])
def test_tree_core_is_the_hsvd_of_the_dense_tensor(rng, tolerance):
    np.random.seed(1)
    tree = tensap.DimensionTree.balanced(4)
    t = tensap.TreeBasedTensor.randn(tree, [1, 3, 3, 3, 2, 3, 2], [4, 5, 3, 6])
    x = tensap.TuckerTensorVector.from_tree_based_tensor(t)
    # redundant representation: x + x has twice the ranks, 0.5 * (x + x) = x
    x = x + x
    x = 0.5 * x

    truncator = tensap.Truncator(tolerance=tolerance)
    y = truncator.truncate(x)
    reference = truncator.hsvd(x.full(), tree)

    assert isinstance(y, TuckerTensorVector)
    assert isinstance(y.core, tensap.TreeBasedTensor)
    npt.assert_array_equal(y.tree_based_tensor().ranks, reference.ranks)
    assert _relative_error(y, x) <= tolerance


def test_tree_based_tensor_round_trip(rng):
    np.random.seed(2)
    t = tensap.TreeBasedTensor.randn(tensap.DimensionTree.linear(4), None, [3, 4, 2, 5])
    x = tensap.TuckerTensorVector.from_tree_based_tensor(t)
    assert isinstance(x.core, tensap.TreeBasedTensor)
    npt.assert_allclose(x.numpy(), t.full().numpy())
    npt.assert_allclose(x.tree_based_tensor().full().numpy(), t.full().numpy())


# --- other cores ------------------------------------------------------------------------


def test_sparse_core_not_implemented(rng):
    frames = TFrameVector([FrameVector(rng.standard_normal((2, 3, 1))) for _ in range(3)])
    x = TuckerTensorVector(tensap.SparseTensor(rng.standard_normal((2, 2, 2))), frames)
    with pytest.raises(NotImplementedError):
        tensap.Truncator().truncate(x)
