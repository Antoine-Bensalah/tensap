"""
Unit tests of the Tucker base operations ported in P0.

normalize_basis, full of an operator, hadamard_product, super_tensor and
convert_cores of tensap.BaseTuckerTensor. Every result is checked against
the dense array of the tensor, built here by an explicit contraction of the
core with the frames.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap import (
    FrameVector, FrameOperator, TFrameVector, TFrameOperator,
    TuckerTensorVector, TuckerTensorOperator,
)

CORE_TYPES = ["full", "diagonal", "sparse"]


def _core(rng, kind, ranks):
    if kind == "diagonal":
        return tensap.DiagonalTensor(rng.standard_normal(ranks[0]), len(ranks))
    data = rng.standard_normal(ranks)
    if kind == "sparse":
        data[rng.random(ranks) < 0.4] = 0.0
        return tensap.SparseTensor(data)
    return tensap.FullTensor(data)


def _vector(rng, kind, shape, ranks):
    if kind == "diagonal":
        ranks = [ranks[0]] * len(shape)
    frames = TFrameVector([FrameVector(rng.standard_normal((r, n, 1)))
                           for r, n in zip(ranks, shape)])
    return TuckerTensorVector(_core(rng, kind, ranks), frames)


def _operator(rng, kind, shape, ranks):
    if kind == "diagonal":
        ranks = [ranks[0]] * len(shape)
    frames = TFrameOperator([FrameOperator(rng.standard_normal((r, n1, n2)))
                             for r, (n1, n2) in zip(ranks, shape)])
    return TuckerTensorOperator(_core(rng, kind, ranks), frames)


def _dense(x):
    """Oracle: sum over k of core[k] * frame_1[k_1] o ... o frame_d[k_d]."""
    ranks = "abcdefgh"[:x.order]
    rows, cols = "ijklmn"[:x.order], "pqrstu"[:x.order]
    # e.g. order 2: "ab,aip,bjq->ipjq"
    spec = ",".join([ranks] + [k + i + j for k, i, j in zip(ranks, rows, cols)])
    spec += "->" + "".join(i + j for i, j in zip(rows, cols))
    ops = [x.core.full().numpy()] + [f.data for f in x.tensor_frame.frames]
    dense = np.einsum(spec, *ops)
    if isinstance(x, TuckerTensorVector):
        dense = dense.reshape([n for n, _ in x.shape])
    return dense


def _matrix(dense, shape):
    """Matrix of an operator from its array (n1, m1, ..., nd, md)."""
    d = len(shape)
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    n = int(np.prod([s[0] for s in shape]))
    m = int(np.prod([s[1] for s in shape]))
    return dense.transpose(perm).reshape(n, m)


# --- full of an operator ------------------------------------------------------


@pytest.mark.parametrize("kind", CORE_TYPES)
@pytest.mark.parametrize("shape", [
    [(3, 4), (2, 5)],
    [(3, 1), (1, 4), (2, 2)],
    [(1, 1), (3, 1)],
])
def test_operator_full_shape_and_entries(rng, kind, shape):
    A = _operator(rng, kind, shape, [2, 3, 2][:len(shape)])
    full = A.full()
    assert isinstance(full, tensap.FullTensor)
    npt.assert_array_equal(full.shape, [n for s in shape for n in s])
    npt.assert_allclose(full.numpy(), _dense(A))


@pytest.mark.parametrize("kind", CORE_TYPES)
def test_operator_full_matches_matvec(rng, kind):
    # the matrix read from full() acts on vectors as A @ x
    shape = [(3, 4), (1, 2), (2, 3)]
    A = _operator(rng, kind, shape, [2, 2, 3])
    x = _vector(rng, "full", [4, 2, 3], [2, 1, 2])
    npt.assert_allclose((A @ x).numpy().ravel(),
                        _matrix(A.numpy(), shape) @ x.numpy().ravel())


@pytest.mark.parametrize("kind", CORE_TYPES)
def test_vector_full_shape(rng, kind):
    x = _vector(rng, kind, [3, 1, 4], [2, 1, 2])
    npt.assert_array_equal(x.full().shape, [3, 1, 4])
    npt.assert_allclose(x.numpy(), _dense(x))


def test_full_of_vectorized_operator(rng):
    A = _operator(rng, "full", [(3, 2), (2, 4)], [2, 3])
    npt.assert_allclose(A.vectorize().numpy(), A.numpy().reshape(6, 8))


# --- contraction with vectors -------------------------------------------------


@pytest.mark.parametrize("kind", CORE_TYPES)
def test_tensor_vector_product_on_some_dimensions(rng, kind):
    # with a diagonal core, the contraction used to fail (no dims argument)
    x = _vector(rng, kind, [3, 4, 5], [2, 2, 2])
    v = rng.standard_normal(4)
    y = x.tensor_vector_product(v, [1])
    npt.assert_allclose(y.numpy(), np.einsum("ijk,j->ik", x.numpy(), v))


# --- normalize_basis ----------------------------------------------------------


@pytest.mark.parametrize("kind", CORE_TYPES)
@pytest.mark.parametrize("make", [_vector, _operator])
def test_normalize_basis(rng, kind, make):
    shape = [4, 3, 2] if make is _vector else [(4, 2), (3, 3), (2, 1)]
    x = make(rng, kind, shape, [2, 3, 2])
    dense = x.numpy()
    frames_before = [f.data.copy() for f in x.tensor_frame.frames]

    y = x.normalize_basis()

    assert y is not x and type(y) is type(x)
    assert type(y.core) is type(x.core)
    npt.assert_allclose(y.numpy(), dense)
    for f in y.tensor_frame.frames:
        npt.assert_allclose(np.linalg.norm(f.data.reshape(f.dim, -1), axis=1), 1.0)
    # self is not modified
    npt.assert_allclose(x.numpy(), dense)
    for f, data in zip(x.tensor_frame.frames, frames_before):
        npt.assert_array_equal(f.data, data)


def test_normalize_basis_zero_element(rng):
    # a zero frame element is kept as is, without a division by zero
    x = _vector(rng, "full", [3, 4], [2, 2])
    data = x.tensor_frame.frames[0].data.copy()
    data[1] = 0.0
    frames = TFrameVector([FrameVector(data), x.tensor_frame.frames[1]])
    x = TuckerTensorVector(x.core, frames)
    y = x.normalize_basis()
    npt.assert_allclose(y.numpy(), x.numpy())
    npt.assert_array_equal(y.tensor_frame.frames[0].data[1], 0.0)


# --- hadamard_product ---------------------------------------------------------


@pytest.mark.parametrize("kind_x, kind_y", [
    ("full", "full"), ("diagonal", "diagonal"), ("full", "diagonal"),
    ("full", "sparse"),
    ("sparse", "sparse"),
    ("diagonal", "sparse"),
])
@pytest.mark.parametrize("make", [_vector, _operator])
def test_hadamard_product(rng, kind_x, kind_y, make):
    shape = [3, 1, 4] if make is _vector else [(3, 2), (1, 4), (2, 2)]
    x = make(rng, kind_x, shape, [2, 1, 3])
    y = make(rng, kind_y, shape, [3, 2, 2])
    z = x.hadamard_product(y)
    assert type(z) is type(x)
    assert z.representation_rank() == tuple(
        rx * ry for rx, ry in zip(x.representation_rank(), y.representation_rank()))
    npt.assert_allclose(z.numpy(), x.numpy() * y.numpy())


def test_hadamard_product_diagonal_cores_stay_diagonal(rng):
    x = _vector(rng, "diagonal", [3, 4], [2])
    y = _vector(rng, "diagonal", [3, 4], [3])
    z = x.hadamard_product(y)
    assert isinstance(z.core, tensap.DiagonalTensor)
    assert z.representation_rank() == (6, 6)


def test_hadamard_product_errors(rng):
    x = _vector(rng, "full", [3, 4], [2, 2])
    with pytest.raises(TypeError):
        x.hadamard_product(x.to_operator())
    with pytest.raises(ValueError):
        x.hadamard_product(_vector(rng, "full", [3, 5], [2, 2]))


# --- super_tensor -------------------------------------------------------------


@pytest.mark.parametrize("kind", CORE_TYPES)
def test_vector_super_tensor(rng, kind):
    x = _vector(rng, kind, [3, 2, 4], [2, 2, 3])
    rows = [np.array([4, 0, 2]), None, np.array([1, 2, 5, 6])]
    s = x.super_tensor([5, 2, 7], rows)
    expected = np.zeros((5, 2, 7))
    expected[np.ix_(rows[0], np.arange(2), rows[2])] = x.numpy()
    npt.assert_allclose(s.numpy(), expected)
    # sub_tensor is the reciprocal operation
    npt.assert_allclose(s.sub_tensor([rows[0], rows[2]], [0, 2]).numpy(), x.numpy())


@pytest.mark.parametrize("kind", CORE_TYPES)
def test_operator_super_tensor(rng, kind):
    A = _operator(rng, kind, [(2, 3), (3, 2)], [2, 3])
    indices = [(np.array([1, 3]), None), (np.array([0, 2, 4]), np.array([3, 1]))]
    s = A.super_tensor([(4, 3), (5, 4)], indices)
    expected = np.zeros((4, 3, 5, 4))
    expected[np.ix_([1, 3], [0, 1, 2], [0, 2, 4], [3, 1])] = A.numpy()
    npt.assert_allclose(s.numpy(), expected)


def test_super_tensor_errors(rng):
    x = _vector(rng, "full", [3, 4], [2, 2])
    with pytest.raises(ValueError):  # None requires an unchanged size
        x.super_tensor([5, 6], [np.arange(3), None])
    with pytest.raises(ValueError):  # one index per entry
        x.super_tensor([5, 4], [np.arange(2), None])
    with pytest.raises(ValueError):  # one entry per dimension
        x.super_tensor([5], [np.arange(3)])


# --- convert_cores ------------------------------------------------------------


@pytest.mark.parametrize("kind_x, kind_y, expected", [
    ("full", "diagonal", tensap.FullTensor),
    ("sparse", "diagonal", tensap.SparseTensor),
    ("diagonal", "diagonal", tensap.DiagonalTensor),
])
def test_convert_cores(rng, kind_x, kind_y, expected):
    x = _vector(rng, kind_x, [3, 4], [2, 2])
    y = _vector(rng, kind_y, [3, 4], [2, 2])
    xc, yc = x.convert_cores(y)
    assert type(xc.core) is expected and type(yc.core) is expected
    npt.assert_allclose(xc.numpy(), x.numpy())
    npt.assert_allclose(yc.numpy(), y.numpy())
    assert xc.tensor_frame is x.tensor_frame


def test_convert_tensors_aligns_cores_of_same_class(rng):
    # two TuckerTensorOperator with a full and a diagonal core: the cores are
    # aligned, they used to be returned unchanged
    A = _operator(rng, "full", [(2, 2), (3, 3)], [2, 2])
    B = _operator(rng, "diagonal", [(2, 2), (3, 3)], [2])
    Ac, Bc = tensap.convert_tensors(A, B)
    assert isinstance(Ac.core, tensap.FullTensor)
    assert isinstance(Bc.core, tensap.FullTensor)
    npt.assert_allclose(Bc.numpy(), B.numpy())


# --- tree-based cores ---------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3, 5])
def test_laplacian_tree_based_tensor(rng, d):
    K = rng.standard_normal((3, 3))
    M = rng.standard_normal((3, 3))
    factory = tensap.LaplacianLikeOperatorFactory(K, M, d)
    A = factory.make_tree_based_tensor()
    assert isinstance(A.core, tensap.TreeBasedTensor)
    npt.assert_allclose(A.numpy(), factory.make_tucker_tensor().numpy())
    x = _vector(rng, "full", [3] * d, [2] * d)
    npt.assert_allclose((A @ x).numpy(), (factory.make_canonical_tensor() @ x).numpy())


def test_laplacian_tree_based_tensor_requires_a_binary_tree():
    factory = tensap.LaplacianLikeOperatorFactory(np.eye(2), np.eye(2), 3)
    with pytest.raises(ValueError):
        factory.make_tree_based_tensor(tensap.DimensionTree.trivial(3))


def _matricization_singular_values(a, mu):
    return np.linalg.svd(np.moveaxis(a, mu, 0).reshape(a.shape[mu], -1), compute_uv=False)


def test_singular_values_tree_core(rng):
    np.random.seed(3)
    t = tensap.TreeBasedTensor.randn(tensap.DimensionTree.balanced(4), None, [4, 3, 5, 2])
    x = TuckerTensorVector.from_tree_based_tensor(t)
    x = x.tensor_matrix_product([rng.standard_normal((n, n)) for n in [4, 3, 5, 2]])
    sv = x.singular_values()
    a = x.numpy()
    tree = t.tree
    for mu in range(4):
        s = sv[tree.dim2ind[mu] - 1]
        # computed from the Gram matrices: accuracy of order sqrt(eps) * s[0]
        npt.assert_allclose(s, _matricization_singular_values(a, mu)[:s.size],
                            atol=1e-6 * s[0])


def test_singular_values_full_and_order_two(rng):
    x = _vector(rng, "full", [4, 3, 5], [2, 3, 2])
    for mu, s in enumerate(x.singular_values()):
        npt.assert_allclose(s, _matricization_singular_values(x.numpy(), mu)[:s.size])
    y = _vector(rng, "diagonal", [4, 3], [2])
    npt.assert_allclose(y.singular_values(), np.linalg.svd(y.numpy(), compute_uv=False)[:2])


def test_singular_values_diagonal_core_not_implemented(rng):
    with pytest.raises(NotImplementedError):
        _vector(rng, "diagonal", [3, 3, 3], [2]).singular_values()


def test_rank_one_constructors_have_a_diagonal_core():
    # as TuckerLikeTensor.ones in MATLAB: no full core of shape (1, ..., 1)
    x = TuckerTensorVector.ones([1, 1, 1], [3, 4, 5])
    assert isinstance(x.core, tensap.DiagonalTensor)
    npt.assert_allclose(x.numpy(), np.ones((3, 4, 5)))
    A = tensap.BaseTuckerTensor.rand([1, 1], [(2, 3), (4, 2)])
    assert isinstance(A.core, tensap.DiagonalTensor)
    assert isinstance(TuckerTensorVector.ones([2, 1], [3, 4]).core, tensap.FullTensor)


# --- sparsity pattern of the vectorization ------------------------------------------


def _laplacian_tucker(n=6, d=3):
    K = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d).make_tucker_tensor()


def test_vectorize_with_sparsity_pattern():
    A = _laplacian_tucker()
    v, patterns = A.vectorize(sparsity=True)
    n = 6
    assert all(len(p) == 3 * n - 2 for p in patterns)  # tridiagonal union
    assert v.tensor_frame.frames[0].shape[0] == 3 * n - 2
    B = v.unvectorize([n] * 3, [n] * 3, patterns)
    npt.assert_allclose(B.full().numpy(), A.full().numpy())
    # without sparsity: unchanged behaviour
    w = A.vectorize()
    assert w.tensor_frame.frames[0].shape[0] == n * n


def test_truncation_of_an_operator_keeps_its_sparsity():
    A = _laplacian_tucker()
    A2 = A + A
    truncated = tensap.Truncator(tolerance=1e-12).truncate(A2)
    npt.assert_allclose(truncated.full().numpy(), 2 * A.full().numpy(), atol=1e-12)
    for frame in truncated.tensor_frame.frames:
        off_band = np.abs(np.triu(frame.data, k=2)).max(axis=(1, 2))
        assert np.all(off_band == 0.0)
