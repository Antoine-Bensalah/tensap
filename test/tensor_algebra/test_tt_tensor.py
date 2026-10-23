"""
Unit tests of tensap.TTTensor and tensap.TTOperator (tensor-train formats).

Every operation is checked against the dense array (small cases) and, for the
conversions, against the same tensor stored as a TreeBasedTensor on the
linear tree (the tree is the oracle of the TT classes, never a fallback).
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap import TTTensor, TTOperator

SHAPE = [3, 4, 2, 5]
RANKS = [2, 3, 2]


def _tt(rng, shape=SHAPE, ranks=RANKS):
    r = [1] + list(ranks) + [1]
    return TTTensor([rng.standard_normal((r[k], n, r[k + 1])) for k, n in enumerate(shape)])


def _tt_operator(rng, shape=((3, 2), (4, 4), (2, 3)), ranks=(2, 3)):
    r = [1] + list(ranks) + [1]
    return TTOperator([rng.standard_normal((r[k], n, m, r[k + 1]))
                       for k, (n, m) in enumerate(shape)])


def _matrix(dense, shape):
    d = len(shape)
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    return dense.transpose(perm).reshape(int(np.prod([s[0] for s in shape])),
                                         int(np.prod([s[1] for s in shape])))


# --- construction --------------------------------------------------------------------


def test_properties(rng):
    x = _tt(rng)
    assert x.order == 4
    npt.assert_array_equal(x.shape, SHAPE)
    assert x.ranks == tuple(RANKS) == x.representation_rank()
    assert not x.is_orth
    assert x.storage() == sum(c.size for c in x.cores)
    assert x.sparse_storage() <= x.storage()
    assert not x.cores[0].flags.writeable  # read-only cores
    assert "TTTensor" in repr(x)


@pytest.mark.parametrize("cores", [
    [np.ones((2, 3, 1))],                       # r_0 != 1
    [np.ones((1, 3, 2)), np.ones((3, 3, 1))],   # ranks do not match
    [np.ones((1, 3))],                          # wrong number of axes
    [],
])
def test_wrong_cores(cores):
    with pytest.raises(ValueError):
        TTTensor(cores)


def test_constructors():
    np.random.seed(0)
    npt.assert_array_equal(TTTensor.ones(SHAPE).numpy(), np.ones(SHAPE))
    npt.assert_array_equal(TTTensor.zeros(SHAPE).numpy(), np.zeros(SHAPE))
    assert TTTensor.randn(SHAPE, RANKS).ranks == tuple(RANKS)
    assert np.all(TTTensor.rand(SHAPE, RANKS).numpy() >= 0)
    with pytest.raises(ValueError):
        TTTensor.randn(SHAPE, [2, 2])


# --- vector-space operations against the dense array ---------------------------------


def test_dot_norm_and_arithmetic(rng):
    x, y = _tt(rng), _tt(rng, ranks=[3, 1, 2])
    a, b = x.numpy(), y.numpy()
    assert np.isclose(x.dot(y), np.sum(a * b))
    assert np.isclose(x.norm(), np.linalg.norm(a))
    npt.assert_allclose((x + y).numpy(), a + b)
    assert (x + y).ranks == (5, 4, 4)
    npt.assert_allclose((x - y).numpy(), a - b)
    npt.assert_allclose((-x).numpy(), -a)
    npt.assert_allclose((2.5 * x).numpy(), 2.5 * a)
    npt.assert_allclose((x / 4).numpy(), a / 4)
    assert (0 + x) is x
    npt.assert_allclose(x.hadamard_product(y).numpy(), a * b)
    assert x.hadamard_product(y).ranks == (6, 3, 4)


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_orth(rng, center):
    x = _tt(rng)
    y = x.orth(center)
    npt.assert_allclose(y.numpy(), x.numpy())
    assert y.orth_center == center and y.is_orth
    for k in range(center):  # left-orthogonal
        c = y.cores[k].reshape(-1, y.cores[k].shape[-1])
        npt.assert_allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-12)
    for k in range(center + 1, 4):  # right-orthogonal
        c = y.cores[k].reshape(y.cores[k].shape[0], -1)
        npt.assert_allclose(c @ c.T, np.eye(c.shape[0]), atol=1e-12)
    assert np.isclose(y.norm(), np.linalg.norm(y.cores[center]))
    # the product by a scalar keeps the orthogonality center
    z = 3.0 * y
    assert z.orth_center == center
    assert np.isclose(z.norm(), 3 * x.norm())


def test_orth_wrong_center(rng):
    with pytest.raises(ValueError):
        _tt(rng).orth(4)


def test_incompatible_tensors(rng):
    x = _tt(rng)
    with pytest.raises(ValueError):
        x.dot(_tt(rng, shape=[3, 4, 2, 6]))
    with pytest.raises(ValueError):
        x + _tt(rng, shape=[3, 4], ranks=[2])
    with pytest.raises(TypeError):
        x.hadamard_product(_tt_operator(rng))
    with pytest.raises(TypeError):
        x + _tt_operator(rng)


def test_order_one(rng):
    x = TTTensor([rng.standard_normal((1, 5, 1))])
    y = TTTensor([rng.standard_normal((1, 5, 1))])
    npt.assert_allclose((x + y).numpy(), x.numpy() + y.numpy())
    assert np.isclose(x.norm(), np.linalg.norm(x.numpy()))
    rounded = tensap.Truncator().truncate(x)
    npt.assert_allclose(rounded.numpy(), x.numpy())


# --- products with matrices and vectors (core protocol of the Tucker format) --------


def test_tensor_matrix_and_vector_products(rng):
    x = _tt(rng)
    a = x.numpy()
    M = [rng.standard_normal((2, 3)), rng.standard_normal((6, 5))]
    npt.assert_allclose(x.tensor_matrix_product(M, [0, 3]).numpy(),
                        np.einsum("ai,dl,ijkl->ajkd", M[0], M[1], a))
    npt.assert_allclose(x.tensor_matrix_product(tensap.FullTensor(M[0]), 0).numpy(),
                        np.einsum("ai,ijkl->ajkl", M[0], a))
    D = rng.standard_normal(4)
    npt.assert_allclose(x.tensor_diagonal_matrix_product(D, 1).numpy(),
                        a * D[None, :, None, None])
    v = [rng.standard_normal(n) for n in SHAPE]
    npt.assert_allclose(x.tensor_vector_product([v[1], v[3]], [1, 3]).numpy(),
                        np.einsum("ijkl,j,l->ik", a, v[1], v[3]))
    npt.assert_allclose(x.tensor_vector_product(v[0], 0).numpy(),
                        np.einsum("ijkl,i->jkl", a, v[0]))
    assert np.isclose(x.tensor_vector_product(v), np.einsum("ijkl,i,j,k,l->", a, *v))
    with pytest.raises(ValueError):
        x.tensor_vector_product(v[:2], [0])


def test_kron_and_cat(rng):
    x, y = _tt(rng), _tt(rng, shape=[2, 2, 3, 1], ranks=[1, 2, 2])
    fx, fy = x.full(), y.full()
    npt.assert_allclose(x.kron(y).numpy(), fx.kron(fy).numpy())
    npt.assert_allclose(x.cat(y).numpy(), fx.cat(fy).numpy())


# --- conversions ---------------------------------------------------------------------


def test_tree_based_tensor_round_trip(rng):
    x = _tt(rng)
    t = x.tree_based_tensor()
    assert isinstance(t, tensap.TreeBasedTensor)
    npt.assert_allclose(t.full().numpy(), x.numpy())
    npt.assert_allclose(TTTensor.from_tree_based_tensor(t).numpy(), x.numpy())
    # the tree is an oracle for the operations
    y = _tt(rng, ranks=[1, 2, 3])
    assert np.isclose(x.dot(y), t.dot(y.tree_based_tensor()))
    assert np.isclose(x.norm(), t.norm())


def test_from_tree_based_tensor_with_active_and_inactive_leaves(rng):
    tree = tensap.DimensionTree.linear(4)
    for seed in range(4):
        np.random.seed(seed)
        t = tensap.TreeBasedTensor.randn(tree, None, SHAPE)
        npt.assert_allclose(TTTensor.from_tree_based_tensor(t).numpy(), t.full().numpy())


def test_from_tree_based_tensor_wrong_tree():
    np.random.seed(0)
    t = tensap.TreeBasedTensor.randn(tensap.DimensionTree.balanced(4), None, SHAPE)
    with pytest.raises(ValueError):
        TTTensor.from_tree_based_tensor(t)
    with pytest.raises(ValueError):
        TTTensor([np.ones((1, 3, 1))]).tree_based_tensor()


def test_from_full_and_from_diagonal(rng):
    a = rng.standard_normal(SHAPE)
    x = TTTensor.from_full(a)
    npt.assert_allclose(x.numpy(), a)
    assert x.ranks == (3, 10, 5)  # exact TT-SVD: ranks of the unfoldings
    low = _tt(rng, ranks=[2, 2, 2]).numpy()
    y = TTTensor.from_full(tensap.FullTensor(low), tolerance=1e-10)
    assert y.ranks == (2, 2, 2)
    npt.assert_allclose(y.numpy(), low, atol=1e-10)
    d = tensap.DiagonalTensor(rng.standard_normal(3), 4)
    npt.assert_allclose(TTTensor.from_diagonal_tensor(d).numpy(), d.full().numpy())
    d1 = tensap.DiagonalTensor(rng.standard_normal(3), 1)
    npt.assert_allclose(TTTensor.from_diagonal_tensor(d1).numpy(), d1.full().numpy())


def test_as_tt_tensor(rng):
    from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor
    x = _tt(rng)
    assert as_tt_tensor(x) is x
    with pytest.raises(NotImplementedError):
        as_tt_tensor(tensap.SparseTensor(np.eye(3)))


# --- operators -----------------------------------------------------------------------


def test_operator_full_matvec_compose_transpose(rng):
    A = _tt_operator(rng)
    shape = A.shape
    assert shape == ((3, 2), (4, 4), (2, 3))
    M = _matrix(A.numpy(), shape)
    x = _tt(rng, shape=[2, 4, 3], ranks=[2, 2])
    y = A @ x
    assert isinstance(y, TTTensor) and y.ranks == (4, 6)
    npt.assert_allclose(y.numpy().ravel(), M @ x.numpy().ravel())
    B = _tt_operator(rng, shape=((2, 3), (4, 2), (3, 2)), ranks=(1, 2))
    C = A @ B
    assert isinstance(C, TTOperator)
    npt.assert_allclose(_matrix(C.numpy(), C.shape), M @ _matrix(B.numpy(), B.shape))
    npt.assert_allclose(_matrix(A.T.numpy(), A.T.shape), M.T)
    with pytest.raises(ValueError):
        A @ _tt(rng, shape=[3, 4, 2], ranks=[2, 2])
    with pytest.raises(ValueError):
        A @ _tt(rng, shape=[2, 4], ranks=[2])
    assert A.__matmul__(np.ones(3)) is NotImplemented


def test_operator_vector_space_operations(rng):
    A, B = _tt_operator(rng), _tt_operator(rng, ranks=(1, 2))
    a, b = A.numpy(), B.numpy()
    npt.assert_allclose((A + B).numpy(), a + b)
    npt.assert_allclose((A - 2 * B).numpy(), a - 2 * b)
    assert np.isclose(A.dot(B), np.sum(a * b))
    assert np.isclose(A.norm(), np.linalg.norm(a))
    npt.assert_allclose(A.orth(1).numpy(), a)
    npt.assert_allclose(A.hadamard_product(B).numpy(), a * b)
    npt.assert_allclose(TTOperator.unvectorize(A.vectorize(), A.shape).numpy(), a)


def test_operator_eye(rng):
    Id = TTOperator.eye([2, 3, 4])
    x = _tt(rng, shape=[2, 3, 4], ranks=[2, 2])
    npt.assert_allclose((Id @ x).numpy(), x.numpy())


def test_laplacian_tt_constructions(rng):
    K, M = rng.standard_normal((3, 3)), rng.standard_normal((3, 3))
    for d in [1, 2, 4]:
        factory = tensap.LaplacianLikeOperatorFactory(K, M, d)
        reference = factory.make_canonical_tensor().numpy()
        tucker = factory.make_tt_tucker_tensor()
        assert isinstance(tucker.core, TTTensor)
        npt.assert_allclose(tucker.numpy(), reference)
        operator = factory.make_tt_operator()
        npt.assert_allclose(operator.numpy(), reference)
        npt.assert_allclose(TTOperator.from_tucker(tucker).numpy(), reference)


# --- Tucker tensors with a TT core ---------------------------------------------------


def _tucker_tt(rng, shape=(6, 5, 7, 4), tucker_ranks=(3, 2, 3, 2), tt_ranks=(2, 3, 2)):
    core = _tt(rng, shape=tucker_ranks, ranks=tt_ranks)
    frames = tensap.TFrameVector([tensap.FrameVector(rng.standard_normal((r, n, 1)))
                                  for r, n in zip(tucker_ranks, shape)])
    return tensap.TuckerTensorVector(core, frames)


def test_tucker_with_tt_core(rng):
    x, y = _tucker_tt(rng), _tucker_tt(rng, tucker_ranks=(2, 2, 2, 3), tt_ranks=(2, 2, 2))
    a, b = x.numpy(), y.numpy()
    npt.assert_allclose(TTTensor.from_tucker(x).numpy(), a)
    assert np.isclose(x.dot(y), np.sum(a * b))
    assert np.isclose(x.norm(), np.linalg.norm(a))
    npt.assert_allclose(x.orth().numpy(), a)
    npt.assert_allclose((x + y).numpy(), a + b)
    npt.assert_allclose((-x).numpy(), -a)
    npt.assert_allclose(x.normalize_basis().numpy(), a)
    ones = tensap.TuckerTensorVector.ones([1] * 4, [6, 5, 7, 4])  # diagonal core
    npt.assert_allclose((x + ones).numpy(), a + 1)
    v = rng.standard_normal(5)
    npt.assert_allclose(x.tensor_vector_product(v, [1]).numpy(), np.einsum("ijkl,j->ikl", a, v))
    # Tucker operator with a TT core applied to a Tucker vector with a TT core
    factory = tensap.LaplacianLikeOperatorFactory(rng.standard_normal((5, 5)), np.eye(5), 4)
    A = factory.make_tt_tucker_tensor()
    z = _tucker_tt(rng, shape=(5, 5, 5, 5))
    M = _matrix(A.numpy(), A.shape)
    npt.assert_allclose((A @ z).numpy().ravel(), M @ z.numpy().ravel())


def test_convert_tensors_diagonal_and_tt(rng):
    x = _tt(rng, shape=[3, 3, 3], ranks=[2, 2])
    d = tensap.DiagonalTensor(rng.standard_normal(3), 3)
    xc, dc = tensap.convert_tensors(x, d)
    assert isinstance(dc, TTTensor)
    npt.assert_allclose(dc.numpy(), d.full().numpy())
    dc, xc = tensap.convert_tensors(d, x)
    assert isinstance(dc, TTTensor) and xc is x
    f = tensap.FullTensor(rng.standard_normal((3, 3, 3)))
    fc, xc = tensap.convert_tensors(f, x)
    assert isinstance(xc, tensap.FullTensor)


# --- truncation ----------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [1e-12, 1e-2])
def test_tt_rounding(rng, tolerance):
    x = _tt(rng, ranks=[2, 3, 2])
    y = x + x + 0.5 * x  # ranks (6, 9, 6), exact rank (2, 3, 2)
    z = tensap.Truncator(tolerance=tolerance).truncate(y)
    assert isinstance(z, TTTensor) and z.orth_center == 3
    assert z.ranks <= (2, 3, 2)
    assert np.linalg.norm(z.numpy() - y.numpy()) <= tolerance * np.linalg.norm(y.numpy()) + 1e-12
    if tolerance < 1e-6:
        assert z.ranks == (2, 3, 2)
    # same ranks as the TT-SVD of the dense tensor
    assert z.ranks == TTTensor.from_full(y.numpy(), tolerance).ranks


def test_tt_rounding_max_rank_and_zero(rng):
    x = _tt(rng, ranks=[2, 3, 2])
    z = tensap.Truncator(max_rank=1).truncate(x)
    assert z.ranks == (1, 1, 1)
    z = tensap.Truncator(max_rank=[2, 1, 2]).truncate(x)
    assert z.ranks == (2, 1, 2)
    zero = tensap.Truncator().truncate(x * 0.0)
    assert zero.norm() == 0


def test_tt_operator_rounding(rng):
    A = _tt_operator(rng)
    B = tensap.Truncator(tolerance=1e-12).truncate(A + A)
    assert isinstance(B, TTOperator) and B.ranks == A.ranks
    npt.assert_allclose(B.numpy(), 2 * A.numpy())


@pytest.mark.parametrize("tolerance", [1e-12, 1e-1])
def test_tucker_tt_truncation(rng, tolerance):
    x = _tucker_tt(rng)
    y = x + x + x  # ranks tripled
    z = tensap.Truncator(tolerance=tolerance).truncate(y)
    assert isinstance(z.core, TTTensor)
    error = np.linalg.norm(z.numpy() - y.numpy()) / np.linalg.norm(y.numpy())
    assert error <= tolerance + 1e-12
    if tolerance < 1e-6:
        # minimal ranks: those of the matricizations of the dense tensor (the
        # Tucker rank of dimension 0 is bounded by the TT rank 2 of the core)
        a = y.numpy()
        tucker_ranks = tuple(np.linalg.matrix_rank(np.moveaxis(a, k, 0).reshape(a.shape[k], -1))
                             for k in range(4))
        assert z.representation_rank() == tucker_ranks == (2, 2, 3, 2)
        assert z.core.ranks == x.core.ranks


def test_tucker_tt_operator_truncation(rng):
    A = tensap.LaplacianLikeOperatorFactory(rng.standard_normal((4, 4)), np.eye(4), 3)
    A = A.make_tt_tucker_tensor()
    B = tensap.Truncator(tolerance=1e-12).truncate(A + A)
    npt.assert_allclose(B.numpy(), 2 * A.numpy(), atol=1e-12)
    assert B.representation_rank() == (2, 2, 2)


# --- solvers on TT formats -----------------------------------------------------------


def _poisson(n, d):
    h = 1 / (n + 1)
    K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
    factory = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d)
    dense = _matrix(factory.make_canonical_tensor().numpy(), [(n, n)] * d)
    x_direct = np.linalg.solve(dense, np.ones(n**d))
    return factory, x_direct


def test_richardson_and_pcg_on_tt_formats():
    n, d = 8, 3
    factory, x_direct = _poisson(n, d)
    A = factory.make_tt_operator()
    b = TTTensor.ones([n] * d)
    h = 1 / (n + 1)
    truncator = tensap.Truncator(tolerance=1e-8)
    s = tensap.TruncatedRichardsonIterations(
        A, b, truncator, step_size=h**2 / (4 * d), max_iterations=2000,
        tolerance=1e-6, stagnation=1e-12, display=False)
    x, output = s.solve()
    assert isinstance(x, TTTensor) and output["flag"] == 0
    assert np.linalg.norm(x.numpy().ravel() - x_direct) / np.linalg.norm(x_direct) < 1e-4
    s = tensap.TruncatedPCG(A, b, truncator, tolerance=1e-8, max_iterations=100,
                            display=False)
    x, output = s.solve()
    assert isinstance(x, TTTensor)
    assert np.linalg.norm(x.numpy().ravel() - x_direct) / np.linalg.norm(x_direct) < 1e-6
