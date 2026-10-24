"""
Unit tests of the QTT tools: quantization (tensap.quantize and friends) and the
QTT primitives of 1D operators (tensap.qtt_*).

Oracles: the dense arrays and matrices (numpy, scipy.sparse), and
tensap.Tensorizer, which describes the same layout for functions.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
import tensap


def _tensorizer(b, L, dim, ordering_type=1):
    t = tensap.Tensorizer(b, L, dim)
    t.ordering_type = ordering_type
    return t


# --- quantization of arrays and matrices -------------------------------------------


@pytest.mark.parametrize("b, L", [(2, 6), (3, 3)])
def test_quantize_vector_round_trip(rng, b, L):
    v = rng.standard_normal(b**L)
    x = tensap.quantize(v, base=b)
    assert isinstance(x, tensap.TTTensor)
    assert x.order == L and np.all(x.shape == b)
    npt.assert_allclose(tensap.dequantize(x), v)
    # most significant digit first: entry i = sum_k i_k b^(L-k)
    digits = tensap.integer2baseb(np.array([5]), b, L)[0]
    assert np.isclose(x.full().numpy()[tuple(digits)], v[5])


@pytest.mark.parametrize("ordering_type", [1, 2])
def test_quantize_array_orderings(rng, ordering_type):
    t = _tensorizer(2, 3, 2, ordering_type)
    a = rng.standard_normal((8, 8))
    x = tensap.quantize(a, t)
    npt.assert_allclose(tensap.dequantize(x, t), a)
    i, j = 5, 3  # digits (1, 0, 1) and (0, 1, 1)
    entry = x.full().numpy()
    if ordering_type == 1:
        assert np.isclose(entry[1, 0, 1, 0, 1, 1], a[i, j])
    else:
        assert np.isclose(entry[1, 0, 0, 1, 1, 1], a[i, j])
    if ordering_type == 1:  # default layout: blocks, deduced from the shape
        npt.assert_allclose(tensap.dequantize(tensap.quantize(a), dim=2), a)


def test_quantize_low_rank_functions():
    t = tensap.Tensorizer(2, 10, 1)
    # exponentials have QTT rank 1, sines and cosines rank 2
    assert tensap.quantize_function(lambda x: np.exp(x[:, 0]), t).ranks == (1,) * 9
    assert max(tensap.quantize_function(lambda x: np.sin(5 * x[:, 0]), t).ranks) == 2


def test_quantize_wrong_shapes():
    with pytest.raises(ValueError):
        tensap.quantize(np.ones(6))
    with pytest.raises(ValueError):
        tensap.quantize(np.ones(8), tensap.Tensorizer(2, 4, 1))
    with pytest.raises(ValueError):
        tensap.dequantize(tensap.quantize(np.ones(8)), dim=2)
    with pytest.raises(ValueError):
        tensap.dequantize(tensap.quantize(np.ones(8)), tensap.Tensorizer(2, 4, 1))
    t = _tensorizer(2, 2, 1, ordering_type=3)
    with pytest.raises(ValueError):
        tensap.quantize(np.ones(4), t)


@pytest.mark.parametrize("ordering_type", [1, 2])
def test_quantize_operator(rng, ordering_type):
    t = _tensorizer(2, 3, 2, ordering_type)
    m = rng.standard_normal((64, 64))
    A = tensap.quantize_operator(m, t)
    assert isinstance(A, tensap.TTOperator) and A.shape == ((2, 2),) * 6
    npt.assert_allclose(tensap.dequantize_operator(A, t), m)
    # the quantized operator acts on the quantized arrays
    w = rng.standard_normal((8, 8))
    npt.assert_allclose(tensap.dequantize(A @ tensap.quantize(w, t), t).ravel(),
                        m @ w.ravel())


def test_quantize_operator_sparse_and_errors():
    m = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(16, 16))
    A = tensap.quantize_operator(m)
    npt.assert_allclose(tensap.dequantize_operator(A), m.toarray(), atol=1e-12)
    assert max(A.ranks) == 3
    with pytest.raises(ValueError):
        tensap.quantize_operator(np.ones((4, 8)))
    with pytest.raises(ValueError):
        tensap.quantize_operator(np.ones((8, 8)), tensap.Tensorizer(2, 2, 2))
    with pytest.raises(ValueError):
        tensap.dequantize_operator(A, tensap.Tensorizer(2, 2, 1))


# --- rectangular grids: one number of digits per axis ------------------------------


@pytest.mark.parametrize("ordering_type", [1, 2])
def test_quantize_rectangle(rng, ordering_type):
    t = _tensorizer(2, [2, 4], 2, ordering_type)
    npt.assert_array_equal(t.levels, [2, 4])
    a = rng.standard_normal((4, 16))
    x = tensap.quantize(a, t)
    assert x.order == 6
    npt.assert_allclose(tensap.dequantize(x, t), a)
    # interleaved: digits aligned on the most significant one,
    # (i_1, j_1, i_2, j_2, j_3, j_4)
    positions = [list(p) for p in t.digit_positions()]
    assert positions == ([[0, 1], [2, 3, 4, 5]] if ordering_type == 1
                         else [[0, 2], [1, 3, 4, 5]])
    i, j = 2, 11  # digits (1, 0) and (1, 0, 1, 1)
    digits = np.zeros(6, dtype=int)
    digits[positions[0]] = [1, 0]
    digits[positions[1]] = [1, 0, 1, 1]
    assert np.isclose(x.full().numpy()[tuple(digits)], a[i, j])
    m = rng.standard_normal((64, 64))
    A = tensap.quantize_operator(m, t)
    npt.assert_allclose(tensap.dequantize_operator(A, t), m)
    w = rng.standard_normal((4, 16))
    npt.assert_allclose(tensap.dequantize(A @ tensap.quantize(w, t), t).ravel(), m @ w.ravel())


def test_default_layout_of_a_rectangle(rng):
    a = rng.standard_normal((4, 8))
    x = tensap.quantize(a)
    t = tensap.qtt.default_tensorizer(a.shape)
    npt.assert_array_equal(t.levels, [2, 3])
    npt.assert_allclose(tensap.dequantize(x, t), a)


def test_tensorizer_with_one_resolution_per_dimension():
    for ordering_type in [1, 2]:
        t = _tensorizer(2, [3, 1, 2], 3, ordering_type)
        x = np.random.default_rng(0).random((20, 3))
        z = t.map(x)
        assert z.shape == (20, 6 + 3)
        npt.assert_allclose(t.inverse_map(z), x)
    with pytest.raises(ValueError):
        tensap.Tensorizer(2, [3, 2], 3)


# --- link with the Tensorizer --------------------------------------------------------


@pytest.mark.parametrize("ordering_type", [1, 2])
@pytest.mark.parametrize("y0", [0.0, 0.5])
@pytest.mark.parametrize("levels", [3, [2, 4]])
def test_quantize_function_is_the_tensorized_function(ordering_type, y0, levels):
    t = _tensorizer(2, levels, 2, ordering_type)

    def f(x):
        return np.cos(x[:, 0]) + x[:, 0] * x[:, 1] ** 2

    x = tensap.quantize_function(f, t, y0=y0)
    g = t.tensorize(f)
    D = int(t.levels.sum())
    digits = tensap.integer2baseb(np.arange(2**D), 2, D)  # all the digit tuples
    z = np.hstack((digits, np.full((digits.shape[0], 2), y0)))
    npt.assert_allclose(x.full().numpy().ravel(), g.fun(z))


def test_grid_points():
    t = tensap.Tensorizer(2, 3, 1, X=tensap.UniformRandomVariable(-1, 3))
    npt.assert_allclose(tensap.grid_points(t)[:, 0], -1 + 4 * np.arange(8) / 8)
    npt.assert_allclose(tensap.grid_points(t, 0.5)[:, 0], -1 + 4 * (np.arange(8) + 0.5) / 8)


def test_to_functional_tensor(rng):
    t = tensap.Tensorizer(2, 5, 1, X=tensap.UniformRandomVariable(0, 2))
    v = rng.standard_normal(32)
    g = tensap.to_functional_tensor(tensap.quantize(v, t), t)
    # piecewise constant: the value at x is the entry of the cell of x
    x = 2 * rng.random(20)
    npt.assert_allclose(g(x), v[np.floor(x / 2 * 32).astype(int)])
    with pytest.raises(ValueError):
        tensap.to_functional_tensor(tensap.quantize(v), tensap.Tensorizer(2, 4, 1))


def test_to_functional_tensor_two_dimensions(rng):
    t = _tensorizer(2, 3, 2, ordering_type=2)
    a = rng.standard_normal((8, 8))
    g = tensap.to_functional_tensor(tensap.quantize(a, t), t)
    x = rng.random((10, 2))
    cells = np.floor(x * 8).astype(int)
    npt.assert_allclose(g(x), a[cells[:, 0], cells[:, 1]])


def test_tensor_matrix_product_eval_diag(rng):
    x = tensap.TTTensor([rng.standard_normal(s) for s in [(1, 3, 2), (2, 4, 1)]])
    M = [rng.standard_normal((5, 3)), rng.standard_normal((5, 4))]
    npt.assert_allclose(x.tensor_matrix_product_eval_diag(M).numpy(),
                        np.einsum("ij,pi,pj->p", x.numpy(), *M))
    with pytest.raises(NotImplementedError):
        x.tensor_matrix_product_eval_diag(M[:1], [0])


# --- QTT primitives ----------------------------------------------------------------


@pytest.mark.parametrize("b, L", [(2, 1), (2, 5), (3, 3)])
@pytest.mark.parametrize("periodic", [False, True])
def test_shift(b, L, periodic):
    N = b**L
    S = tensap.qtt_shift(L, b, periodic)
    expected = np.eye(N, k=1)
    if periodic:
        expected[N - 1, 0] = 1.0
    npt.assert_allclose(tensap.dequantize_operator(S), expected)
    assert max(S.ranks, default=1) <= 2
    npt.assert_allclose(tensap.dequantize_operator(S.T), expected.T)


def test_identity_elementary_and_ones():
    L, N = 4, 16
    npt.assert_allclose(tensap.dequantize_operator(tensap.qtt_identity(L)), np.eye(N))
    E = tensap.qtt_elementary_matrix(L, 3, 12)
    expected = np.zeros((N, N))
    expected[3, 12] = 1.0
    npt.assert_allclose(tensap.dequantize_operator(E), expected)
    assert E.ranks == (1, 1, 1)
    npt.assert_allclose(tensap.dequantize_operator(tensap.qtt_ones_matrix(L)), np.ones((N, N)))


@pytest.mark.parametrize("periodic", [False, True])
def test_banded(periodic):
    L, N = 6, 64
    diagonals = {-2: 0.5, -1: -1.0, 0: 2.0, 1: -1.0, 3: 0.25}
    B = tensap.qtt_banded(diagonals, L, periodic=periodic)
    if periodic:
        expected = sum(c * np.roll(np.eye(N), k, axis=1) for k, c in diagonals.items())
    else:
        expected = sp.diags(list(diagonals.values()), list(diagonals.keys()),
                            shape=(N, N)).toarray()
    npt.assert_allclose(tensap.dequantize_operator(B), expected, atol=1e-12)
    assert max(B.ranks) <= 5  # small rank, independent of N
    tridiagonal = tensap.qtt_banded({-1: -1.0, 0: 2.0, 1: -1.0}, L)
    assert max(tridiagonal.ranks) == 3


def test_banded_errors_and_zero():
    with pytest.raises(ValueError):
        tensap.qtt_banded({16: 1.0}, 4)
    Z = tensap.qtt_banded({1: 0.0}, 3)
    npt.assert_allclose(tensap.dequantize_operator(Z), np.zeros((8, 8)))


@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_corner(rng, corner):
    L, N = 5, 32
    block = rng.standard_normal((2, 3))
    C = tensap.qtt_corner(block, L, corner)
    expected = np.zeros((N, N))
    rows = slice(0, 2) if corner.startswith("top") else slice(N - 2, N)
    cols = slice(0, 3) if corner.endswith("left") else slice(N - 3, N)
    expected[rows, cols] = block
    npt.assert_allclose(tensap.dequantize_operator(C), expected, atol=1e-12)
    with pytest.raises(ValueError):
        tensap.qtt_corner(block, L, "middle-left")


def test_banded_plus_corners_is_a_low_rank_correction():
    # a tridiagonal matrix with modified first and last rows: rank stays small
    L, N = 8, 256
    A = tensap.qtt_banded({-1: 1.0, 0: -2.0, 1: 1.0}, L)
    A = A + tensap.qtt_corner([[1.0, -0.5]], L, "top-left")
    A = A + tensap.qtt_corner([[-0.5, 1.0]], L, "bottom-right")
    A = tensap.Truncator(tolerance=1e-14).truncate(A)
    expected = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N)).toarray()
    expected[0, :2] += [1.0, -0.5]
    expected[-1, -2:] += [-0.5, 1.0]
    npt.assert_allclose(tensap.dequantize_operator(A), expected, atol=1e-12)
    assert max(A.ranks) <= 5


# --- assembly in dimension d by tensor products ------------------------------------


def test_tensor_product_assembly_2d():
    # 2D operator I (x) T + T (x) I, block ordering: chains concatenated
    L, n = 3, 8
    T = tensap.qtt_banded({-1: -1.0, 0: 2.0, 1: -1.0}, L)
    Id = tensap.qtt_identity(L)
    A = Id.tensor_product(T) + T.tensor_product(Id)
    A = tensap.Truncator(tolerance=1e-14).truncate(A)
    Td = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    expected = (sp.kron(sp.identity(n), Td) + sp.kron(Td, sp.identity(n))).toarray()
    t = tensap.Tensorizer(2, L, 2)
    npt.assert_allclose(tensap.dequantize_operator(A, t), expected, atol=1e-12)
    # and on vectors: separable right-hand side f(x) g(y)
    f, g = np.arange(1.0, 9.0), np.cos(np.arange(8.0))
    fg = tensap.quantize(f).tensor_product(tensap.quantize(g))
    npt.assert_allclose(tensap.dequantize(fg, t), np.outer(f, g))
    with pytest.raises(TypeError):
        Id.tensor_product(tensap.quantize(f))
