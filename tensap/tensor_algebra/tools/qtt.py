# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module qtt: quantized tensor-train (QTT) representations.

An array of size b^L_k along its axis k (k = 1, ..., dim) is reshaped into a
tensor of order L_1 + ... + L_dim with dimensions of size b, then stored as a TTTensor; a matrix
acting on such arrays becomes a TTOperator with cores of size b x b.

The layout (base b, number L of digits per axis, number dim of axes, order of
the digits) is the one of tensap.Tensorizer, which describes the same
quantization for functions: the index i = sum_k i_k b^(L-k) of a grid point
along an axis is written with its most significant digit first (C order),
and the digits of the axes are ordered by blocks (ordering_type = 1: all the
digits of the first axis, then those of the second one...) or interleaved
(ordering_type = 2: the first digit of every axis, then the second one...).
The Tensorizer is optional for the algebra (the layout is then deduced from
the shape); it is required to interpret a QTT vector as a function, since it
carries the geometry of the grid.

The module also provides the QTT primitives of 1D operators on b^L points:
identity, shift, elementary matrices, the matrix of ones, banded Toeplitz
matrices and corner blocks. They contain no discretization choice (stencils
and boundary conditions are built on top of them).
"""

import numpy as np
import tensap


# --- layout -----------------------------------------------------------------------


def default_tensorizer(shape, base: int = 2) -> "tensap.Tensorizer":
    """Tensorizer of the default layout of an array: base b, the number of
    digits of each axis deduced from its size, blocks, [0, 1]^dim."""
    levels = [_number_of_digits(int(n), base) for n in shape]
    return tensap.Tensorizer(base, levels if len(set(levels)) > 1 else levels[0], len(levels))


def _number_of_digits(n: int, b: int) -> int:
    L = int(round(np.log(n) / np.log(b))) if n > 1 else 0
    if L < 1 or b**L != n:
        raise ValueError(f"The size {n} is not a power b^L of the base b = {b}.")
    return L


def _check_shape(shape, tensorizer):
    expected = tuple(int(tensorizer.b) ** int(L) for L in tensorizer.levels)
    if tuple(shape) != expected:
        raise ValueError(f"Expected an array of shape {expected}, got {tuple(shape)}.")


def _permutation(tensorizer) -> np.ndarray:
    """perm such that the digits in the layout of the tensorizer are the
    block-ordered digits (C-order reshape) transposed by perm."""
    positions = np.concatenate(tensorizer.digit_positions())
    return np.argsort(positions)

# --- vectors and arrays -----------------------------------------------------------


def quantize(array, tensorizer=None, tolerance: float = 1e-12, base: int = 2):
    """QTT representation of an array of shape (b^L_1, ..., b^L_dim).

    Parameters
    ----------
    array : numpy.ndarray or tensap.FullTensor
        Vector of size b^L, or array with axes of sizes b^L_k.
    tensorizer : tensap.Tensorizer, optional
        Layout (b, L, dim, ordering_type). The default is deduced from the
        shape: base `base`, blocks.
    tolerance : float, optional
        Relative precision of the TT-SVD. The default 1e-12 only removes the
        round-off.
    base : int, optional
        Base used without tensorizer. The default is 2.

    Returns
    -------
    tensap.TTTensor
        Tensor of order L * dim with dimensions of size b.
    """
    a = np.asarray(getattr(array, "data", array), dtype=float)
    if tensorizer is None:
        tensorizer = default_tensorizer(a.shape, base)
    _check_shape(a.shape, tensorizer)
    b = tensorizer.b
    digits = a.reshape([b] * int(tensorizer.levels.sum())).transpose(_permutation(tensorizer))
    return tensap.TTTensor.from_full(digits, tolerance)


def dequantize(tensor, tensorizer=None, dim: int = 1) -> np.ndarray:
    """Inverse of quantize: the array of shape (b^L_1, ..., b^L_dim).

    Parameters
    ----------
    tensor : tensap.TTTensor
        QTT tensor of order L * dim.
    tensorizer : tensap.Tensorizer, optional
        Layout. The default is blocks, with the base given by the cores and
        `dim` axes.
    dim : int, optional
        Number of axes when no tensorizer is given. The default is 1.
    """
    if tensorizer is None:
        if tensor.order % dim:
            raise ValueError("The order must be a multiple of dim.")
        tensorizer = tensap.Tensorizer(int(tensor.shape[0]), tensor.order // dim, dim)
    b, levels = tensorizer.b, tensorizer.levels
    if tensor.order != levels.sum():
        raise ValueError(f"Expected a tensor of order {levels.sum()}, got {tensor.order}.")
    digits = tensor.numpy().transpose(np.argsort(_permutation(tensorizer)))
    return digits.reshape([b**int(L) for L in levels])


def quantize_function(fun, tensorizer, y0: float = 0.0, tolerance: float = 1e-12):
    """QTT representation of the values of fun on the grid of tensorizer.

    The grid points are x = F_X^{-1}((i + F_Y(y0)) b^(-L)) along each axis,
    for the b^L indices i (the inverse map of the Tensorizer at y = y0): nodes
    for y0 = 0, cell centres for y0 = 1/2 (uniform X and Y). The function is
    sampled on the whole grid (b^(L dim) points): moderate sizes only.

    Parameters
    ----------
    fun : callable
        Function of points of shape (n, dim), returning n values.
    tensorizer : tensap.Tensorizer
    y0 : float, optional
        Position in the cells. The default is 0.
    tolerance : float, optional
        Relative precision of the TT-SVD.

    Returns
    -------
    tensap.TTTensor
    """
    points = grid_points(tensorizer, y0)
    values = np.asarray(fun(points.reshape(-1, tensorizer.dim)), dtype=float)
    return quantize(values.reshape(points.shape[:-1]), tensorizer, tolerance)


def grid_points(tensorizer, y0: float = 0.0) -> np.ndarray:
    """Points of the grid of tensorizer, array of shape (b^L_1, ..., b^L_dim, dim)."""
    b = tensorizer.b
    axes = []
    for k, L in enumerate(tensorizer.levels):
        u = (np.arange(b**L) + tensorizer.Y.random_variables[k].cdf(np.array([y0]))) / b**L
        axes.append(tensorizer.X.random_variables[k].icdf(u))
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def to_functional_tensor(tensor, tensorizer):
    """The QTT vector as a function: piecewise constant on the grid cells.

    Returns the tensap.TensorizedFunction whose value at x is the entry of the
    QTT vector at the digits of the cell of x (delta basis on the digits,
    constant basis in y). The geometry comes from the tensorizer, which is
    therefore required.
    """
    b, dim, D = tensorizer.b, tensorizer.dim, int(tensorizer.levels.sum())
    if tensor.order != D or np.any(tensor.shape != b):
        raise ValueError("The tensor does not have the layout of the tensorizer.")
    # one core of size 1 per axis for the variable y (constant basis)
    cores = list(tensor.cores) + [np.ones((1, 1, 1))] * dim
    full = tensap.TTTensor(cores)
    digits = tensap.DeltaFunctionalBasis(np.arange(b))
    constant = tensap.FunctionalBases(
        [tensap.PolynomialFunctionalBasis(y.orthonormal_polynomials(), [0])
         for y in tensorizer.Y.random_variables]
    )
    bases = tensap.FunctionalBases([digits] * D + list(constant.bases))
    return tensap.TensorizedFunction(tensap.FunctionalTensor(full, bases), tensorizer)


# --- matrices ---------------------------------------------------------------------


def quantize_operator(matrix, tensorizer=None, tolerance: float = 1e-12, base: int = 2):
    """QTT representation of a square matrix acting on quantized arrays.

    Rows and columns are quantized with the same layout and paired digit by
    digit: core k of the TTOperator has shape (r, b, b, r'), indexed by the
    k-th digit of the row and of the column. The matrix is dense: moderate
    sizes only (mostly a test oracle).

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse matrix
        Matrix of shape (N, N), N = b^(L dim).
    tensorizer : tensap.Tensorizer, optional
        Layout of the arrays. The default is one axis (dim = 1) of base `base`.
    """
    m = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)
    N = m.shape[0]
    if m.shape != (N, N):
        raise ValueError("The matrix must be square.")
    if tensorizer is None:
        tensorizer = default_tensorizer([N], base)
    b = tensorizer.b
    D = int(tensorizer.levels.sum())
    if N != b**D:
        raise ValueError(f"Expected a matrix of size {b**D}, got {N}.")
    perm = _permutation(tensorizer)
    digits = m.reshape([b] * (2 * D)).transpose(np.concatenate((perm, perm + D)))
    # pair the digits (i_k, j_k), then TT-SVD on the paired indices of size b^2
    paired = digits.transpose(np.ravel(np.column_stack((np.arange(D), np.arange(D) + D))))
    tt = tensap.TTTensor.from_full(paired.reshape([b * b] * D), tolerance)
    return tensap.TTOperator.unvectorize(tt, [(b, b)] * D)


def dequantize_operator(operator, tensorizer=None) -> np.ndarray:
    """Inverse of quantize_operator: the matrix of shape (N, N)."""
    D = operator.order
    b = operator.shape[0][0]
    if tensorizer is None:
        tensorizer = tensap.Tensorizer(b, D, 1)
    if D != tensorizer.levels.sum():
        raise ValueError(f"Expected an operator of order {tensorizer.levels.sum()}, got {D}.")
    # full: (i_1, j_1, ..., i_D, j_D) -> (i_1, ..., i_D, j_1, ..., j_D)
    a = operator.numpy().transpose(list(range(0, 2 * D, 2)) + list(range(1, 2 * D, 2)))
    inverse = np.argsort(_permutation(tensorizer))
    a = a.transpose(np.concatenate((inverse, inverse + D)))
    return a.reshape(b**D, b**D)


# --- tensor products in a layout ------------------------------------------------


def qtt_layout_product(factors, tensorizer):
    """Tensor product of one QTT factor per axis, in the layout of tensorizer.

    factors[k] is a TTTensor (or TTOperator) with tensorizer.levels[k] cores,
    acting on axis k; the result is their tensor product v_1 (x) ... (x) v_m
    on the quantized arrays, its cores being placed at the positions of the
    digits of their axis (Tensorizer.digit_positions). The bond between two
    cores carries the current rank of every axis (the product of these
    ranks): with the block ordering it is the concatenation of the chains,
    with the interleaved ordering the ranks of the axes are multiplied where
    their digits alternate.

    Parameters
    ----------
    factors : list of tensap.TTTensor or list of tensap.TTOperator
    tensorizer : tensap.Tensorizer

    Returns
    -------
    tensap.TTTensor or tensap.TTOperator
    """
    m = tensorizer.dim
    if len(factors) != m:
        raise ValueError(f"Expected one factor per axis ({m}), got {len(factors)}.")
    kind = type(factors[0])
    if any(type(f) is not kind for f in factors):
        raise TypeError("The factors must all be TTTensor or all TTOperator.")
    positions = tensorizer.digit_positions()
    for k, f in enumerate(factors):
        if f.order != len(positions[k]):
            raise ValueError(f"Factor {k} has {f.order} cores, expected {len(positions[k])}.")
    D = int(tensorizer.levels.sum())
    owner, index = np.empty(D, dtype=int), np.empty(D, dtype=int)
    for k, p in enumerate(positions):
        owner[p], index[p] = k, np.arange(len(p))
    ranks = [1] * m
    cores = []
    for q in range(D):
        k = owner[q]
        core = factors[k].cores[index[q]]
        n_phys = core.ndim - 2
        others = [j for j in range(m) if j != k]
        # core (a_k, phys, a_k') times identities (a_j, a_j') of the other axes
        full = core
        for j in others:
            full = np.multiply.outer(full, np.eye(ranks[j]))
        # axes of full: a_k, phys..., a_k', then (a_j, a_j') for j in others
        left = {k: 0}
        right = {k: n_phys + 1}
        for pos, j in enumerate(others):
            left[j] = n_phys + 2 + 2 * pos
            right[j] = n_phys + 3 + 2 * pos
        order = ([left[j] for j in range(m)] + list(range(1, n_phys + 1))
                 + [right[j] for j in range(m)])
        new_ranks = list(ranks)
        new_ranks[k] = core.shape[-1]
        full = full.transpose(order).reshape(
            int(np.prod(ranks)), *core.shape[1:-1], int(np.prod(new_ranks)))
        cores.append(full)
        ranks = new_ranks
    return kind(cores)


def qtt_embed(operator, axis: int, tensorizer):
    """A 1D TTOperator acting on one axis of the layout (identity elsewhere)."""
    factors = [qtt_identity(int(L), tensorizer.b) for L in tensorizer.levels]
    factors[axis] = operator
    return qtt_layout_product(factors, tensorizer)


# --- QTT primitives of 1D operators on N = b^L points ------------------------------


def _shift_blocks(b: int):
    """A: digit a <- a + 1 (no carry), B: digit b - 1 <- 0 (carry)."""
    A = np.eye(b, k=1)
    B = np.zeros((b, b))
    B[b - 1, 0] = 1.0
    return A, B


def qtt_identity(L: int, b: int = 2) -> "tensap.TTOperator":
    """Identity of R^(b^L), of rank 1."""
    return tensap.TTOperator.eye([b] * L)


def qtt_shift(L: int, b: int = 2, periodic: bool = False) -> "tensap.TTOperator":
    """Shift S of R^(b^L), (S x)_j = x_{j+1} (ones on the superdiagonal).

    Adding 1 to an index written in base b changes its last digits by a
    carry: S = sum_k I x ... x I x A x B x ... x B (A at digit k), with A the
    shift of one digit and B = e_{b-1} e_0^T the carry. TT rank 2. With
    periodic=True, the corner e_{N-1} e_0^T (= B x ... x B) is added, with the
    same rank: (S x)_{N-1} = x_0.
    """
    A, B = _shift_blocks(b)
    if L == 1:
        return tensap.TTOperator([(A + periodic * B)[None, :, :, None]])
    first = np.zeros((1, b, b, 2))
    first[0, :, :, 0] = np.eye(b)
    first[0, :, :, 1] = A + periodic * B
    middle = np.zeros((2, b, b, 2))
    middle[0, :, :, 0] = np.eye(b)
    middle[0, :, :, 1] = A
    middle[1, :, :, 1] = B
    last = np.zeros((2, b, b, 1))
    last[0, :, :, 0] = A
    last[1, :, :, 0] = B
    return tensap.TTOperator([first] + [middle] * (L - 2) + [last])


def qtt_elementary_matrix(L: int, i: int, j: int, b: int = 2) -> "tensap.TTOperator":
    """Elementary matrix e_i e_j^T of size b^L, of rank 1."""
    di = tensap.integer2baseb(np.array([i]), b, L)[0]
    dj = tensap.integer2baseb(np.array([j]), b, L)[0]
    cores = []
    for p, q in zip(di, dj):
        core = np.zeros((1, b, b, 1))
        core[0, p, q, 0] = 1.0
        cores.append(core)
    return tensap.TTOperator(cores)


def qtt_ones_matrix(L: int, b: int = 2) -> "tensap.TTOperator":
    """The matrix of ones 1 1^T of size b^L, of rank 1."""
    return tensap.TTOperator([np.ones((1, b, b, 1))] * L)


def qtt_banded(diagonals: dict, L: int, b: int = 2, periodic: bool = False,
               tolerance: float = 1e-14) -> "tensap.TTOperator":
    """Banded Toeplitz matrix sum_k c_k S^k of size b^L.

    Parameters
    ----------
    diagonals : dict
        {offset k: value c_k}: c_k on the k-th diagonal (k > 0 above the main
        diagonal, k < 0 below). |k| < b^L.
    L, b : int
        Size b^L.
    periodic : bool, optional
        Circulant matrix (the diagonals wrap around). The default is False.
    tolerance : float, optional
        Precision of the rounding of the sum (exact up to round-off).

    Returns
    -------
    tensap.TTOperator
        Of small rank: the sum of the powers of the shift, rounded.
    """
    shift = qtt_shift(L, b, periodic)
    result = None
    for k, c in diagonals.items():
        if c == 0:
            continue
        if abs(k) >= b**L:
            raise ValueError(f"Offset {k} out of range for a matrix of size {b**L}.")
        term = qtt_identity(L, b)
        power = shift if k > 0 else shift.T
        for _ in range(abs(k)):
            term = _round(power @ term, tolerance)
        term = c * term
        result = term if result is None else _round(result + term, tolerance)
    return result if result is not None else 0.0 * qtt_identity(L, b)


def qtt_corner(block, L: int, corner: str = "top-left", b: int = 2,
               tolerance: float = 1e-14) -> "tensap.TTOperator":
    """Matrix of size b^L, zero except a small dense block in a corner.

    Parameters
    ----------
    block : numpy.ndarray
        Block of shape (p, q), placed at rows 0..p-1 ("top") or N-p..N-1
        ("bottom") and columns 0..q-1 ("left") or N-q..N-1 ("right").
    corner : str
        "top-left", "top-right", "bottom-left" or "bottom-right".

    Returns
    -------
    tensap.TTOperator
        Sum of elementary matrices, rounded: rank at most min(p, q) per bond
        in practice (a corner block is a low-rank correction).
    """
    block = np.atleast_2d(np.asarray(block, dtype=float))
    N = b**L
    vertical, horizontal = corner.split("-")
    if vertical not in ("top", "bottom") or horizontal not in ("left", "right"):
        raise ValueError("corner must be top-left, top-right, bottom-left or bottom-right.")
    p, q = block.shape
    rows = np.arange(p) if vertical == "top" else N - p + np.arange(p)
    cols = np.arange(q) if horizontal == "left" else N - q + np.arange(q)
    result = 0.0 * qtt_identity(L, b)
    for a, i in enumerate(rows):
        for c, j in enumerate(cols):
            if block[a, c] != 0:
                term = block[a, c] * qtt_elementary_matrix(L, int(i), int(j), b)
                result = _round(result + term, tolerance)
    return result


def _round(tensor, tolerance):
    return tensap.Truncator(tolerance=tolerance).truncate(tensor)
