# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tt_tensor: tensors and operators in tensor-train (TT) format.

Port of TTTensor.m and OperatorTTTensor.m of ApproximationToolbox. A TT
tensor of order d is a list of cores, contiguous read-only numpy arrays,

- TTTensor: x(i_1, ..., i_d) = X_1(i_1) X_2(i_2) ... X_d(i_d), with cores of
  shape (r_{k-1}, n_k, r_k);
- TTOperator: A(i_1, j_1, ..., i_d, j_d) = A_1(i_1, j_1) ... A_d(i_d, j_d),
  with cores of shape (r_{k-1}, n_k, m_k, r_k) (output, input),

and r_0 = r_d = 1. The operations of a vector space are written once in
BaseTTTensor, on the cores seen as (r_{k-1}, n_k m_k, r_k) (a reshape without
copy). The object keeps the index of its orthogonality center, if any: the
cores on its left are left-orthogonal, those on its right right-orthogonal,
so that the norm of the tensor is the norm of the center core.
"""

from typing import List, Optional, Sequence

import numpy as np
import tensap


def _read_only(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


class BaseTTTensor:
    """
    Base class of the TT formats: cores and vector-space operations.

    Attributes
    ----------
    cores : list of numpy.ndarray
        The d cores, of shape (r_{k-1}, *physical shape, r_k).
    orth_center : int or None
        Index of the orthogonality center, None if the representation is
        not orthogonalized.
    """

    # number of physical axes of a core (1 for a vector, 2 for an operator)
    _physical_ndim = 1

    def __init__(self, cores: Sequence[np.ndarray], orth_center: Optional[int] = None):
        cores = [_read_only(np.asarray(c, dtype=float)) for c in cores]
        if len(cores) == 0:
            raise ValueError("A TT tensor needs at least one core.")
        for k, core in enumerate(cores):
            if core.ndim != self._physical_ndim + 2:
                raise ValueError(
                    f"Core {k} has {core.ndim} axes, expected {self._physical_ndim + 2}."
                )
        if cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
            raise ValueError("The boundary ranks r_0 and r_d must be 1.")
        for k in range(len(cores) - 1):
            if cores[k].shape[-1] != cores[k + 1].shape[0]:
                raise ValueError(f"Ranks of the cores {k} and {k + 1} do not match.")
        self.cores = cores
        self.orth_center = orth_center

    # --- properties ---

    @property
    def order(self) -> int:
        """Number of dimensions d."""
        return len(self.cores)

    @property
    def ranks(self) -> tuple:
        """TT ranks (r_1, ..., r_{d-1})."""
        return tuple(core.shape[-1] for core in self.cores[:-1])

    def representation_rank(self) -> tuple:
        """TT ranks (r_1, ..., r_{d-1}), as for the other formats."""
        return self.ranks

    @property
    def is_orth(self) -> bool:
        """True if the representation has an orthogonality center."""
        return self.orth_center is not None

    def storage(self) -> int:
        """Number of stored entries."""
        return int(sum(core.size for core in self.cores))

    def sparse_storage(self) -> int:
        """Number of non-zero stored entries."""
        return int(sum(np.count_nonzero(core) for core in self.cores))

    # --- vectorized view of the cores ---

    def _vec_cores(self) -> List[np.ndarray]:
        """Cores seen as (r_{k-1}, prod(physical shape), r_k), without copy."""
        return [c.reshape(c.shape[0], -1, c.shape[-1]) for c in self.cores]

    def _from_vec_cores(self, cores, orth_center=None):
        """Same class and physical shapes as self, from vectorized cores."""
        shapes = [c.shape[1:-1] for c in self.cores]
        return self.__class__(
            [v.reshape(v.shape[0], *s, v.shape[-1]) for v, s in zip(cores, shapes)],
            orth_center,
        )

    # --- conversion to dense ---

    def _full_vec(self) -> np.ndarray:
        """Dense array with one axis of size prod(physical shape) per core."""
        cores = self._vec_cores()
        result = cores[0][0]
        for core in cores[1:]:
            result = np.tensordot(result, core, axes=(-1, 0))
        return result[..., 0]

    def numpy(self) -> np.ndarray:
        """Dense numpy array."""
        return self.full().numpy()

    # --- vector-space operations ---

    def dot(self, other: "BaseTTTensor") -> float:
        """Canonical (Frobenius) inner product, by a sweep over the cores."""
        self._check_compatible(other)
        z = np.ones((1, 1))
        for x, y in zip(self._vec_cores(), other._vec_cores()):
            z = np.tensordot(np.tensordot(z, x, axes=(0, 0)), y, axes=([0, 1], [0, 1]))
        return float(z[0, 0])

    def norm(self) -> float:
        """Frobenius norm: norm of the center core of the orthogonalized tensor."""
        x = self if self.is_orth else self.orth()
        return float(np.linalg.norm(x.cores[x.orth_center]))

    def orth(self, center: int = 0) -> "BaseTTTensor":
        """Orthogonalization with the orthogonality center at the core `center`.

        The cores 0, ..., center - 1 are left-orthogonal (QR from the left),
        the cores center + 1, ..., d - 1 right-orthogonal (QR from the right).
        The ranks may decrease (reduced QR).
        """
        d = self.order
        if not 0 <= center < d:
            raise ValueError(f"center must be in [0, {d - 1}].")
        cores = list(self._vec_cores())
        for k in range(center):
            r0, n, r1 = cores[k].shape
            q, r = np.linalg.qr(cores[k].reshape(r0 * n, r1))
            cores[k] = q.reshape(r0, n, -1)
            cores[k + 1] = np.tensordot(r, cores[k + 1], axes=(1, 0))
        for k in range(d - 1, center, -1):
            r0, n, r1 = cores[k].shape
            q, r = np.linalg.qr(cores[k].reshape(r0, n * r1).T)
            cores[k] = q.T.reshape(-1, n, r1)
            cores[k - 1] = np.tensordot(cores[k - 1], r.T, axes=(-1, 0))
        return self._from_vec_cores(cores, center)

    def __add__(self, other: "BaseTTTensor") -> "BaseTTTensor":
        """Sum: cores in block-diagonal position, ranks r_k + r'_k."""
        if type(self) is not type(other):
            return NotImplemented
        self._check_compatible(other)
        if self.order == 1:
            return self.__class__([self.cores[0] + other.cores[0]])
        x, y = self._vec_cores(), other._vec_cores()
        d = self.order
        cores = [np.concatenate((x[0], y[0]), axis=2)]
        for k in range(1, d - 1):
            a, n, b = x[k].shape
            c, _, e = y[k].shape
            core = np.zeros((a + c, n, b + e))
            core[:a, :, :b] = x[k]
            core[a:, :, b:] = y[k]
            cores.append(core)
        cores.append(np.concatenate((x[-1], y[-1]), axis=0))
        return self._from_vec_cores(cores)

    def __radd__(self, other):
        if np.isscalar(other) and other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: "BaseTTTensor") -> "BaseTTTensor":
        if type(self) is not type(other):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> "BaseTTTensor":
        return self * -1.0

    def __mul__(self, scalar: float) -> "BaseTTTensor":
        """Product by a scalar, applied to the center core (or the first one)."""
        if not np.isscalar(scalar):
            return NotImplemented
        k = self.orth_center if self.is_orth else 0
        cores = list(self.cores)
        cores[k] = cores[k] * scalar
        return self.__class__(cores, self.orth_center)

    def __rmul__(self, scalar: float) -> "BaseTTTensor":
        return self * scalar

    def __truediv__(self, scalar: float) -> "BaseTTTensor":
        if not np.isscalar(scalar):
            return NotImplemented
        return self * (1.0 / scalar)

    def hadamard_product(self, other: "BaseTTTensor") -> "BaseTTTensor":
        """Entrywise product: Kronecker product of the cores along the ranks."""
        if type(self) is not type(other):
            raise TypeError("hadamard_product requires two tensors of the same type.")
        self._check_compatible(other)
        cores = []
        for x, y in zip(self._vec_cores(), other._vec_cores()):
            a, n, b = x.shape
            c, _, e = y.shape
            cores.append(np.einsum("anb,cnd->acnbd", x, y).reshape(a * c, n, b * e))
        return self._from_vec_cores(cores)

    def tensor_product(self, other: "BaseTTTensor") -> "BaseTTTensor":
        """Tensor product x (x) y on V (x) W: the chains of cores are
        concatenated (order d + d', ranks (r, 1, r')). For QTT tensors with the
        block ordering, the tensor product of the axes."""
        if type(self) is not type(other):
            raise TypeError("tensor_product requires two tensors of the same type.")
        return self.__class__(list(self.cores) + list(other.cores))

    # --- helpers ---

    def _check_compatible(self, other):
        if self.order != other.order:
            raise ValueError("The tensors must have the same order.")
        for x, y in zip(self.cores, other.cores):
            if x.shape[1:-1] != y.shape[1:-1]:
                raise ValueError("The tensors must have the same physical shape.")

    def __repr__(self):
        return (
            f"<{type(self).__name__}:\n"
            f"\torder = {self.order},\n"
            f"\tshape = {self.shape},\n"
            f"\tranks = {self.ranks},\n"
            f"\torth_center = {self.orth_center}>"
        )


class TTTensor(BaseTTTensor):
    """
    Tensor in TT format, cores of shape (r_{k-1}, n_k, r_k).

    Also used as the core of a Tucker tensor (then n_k is the Tucker rank of
    dimension k): it implements the methods that BaseTuckerTensor calls on
    its core.
    """

    _physical_ndim = 1

    @property
    def shape(self) -> np.ndarray:
        """Sizes (n_1, ..., n_d)."""
        return np.array([core.shape[1] for core in self.cores])

    def full(self) -> "tensap.FullTensor":
        """Dense tensor of shape (n_1, ..., n_d)."""
        return tensap.FullTensor(self._full_vec())

    # --- products with matrices and vectors (core protocol) ---

    def tensor_matrix_product(self, matrices, dims=None) -> "TTTensor":
        """Product of dimension dims[i] with the matrix matrices[i] (M @ x)."""
        dims, matrices = _dims_and_list(matrices, dims, self.order)
        cores = list(self.cores)
        for k, matrix in zip(dims, matrices):
            matrix = getattr(matrix, "data", matrix)
            cores[k] = np.einsum("mn,anb->amb", matrix, cores[k])
        return TTTensor(cores)

    def tensor_diagonal_matrix_product(self, matrices, dims=None) -> "TTTensor":
        """Product with diagonal matrices given by their diagonals."""
        dims, matrices = _dims_and_list(matrices, dims, self.order)
        cores = list(self.cores)
        for k, diagonal in zip(dims, matrices):
            cores[k] = cores[k] * np.ravel(diagonal)[None, :, None]
        return TTTensor(cores)

    def tensor_vector_product(self, vectors, dims=None):
        """Contraction of dimension dims[i] with the vector vectors[i].

        The contracted cores become matrices, absorbed into the next kept core
        (or the previous one at the end). Returns a float if every dimension
        is contracted.
        """
        dims, vectors = _dims_and_list(vectors, dims, self.order)
        contracted = dict(zip(dims, vectors))
        cores, pending = [], None
        for k, core in enumerate(self.cores):
            if k in contracted:
                matrix = np.tensordot(core, np.ravel(contracted[k]), axes=(1, 0))
                pending = matrix if pending is None else pending @ matrix
            else:
                if pending is not None:
                    core = np.tensordot(pending, core, axes=(1, 0))
                    pending = None
                cores.append(core)
        if not cores:
            return float(pending[0, 0])
        if pending is not None:
            cores[-1] = np.tensordot(cores[-1], pending, axes=(-1, 0))
        return TTTensor(cores)

    def tensor_matrix_product_eval_diag(self, matrices, dims=None):
        """Evaluations sum_i x(i) prod_k M_k[p, i_k] for every point p.

        Used by FunctionalTensor.eval, with M_k the evaluations of the basis
        of dimension k at the points (shape (P, n_k)). Only the evaluation in
        all the dimensions is implemented.

        Returns
        -------
        tensap.FullTensor
            Of order 1, the P evaluations.
        """
        if dims is not None and len(np.atleast_1d(dims)) != self.order:
            raise NotImplementedError("Partial evaluations are not implemented.")
        matrices = [np.asarray(getattr(m, "data", m)) for m in matrices]
        z = np.ones((matrices[0].shape[0], 1))
        for core, m in zip(self.cores, matrices):
            z = np.einsum("pa,anb,pn->pb", z, core, m)
        return tensap.FullTensor(z[:, 0])

    def tensordot_matrix_product_except_dim(self, other, matrices, dim) -> np.ndarray:
        """Contraction of self and other in every dimension but dim, through
        the matrices: z(i, j) = sum x(..., i, ...) M_k(i_k, j_k) y(..., j, ...),
        the sum running over the indices of the dimensions k != dim, as
        FullTensor.tensordot_matrix_product_except_dim.

        Parameters
        ----------
        other : TTTensor
        matrices : list of numpy.ndarray
            One matrix per dimension, of shape (self.shape[k], other.shape[k])
            (the matrix of dimension dim is not used).
        dim : int

        Returns
        -------
        numpy.ndarray
            Of shape (self.shape[dim], other.shape[dim]).
        """
        if len(matrices) != self.order:
            raise ValueError("One matrix per dimension is required.")

        def step(z, x, y, matrix):
            t = np.tensordot(z, x, axes=(0, 0))  # (c, i, b)
            t = np.tensordot(t, matrix, axes=(1, 0))  # (c, b, j)
            return np.tensordot(t, y, axes=([0, 2], [0, 1]))  # (b, d)

        left = np.ones((1, 1))
        for k in range(dim):
            left = step(left, self.cores[k], other.cores[k], matrices[k])
        right = np.ones((1, 1))
        for k in range(self.order - 1, dim, -1):
            x, y = self.cores[k], other.cores[k]
            t = np.tensordot(x, right, axes=(2, 0))  # (a, i, d)
            t = np.tensordot(t, matrices[k], axes=(1, 0))  # (a, d, j)
            right = np.tensordot(t, y, axes=([1, 2], [2, 1]))  # (a, c)
        t = np.tensordot(left, self.cores[dim], axes=(0, 0))  # (c, i, b)
        t = np.tensordot(t, right, axes=(2, 0))  # (c, i, c')
        return np.tensordot(t, other.cores[dim], axes=([0, 2], [0, 2]))

    def kron(self, other: "TTTensor") -> "TTTensor":
        """Kronecker product, z(i n' + j) = x(i) y(j) in every dimension."""
        cores = []
        for x, y in zip(self.cores, other.cores):
            a, n, b = x.shape
            c, m, e = y.shape
            cores.append(np.einsum("aib,cjd->acijbd", x, y).reshape(a * c, n * m, b * e))
        return TTTensor(cores)

    def cat(self, other: "TTTensor") -> "TTTensor":
        """Block-diagonal concatenation: z has size n_k + m_k in dimension k,
        equal to x on the first block of indices and to y on the last one."""
        d = self.order
        cores = []
        for k, (x, y) in enumerate(zip(self.cores, other.cores)):
            a, n, b = x.shape
            c, m, e = y.shape
            rows = 1 if k == 0 else a + c
            cols = 1 if k == d - 1 else b + e
            core = np.zeros((rows, n + m, cols))
            row_y = 0 if k == 0 else a
            col_y = 0 if k == d - 1 else b
            core[:a, :n, :b] = x
            core[row_y:row_y + c, n:, col_y:col_y + e] = y
            cores.append(core)
        return TTTensor(cores)

    # --- conversions ---

    def tree_based_tensor(self) -> "tensap.TreeBasedTensor":
        """Tree-based tensor on the linear tree (TreeBasedTensor.tensor_train)."""
        cores = list(self.cores)
        if self.order == 1:
            raise ValueError("The linear tree needs an order of at least 2.")
        tensors = [tensap.FullTensor(cores[0][0])]
        tensors += [tensap.FullTensor(c) for c in cores[1:-1]]
        tensors.append(tensap.FullTensor(cores[-1][:, :, 0]))
        return tensap.TreeBasedTensor.tensor_train(np.array(tensors, dtype=object))

    @classmethod
    def from_tree_based_tensor(cls, tree_tensor) -> "TTTensor":
        """Inverse of tree_based_tensor, for a tensor on DimensionTree.linear.

        Active leaves (other than the one of the first dimension) are absorbed
        into the cores; an inactive first leaf is the identity.
        """
        d = tree_tensor.order
        tree = tree_tensor.tree
        linear = tensap.DimensionTree.linear(d)
        if not (np.array_equal(tree.dim2ind, linear.dim2ind)
                and np.array_equal(tree.adjacency_matrix, linear.adjacency_matrix)):
            raise ValueError("The tree must be DimensionTree.linear(order).")
        tensors = tree_tensor.tensors
        active = tree_tensor.is_active_node
        leaf = tree.dim2ind[0] - 1
        first = tensors[leaf].data if active[leaf] else np.eye(tree_tensor.shape[0])
        cores = [first[None]]
        nod = tree.parent(tree.dim2ind[0])
        for k in range(1, d):
            core = tensors[nod - 1].data
            if core.ndim == 2:  # root: no parent rank
                core = core[:, :, None]
            leaf = tree.dim2ind[k] - 1
            if active[leaf]:
                core = np.einsum("anb,in->aib", core, tensors[leaf].data)
            cores.append(core)
            nod = tree.parent(nod) if k < d - 1 else nod
        return cls(cores)

    @classmethod
    def from_diagonal_tensor(cls, diagonal) -> "TTTensor":
        """TT representation of a diagonal tensor: sum_k s_k e_k x ... x e_k,
        ranks R (MATLAB TTTensor(DiagonalTensor))."""
        data = np.ravel(diagonal.data)
        R, d = data.size, diagonal.order
        if d == 1:
            return cls([data.reshape(1, R, 1)])
        delta = np.zeros((R, R, R))
        delta[np.arange(R), np.arange(R), np.arange(R)] = 1.0
        first = np.diag(data)[None]
        last = np.eye(R)[:, :, None]
        return cls([first] + [delta] * (d - 2) + [last])

    @classmethod
    def from_full(cls, tensor, tolerance: float = 0.0) -> "TTTensor":
        """TT-SVD of a dense tensor (exact for tolerance = 0)."""
        a = np.asarray(getattr(tensor, "data", tensor), dtype=float)
        if not np.any(a):
            # zero tensor: rank one (the SVDs of zero matrices are not defined
            # for the relative truncation)
            return cls([np.zeros((1, n, 1)) for n in a.shape])
        truncator = tensap.Truncator(tolerance=tolerance)
        local_tolerance = tolerance / np.sqrt(max(a.ndim - 1, 1))
        cores, r = [], 1
        rest = a.reshape(1, -1)
        for n in a.shape[:-1]:
            rest = rest.reshape(r * n, -1)
            svd = _truncated_svd(truncator, rest, local_tolerance)
            cores.append(svd[0].reshape(r, n, -1))
            rest = svd[1]
            r = rest.shape[0]
        cores.append(rest.reshape(r, a.shape[-1], 1))
        return cls(cores)

    @classmethod
    def from_tucker(cls, tucker) -> "TTTensor":
        """Exact TT representation of a TuckerTensorVector: each core absorbs
        its frame, X_k(a, i, b) = sum_l G_k(a, l, b) U_k(i, l). The core of
        the Tucker tensor is converted to a TT tensor first."""
        core = as_tt_tensor(tucker.core)
        factors = tucker.tensor_frame.get_vectorize_factors()
        return cls([np.einsum("alb,il->aib", g, u) for g, u in zip(core.cores, factors)])

    # --- constructors ---

    @classmethod
    def create(cls, generator, shape, ranks) -> "TTTensor":
        """Cores filled by generator(shape of the core); ranks (r_1, ..., r_{d-1})."""
        r = [1] + list(np.atleast_1d(ranks).astype(int)) + [1]
        if len(r) != len(shape) + 1:
            raise ValueError("ranks must have order - 1 entries.")
        return cls([generator((r[k], n, r[k + 1])) for k, n in enumerate(shape)])

    @classmethod
    def randn(cls, shape, ranks) -> "TTTensor":
        return cls.create(lambda s: np.random.randn(*s), shape, ranks)

    @classmethod
    def rand(cls, shape, ranks) -> "TTTensor":
        return cls.create(lambda s: np.random.rand(*s), shape, ranks)

    @classmethod
    def zeros(cls, shape, ranks=None) -> "TTTensor":
        ranks = [1] * (len(shape) - 1) if ranks is None else ranks
        return cls.create(np.zeros, shape, ranks)

    @classmethod
    def ones(cls, shape) -> "TTTensor":
        """The tensor of ones, of rank one."""
        return cls.create(np.ones, shape, [1] * (len(shape) - 1))


class TTOperator(BaseTTTensor):
    """
    Operator in TT format, cores of shape (r_{k-1}, n_k, m_k, r_k): output
    size n_k, input size m_k.
    """

    _physical_ndim = 2

    @property
    def shape(self) -> tuple:
        """Shapes ((n_1, m_1), ..., (n_d, m_d))."""
        return tuple((core.shape[1], core.shape[2]) for core in self.cores)

    def full(self) -> "tensap.FullTensor":
        """Dense array of shape (n_1, m_1, ..., n_d, m_d), as
        TuckerTensorOperator.full."""
        result = self._full_vec()
        return tensap.FullTensor(result.reshape([n for s in self.shape for n in s]))

    def __matmul__(self, other):
        """Application to a TTTensor, or composition with a TTOperator: the
        cores are multiplied, ranks r_k r'_k."""
        if not isinstance(other, BaseTTTensor):
            return NotImplemented
        if self.order != other.order:
            raise ValueError("The tensors must have the same order.")
        cores = []
        for a, x in zip(self.cores, other.cores):
            if a.shape[2] != x.shape[1]:
                raise ValueError("Incompatible input and output sizes.")
            if isinstance(other, TTTensor):
                z = np.einsum("aijb,cjd->acibd", a, x)
                cores.append(z.reshape(a.shape[0] * x.shape[0], a.shape[1], -1))
            else:
                z = np.einsum("aijb,cjkd->acikbd", a, x)
                cores.append(z.reshape(a.shape[0] * x.shape[0], a.shape[1], x.shape[2], -1))
        return other.__class__(cores)

    def transpose(self) -> "TTOperator":
        """Transpose: every core matrix is transposed."""
        return TTOperator([core.transpose(0, 2, 1, 3) for core in self.cores], self.orth_center)

    @property
    def T(self) -> "TTOperator":
        return self.transpose()

    def vectorize(self) -> TTTensor:
        """The operator as a TTTensor with cores (r, n m, r')."""
        return TTTensor(self._vec_cores(), self.orth_center)

    @classmethod
    def unvectorize(cls, tensor: TTTensor, shape) -> "TTOperator":
        """Inverse of vectorize, shape = ((n_1, m_1), ..., (n_d, m_d))."""
        return cls(
            [c.reshape(c.shape[0], n, m, c.shape[-1]) for c, (n, m) in zip(tensor.cores, shape)],
            tensor.orth_center,
        )

    @classmethod
    def from_tucker(cls, tucker) -> "TTOperator":
        """Exact TT representation of a TuckerTensorOperator: each core absorbs
        its frame, W_k(a, i, j, b) = sum_l G_k(a, l, b) F^(k)_l(i, j)."""
        core = as_tt_tensor(tucker.core)
        frames = [f.data for f in tucker.tensor_frame.frames]
        return cls([np.einsum("alb,lij->aijb", g, f) for g, f in zip(core.cores, frames)])

    @classmethod
    def eye(cls, shape) -> "TTOperator":
        """Identity of R^{n_1} x ... x R^{n_d}, of rank one."""
        return cls([np.eye(n)[None, :, :, None] for n in shape])


# --- helpers ---


def as_tt_tensor(tensor) -> TTTensor:
    """A TTTensor representing the same tensor (TT, diagonal or full)."""
    if isinstance(tensor, TTTensor):
        return tensor
    if isinstance(tensor, tensap.DiagonalTensor):
        return TTTensor.from_diagonal_tensor(tensor)
    if isinstance(tensor, tensap.FullTensor):
        return TTTensor.from_full(tensor)
    raise NotImplementedError(f"Cannot convert a {type(tensor).__name__} to a TTTensor.")


def _dims_and_list(items, dims, order):
    if dims is None:
        dims = list(range(order))
    dims = [int(k) for k in np.atleast_1d(dims)]
    if not isinstance(items, list):
        items = [items]
    if len(items) != len(dims):
        raise ValueError("One matrix or vector per dimension is required.")
    return dims, items


def _truncated_svd(truncator, matrix, tolerance):
    """(U, S V^T) of the truncated SVD of matrix, with tensap.Truncator."""
    svd = truncator.trunc_svd(matrix, tolerance)
    left, right = svd.space
    return left, svd.core.data[:, None] * right.T
