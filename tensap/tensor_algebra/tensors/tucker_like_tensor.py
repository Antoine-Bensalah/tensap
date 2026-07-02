# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tucker_like_tensor.
"""

import numpy as np
import tensap
from typing import Union, List, Tuple, Optional
from .tensor_frame import TFrame, TFrameVector, TFrameOperator
from .frame import Frame, FrameVector, FrameOperator


class TuckerLikeTensor:
    """
    Class TuckerLikeTensor.
    Algebraic tensors in Tucker-like tensor format.

    Attributes
    ----------
    core : tensap.FullTensor, tensap.TreeBasedTensor, tensap.DiagonalTensor,
           or tensap.SparseTensor
        The core of the Tucker tensor.
    space : tensap.TSpaceVectors or tensap.TSpaceOperators
        The tensor product space (factor matrices).
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor in the full physical space.
    ranks : numpy.ndarray
        The Tucker ranks (subspace dimensions).
    is_orth : bool
        Boolean indicating if the representation of the tensor is orthogonal.
    """

    def __init__(self, core, space=None):
        """
        Constructor for the class TuckerLikeTensor.

        Parameters
        ----------
        core : tensap.FullTensor, tensap.TreeBasedTensor, tensap.DiagonalTensor,
               tensap.SparseTensor, or numpy.ndarray
            The core tensor. If a numpy.ndarray, cast to FullTensor.
            If a TreeBasedTensor and space is None, converts the
            TreeBasedTensor into Tucker-like format: for each dimension,
            the factor matrix is extracted from the corresponding leaf
            node (if active) or set to the identity (if inactive).
        space : tensap.TSpaceVectors, tensap.TSpaceOperators, or list, optional
            The tensor product space. If a list of arrays, cast to
            TSpaceVectors.
        """
        if isinstance(core, np.ndarray):
            core = tensap.FullTensor(core)

        if space is not None:
            if isinstance(space, (list, tuple)):
                space = tensap.TSpaceVectors(space)
            self.core = core
            self.space = space
        elif isinstance(core, tensap.TreeBasedTensor):
            y = core
            tree = y.tree
            space_list = [None] * y.order
            for mu in range(y.order):
                nod = tree.dim2ind[mu]
                idx = nod - 1
                if y.is_active_node[idx]:
                    space_list[mu] = y.tensors[idx].data
                else:
                    space_list[mu] = np.eye(y.shape[mu])
            self.core = y.tensors[0]
            self.space = tensap.TSpaceVectors(space_list)
            self.is_orth = y.is_orth
        else:
            self.core = core
            self.space = space

        self._update_properties()

    def _update_properties(self):
        """Update order, shape, ranks, and orthogonality flags."""
        self.order = self.space.order
        if isinstance(self.space, tensap.TSpaceOperators):
            self.shape = np.column_stack(
                [self.space.dims_out, self.space.dims_in]
            )
        else:
            self.shape = self.space.dims_out.copy()
        self.ranks = self.space.ranks.copy()
        self.is_orth = self.core.is_orth and self.space.is_orth

    # ---- Conversion ----

    def full(self):
        """
        Convert the object to a dense tensap.FullTensor.

        Returns
        -------
        tensap.FullTensor
            The Tucker tensor reconstructed as a dense tensor.
        """
        result = self.core.tensor_matrix_product(
            self.space.vectorize_factors()
        ).full()
        return result.reshape(self.space.unvectorized_shape)

    def numpy(self):
        """
        Convert the TuckerLikeTensor to a dense numpy.ndarray.
        """
        return self.full().numpy()

    def tree_based_tensor(self):
        """Convert into a TreeBasedTensor.

        Returns
        -------
        tensap.TreeBasedTensor
        """
        if isinstance(self.space, tensap.TSpaceOperators):
            raise NotImplementedError(
                "Method not implemented for TSpaceOperators."
            )
        if isinstance(self.core, tensap.TreeBasedTensor):
            x = self.core
            x.tensors[x.tree.dim2ind] = self.space.spaces
            return x
        elif isinstance(self.core, tensap.FullTensor):
            tree = tensap.DimensionTree.trivial(self.order)
            tensors = [self.core] + [
                tensap.FullTensor(m) for m in self.space.vectorize_factors()
            ]
            return tensap.TreeBasedTensor(tensors, tree)
        elif isinstance(self.core, tensap.DiagonalTensor):
            from functools import reduce
            x = reduce(lambda a, b: a + b, self.space.spaces)
            # TODO: proper DiagonalTensor case
            raise NotImplementedError(
                "DiagonalTensor core to TreeBasedTensor not yet implemented."
            )
        else:
            raise TypeError(
                f"Unsupported core type: {type(self.core)}"
            )

    # ---- Unary operators ----

    def __neg__(self):
        """Return the negative of the tensor."""
        return TuckerLikeTensor(-self.core, self.space)

    def __abs__(self):
        return TuckerLikeTensor(abs(self.core.data), self.space)

    # ---- Arithmetic operators ----

    def __add__(self, other):
        if isinstance(other, TuckerLikeTensor):
            core_self, core_other = tensap.convert_tensors(self.core, other.core)
            new_core = core_self.cat(core_other)
            new_space = self.space.cat(other.space)
            return TuckerLikeTensor(new_core, new_space)
        return NotImplemented

    def __radd__(self, other):
        """Only to allowed the Python sum function over TuckerLikeTensor
        e.g.: sum([T1, T2, T3]) which initialize the sum with 0
        """
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TuckerLikeTensor):
            return self + (-other)
        return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other):
            return TuckerLikeTensor(self.core * other, self.space)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            return TuckerLikeTensor(self.core / other, self.space)
        return NotImplemented

    # ---- Core operations ----

    def dot(self, tensor_2):
        """
        Compute the inner product of self (A) and tensor_2 (B): <A, B>.

        Denoting the two tensors in Tucker-like format:
        A = sum_{k} A(k_1, ..., k_d) * A^(1)_{k_1} x ... x A^(d)_{k_d}
        B = sum_{k'} B(k'_1, ..., k'_d) * B^(1)_{k'_1} x ... x B^(d)_{k'_d}

        The inner product is computed efficiently by pushing the
        local space inner products into the core tensor of B, avoiding
        any full tensor reconstruction or Kronecker products:

        1. Compute the Gram matrices M^(v) for each dimension v:
           M^(v)_{k_v, k'_v} = < A^(v)_{k_v}, B^(v)_{k'_v} >
        2. Contract the core of B with the Gram matrices along all
           dimensions to obtain the projected core B_tilde:
           B_tilde(k_1, ..., k_d) = sum_{k'} B(k'_1, ..., k'_d) * M^(1) ... M^(d)
        3. Compute the standard dot product between the core A and B_tilde:
           <A, B> = sum_{k} A(k_1, ..., k_d) * B_tilde(k_1, ..., k_d)

        Parameters
        ----------
        tensor_2 : TuckerLikeTensor
            The second tensor to compute the inner product with.

        Returns
        -------
        float
            The inner product <A, B>.
        """
        assert isinstance(
            tensor_2, TuckerLikeTensor
        ), "Argument must be a TuckerLikeTensor."

        core_self, core_other = tensap.convert_tensors(self.core, tensor_2.core)
        M = self.space.dot(tensor_2.space)
        core_2_projected = core_other.tensor_matrix_product(M)
        return core_self.dot(core_2_projected)

    def dot_with_rank_one_metric(self, tensor_2, matrices):
        """
        Weighted inner product: dot(self, times_matrix(tensor_2, matrices)).

        Parameters
        ----------
        tensor_2 : TuckerLikeTensor
        matrices : list of numpy.ndarray

        Returns
        -------
        float
        """
        tmp = tensor_2.tensor_matrix_product(matrices)
        return self.dot(tmp)

    def norm(self):
        """Compute the canonical Frobenius norm."""
        if self.is_orth:
            return self.core.norm()
        else:
            return np.sqrt(np.abs(self.dot(self)))

    def orth(self):
        """
        Orthogonalize the TuckerLikeTensor.

        Orthonormalize the factor matrices (TSpace), absorbs
        the transformation into the core, then orthonormalize the
        core itself (matching MATLAB behaviour).

        Returns
        -------
        TuckerLikeTensor
            The orthonormalized tensor (self, modified in-place).
        """
        self.space, M = self.space.orth()
        self.core = self.core.tensor_matrix_product(M)
        core_orth = self.core.orth()
        if isinstance(core_orth, tuple):
            self.core = core_orth[0]
        else:
            self.core = core_orth
        self._update_properties()
        return self

    # ---- Storage ----

    def storage(self):
        """Return the storage complexity."""
        return self.space.storage() + self.core.storage()

    def sparse_storage(self):
        """Return the number of non-zero entries."""
        return self.space.sparse_storage() + self.core.sparse_storage()

    def representation_rank(self):
        """Return the representation rank (prod of subspace dims)."""
        return int(np.prod(self.ranks))

    # ---- Contractions ----

    def tensor_matrix_product(self, matrices, dims=None):
        """
        Contract the Tucker tensor with matrices along physical dimensions.

        Left-multiplies the factor matrices by ``matrices[mu]`` along
        dimensions ``dims``.

        Parameters
        ----------
        matrices : list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.matrix_times_space(matrices, dims)
        return TuckerLikeTensor(self.core, new_space)

    def tensor_vector_product(self, vectors, dims=None):
        """
        Contract the tensor with vectors along given dimensions.

        Parameters
        ----------
        vectors : list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor or numpy.ndarray (if all dims contracted)
        """
        assert isinstance(self.space, tensap.TSpaceVectors), \
            "The TSpace must be of TSpaceVectors type."

        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        if isinstance(vectors, np.ndarray):
            vectors = [vectors]
        vectors = [np.atleast_2d(v.ravel()) for v in vectors]

        # Left-multiply space by row vectors: v(1,N) @ space(N,1,R) -> (1,1,R)
        new_space = self.space.matrix_times_space(vectors, dims)
        # Reshape contracted factor mats from (1,1,R) to (R,1,1) for core
        new_spaces = list(new_space.spaces)
        for i, d in enumerate(dims):
            new_spaces[d] = new_spaces[d].transpose(2, 1, 0)
        xs = tensap.TSpaceVectors(new_spaces, is_orth=False)

        # Contract core with the transposed factor matrices
        xc = self.core.tensor_vector_product(xs.spaces, dims)

        if len(dims) != self.order:
            xs = xs.remove_space(dims)
            return TuckerLikeTensor(xc, xs)
        else:
            return xc

    def tensor_diag_matrix_product(self, M, dims=None):
        """
        Contract the tensor with diagonal matrices.

        Parameters
        ----------
        M : numpy.ndarray or list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor
        """
        if isinstance(M, np.ndarray):
            M = [M]
        if dims is None:
            dims = range(self.order)
        diag_mats = [np.diag(m.ravel()) for m in M]
        return self.tensor_matrix_product(diag_mats, dims)

    # ---- Tensor algebra ----

    def cat(self, other):
        """
        Concatenate two TuckerLikeTensors.

        Uses block-diagonal concatenation for spaces and core concatenation
        along all dimensions.

        Parameters
        ----------
        other : TuckerLikeTensor

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.cat(other.core)
        new_space = self.space.diag_cat(other.space)
        return TuckerLikeTensor(new_core, new_space)

    def kron(self, other):
        """
        Kronecker product of two TuckerLikeTensors.

        Parameters
        ----------
        other : TuckerLikeTensor

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.kron(other.core)
        new_space = self.space.kron(other.space)
        return TuckerLikeTensor(new_core, new_space)

    # ---- Dimension manipulation ----

    def permute(self, dims):
        """
        Permute the dimensions of the tensor.

        Parameters
        ----------
        dims : array_like
            New ordering of dimensions.

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.transpose(dims)
        new_space = self.space.permute(dims)
        return TuckerLikeTensor(new_core, new_space)

    def squeeze(self, dims=None):
        """
        Remove singleton dimensions.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to remove. Defaults to all singleton dims.

        Returns
        -------
        TuckerLikeTensor or numpy scalar
        """
        if dims is None:
            if self.shape.ndim == 2:
                dims = np.where(np.all(self.shape == 1, axis=1))[0]
            else:
                dims = np.where(self.shape == 1)[0]
        dims = np.atleast_1d(dims)

        if len(dims) == 0:
            return self

        # Contract singleton dimensions into the core
        new_core = self.core.tensor_vector_product(
            self.space.transposed_factors(dims), dims
        )

        # Remove those dimensions from space
        keep_dims = [d for d in range(self.order) if d not in dims]
        new_space = self.space.keep_space(keep_dims)
        return TuckerLikeTensor(new_core, new_space)

    # ---- Subspace extraction ----

    def sub_tensor(self, *args):
        """
        Extract a subtensor.

        Parameters
        ----------
        *args : indices for each dimension (arrays or ':')
        """
        new_spaces = list(self.space.spaces)
        if isinstance(self.space, tensap.TSpaceVectors):
            for k in range(self.order):
                if not isinstance(args[k], str) or args[k] != ':':
                    idx = np.atleast_1d(args[k])
                    new_spaces[k] = self.space.subspace(k, idx)
        elif isinstance(self.space, tensap.TSpaceOperators):
            for k in range(self.order):
                I1 = args[2 * k]
                I2 = args[2 * k + 1]
                new_spaces[k] = self.space.subspace(k, I1, I2)
        self.space = self.space.__class__(new_spaces, is_orth=False)
        self._update_properties()

    # ---- Operators specific ----

    def transpose(self):
        """Transpose of a TuckerLikeTensor of type Operator."""
        assert isinstance(self.space, tensap.TSpaceOperators), \
            "The TSpace must be of TSpaceOperators type."
        return TuckerLikeTensor(self.core, self.space.T)

    @property
    def T(self):
        return self.transpose()

    def ctranspose(self):
        """Conjugate transpose of a TuckerLikeTensor of type Operator."""
        assert isinstance(self.space, tensap.TSpaceOperators), \
            "The TSpace must be of TSpaceOperators type."
        return TuckerLikeTensor(self.core, self.space.H)

    @property
    def H(self):
        return self.ctranspose()

    def to_operator(self):
        """Convert a vector TuckerLikeTensor to an operator TuckerLikeTensor.

        Each basis vector v is replaced by the diagonal operator
        diag(v).
        """
        assert isinstance(self.space, tensap.TSpaceVectors), \
            "The TSpace must be of TSpaceVectors type."
        new_spaces = [s.copy() for s in self.space.spaces]
        for mu in range(self.order):
            n_out, n_in, r = new_spaces[mu].shape
            ops = np.zeros((n_out, n_out, r))
            for k in range(r):
                ops[:, :, k] = np.diag(new_spaces[mu][:, 0, k])
            new_spaces[mu] = ops
        new_space = tensap.TSpaceOperators(new_spaces, is_orth=False)
        return TuckerLikeTensor(self.core, new_space)

    def vectorize(self):
        """
        Vectorize an operator TuckerLikeTensor into a vector TuckerLikeTensor.

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.vectorize()
        return TuckerLikeTensor(self.core, new_space)

    def unvectorize(self, sz):
        """
        Unvectorize a vector TuckerLikeTensor into an operator one.

        Parameters
        ----------
        sz : numpy.ndarray of shape (2, K)

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.unvectorize(sz)
        return TuckerLikeTensor(self.core, new_space)

    # ---- Evaluation ----

    def eval_diag(self, dims=None):
        """
        Extract the diagonal of the tensor.

        Parameters
        ----------
        dims : list of int, optional

        Returns
        -------
        numpy.ndarray or TuckerLikeTensor
        """
        if dims is None:
            dims = list(range(self.order))
        else:
            dims = np.atleast_1d(dims).tolist()

        if len(dims) == 1:
            return self

        if isinstance(self.core, tensap.DiagonalTensor) and \
                isinstance(self.space, tensap.TSpaceVectors):
            s = self.space.hadamard(dims)
            if len(dims) == self.order:
                return s[:, 0, :] @ self.core.data
            keep = [k for k in range(self.order) if k not in dims]
            new_spaces = [s] + [self.space.spaces[k] for k in keep]
            ns = self.space.__class__(new_spaces, is_orth=False)
            n_order = 1 + (self.order - len(dims))
            new_core = tensap.DiagonalTensor(
                self.core.data, order=n_order
            )
            return TuckerLikeTensor(new_core, ns)

        s = self.core.tensor_matrix_product(
            self.space.column_vectors()
        )
        return s.eval_diag(dims)

    def singular_values(self):
        """
        Return the singular values of the tensor.

        Returns
        -------
        numpy.ndarray or list
        """
        if self.order == 2:
            self.orth()
            return np.linalg.svd(self.core.data, compute_uv=False)
        else:
            if isinstance(self.core, tensap.FullTensor):
                self.orth()
                return self.core.singular_values()
            else:
                raise NotImplementedError(
                    f"singular_values not implemented for "
                    f"core type {type(self.core)}."
                )

    # ---- Normalization ----

    def normalize_basis(self):
        """
        Normalize the elements of the bases of subspaces.
        """
        N = self.space.dot(self.space)
        N = [np.sqrt(np.diag(n)) for n in N]
        diag_mats = [np.diag(n) for n in N]
        self.core = self.core.tensor_matrix_product(diag_mats)
        inv_mats = [np.diag(1.0 / n) for n in N]
        self.space = self.space.space_times_matrix(inv_mats)
        return self

    # ---- Static constructors ----

    @staticmethod
    def create(generator, ranks, shape):
        """
        Create a TuckerLikeTensor using a given generator.

        Parameters
        ----------
        generator : callable
            Function generating a numpy array from a shape tuple.
        ranks : array_like
            Tucker ranks (ranks of the core for each dimension).
        shape : array_like
            Physical shape of the tensor.

        Returns
        -------
        TuckerLikeTensor
        """
        ranks = np.atleast_1d(np.asarray(ranks, dtype=int))
        shape = np.atleast_1d(np.asarray(shape, dtype=int))
        space = tensap.TSpaceVectors.create(generator, shape, ranks)
        core = tensap.FullTensor(generator(tuple(ranks)))
        return TuckerLikeTensor(core, space)

    @staticmethod
    def zeros(ranks, shape):
        return TuckerLikeTensor.create(np.zeros, ranks, shape)

    @staticmethod
    def ones(ranks, shape):
        return TuckerLikeTensor.create(np.ones, ranks, shape)

    @staticmethod
    def rand(ranks, shape):
        return TuckerLikeTensor.create(
            lambda x: np.random.rand(*x), ranks, shape
        )

    @staticmethod
    def randn(ranks, shape):
        return TuckerLikeTensor.create(
            lambda x: np.random.randn(*x), ranks, shape
        )

    @staticmethod
    def eye(shape):
        """
        Construct the identity operator in TuckerLikeTensor format.

        Parameters
        ----------
        shape : array_like
            Size along each dimension.

        Returns
        -------
        TuckerLikeTensor
        """
        shape = np.atleast_1d(np.asarray(shape, dtype=int))
        d = len(shape)
        core = tensap.DiagonalTensor(np.array([1.0]), order=d)
        space = tensap.TSpaceOperators.eye(shape)
        return TuckerLikeTensor(core, space)

    # ---- Display ----

    def __repr__(self):
        if self.shape.ndim == 2:
            shape_str = "x".join(
                f"{out}x{inn}" for out, inn in self.shape
            )
        else:
            shape_str = "x".join(map(str, self.shape))
        return (
            "<{} TuckerLikeTensor:\n"
            + "\torder = {},\n"
            + "\tshape = {},\n"
            + "\tranks = {},\n"
            + "\tis_orth = {}>"
        ).format(
            shape_str,
            self.order,
            self.shape,
            self.ranks,
            self.is_orth,
        )


class BaseTuckerTensor:
    """
    Abstract Base Class for Tucker tensors.
    Contains universal algebraic operations independent of the physical nature
    (vector or operator) of the tensor.
    """

    def __init__(self, core, frames: 'TFrame'):
        self.core = core
        self.frames = frames

    @property
    def order(self) -> int:
        return self.frames.order

    @property
    def ranks(self) -> tuple:
        return self.frames.representation_ranks

    @property
    def shape(self) -> tuple:
        return self.frames.shape

    @property
    def is_orth(self) -> bool:
        return self.core.is_orth and self.frames.is_orth

    # --- storage and metrics ---

    def storage(self) -> int:
        """Return the total storage complexity"""
        return self.frames.storage() + self.core.storage()

    def sparse_storage(self) -> int:
        """Return the number of non-zero entries"""
        return self.frames.sparse_storage() + self.core.sparse_storage()

    def representation_rank(self) -> int:
        """Return the representation rank (product of subspace dimensions)"""
        return int(np.prod(self.ranks))

    # --- conversion to dense ---

    def full(self):
        """
        Convert the object to a dense tensap.FullTensor.
        """
        # 1. Extract matrices of shape (Physical_size, Rank)
        matrices = self.frames.get_vectorize_factors()

        # 2. Contract the core with these matrices
        result = self.core.tensor_matrix_product(matrices).full()

        # 3. Reshape the resulting 1D/flat array into the actual physical tensor geometry
        target_shape = []
        for frame in self.frames.frames:
            if frame.dim2 == 1:
                target_shape.append(frame.dim1)
            else:
                target_shape.extend([frame.dim1, frame.dim2])

        return result.reshape(target_shape)

    # --- routing and complex constructors ---

    @classmethod
    def _factory(cls, core, frames: "TFrame") -> "BaseTuckerTensor":
        """Internal router to instantiate the correct subclass based on frame type."""
        if isinstance(frames, TFrameVector):
            return TuckerTensorVector(core, frames)
        elif isinstance(frames, TFrameOperator):
            return TuckerTensorOperator(core, frames)
        raise TypeError(f"Unknown TFrame type: {type(frames)}")

    @classmethod
    def from_tree_based_tensor(cls, tree_tensor) -> "BaseTuckerTensor":
        """
        Convert a TreeBasedTensor into a Tucker format.
        """
        tree = tree_tensor.tree
        frames_data = []

        for mu in range(tree_tensor.order):
            nod = tree.dim2ind[mu]
            idx = nod - 1

            # get the matrix (N, R)
            if tree_tensor.is_active_node[idx]:
                mat = tree_tensor.tensors[idx].data
            else:
                mat = np.eye(tree_tensor.shape[mu])

            # For FrameVector: (N, R) -> (R, N, 1)
            N, R = mat.shape
            data = mat.T.reshape(R, N, 1)
            frames_data.append(FrameVector(data))

        core = tree_tensor.tensors[0]
        frames = TFrameVector(frames_data)

        return cls._factory(core, frames)

    # ---- Fundamental Algebra ----

    def dot(self, other: 'BaseTuckerTensor') -> float:
        """
        Compute the inner product <self, other>.
        Pushes local inner products into the core of 'other' before contraction.
        """
        if self.order != other.order:
            raise ValueError("Tensors must have the same order.")

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute dot product between {type(self).__name__} "
                f"and {type(other).__name__}."
            )

        core_self, core_other = tensap.convert_tensors(self.core, other.core)

        gram_matrices = self.frames.dot(other.frames)
        projected_other_core = core_other.tensor_matrix_product(gram_matrices)
        return core_self.dot(projected_other_core)

    def norm(self) -> float:
        """Compute the Frobenius norm."""
        if self.is_orth:
            return float(self.core.norm())
        else:
            return float(np.sqrt(np.abs(self.dot(self))))

    def orth(self) -> 'BaseTuckerTensor':
        """
        Orthonormalize the tensor spaces and absorb the change into the core.
        Returns a new orthonormalized tensor.
        """
        new_frames, r_matrices = self.frames.orth()

        # Absorb the R factors into the core
        new_core = self.core.tensor_matrix_product(r_matrices)

        if type(new_core).__name__ == "FullTensor":
            raise NotImplementedError(
                "Global orthogonalization of a Tucker tensor with a FullTensor core "
                "is not mathematically well-defined. The core must support global "
                "orthogonalization (e.g., TreeBasedTensor)."
            )

        core_orth = new_core.orth()
        return self.__class__(core_orth, new_frames)

    def tensor_matrix_product(
        self, matrices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "BaseTuckerTensor":
        """Contract the tensor with matrices along physical dimensions."""
        new_frames = self.frames.matrix_times_space(matrices, dims)
        return self.__class__(self.core, new_frames)

    # ---- Magic Methods ----

    def __add__(self, other: "BaseTuckerTensor") -> "BaseTuckerTensor":
        if type(self) is not type(other):
            return NotImplemented

        core_self, core_other = tensap.convert_tensors(self.core, other.core)

        new_core = core_self.cat(core_other)
        new_frames = self.frames.cat(other.frames)

        return self.__class__(new_core, new_frames)

    def __neg__(self) -> 'BaseTuckerTensor':
        return self.__class__(-self.core, self.frames)

    def __sub__(self, other: 'BaseTuckerTensor') -> 'BaseTuckerTensor':
        return self + (-other)

    def __mul__(self, scalar: float) -> 'BaseTuckerTensor':
        if not np.isscalar(scalar):
            return NotImplemented
        return self.__class__(self.core * scalar, self.frames)

    def __rmul__(self, scalar: float) -> 'BaseTuckerTensor':
        return self * scalar

    # ---- The Factory (Static Constructors) ----

    @classmethod
    def _factory(cls, core, frames: 'TFrame') -> 'BaseTuckerTensor':
        """Internal router to instantiate the correct subclass based on frame type."""
        # Note: TFrameVectors and TFrameOperators must be imported
        if isinstance(frames, TFrameVector):
            return TuckerTensorVector(core, frames)
        elif isinstance(frames, TFrameOperator):
            return TuckerTensorOperator(core, frames)
        raise TypeError("Unknown TFrames type.")

    @staticmethod
    def create(generator: callable, ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """
        Create a Tucker tensor.
        Automatically returns a TuckerTensorVector or TuckerTensorOperator depending on
        'shape'.
        """

        ranks = np.atleast_1d(np.asarray(ranks, dtype=int))

        # Dynamic Dispatch: check if shape elements are integers (Vector) or tuples
        # (Operator)
        if isinstance(shape[0], (int, np.integer)):
            # It's a State/Vector
            frames = TFrameVector.create(generator, shape, ranks)
        elif isinstance(shape[0], (tuple, list)):
            # It's an Operator
            frames = TFrameOperator.create(generator, shape, ranks)
        else:
            raise ValueError("Unrecognized shape format.")

        # For simplicity, assuming a FullTensor core generator exists
        # core = FullTensor(generator(tuple(ranks)))
        core = "MOCK_CORE"

        return BaseTuckerTensor._factory(core, frames)

    @staticmethod
    def rand(ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        return BaseTuckerTensor.create(lambda x: np.random.rand(*x), ranks, shape)


class TuckerTensorVector(BaseTuckerTensor):
    """
    Tucker tensor representing a state or function in a vector space.
    """

    frames: TFrameVector

    def __init__(self, core, frames: 'TFrameVector'):
        super().__init__(core, frames)
        if not isinstance(frames, TFrameVector):
            raise TypeError("TuckerTensorVector requires a TFrameVectors object.")

    def sub_tensor(self, indices: List[np.ndarray], dims: Optional[List[int]] = None) -> 'TuckerTensorVector':
        """Extract a sub-tensor by evaluating on a grid of indices."""
        new_frames = self.frames.eval_at_indices(indices, dims)
        return TuckerTensorVector(self.core, new_frames)

    def tensor_vector_product(self, vectors: List[np.ndarray], dims: Optional[List[int]] = None) -> 'BaseTuckerTensor':
        """Contract the physical dimension with external vectors."""
        # Note: the exact math logic depends on how your cores handle vector contraction
        # Typically, this reduces the order of the tensor if all dims are contracted.
        new_frames = self.frames.matrix_times_space(vectors, dims) # Reduces dim1 to 1

        # Usually requires core.tensor_vector_product logic here
        # ...
        return TuckerTensorVector(self.core, new_frames)

    def to_operator(self) -> 'TuckerTensorOperator':
        """
        Isomorphism: Convert the state vector into a diagonal operator.
        Each basis vector v becomes a diagonal matrix diag(v).
        """
        new_frames = self.frames.to_diagonal_operators()
        return TuckerTensorOperator(self.core, new_frames)


class TuckerTensorOperator(BaseTuckerTensor):
    """
    Tucker tensor representing a linear operator (matrices mapping N_in to N_out).
    """

    frames: TFrameOperator

    def __init__(self, core, frames: "TFrameOperator"):
        super().__init__(core, frames)
        if not isinstance(frames, TFrameOperator):
            raise TypeError("TuckerTensorOperator requires a TFrameOperators object.")

    def __matmul__(self, other: BaseTuckerTensor) -> BaseTuckerTensor:
        """
        Compose this operator with another tensor using the '@' operator.
        Equivalent to mtimes.
        """
        if not isinstance(other, BaseTuckerTensor):
            return NotImplemented

        # 1. Kronecker product of the cores
        new_core = self.core.kron(other.core)

        # 2. Composition of the physical frames (which handles the rank multiplication)
        new_frames = self.frames.mtimes(other.frames)

        # 3. Dynamic return type based on the right operand
        if isinstance(other, TuckerTensorVector):
            return TuckerTensorVector(new_core, new_frames)
        return TuckerTensorOperator(new_core, new_frames)

    def transpose(self) -> "TuckerTensorOperator":
        """Transpose the physical dimensions."""
        return TuckerTensorOperator(self.core, self.frames.transpose(conjugate=False))

    @property
    def T(self) -> "TuckerTensorOperator":
        """Syntactic sugar for transpose."""
        return self.transpose()

    def ctranspose(self) -> "TuckerTensorOperator":
        """Conjugate transpose of the operator."""
        # Assuming the core has a conj() method
        new_core = self.core.conj() if hasattr(self.core, "conj") else self.core
        return TuckerTensorOperator(new_core, self.frames.transpose(conjugate=True))

    @property
    def H(self) -> "TuckerTensorOperator":
        """Syntactic sugar for conjugate transpose."""
        return self.ctranspose()

    def vectorize(self) -> TuckerTensorVector:
        """Flatten the operator into a column vector space."""
        new_frames = self.frames.vectorize()
        return TuckerTensorVector(self.core, new_frames)
