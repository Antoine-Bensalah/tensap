# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tucker_like_tensor.
"""

import numpy as np
import tensap
from typing import Union, List, Tuple, Optional
from .tensor_frame import TFrame, TFrameVector, TFrameOperator
from .frame import FrameVector


class BaseTuckerTensor:
    """
    Abstract Base Class for Tucker tensors.

    Contains universal algebraic operations independent of the physical nature
    (vector or operator) of the tensor. A Tucker tensor is defined by:
    - a core tensor (FullTensor, DiagonalTensor, or TreeBasedTensor)
    - a tensor product of frames (TFrame), one per dimension
    """

    def __init__(self, core, frames: 'TFrame'):
        self.core = core
        self.tensor_frame = frames

    @property
    def order(self) -> int:
        """Return the number of tensor dimensions.

        Returns
        -------
        int
        """
        return self.tensor_frame.order

    @property
    def shape(self) -> tuple:
        """Return the physical shape for each dimension.

        Returns
        -------
        tuple of tuple of int
            Each entry is (dim1, dim2) for that dimension.
        """
        return self.tensor_frame.shape

    @property
    def is_orth(self) -> bool:
        """Return whether the tensor is fully orthonormalized.

        Returns
        -------
        bool
        """
        return self.core.is_orth and self.tensor_frame.is_orth

    # --- storage and metrics ---

    def storage(self) -> int:
        """Return the total storage complexity.

        Sum of storage of the frames and the core.

        Returns
        -------
        int
        """
        return self.tensor_frame.storage() + self.core.storage()

    def sparse_storage(self) -> int:
        """Return the number of non-zero entries.

        Returns
        -------
        int
        """
        return self.tensor_frame.sparse_storage() + self.core.sparse_storage()

    def representation_rank(self) -> Tuple[int, ...]:
        """Return the representation ranks.

        Returns
        -------
        tuple(int)
        """
        return self.tensor_frame.representation_ranks

    # --- conversion to dense ---

    def full(self):
        """Convert the object to a dense tensap.FullTensor.

        The core is contracted with the vectorized frames: the result has one
        dimension of size N1 * N2 per tensor dimension. TuckerTensorOperator
        reshapes it into the matrix geometry.

        Returns
        -------
        FullTensor
        """
        matrices = self.tensor_frame.get_vectorize_factors()
        return self.core.tensor_matrix_product(matrices).full()

    def numpy(self) -> np.ndarray:
        """Convert to a dense numpy.ndarray.

        Returns
        -------
        ndarray
        """
        return self.full().numpy()

    # --- routing and complex constructors ---

    @classmethod
    def _factory(cls, core, frames: "TFrame") -> "BaseTuckerTensor":
        """Internal router to instantiate the correct subclass based on frame type.

        Parameters
        ----------
        core : FullTensor or DiagonalTensor or TreeBasedTensor
        frames : TFrame, TFrameVector, or TFrameOperator

        Returns
        -------
        BaseTuckerTensor
            A TuckerTensorVector or TuckerTensorOperator instance.

        Raises
        ------
        TypeError
            If the frame type is unknown.
        """
        if isinstance(frames, TFrameVector):
            return TuckerTensorVector(core, frames)
        elif isinstance(frames, TFrameOperator):
            return TuckerTensorOperator(core, frames)
        raise TypeError(f"Unknown TFrame type: {type(frames)}")

    @classmethod
    def from_tree_based_tensor(cls, tree_tensor) -> "BaseTuckerTensor":
        """Convert a TreeBasedTensor into a Tucker format.

        The leaf of each dimension becomes its frame (the identity if the leaf
        is inactive). With a trivial tree the core is the root tensor
        (FullTensor); otherwise it is the TreeBasedTensor with inactive leaves.

        Parameters
        ----------
        tree_tensor : TreeBasedTensor

        Returns
        -------
        TuckerTensorVector
        """
        tree = tree_tensor.tree
        tensors = np.array(tree_tensor.tensors)
        frames_data = []

        for mu in range(tree_tensor.order):
            idx = tree.dim2ind[mu] - 1

            # get the matrix (N, R)
            if tree_tensor.is_active_node[idx]:
                mat = tensors[idx].data
                tensors[idx] = tensap.FullTensor([])
            else:
                mat = np.eye(tree_tensor.shape[mu])

            # For FrameVector: (N, R) -> (R, N, 1)
            N, R = mat.shape
            data = mat.T.reshape(R, N, 1)
            frames_data.append(FrameVector(data))

        if all(tree.parent(nod) == tree.root for nod in tree.dim2ind):
            core = tensors[tree.root - 1]
        else:
            core = tensap.TreeBasedTensor(tensors, tree)
        frames = TFrameVector(frames_data)

        return cls._factory(core, frames)

    # ---- Fundamental Algebra ----

    def dot(self, other: 'BaseTuckerTensor') -> float:
        """Compute the inner product <self, other>.

        Local Gram matrices are pushed into the core of 'other' before
        contraction.

        Parameters
        ----------
        other : BaseTuckerTensor
            Must have the same order and type.

        Returns
        -------
        float
        """
        if self.order != other.order:
            raise ValueError("Tensors must have the same order.")

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute dot product between {type(self).__name__} "
                f"and {type(other).__name__}."
            )

        core_self, core_other = tensap.convert_tensors(self.core, other.core)

        gram_matrices = self.tensor_frame.dot(other.tensor_frame)

        # Canonical format: <x, y> = sigma^T (G^(1) o ... o G^(d)) tau, without
        # forming any full core
        if isinstance(core_self, tensap.DiagonalTensor):
            gram = np.prod(np.stack(gram_matrices), axis=0)
            return float(core_self.data @ gram @ core_other.data)

        projected_other_core = core_other.tensor_matrix_product(gram_matrices)
        # the projection may change the type of the core (e.g. to a FullTensor)
        core_self, projected_other_core = tensap.convert_tensors(
            core_self, projected_other_core
        )
        return float(core_self.dot(projected_other_core))

    def norm(self) -> float:
        """Compute the Frobenius norm.

        Uses the core norm when orthonormalized for efficiency.

        Returns
        -------
        float
        """
        if self.is_orth:
            return float(self.core.norm())
        else:
            return float(np.sqrt(np.abs(self.dot(self))))

    def orth(self) -> 'BaseTuckerTensor':
        """Orthonormalize the tensor spaces and absorb the change into the core.

        Returns
        -------
        BaseTuckerTensor
            A new orthonormalized tensor.
        """
        new_frames, r_matrices = self.tensor_frame.orth()

        # Absorb the R factors into the core
        new_core = self.core.tensor_matrix_product(r_matrices)

        core_orth = new_core.orth()
        if isinstance(core_orth, tuple):
            core_orth = core_orth[0]
        return self.__class__(core_orth, new_frames)

    # --- contractions ---

    def tensor_matrix_product(
        self, matrices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "BaseTuckerTensor":
        """Contract the tensor with matrices along physical dimensions.

        Parameters
        ----------
        matrices : list of ndarray
            One matrix per specified dimension.
        dims : list of int, optional
            Dimensions to contract. If None, all dimensions.

        Returns
        -------
        BaseTuckerTensor
        """
        new_frames = self.tensor_frame.matrix_times_space(matrices, dims)
        return self.__class__(self.core, new_frames)

    def tensor_diag_matrix_product(
        self, M: Union[np.ndarray, List[np.ndarray]], dims: Optional[List[int]] = None
    ) -> "BaseTuckerTensor":
        """Contract the tensor with diagonal matrices.

        Parameters
        ----------
        M : ndarray or list of ndarray
            Diagonal entries (one per dimension).
        dims : list of int, optional
            Dimensions to contract. If None, all dimensions.

        Returns
        -------
        BaseTuckerTensor
        """
        if isinstance(M, np.ndarray):
            M = [M]
        if dims is None:
            dims = list(range(self.order))

        diag_mats = [np.diag(m.ravel()) for m in M]
        return self.tensor_matrix_product(diag_mats, dims)

    def squeeze(self, dims: Optional[List[int]] = None):
        """Remove singleton dimensions.

        Singleton dimensions (dim1 == dim2 == 1) are contracted into the
        core and removed from the tensor space.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to squeeze. If None, all singleton dimensions.

        Returns
        -------
        BaseTuckerTensor or ndarray
            If the result has order 0, returns a scalar ndarray.
        """
        if dims is None:
            dims = [i for i, (d1, d2) in enumerate(self.shape) if d1 == 1 and d2 == 1]

        if len(dims) == 0:
            return self

        squeeze_matrices = self.tensor_frame.get_vectorize_factors(dims)
        new_core = self.core.tensor_vector_product(squeeze_matrices, dims)
        new_frames = self.tensor_frame.remove_space(dims)

        if new_frames.order == 0:
            return new_core.numpy()

        return self.__class__(new_core, new_frames)

    # ---- Tensor Algebra ----

    def cat(self, other: 'BaseTuckerTensor') -> 'BaseTuckerTensor':
        """Concatenate two Tucker tensors using block-diagonal concatenation.

        Parameters
        ----------
        other : BaseTuckerTensor
            Must have the same order and type.

        Returns
        -------
        BaseTuckerTensor
        """
        new_core = self.core.cat(other.core)
        new_frames = self.tensor_frame.diag_cat(other.tensor_frame)
        return self.__class__(new_core, new_frames)

    def kron(self, other: 'BaseTuckerTensor') -> 'BaseTuckerTensor':
        """Kronecker product of two Tucker tensors.

        Parameters
        ----------
        other : BaseTuckerTensor

        Returns
        -------
        BaseTuckerTensor
        """
        core_self, core_other = tensap.convert_tensors(self.core, other.core)
        new_core = core_self.kron(core_other)
        new_frames = self.tensor_frame.kron(other.tensor_frame)
        return self.__class__(new_core, new_frames)

    def permute(self, dims: List[int]) -> 'BaseTuckerTensor':
        """Permute the dimensions of the tensor.

        Parameters
        ----------
        dims : list of int
            A permutation of the integers from 0 to order - 1.

        Returns
        -------
        BaseTuckerTensor
        """
        new_core = self.core.transpose(dims)
        new_frames = self.tensor_frame.permute(dims)
        return self.__class__(new_core, new_frames)

    def singular_values(self):
        """Return the singular values of the tensor.

        The frames are orthonormalized first, so that the singular values of
        the tensor are those of its core. An operator is treated as its
        vectorization.

        Returns
        -------
        ndarray or list of ndarray
            Order 2: the singular values of the matrix. FullTensor core: the
            singular values of the matricizations along each dimension.
            TreeBasedTensor core: the tree-based singular values, one array
            per node of the tree (None for the root), the leaves giving the
            singular values along each dimension. TTTensor core: the
            singular values of the matricizations along each dimension, one
            array per dimension (Matlab returns them in sv{2}, with the TT
            singular values of the core in sv{1}).

        Raises
        ------
        NotImplementedError
            For another type of core (as in MATLAB).
        """
        if self.order == 2:
            orth_self = self.orth()
            return np.linalg.svd(orth_self.core.full().numpy(), compute_uv=False)
        if isinstance(self.core, tensap.TreeBasedTensor):
            vector = self.vectorize() if isinstance(self, TuckerTensorOperator) else self
            return vector.tree_based_tensor().singular_values()
        if isinstance(self.core, tensap.FullTensor):
            return self.orth().core.singular_values()
        if isinstance(self.core, tensap.TTTensor):
            # orthonormal frames, then the center of the core on mu: the
            # mu-singular values are those of the middle matricization of
            # the core mu
            core = self.orth().core
            values = []
            for mu in range(self.order):
                center = core.orth(mu).cores[mu]
                matrix = np.moveaxis(center, 1, 0).reshape(center.shape[1], -1)
                values.append(np.linalg.svd(matrix, compute_uv=False))
            return values
        raise NotImplementedError(
            f"singular_values is not implemented for a {type(self.core).__name__} core."
        )

    def normalize_basis(self) -> 'BaseTuckerTensor':
        """Normalize the elements of the bases of subspaces.

        Each frame element is divided by its Frobenius norm and the core is
        multiplied by the norms, so that the tensor is unchanged. A zero frame
        element is left as it is.

        Returns
        -------
        BaseTuckerTensor
            A new tensor, equal to self, whose frame elements have unit norm.
        """
        norms = [
            np.sqrt(np.diag(gram)) for gram in self.tensor_frame.dot(self.tensor_frame)
        ]
        # a zero frame element has no direction: scale it (and its core
        # coefficients) by 1 instead of dividing by 0
        norms = [np.where(n > 0, n, 1.0) for n in norms]
        new_core = self.core.tensor_diagonal_matrix_product(norms)
        new_frames = self.tensor_frame.space_times_matrix(
            [np.diag(1.0 / n) for n in norms]
        )
        return self.__class__(new_core, new_frames)

    def hadamard_product(self, other: 'BaseTuckerTensor') -> 'BaseTuckerTensor':
        """Entrywise (Hadamard) product of two Tucker tensors.

        The core is the Kronecker product of the cores and, in each dimension,
        the frame contains the entrywise products of every pair of frame
        elements. The ranks are multiplied.

        Parameters
        ----------
        other : BaseTuckerTensor
            Tensor of the same type (vector or operator) and shape.

        Returns
        -------
        BaseTuckerTensor
        """
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute the Hadamard product of {type(self).__name__} "
                f"and {type(other).__name__}."
            )
        core_self, core_other = tensap.convert_tensors(self.core, other.core)
        new_core = core_self.kron(core_other)
        new_frames = self.tensor_frame.hadamard_product(other.tensor_frame)
        return self.__class__(new_core, new_frames)

    def convert_cores(self, other: 'BaseTuckerTensor'):
        """Convert the cores of two Tucker tensors into a common format.

        Parameters
        ----------
        other : BaseTuckerTensor

        Returns
        -------
        tuple of BaseTuckerTensor
            self and other, with cores of the same type (see convert_tensors);
            the frames are unchanged.
        """
        return tensap.convert_tensors(self, other)

    # ---- Magic Methods ----

    def __add__(self, other: "BaseTuckerTensor") -> "BaseTuckerTensor":
        """Addition of two Tucker tensors."""
        if type(self) is not type(other):
            return NotImplemented

        core_self, core_other = tensap.convert_tensors(self.core, other.core)

        new_core = core_self.cat(core_other)
        new_frames = self.tensor_frame.cat(other.tensor_frame)

        return self.__class__(new_core, new_frames)

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: 'BaseTuckerTensor') -> 'BaseTuckerTensor':
        if type(self) is not type(other):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> 'BaseTuckerTensor':
        return self.__class__(-self.core, self.tensor_frame)

    def _scaled_core(self, scalar: float):
        """Return the core multiplied by a scalar.

        DiagonalTensor.__mul__ is the Hadamard product with another tensor, so
        a diagonal core is scaled through its data.
        """
        if isinstance(self.core, tensap.DiagonalTensor):
            return tensap.DiagonalTensor(self.core.data * scalar, self.core.order)
        return self.core * scalar

    def __mul__(self, scalar: float) -> 'BaseTuckerTensor':
        if not np.isscalar(scalar):
            return NotImplemented
        return self.__class__(self._scaled_core(scalar), self.tensor_frame)

    def __rmul__(self, scalar: float) -> 'BaseTuckerTensor':
        return self * scalar

    def __truediv__(self, scalar: float) -> "BaseTuckerTensor":
        if not np.isscalar(scalar):
            return NotImplemented
        return self.__class__(self._scaled_core(1.0 / scalar), self.tensor_frame)

    def __repr__(self):
        class_name = type(self).__name__
        shape_str = "x".join(
            f"{d1}x{d2}" if d2 > 1 else str(d1) for d1, d2 in self.shape
        )
        return (
            f"<{shape_str} {class_name}:\n"
            f"\torder = {self.order},\n"
            f"\tshape = {self.shape},\n"
            f"\tranks = {self.representation_rank()},\n"
            f"\tis_orth = {self.is_orth}>"
        )

    @staticmethod
    def create(generator: callable, ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """Create a Tucker tensor using a generator function.

        Automatically returns a TuckerTensorVector or TuckerTensorOperator
        depending on the shape entries (integer for vector, tuple for operator).
        The core is full, filled by the generator, except for ranks all equal
        to 1: then it is a DiagonalTensor of order d, as in MATLAB.

        Parameters
        ----------
        generator : Callable
            Function that takes a shape tuple and returns an ndarray.
        ranks : list of int
            Representation ranks for each dimension.
        shape : list of int or list of tuple
            Physical shape. Use int entries for vectors, (dim1, dim2) tuples
            for operators.

        Returns
        -------
        BaseTuckerTensor
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

        if np.all(ranks == 1):
            # rank one: diagonal core, as TuckerLikeTensor.create in MATLAB (a
            # full core would be made full again by convert_tensors when
            # combined with canonical tensors, with R^d entries)
            core = tensap.DiagonalTensor(generator((1,)).ravel(), len(ranks))
        else:
            core = tensap.FullTensor(generator(tuple(ranks)))

        return BaseTuckerTensor._factory(core, frames)

    @staticmethod
    def rand(ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """Create a random Tucker tensor with uniform entries.

        Parameters
        ----------
        ranks : list of int
        shape : list of int or list of tuple

        Returns
        -------
        BaseTuckerTensor
        """
        return BaseTuckerTensor.create(lambda x: np.random.rand(*x), ranks, shape)

    @staticmethod
    def randn(ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """Create a random Tucker tensor with standard normal entries.

        Parameters
        ----------
        ranks : list of int
        shape : list of int or list of tuple

        Returns
        -------
        BaseTuckerTensor
        """
        return BaseTuckerTensor.create(lambda x: np.random.randn(*x), ranks, shape)

    @staticmethod
    def zeros(ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """Create a Tucker tensor filled with zeros.

        Parameters
        ----------
        ranks : list of int
        shape : list of int or list of tuple

        Returns
        -------
        BaseTuckerTensor
        """
        return BaseTuckerTensor.create(np.zeros, ranks, shape)

    @staticmethod
    def ones(ranks: List[int], shape: List) -> 'BaseTuckerTensor':
        """Create a Tucker tensor filled with ones.

        Parameters
        ----------
        ranks : list of int
        shape : list of int or list of tuple

        Returns
        -------
        BaseTuckerTensor
        """
        return BaseTuckerTensor.create(np.ones, ranks, shape)

    @staticmethod
    def eye(shape: List[int]) -> 'BaseTuckerTensor':
        """Construct the identity operator in Tucker format.

        Parameters
        ----------
        shape : list of int
            Physical dimensions for each tensor dimension.

        Returns
        -------
        BaseTuckerTensor
            A TuckerTensorOperator representing the identity.
        """
        shape = np.atleast_1d(np.asarray(shape, dtype=int))
        d = len(shape)
        core = tensap.DiagonalTensor(np.array([1.0]), order=d)
        frames = TFrameOperator.eye(shape, [1] * d)
        return BaseTuckerTensor._factory(core, frames)


class TuckerTensorVector(BaseTuckerTensor):
    """
    Tucker tensor representing a state or function in a vector space.
    """

    tensor_frame: TFrameVector

    def __init__(self, core, frames: 'TFrameVector'):
        super().__init__(core, frames)
        if not isinstance(frames, TFrameVector):
            raise TypeError("TuckerTensorVector requires a TFrameVectors object.")

    def sub_tensor(
        self, indices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> 'TuckerTensorVector':
        """Extract a sub-tensor by evaluating on a grid of indices.

        Parameters
        ----------
        indices : list of ndarray
            Index arrays, one per specified dimension.
        dims : list of int, optional
            Dimensions to evaluate. If None, all dimensions.

        Returns
        -------
        TuckerTensorVector
        """
        new_frames = self.tensor_frame.eval_at_indices(indices, dims)
        return self.__class__(self.core, new_frames)

    def super_tensor(
        self, shape: List[int], indices: List[Optional[np.ndarray]]
    ) -> 'TuckerTensorVector':
        """Embed the tensor in a larger tensor filled with zeros.

        Returns s of shape (M1, ..., Md) with s[I1[k1], ..., Id[kd]] =
        self[k1, ..., kd] and s = 0 outside I1 x ... x Id. Reciprocal of
        sub_tensor. The frames are stored densely.

        Parameters
        ----------
        shape : list of int
            New size of each dimension.
        indices : list of ndarray or None
            For each dimension, the indices I_mu of the current entries; None
            keeps the dimension unchanged.

        Returns
        -------
        TuckerTensorVector
        """
        new_frames = self.tensor_frame.super_frame(
            [(m, 1) for m in shape], [(idx, None) for idx in indices]
        )
        return self.__class__(self.core, new_frames)

    def tensor_vector_product(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[List[int]] = None,
    ):
        """Contract the tensor with external vectors along given dimensions.

        When all dimensions are contracted, returns a scalar ndarray;
        otherwise returns a lower-order TuckerTensorVector.

        Parameters
        ----------
        vectors : ndarray or list of ndarray
            Vectors to contract with, one per specified dimension.
        dims : list of int, optional
            Dimensions to contract. If None, all dimensions.

        Returns
        -------
        TuckerTensorVector or ndarray
        """
        if dims is None:
            dims = list(range(self.order))
        else:
            dims = np.atleast_1d(dims).tolist()

        if isinstance(vectors, np.ndarray):
            vectors = [vectors]

        # 1. transform the vectors in a row-vector (1, N)
        row_vectors = [np.atleast_2d(v.ravel()) for v in vectors]

        # 2. Contraction
        # (1, N) @ (Rank, N, 1) -> (Rank, 1, 1)
        new_frames = self.tensor_frame.matrix_times_space(row_vectors, dims)

        # 3. extract the contracted factors for the core
        core_factors = [
            # (Rank, 1, 1) -> (Rank, 1)
            new_frames.frames[d].data.reshape(-1, 1) for d in dims
        ]

        # 4. contraction of with the core
        new_core = self.core.tensor_vector_product(core_factors, dims)

        # 5. Resolution of the output geometry
        if len(dims) != self.order:
            # if the remains some dimensions, one keep them
            final_frames = new_frames.remove_space(dims)
            return self.__class__(new_core, final_frames)
        else:
            # all dimensions are contracted
            return new_core

    def to_operator(self) -> 'TuckerTensorOperator':
        """Isomorphism converting the state vector into a diagonal operator.

        Each basis vector v becomes a diagonal matrix diag(v).

        Returns
        -------
        TuckerTensorOperator
        """
        new_frames = self.tensor_frame.to_diagonal_operators()
        return TuckerTensorOperator(self.core, new_frames)

    def dot_with_rank_one_metric(
        self, tensor_2: "TuckerTensorVector", matrices: List[np.ndarray]
    ) -> float:
        """Weighted inner product using matrices: <A, M @ B>.

        Parameters
        ----------
        tensor_2 : TuckerTensorVector
        matrices : list of ndarray
            Matrices defining the metric, one per dimension.

        Returns
        -------
        float
        """
        tmp = tensor_2.tensor_matrix_product(matrices)
        return self.dot(tmp)

    def unvectorize(self, dims1: List[int], dims2: List[int]) -> 'TuckerTensorOperator':
        """Reshape the vector Tucker tensor into an operator Tucker tensor.

        Parameters
        ----------
        dims1 : list of int
            Output dimensions for each space.
        dims2 : list of int
            Input dimensions for each space.

        Returns
        -------
        TuckerTensorOperator
        """
        new_frames = self.tensor_frame.unvectorize(dims1, dims2)
        return TuckerTensorOperator(self.core, new_frames)

    def tree_based_tensor(self):
        """Convert into a TreeBasedTensor.

        With a TreeBasedTensor core, the frames are absorbed into the leaves
        of its tree (an inactive leaf receives the frame itself); otherwise
        the tree is trivial, with the core at the root and the frames as
        leaves.

        Returns
        -------
        TreeBasedTensor
        """
        factors = self.tensor_frame.get_vectorize_factors()
        if isinstance(self.core, tensap.TreeBasedTensor):
            tree = self.core.tree
            tensors = np.array(self.core.tensors)
            for mu, factor in enumerate(factors):
                idx = tree.dim2ind[mu] - 1
                if self.core.is_active_node[idx]:
                    factor = factor @ tensors[idx].data
                tensors[idx] = tensap.FullTensor(factor)
            return tensap.TreeBasedTensor(tensors, tree)
        tree = tensap.DimensionTree.trivial(self.order)
        tensors = [self.core] + [tensap.FullTensor(m) for m in factors]
        return tensap.TreeBasedTensor(tensors, tree)

    def eval_diag(self, dims: Optional[List[int]] = None):
        """Extract the diagonal of the tensor.

        For a CP (diagonal core) decomposition this is efficient.
        For a full core, contraction is performed first.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions over which to evaluate the diagonal.
            If None, all dimensions.

        Returns
        -------
        TuckerTensorVector or ndarray
        """
        if dims is None:
            dims = list(range(self.order))

        if len(dims) <= 1:
            return self

        # 1. Case of a Diagonal Core (CP decomposition)
        # use the name of the class in string form to avoid circular import
        if type(self.core).__name__ == "DiagonalTensor":

            s_frame = self.tensor_frame.elementwise_hadamard(dims)

            if len(dims) == self.order:
                # total evaluation
                # core.data: (R, )
                # s_frame.data: (R, N, 1)
                # res: (N, 1)
                res = np.tensordot(self.core.data, s_frame.data, axes=(0, 0))
                return res.squeeze()  # (N, )

            # partial evaluation
            keep = [k for k in range(self.order) if k not in dims]
            new_frames_list = [s_frame] + [self.tensor_frame.frames[k] for k in keep]
            new_frames = self.tensor_frame.__class__(new_frames_list)

            # the new core is still diagonal, but its order decreases
            n_order = 1 + len(keep)
            new_core = self.core.__class__(self.core.data, order=n_order)

            return self.__class__(new_core, new_frames)

        # 2. General case: FullTensor or TreeBasedTensor
        # one gets the factors in the form (N, R)
        factors = self.tensor_frame.get_vectorize_factors()

        # contraction
        s = self.core.tensor_matrix_product(factors)
        return s.eval_diag(dims)


class TuckerTensorOperator(BaseTuckerTensor):
    """
    Tucker tensor representing a linear operator (matrices mapping N_in to N_out).
    """

    tensor_frame: TFrameOperator

    def __init__(self, core, frames: "TFrameOperator"):
        super().__init__(core, frames)
        if not isinstance(frames, TFrameOperator):
            raise TypeError("TuckerTensorOperator requires a TFrameOperators object.")

    def full(self):
        """Convert the operator to a dense tensap.FullTensor.

        The result has shape (N1_1, N2_1, ..., N1_d, N2_d): the entry
        (i_1, j_1, ..., i_d, j_d) is the coefficient of the operator between
        the input index (j_1, ..., j_d) and the output index (i_1, ..., i_d).
        Dimensions of size 1 are kept.

        Returns
        -------
        FullTensor
        """
        result = super().full().numpy()
        target_shape = [n for frame_shape in self.shape for n in frame_shape]
        # C order, as in get_vectorize_factors (FullTensor.reshape uses the
        # Fortran order, which would transpose every operator factor)
        return tensap.FullTensor(result.reshape(target_shape))

    def super_tensor(self, shape: List[Tuple[int, int]], indices: List[tuple]
                     ) -> 'TuckerTensorOperator':
        """Embed the operator in a larger operator filled with zeros.

        In each dimension mu, every frame matrix A becomes a matrix S of shape
        shape[mu] with S[I_mu, J_mu] = A and S = 0 elsewhere. The frames are
        stored densely.

        Parameters
        ----------
        shape : list of tuple of int
            New shape (M1, M2) of each dimension.
        indices : list of tuple
            For each dimension, the pair (I_mu, J_mu) of row and column indices
            of the current entries; None keeps an axis unchanged.

        Returns
        -------
        TuckerTensorOperator
        """
        new_frames = self.tensor_frame.super_frame(shape, indices)
        return self.__class__(self.core, new_frames)

    def __matmul__(self, other: BaseTuckerTensor) -> BaseTuckerTensor:
        """
        Compose this operator with another tensor using the '@' operator.
        Equivalent to mtimes.
        """
        if not isinstance(other, BaseTuckerTensor):
            return NotImplemented

        # 1. Kronecker product of the cores, aligned to a common type first
        # (DiagonalTensor.kron assumes a diagonal argument)
        core_self, core_other = tensap.convert_tensors(self.core, other.core)
        new_core = core_self.kron(core_other)

        # 2. Composition of the physical frames (which handles the rank multiplication)
        new_frames = self.tensor_frame.mtimes(other.tensor_frame)

        # 3. Dynamic return type based on the right operand
        if isinstance(other, TuckerTensorVector):
            return TuckerTensorVector(new_core, new_frames)
        return TuckerTensorOperator(new_core, new_frames)

    def transpose(self) -> "TuckerTensorOperator":
        """Transpose the physical dimensions.

        Returns
        -------
        TuckerTensorOperator
        """
        return TuckerTensorOperator(self.core, self.tensor_frame.transpose(conjugate=False))

    @property
    def T(self) -> "TuckerTensorOperator":
        """Syntactic sugar for transpose."""
        return self.transpose()

    def ctranspose(self) -> "TuckerTensorOperator":
        """Conjugate transpose of the operator.

        Returns
        -------
        TuckerTensorOperator
        """
        # Assuming the core has a conj() method
        new_core = self.core.conj() if hasattr(self.core, "conj") else self.core
        return TuckerTensorOperator(new_core, self.tensor_frame.transpose(conjugate=True))

    @property
    def H(self) -> "TuckerTensorOperator":
        """Syntactic sugar for conjugate transpose."""
        return self.ctranspose()

    def vectorize(self) -> TuckerTensorVector:
        """Flatten the operator into a column vector space.

        Returns
        -------
        TuckerTensorVector
        """
        new_frames = self.tensor_frame.vectorize()
        return TuckerTensorVector(self.core, new_frames)
