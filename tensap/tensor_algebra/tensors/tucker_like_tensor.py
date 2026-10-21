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

        Returns
        -------
        FullTensor
        """
        # TODO : faire un full() spécifique aux TuckerTensorOperator TuckerTensorVector
        #        ne pas tester les dims = 1
        # 1. Extract matrices of shape (Physical_size, Rank)
        matrices = self.tensor_frame.get_vectorize_factors()

        # 2. Contract the core with these matrices
        result = self.core.tensor_matrix_product(matrices).full()

        # 3. Reshape the resulting 1D/flat array into the actual physical tensor geometry
        target_shape = []
        for frame in self.tensor_frame.frames:
            if frame.shape[1] == 1:
                target_shape.append(frame.shape[0])
            else:
                target_shape.extend([frame.shape[0], frame.shape[1]])

        return result.reshape(target_shape)

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

        Parameters
        ----------
        tree_tensor : TreeBasedTensor

        Returns
        -------
        BaseTuckerTensor
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
        new_core = self.core.kron(other.core)
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

        Uses orthonormalization before SVD for numerical stability.

        Returns
        -------
        ndarray or list of ndarray or list of None
            If order == 2, returns the singular values of the core after orth.
            If order > 2, delegates to core.singular_values() if available.
        """
        if self.order == 2:
            orth_self = self.orth()
            return np.linalg.svd(orth_self.core.data, compute_uv=False)
        else:
            orth_self = self.orth()
            if hasattr(orth_self.core, 'singular_values'):
                return orth_self.core.singular_values()
            # fallback: unfold and compute SVD for each dimension
            return [None] * self.order

    def normalize_basis(self):
        """Normalize the elements of the bases of subspaces in-place.

        Returns
        -------
        BaseTuckerTensor
            Self, after in-place normalization.
        """
        N = self.tensor_frame.dot(self.tensor_frame)
        N = [np.sqrt(np.diag(n)) for n in N]
        diag_mats = [np.diag(n) for n in N]
        self.core = self.core.tensor_matrix_product(diag_mats)
        inv_mats = [np.diag(1.0 / n) for n in N]
        self.tensor_frame = self.tensor_frame.space_times_matrix(inv_mats)
        return self

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

    def sub_tensor(self, indices: List[np.ndarray], dims: Optional[List[int]] = None) -> 'TuckerTensorVector':
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

        Returns
        -------
        TreeBasedTensor
        """
        tree = tensap.DimensionTree.trivial(self.order)
        factors = self.tensor_frame.get_vectorize_factors()
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
