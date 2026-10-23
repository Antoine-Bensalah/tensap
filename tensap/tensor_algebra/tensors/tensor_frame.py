from typing import List, Optional, Tuple, Sequence, Callable
import numpy as np

from .frame import Frame, FrameVector, FrameOperator


class TFrame:
    """
    Base class representing a tensor product of frames.

    Orchestrates local operations across multiple tensor dimensions.
    Each dimension stores a :class:`Frame` object.
    """

    def __init__(self, frames: List[Frame]):
        self.frames = list(frames)

    @property
    def order(self) -> int:
        """Return the number of tensor dimensions (d).

        Returns
        -------
        int
        """
        return len(self.frames)

    @property
    def representation_ranks(self) -> Tuple[int, ...]:
        """Return the representation ranks for each dimension.

        Returns
        -------
        tuple of int
        """
        return tuple(frame.dim for frame in self.frames)

    @property
    def shape(self) -> Tuple[Tuple[int, int], ...]:
        """Return the physical shape (dim1, dim2) of each frame.

        Returns
        -------
        tuple of tuple of int
        """
        return tuple(frame.shape for frame in self.frames)

    @property
    def is_orth(self):
        """Return whether all frames are orthonormal.

        Returns
        -------
        bool
        """
        return all(frame.is_orth for frame in self.frames)

    def storage(self) -> int:
        """Return the total storage complexity of all frames combined.

        Returns
        -------
        int
        """
        return sum(frame.storage() for frame in self.frames)

    def sparse_storage(self) -> int:
        """Return the total number of non-zero entries across all frames.

        Returns
        -------
        int
        """
        return sum(frame.sparse_storage() for frame in self.frames)

    def matrix_times_space(
        self, matrices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "TFrame":
        """Left multiply the physical dimension of frames on specified dims.

        Parameters
        ----------
        matrices : list of ndarray
            Matrices to apply, one per specified dimension.
        dims : list of int, optional
            Dimensions to contract. If None, all dimensions are contracted.

        Returns
        -------
        TFrame
        """
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of specified dimensions."
            )

        new_frames = list(self.frames)  # shallow copy
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].matrix_times_space(mat)

        return self.__class__(new_frames)

    def space_times_matrix(
            self,
            matrices: List[np.ndarray],
            dims: Optional[List[int]] = None
    ) -> 'TFrame':
        """Apply rank contraction (change of basis) on specified dimensions.

        Parameters
        ----------
        matrices : list of ndarray
            Matrices defining the linear combination of basis elements.
        dims : list of int, optional
            Dimensions to transform. If None, all dimensions are transformed.

        Returns
        -------
        TFrame
        """
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of specified dimensions."
            )

        new_frames = list(self.frames)  # shallow copy
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].space_times_matrix(mat)

        return self.__class__(new_frames)

    def dot(
            self, other: 'TFrame', dims: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Compute the local Gram matrices between self and other.

        Parameters
        ----------
        other : TFrame
        dims : list of int, optional
            Dimensions to compute Gram matrices for. If None, all dimensions.

        Returns
        -------
        list of ndarray
            List of local Gram matrices, one per specified dimension.
        """
        if dims is None:
            dims = list(range(self.order))

        gram_matrices = []
        for dim in dims:
            gram_matrices.append(self.frames[dim].dot(other.frames[dim]))

        return gram_matrices

    def orth(
            self, dims: Optional[List[int]] = None
    ) -> Tuple['TFrame', List[Optional[np.ndarray]]]:
        """Orthonormalize the specified frames.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to orthonormalize. If None, all dimensions.

        Returns
        -------
        TFrame
            A new TFrame with orthonormalized frames.
        list of ndarray or None
            The R factors from the QR decomposition for each orthonormalized
            dimension; None for skipped dimensions.
        """
        if dims is None:
            dims = list(range(self.order))

        new_frames = list(self.frames)
        r_matrices = []

        for dim in dims:
            new_frames[dim], r_mat = self.frames[dim].orth()
            r_matrices.append(r_mat)

        return self.__class__(new_frames), r_matrices

    def get_vectorize_factors(
        self, dims: Optional[Sequence[int]] = None
    ) -> List[np.ndarray]:
        """Return the Tucker factors in vectorized (2D) format for specified dims.

        Each 3D array (Rank, N1, N2) is flattened to 2D (N1 * N2, Rank).
        # todo : C-order -> le documenter ! sens pour les opérateurs ?

        Parameters
        ----------
        dims : Sequence[int], optional
            Dimensions to vectorize. If None, all dimensions.

        Returns
        -------
        list of ndarray
        """
        if dims is None:
            dims = range(self.order)

        return [
            self.frames[d].data.transpose(1, 2, 0).reshape(-1, self.frames[d].dim)
            for d in dims
        ]

    def cat(self, other: "TFrame") -> "TFrame":
        """Concatenate the basis elements of two TFrame objects.

        Parameters
        ----------
        other : TFrame
            Must have the same order.

        Returns
        -------
        TFrame
        """
        if self.order != other.order:
            raise ValueError("Spaces must have the same order.")

        new_frames = [
            f_self.cat(f_other) for f_self, f_other in zip(self.frames, other.frames)
        ]
        return self.__class__(new_frames)

    def kron(self, other: "TFrame") -> "TFrame":
        """Kronecker product with another TFrame.

        Parameters
        ----------
        other : TFrame
            Must have the same order.

        Returns
        -------
        TFrame
        """
        if self.order != other.order:
            raise ValueError("Spaces must have the same order.")
        new_frames = [
            f_self.kron(f_other) for f_self, f_other in zip(self.frames, other.frames)
        ]
        return self.__class__(new_frames)

    def hadamard_product(self, other: "TFrame") -> "TFrame":
        """Entrywise product of every pair of frame elements, dimension by dimension.

        Parameters
        ----------
        other : TFrame
            Must have the same order and physical shape.

        Returns
        -------
        TFrame
        """
        if self.order != other.order:
            raise ValueError("Spaces must have the same order.")
        new_frames = [
            f_self.hadamard_product(f_other)
            for f_self, f_other in zip(self.frames, other.frames)
        ]
        return self.__class__(new_frames)

    def super_frame(self, shape, indices) -> "TFrame":
        """Embed the frame elements of every dimension in larger zero matrices.

        Parameters
        ----------
        shape : sequence of tuple of int
            New physical shape (M1, M2) of each dimension.
        indices : sequence of tuple
            For each dimension, the pair (rows, cols) of indices of the current
            entries (see Frame.super_frame); None keeps an axis unchanged.

        Returns
        -------
        TFrame
        """
        if len(shape) != self.order or len(indices) != self.order:
            raise ValueError(
                "shape and indices must have one entry per dimension."
            )
        new_frames = [
            frame.super_frame(sz, rows, cols)
            for frame, sz, (rows, cols) in zip(self.frames, shape, indices)
        ]
        return self.__class__(new_frames)

    def eval_in_space(self, dim: int, coefs: np.ndarray) -> np.ndarray:
        """Evaluate a vector/operator from its coefficients on a given dim.

        Parameters
        ----------
        dim : int
            Dimension index.
        coefs : ndarray
            1D array of coefficients.

        Returns
        -------
        ndarray
        """
        return self.frames[dim].eval_in_space(coefs)

    def permute(self, dims: Sequence[int]) -> "TFrame":
        """Reorder the tensor dimensions.

        Parameters
        ----------
        dims : Sequence[int]
            A permutation of the integers from 0 to order - 1.

        Returns
        -------
        TFrame
            A new TFrame with reordered frames.
        """
        if len(dims) != self.order:
            raise ValueError(
                f"permute: 'dims' must have the same lenght as the tensor oder"
                f"(expected {self.order}, got {len(dims)}"
            )

        if set(dims) != set(range(self.order)):
            raise ValueError(
                "permute: 'dims' must be a valid permutation of indices from 0 to "
                f"order - 1, got {dims}"
            )

        new_frames = [self.frames[i] for i in dims]
        return self.__class__(new_frames)

    def keep_space(self, dims: Sequence[int]) -> "TFrame":
        """Keep only the specified dimensions, reducing the tensor order.

        Parameters
        ----------
        dims : Sequence[int]
            Indices of the dimensions to keep.

        Returns
        -------
        TFrame
            A new TFrame containing only the kept dimensions.
        """
        if any(i < 0 or i >= self.order for i in dims):
            raise IndexError("keep_space: dimension index out of bounds.")

        new_frames = [self.frames[i] for i in dims]
        return self.__class__(new_frames)

    def remove_space(self, dims: Sequence[int]) -> "TFrame":
        """Remove the specified dimensions, reducing the tensor order.

        Parameters
        ----------
        dims : Sequence[int]
            Indices of the dimensions to remove.

        Returns
        -------
        TFrame
            A new TFrame without the specified dimensions.
        """
        keep = [i for i in range(self.order) if i not in dims]
        return self.keep_space(keep)


class TFrameVector(TFrame):
    """Orchestrates tensor product of vector spaces (states, functions)."""

    frames: List[FrameVector]

    def __init__(self, frames: List[FrameVector]):
        super().__init__(frames)

    def eval_at_indices(
        self, indices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "TFrameVector":
        """Extract a sub-tensor space by evaluating at specific indices.

        Parameters
        ----------
        indices : list of ndarray
            Arrays of indices, one per specified dimension.
        dims : list of int, optional
            Dimensions to evaluate. If None, all dimensions.

        Returns
        -------
        TFrameVector
        """
        if dims is None:
            dims = list(range(self.order))

        if len(indices) != len(dims):
            raise ValueError(
                "The number of index arrays must match the number of specified "
                "dimensions."
            )

        new_frames = list(self.frames)
        for idx, dim in zip(indices, dims):
            new_frames[dim] = self.frames[dim].eval_at_indices(idx)

        return TFrameVector(new_frames)

    def dot_with_metrics(
        self,
        other: "TFrameVector",
        metrics: "TFrameOperator",
        dims: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Compute inner products of basis vectors with a given metric.

        Parameters
        ----------
        other : TFrameVector
        metrics : TFrameOperator
            Metric operator for each dimension.
        dims : list of int, optional
            Dimensions to compute for. If None, all dimensions.

        Returns
        -------
        list of ndarray
            Each entry is a 3D array of shape (R_metric, R_self, R_other).
        """
        if dims is None:
            dims = list(range(self.order))

        gram_matrices = []
        for dim in dims:
            gram_matrices.append(
                self.frames[dim].dot_with_metric(other.frames[dim], metrics.frames[dim])
            )

        return gram_matrices

    def diag_cat(self, other: "TFrameVector") -> "TFrameVector":
        """Concatenate two TFrameVector objects block-diagonally.

        Parameters
        ----------
        other : TFrameVector
            Must have the same order.

        Returns
        -------
        TFrameVector
        """
        if self.order != other.order:
            raise ValueError("Spaces must have the same order.")

        new_frames = [
            f_self.diag_cat(f_other)
            for f_self, f_other in zip(self.frames, other.frames)
        ]
        return self.__class__(new_frames)

    def unvectorize(
        self, dims1: List[int], dims2: List[int]
    ) -> "TFrameOperator":
        """Reshape the vector spaces into operator spaces.

        Parameters
        ----------
        dims1 : list of int
            Output dimensions for each space.
        dims2 : list of int
            Input dimensions for each space.
            Must satisfy dims1[i] * dims2[i] == self.frames[i].shape[0].

        Returns
        -------
        TFrameOperator
        """
        if len(dims1) != self.order or len(dims2) != self.order:
            raise ValueError(
                "The length of dims1 and dims2 must match the order of the tensor."
            )

        new_frames = []
        for i, frame in enumerate(self.frames):
            new_frames.append(frame.unvectorize(dims1[i], dims2[i]))
        return TFrameOperator(new_frames)

    def to_diagonal_operators(self) -> "TFrameOperator":
        """Convert all vector frames into diagonal operator frames.

        Returns
        -------
        TFrameOperator
        """
        new_frames = [frame.to_diagonal_operator() for frame in self.frames]
        return TFrameOperator(new_frames)

    def elementwise_hadamard(self, dims: List[int]) -> "FrameVector":
        """Compute the element-wise product of several dimensions.

        Parameters
        ----------
        dims : list of int
            Dimensions to multiply together.

        Returns
        -------
        FrameVector
            A single FrameVector resulting from the Hadamard product.
        """
        if not dims:
            raise ValueError("Must provide at least one dimension.")

        result = self.frames[dims[0]]
        for dim in dims[1:]:
            result = result.elementwise_hadamard(self.frames[dim])

        return result

    @classmethod
    def create(
        cls, generator: Callable, dims: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameVector":
        """Create a TFrameVector using a generator function.

        Parameters
        ----------
        generator : Callable
            Function that takes a shape tuple and returns an ndarray.
        dims : Sequence[int]
            Physical dimensions for each tensor dimension.
        ranks : Sequence[int]
            Representation ranks for each tensor dimension.

        Returns
        -------
        TFrameVector
        """
        return cls._generate(dims, ranks, generator)

    @classmethod
    def _generate(
        cls, dims: Sequence[int], ranks: Sequence[int], generator: Callable
    ) -> "TFrameVector":
        """Utility method to generate TFrameVectors.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]
        generator : Callable

        Returns
        -------
        TFrameVector
        """
        if len(dims) != len(ranks):
            raise ValueError("Length of 'dims' and 'ranks' must match.")

        frames = []
        for d, r in zip(dims, ranks):
            # convention (Rank, dim1, dim2=1)
            data = generator((r, d, 1))
            frames.append(FrameVector(data))
        return cls(frames)

    @classmethod
    def zeros(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Create a TFrameVector filled with zeros.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameVector
        """
        return cls._generate(dims, ranks, np.zeros)

    @classmethod
    def ones(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Create a TFrameVector filled with ones.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameVector
        """
        return cls._generate(dims, ranks, np.ones)

    @classmethod
    def rand(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Create a TFrameVector with uniformly random entries.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameVector
        """
        return cls._generate(dims, ranks, lambda shape: np.random.uniform(size=shape))

    @classmethod
    def randn(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Create a TFrameVector with standard normal random entries.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameVector
        """
        return cls._generate(dims, ranks, lambda shape: np.random.normal(size=shape))

    @classmethod
    def eye(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Create a canonical basis TFrameVector.

        The first rank elements of each dimension are the canonical basis
        vectors.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameVector
        """
        if len(dims) != len(ranks):
            raise ValueError("Length of 'dims' and 'ranks' must match.")

        frames = []
        for d, r in zip(dims, ranks):
            data = np.eye(r, d).reshape(r, d, 1)
            frames.append(FrameVector(data))
        return cls(frames)


class TFrameOperator(TFrame):
    """Orchestrates tensor product of operator spaces."""

    frames: List[FrameOperator]

    def __init__(self, frames: List[FrameOperator]):
        super().__init__(frames)

    def mtimes(self, other: TFrame) -> TFrame:
        """Compose this tensor operator space with another space.

        Parameters
        ----------
        other : TFrame or TFrameVector or TFrameOperator
            Must have the same order.

        Returns
        -------
        TFrame
            Result of the composition.
        """
        if self.order != other.order:
            raise ValueError("Spaces must have the same order for composition.")

        new_frames = []
        for f_self, f_other in zip(self.frames, other.frames):
            new_frames.append(f_self.mtimes(f_other))

        return other.__class__(new_frames)

    def transpose(self, conjugate: bool = False) -> "TFrameOperator":
        """Transpose (or conjugate transpose) all physical dimensions.

        Parameters
        ----------
        conjugate : bool, optional
            If True, also take the complex conjugate. Default is False.

        Returns
        -------
        TFrameOperator
        """
        new_frames = [frame.transpose(conjugate) for frame in self.frames]
        return TFrameOperator(new_frames)

    @property
    def T(self) -> "TFrameOperator":
        """Syntactic sugar for transpose."""
        return self.transpose(conjugate=False)

    @property
    def H(self) -> "TFrameOperator":
        """Syntactic sugar for conjugate transpose (Hermitian)."""
        return self.transpose(conjugate=True)

    def vectorize(self) -> TFrameVector:
        """Flatten all operators into column vector spaces.

        Returns
        -------
        TFrameVector
        """
        new_frames = [frame.vectorize() for frame in self.frames]
        return TFrameVector(new_frames)

    @classmethod
    def create(
        cls, generator: Callable, shape, ranks: Sequence[int]
    ) -> "TFrameOperator":
        """Create a TFrameOperator using a generator function.

        Parameters
        ----------
        generator : Callable
            Function that takes a shape tuple and returns an ndarray.
        shape : list of tuple of int
            Physical shape for each dimension, e.g. [(n1, m1), (n2, m2)].
        ranks : Sequence[int]
            Representation ranks for each dimension.

        Returns
        -------
        TFrameOperator
        """
        dims1 = [s[0] for s in shape]
        dims2 = [s[1] for s in shape]
        return cls._generate(dims1, dims2, ranks, generator)

    @classmethod
    def _generate(
        cls,
        dims1: Sequence[int],
        dims2: Sequence[int],
        ranks: Sequence[int],
        generator: Callable,
    ) -> "TFrameOperator":
        """Utility method to generate TFrameOperators.

        Parameters
        ----------
        dims1 : Sequence[int]
            Output dimensions.
        dims2 : Sequence[int]
            Input dimensions.
        ranks : Sequence[int]
        generator : Callable

        Returns
        -------
        TFrameOperator
        """
        if not (len(dims1) == len(dims2) == len(ranks)):
            raise ValueError("Lengths of dims1, dims2 and ranks must match.")

        frames = []
        for d1, d2, r in zip(dims1, dims2, ranks):
            # convention (Rank, dim1, dim2)
            data = generator((r, d1, d2))
            frames.append(FrameOperator(data))
        return cls(frames)

    @classmethod
    def zeros(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        """Create a TFrameOperator filled with zeros.

        Parameters
        ----------
        dims1 : Sequence[int]
        dims2 : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameOperator
        """
        return cls._generate(dims1, dims2, ranks, np.zeros)

    @classmethod
    def ones(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        """Create a TFrameOperator filled with ones.

        Parameters
        ----------
        dims1 : Sequence[int]
        dims2 : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameOperator
        """
        return cls._generate(dims1, dims2, ranks, np.ones)

    @classmethod
    def rand(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        """Create a TFrameOperator with uniformly random entries.

        Parameters
        ----------
        dims1 : Sequence[int]
        dims2 : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameOperator
        """
        return cls._generate(
            dims1, dims2, ranks, lambda shape: np.random.uniform(size=shape)
        )

    @classmethod
    def randn(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        """Create a TFrameOperator with standard normal random entries.

        Parameters
        ----------
        dims1 : Sequence[int]
        dims2 : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameOperator
        """
        return cls._generate(
            dims1, dims2, ranks, lambda shape: np.random.normal(size=shape)
        )

    @classmethod
    def eye(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameOperator":
        """Create an identity operator on the tensor space.

        For each dimension, the first basis element is the identity matrix;
        remaining elements are zero.

        Parameters
        ----------
        dims : Sequence[int]
        ranks : Sequence[int]

        Returns
        -------
        TFrameOperator
        """
        if len(dims) != len(ranks):
            raise ValueError("Length of 'dims' and 'ranks' must match.")

        frames = []
        for d, r in zip(dims, ranks):
            # (Rank, dim1, dim2) -> (r, d, d)
            data = np.zeros((r, d, d))
            data[0, :, :] = np.eye(d)
            frames.append(FrameOperator(data))
        return cls(frames)
