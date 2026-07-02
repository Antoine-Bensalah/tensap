from typing import List, Optional, Tuple, Sequence, Callable
import numpy as np

from .frame import Frame, FrameVector, FrameOperator


class TFrame:
    """
    Base class representing a tensor product of frames.
    Orchestrates local operations across multiple tensor dimensions.
    """

    def __init__(self, frames: List[Frame]):
        self.frames = list(frames)

    @property
    def order(self) -> int:
        """The number of tensor dimensions (d)."""
        return len(self.frames)

    @property
    def representation_ranks(self) -> Tuple[int, ...]:
        """Tuple of the representation ranks for each dimension."""
        return tuple(frame.rank for frame in self.frames)

    @property
    def shape(self) -> Tuple[Tuple[int, int], ...]:
        """Tuple containing the physical shape (dim1, dim2) of each frame."""
        return tuple(frame.shape for frame in self.frames)

    @property
    def is_orth(self):
        return all(frame.is_orth for frame in self.frames)

    def storage(self) -> int:
        """Return the total storage complexity of all tensor spaces combined"""
        return sum(frame.storage() for frame in self.frames)

    def sparse_storage(self) -> int:
        """Return the total number of non-zero entries across all spaces"""
        return sum(frame.sparse_storage() for frame in self.frames)

    def matrix_times_space(
        self, matrices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "TFrame":
        """Apply physical contraction on specified dimensions."""
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of specified dimensions."
            )

        new_frames = list(self.frames) # shallow copy
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].matrix_times_space(mat)

        return self.__class__(new_frames)

    def space_times_matrix(
            self,
            matrices: List[np.ndarray],
            dims: Optional[List[int]] = None
    ) -> 'TFrame':
        """Apply rank contraction (change of basis) on specified dimensions."""
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of specified dimensions."
            )

        new_frames = list(self.frames) # shallow copy
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].space_times_matrix(mat)

        return self.__class__(new_frames)

    def dot(
            self, other: 'TFrame', dims: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Compute the local Gram matrices between self and other.
        Returns a list of local Gram matrices (one for each specified dimension).
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
        """
        Orthonormalize the specified frames.

        Returns
        -------
        new_tframes : TFrame
            A new TFrames object with orthonormalized frames.
        r_matrices : List[np.ndarray or None]
            A list of length `order` containing the R factors from the QR
            decomposition for the orthonormalized dimensions, and None for the others.
        """
        if dims is None:
            dims = list(range(self.order))

        new_frames = list(self.frames)
        r_matrices = []

        for dim in dims:
            new_frames[dim], r_mat = self.frames[dim].orth()
            r_matrices.append(r_mat)

        return self.__class__(new_frames), r_matrices

    def get_vectorize_factors(self) -> List[np.ndarray]:
        """Return the Tucker factors in vectorized (2D) format.

        Each 3D array is flattened to 2D ``(-1, Rank)``:
        ``(Rank, N1, N2)`` becomes ``(N1 * N2, Rank)``.

        Returns
        -------
        list of numpy.ndarray
        """
        return [
            s.data.transpose(1, 2, 0).reshape(d1 * d2, r)
            for s, (d1, d2), r in zip(
                self.frames, self.shape, self.representation_ranks
            )
        ]

    def cat(self, other: "TFrame") -> "TFrame":
        """Concatenates the elements of two TFrames."""
        if self.order != other.order:
            raise ValueError("Spaces must have the same order.")

        new_frames = [
            f_self.cat(f_other) for f_self, f_other in zip(self.frames, other.frames)
        ]
        return self.__class__(new_frames)

    def eval_in_space(self, dim: int, coefs: np.ndarray) -> np.ndarray:
        """Evaluates a vector/operator from its coefficients on the basis."""
        return self.frames[dim].eval_in_space(coefs)

    def permute(self, dims: Sequence[int]) -> "TFrame":
        """
        Reorder the tensor dimensions.

        Parameters
        ----------
        dims: Sequence[int]
            A permutation of the integers from 0 to order-1

        Returns
        -------
        TFrame
            A new TFrames instance with reordered frames
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
        """
        Keep only the specified dimensions, reducing the tensor order.

        Parameters
        ----------
        dims: Sequence[int]
            The indices of the dimensions to keep

        Returns
        -------
        TFrame
            A new TFrames instance containing only the kept dimensions
        """
        if any(i < 0 or i >= self.order for i in dims):
            raise IndexError("keep_space: dimension index out of bounds.")

        new_frames = [self.frames[i] for i in dims]
        return self.__class__(new_frames)

    def remove_space(self, dims: Sequence[int]) -> "TFrame":
        """
        Remove the specified dimensions, reducing the tensor order.

        Parameters
        ----------
        dims : Sequence[int]
            The indices of the dimensions to remove.

        Returns
        -------
        TFrame
            A new TFrames instance without the specified dimensions.
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
        """Extract a sub-tensor space (evaluate at specific indices)."""
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
        """
        Inner products of basis vectors with a given metric provided by a TFrameOperators.
        Returns a list of 3D arrays of shape (Rank_metric, Rank_self, Rank_other).
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
        """Concatenates the elements of two TFrames in a block-diagonal way."""
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
        """Reshape the vector spaces into operator spaces."""
        if len(dims1) != self.order or len(dims2) != self.order:
            raise ValueError(
                "The length of dims1 and dims2 must match the order of the tensor."
            )

        new_frames = []
        for i, frame in enumerate(self.frames):
            new_frames.append(frame.unvectorize(dims1[i], dims2[i]))
        return TFrameOperator(new_frames)

    def to_diagonal_operators(self) -> "TFrameOperator":
        """Convert all vector frames into diagonal operator frames."""
        new_frames = [frame.to_diagonal_operator() for frame in self.frames]
        return TFrameOperator(new_frames)

    @classmethod
    def _generate(
        cls, dims: Sequence[int], ranks: Sequence[int], generator: Callable
    ) -> "TFrameVector":
        """Utility method to generate TFrameVectors."""
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
        return cls._generate(dims, ranks, np.zeros)

    @classmethod
    def ones(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        return cls._generate(dims, ranks, np.ones)

    @classmethod
    def rand(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        return cls._generate(dims, ranks, lambda shape: np.random.uniform(size=shape))

    @classmethod
    def randn(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        return cls._generate(dims, ranks, lambda shape: np.random.normal(size=shape))

    @classmethod
    def eye(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameVector":
        """Canonical basis for each Frame"""
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
        """Compose this tensor operator space with another space (Operator or Vector)."""
        if self.order != other.order:
            raise ValueError("Spaces must have the same order for composition.")

        new_frames = []
        for f_self, f_other in zip(self.frames, other.frames):
            new_frames.append(f_self.mtimes(f_other))

        return other.__class__(new_frames)

    def transpose(self, conjugate: bool = False) -> "TFrameOperator":
        """Transpose (or conjugate transpose) all physical dimensions."""
        new_frames = [frame.transpose(conjugate) for frame in self.frames]
        return TFrameOperator(new_frames)

    @property
    def T(self) -> "TFrameOperator":
        return self.transpose(conjugate=False)

    @property
    def H(self) -> "TFrameOperator":
        return self.transpose(conjugate=True)

    def vectorize(self) -> TFrameVector:
        """Flatten all operators into column vector spaces."""
        new_frames = [frame.vectorize() for frame in self.frames]
        return TFrameVector(new_frames)

    @classmethod
    def _generate(
        cls,
        dims1: Sequence[int],
        dims2: Sequence[int],
        ranks: Sequence[int],
        generator: Callable,
    ) -> "TFrameOperator":
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
        return cls._generate(dims1, dims2, ranks, np.zeros)

    @classmethod
    def ones(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        return cls._generate(dims1, dims2, ranks, np.ones)

    @classmethod
    def rand(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        return cls._generate(
            dims1, dims2, ranks, lambda shape: np.random.uniform(size=shape)
        )

    @classmethod
    def randn(
        cls, dims1: Sequence[int], dims2: Sequence[int], ranks: Sequence[int]
    ) -> "TFrameOperator":
        return cls._generate(
            dims1, dims2, ranks, lambda shape: np.random.normal(size=shape)
        )

    @classmethod
    def eye(cls, dims: Sequence[int], ranks: Sequence[int]) -> "TFrameOperator":
        """
        Create an identity operator on the tensor space.
        The frames are generated by only one operator for each dimension: the identity.
        If ranks are more than 1, one completes by zero operators.
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
