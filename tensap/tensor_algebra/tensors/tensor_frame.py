from typing import List, Optional, Tuple
import numpy as np

from .frame import Frame, FrameVector, FrameOperator


class TFrames:
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
    def ranks(self) -> Tuple[int, ...]:
        """Tuple of the representation ranks for each dimension."""
        return tuple(frame.rank for frame in self.frames)

    @property
    def shape(self) -> Tuple[Tuple[int, int], ...]:
        """Tuple containing the physical shape (dim1, dim2) of each frame."""
        return tuple(frame.shape for frame in self.frames)

    def matrix_times_space(
        self, matrices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "TFrames":
        """Apply physical contraction on specified dimensions."""
        # todo: rename contract_physical ?
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of "
                "specified dimensions."
            )

        new_frames = list(self.frames)
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].matrix_times_space(mat)

        return self.__class__(new_frames)

    def space_times_matrix(
            self,
            matrices: List[np.ndarray],
            dims: Optional[List[int]] = None
    ) -> 'TFrames':
        """Apply rank contraction (change of basis) on specified dimensions."""
        # todo: rename contract_rank ?
        if dims is None:
            dims = list(range(self.order))

        if len(matrices) != len(dims):
            raise ValueError("The number of matrices must match the number of "
                             "specified dimensions.")

        new_frames = list(self.frames)
        for dim, mat in zip(dims, matrices):
            new_frames[dim] = self.frames[dim].space_times_matrix(mat)

        return self.__class__(new_frames)

    def dot(
            self, other: 'TFrames', dims: Optional[List[int]] = None
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
    ) -> Tuple['TFrames', List[Optional[np.ndarray]]]:
        """
        Orthonormalize the specified frames.

        Returns
        -------
        new_tframes : TFrames
            A new TFrames object with orthonormalized frames.
        r_matrices : List[np.ndarray or None]
            A list of length `order` containing the R factors from the QR
            decomposition for the orthonormalized dimensions, and None for the others.
        """
        if dims is None:
            dims = list(range(self.order))

        new_frames = list(self.frames)
        r_matrices = [None] * self.order

        for dim in dims:
            new_frames[dim], r_matrices[dim] = self.frames[dim].orth()

        return self.__class__(new_frames), r_matrices


class TFrameVectors(TFrames):
    """Orchestrates tensor product of vector spaces (states, functions)."""

    def __init__(self, frames: List[FrameVector]):
        super().__init__(frames)

    def subspace(
        self, indices: List[np.ndarray], dims: Optional[List[int]] = None
    ) -> "TFrameVectors":
        """Extract a sub-tensor space (evaluate at specific indices)."""
        if dims is None:
            dims = list(range(self.order))

        new_frames = list(self.frames)
        for i, dim in enumerate(dims):
            new_frames[dim] = self.frames[dim].subspace(indices[i])

        return TFrameVectors(new_frames)

    def unvectorize(
        self, shapes_out: List[int], shapes_in: List[int]
    ) -> "TFrameOperators":
        """Reshape the vector spaces into operator spaces."""
        new_frames = []
        for i, frame in enumerate(self.frames):
            new_frames.append(frame.unvectorize(shapes_out[i], shapes_in[i]))
        return TFrameOperators(new_frames)

    def to_diagonal_operators(self) -> "TFrameOperators":
        """Convert all vector frames into diagonal operator frames."""
        new_frames = [frame.to_diagonal_operator() for frame in self.frames]
        return TFrameOperators(new_frames)


class TFrameOperators(TFrames):
    """Orchestrates tensor product of operator spaces."""

    def __init__(self, frames: List[FrameOperator]):
        super().__init__(frames)

    def mtimes(self, other: TFrames) -> TFrames:
        """Compose this tensor operator space with another space (Operator or Vector)."""
        if self.order != other.order:
            raise ValueError("Spaces must have the same order for composition.")

        new_frames = []
        for f_self, f_other in zip(self.frames, other.frames):
            new_frames.append(f_self.mtimes(f_other))

        return other.__class__(new_frames)

    def transpose(self, conjugate: bool = False) -> "TFrameOperators":
        """Transpose (or conjugate transpose) all physical dimensions."""
        new_frames = [frame.transpose(conjugate) for frame in self.frames]
        return TFrameOperators(new_frames)

    @property
    def T(self) -> "TFrameOperators":
        return self.transpose(conjugate=False)

    @property
    def H(self) -> "TFrameOperators":
        return self.transpose(conjugate=True)

    def vectorize(self) -> TFrameVectors:
        """Flatten all operators into column vector spaces."""
        new_frames = [frame.vectorize() for frame in self.frames]
        return TFrameVectors(new_frames)
