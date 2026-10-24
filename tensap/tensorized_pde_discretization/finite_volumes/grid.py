# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module grid: cell-centred finite volume grids in QTT format.

An axis (0, L) is divided into N = 2^l cells of size h = L / N, whatever the
boundary conditions; the unknowns are the values at the cell centres
x_i = (i + 1/2) h, i = 0, ..., N - 1. These are the cells of a
tensap.Tensorizer with X uniform on [0, L], at y = 1/2: the digits of i are
the levels of the dyadic refinement of the axis.

The second difference is the three-point flux difference
(u_{i-1} - 2 u_i + u_{i+1}) / h^2. At a face, the missing neighbour is a
ghost value set by the boundary condition (the closure of the finite volume
method):

- Dirichlet u = g at the face: u_ghost = 2 g - u_0 (the face value is the mean
  of the ghost and the first cell), a correction -1 on the diagonal and the
  source 2 g / h^2;
- periodic axis: the neighbour across the face is the cell at the other end
  (circulant matrix).

The operators are tensap.TTOperator objects built from the QTT primitives of
tensap (qtt_banded, qtt_corner), assembled in dimension d by qtt_embed.
"""

import numpy as np
import tensap
from ..boundary_conditions import BoundaryCondition, Dirichlet, Neumann


class Axis:
    """One axis (0, length) of a tensor-product grid, with 2^levels cells.

    Attributes
    ----------
    length : float
        Length L of the axis.
    levels : int
        Number of digits: N = 2^levels cells.
    left, right : BoundaryCondition or None
        Conditions at x = 0 and x = L (None for a periodic axis).
    periodic : bool
        True if the two faces are identified.
    """

    def __init__(self, length: float, levels: int, left: BoundaryCondition = None,
                 right: BoundaryCondition = None, periodic: bool = False):
        if length <= 0:
            raise ValueError("The length must be positive.")
        if levels < 1:
            raise ValueError("At least one digit (two cells) is required.")
        if periodic and (left is not None or right is not None):
            raise ValueError("A periodic axis has no boundary condition.")
        if not periodic:
            left = Dirichlet() if left is None else left
            right = Dirichlet() if right is None else right
            for condition in (left, right):
                if not isinstance(condition, BoundaryCondition):
                    raise TypeError(f"Not a boundary condition: {condition!r}.")
        self.length = float(length)
        self.levels = int(levels)
        self.left, self.right = left, right
        self.periodic = periodic

    @property
    def size(self) -> int:
        """Number N of cells."""
        return 2**self.levels

    @property
    def h(self) -> float:
        """Size L / N of the cells."""
        return self.length / self.size

    def cell_centers(self) -> np.ndarray:
        """The N cell centres (i + 1/2) h."""
        return (np.arange(self.size) + 0.5) * self.h

    def random_variable(self) -> "tensap.UniformRandomVariable":
        """Uniform law on [0, L]: the geometry of the axis for a Tensorizer."""
        return tensap.UniformRandomVariable(0.0, self.length)

    def second_difference(self, tolerance: float = 1e-14) -> "tensap.TTOperator":
        """The matrix D2 ~ d^2/dx^2 on the cells, with the closures of the
        boundary conditions (homogeneous part: the data only enter the
        boundary source), as a TTOperator of small rank."""
        band = {-1: 1.0, 0: -2.0, 1: 1.0}
        if self.periodic:
            D2 = tensap.qtt_banded(band, self.levels, periodic=True, tolerance=tolerance)
        else:
            D2 = tensap.qtt_banded(band, self.levels, tolerance=tolerance)
            for condition, corner in ((self.left, "top-left"), (self.right, "bottom-right")):
                D2 = D2 + tensap.qtt_corner([[_diagonal_correction(condition)]],
                                            self.levels, corner)
            D2 = tensap.Truncator(tolerance=tolerance).truncate(D2)
        return D2 * (1.0 / self.h**2)

    def second_difference_matrix(self):
        """The same matrix with scipy.sparse (reference for the tests)."""
        import scipy.sparse as sp
        N = self.size
        D2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N), format="lil")
        if self.periodic:
            D2[0, N - 1] = D2[N - 1, 0] = 1.0
        else:
            D2[0, 0] += _diagonal_correction(self.left)
            D2[N - 1, N - 1] += _diagonal_correction(self.right)
        return (D2 / self.h**2).tocsr()

    def __repr__(self):
        ends = "periodic" if self.periodic else f"left={self.left}, right={self.right}"
        return f"Axis(length={self.length}, levels={self.levels}, {ends})"


def _diagonal_correction(condition: BoundaryCondition) -> float:
    """Diagonal term added at a face by the closure of the condition."""
    if isinstance(condition, Dirichlet):
        return -1.0  # u_ghost = 2 g - u_0
    if isinstance(condition, Neumann):
        raise NotImplementedError("Neumann conditions: step 3 of P4c.")
    raise NotImplementedError(f"No finite volume closure for {condition!r}.")


class Grid:
    """Tensor product of axes, in the layout of a tensap.Tensorizer.

    Attributes
    ----------
    axes : list of Axis
    ordering : str
        "blocks" (the digits of the first axis, then of the second one...) or
        "interleaved" (the first digit of every axis, then the second one...).
    tensorizer : tensap.Tensorizer
        Layout and geometry of the unknowns: base 2, the levels of the axes,
        X uniform on [0, L_k]; the unknowns are its cells at y = 1/2.
    """

    _ORDERINGS = {"blocks": 1, "interleaved": 2}

    def __init__(self, axes, ordering: str = "blocks"):
        if ordering not in self._ORDERINGS:
            raise ValueError(f"ordering must be one of {list(self._ORDERINGS)}.")
        self.axes = list(axes)
        self.ordering = ordering
        X = tensap.RandomVector([axis.random_variable() for axis in self.axes])
        self.tensorizer = tensap.Tensorizer(
            2, [axis.levels for axis in self.axes], len(self.axes), X=X)
        self.tensorizer.ordering_type = self._ORDERINGS[ordering]

    @property
    def dim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def shape(self) -> tuple:
        """Numbers of cells (N_1, ..., N_d)."""
        return tuple(axis.size for axis in self.axes)

    def cell_centers(self) -> np.ndarray:
        """The cell centres, array of shape (N_1, ..., N_d, d)."""
        return tensap.grid_points(self.tensorizer, 0.5)

    def laplacian(self, tolerance: float = 1e-14) -> "tensap.TTOperator":
        """The discrete Laplacian sum_k I x ... x D2_k x ... x I, with the
        closures of the boundary conditions, as a TTOperator (rounded)."""
        terms = [tensap.qtt_embed(axis.second_difference(tolerance), k, self.tensorizer)
                 for k, axis in enumerate(self.axes)]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return tensap.Truncator(tolerance=tolerance).truncate(result)

    def laplacian_matrix(self):
        """The same matrix with scipy.sparse, unknowns in C order (reference)."""
        import scipy.sparse as sp
        result = None
        for k, axis in enumerate(self.axes):
            term = sp.identity(1)
            for j, other in enumerate(self.axes):
                block = other.second_difference_matrix() if j == k else sp.identity(other.size)
                term = sp.kron(term, block)
            result = term if result is None else result + term
        return result.tocsr()

    def quantize(self, values, tolerance: float = 1e-12) -> "tensap.TTTensor":
        """QTT of a function at the cell centres (sampled on the whole grid),
        or of an array of shape self.shape."""
        if callable(values):
            return tensap.quantize_function(values, self.tensorizer, 0.5, tolerance)
        return tensap.quantize(values, self.tensorizer, tolerance)

    def dequantize(self, tensor) -> np.ndarray:
        """The array of shape self.shape of a QTT on the grid."""
        return tensap.dequantize(tensor, self.tensorizer)

    def __repr__(self):
        return f"Grid({self.axes}, ordering={self.ordering!r})"
