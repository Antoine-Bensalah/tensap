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
- Neumann du/dn = g at the face (n the outward normal): the centred flux at the
  face, (u_ghost - u_0) / h = g, so u_ghost = u_0 + h g: a correction +1 on the
  diagonal and the source g / h;
- periodic axis: the neighbour across the face is the cell at the other end
  (circulant matrix).

The operators only depend on the types of the conditions; the data g enter the
boundary source s, so that Delta u ~ Delta_h u + s at the cell centres, and
-Delta u = f becomes -Delta_h u = f + s.

The operators are tensap.TTOperator objects built from the QTT primitives of
tensap (qtt_banded, qtt_corner), assembled in dimension d by qtt_embed; the
source of a face is e_0 (or e_{N-1}) of the normal axis times the data on the
face, built in QTT by qtt_insert_axis (no array of the size of the grid).
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

    @property
    def is_singular(self) -> bool:
        """True if the constants are in the kernel of the second difference:
        periodic axis, or Neumann on both faces."""
        return self.periodic or (isinstance(self.left, Neumann)
                                 and isinstance(self.right, Neumann))

    def smallest_nonzero_eigenvalue(self) -> float:
        """Smallest non-zero eigenvalue of -D2, for a singular axis.

        The eigenvalues of -D2 are (4 / h^2) sin^2(pi k / (2 N)), k = 0, ...,
        N - 1, with Neumann on both faces (cosine modes), and
        (4 / h^2) sin^2(pi k / N) for a periodic axis (Fourier modes).
        """
        if not self.is_singular:
            raise ValueError("The second difference of this axis is not singular.")
        angle = np.pi / self.size if self.periodic else np.pi / (2 * self.size)
        return 4.0 / self.h**2 * np.sin(angle) ** 2

    def boundary_weight(self, side: str) -> float:
        """Weight of the data g of the condition of a side in the source of the
        first (left) or last (right) cell: 2 / h^2 for Dirichlet, 1 / h for
        Neumann."""
        condition = self._condition(side)
        if isinstance(condition, Dirichlet):
            return 2.0 / self.h**2
        if isinstance(condition, Neumann):
            return 1.0 / self.h
        raise NotImplementedError(f"No finite volume closure for {condition!r}.")

    def _condition(self, side: str) -> BoundaryCondition:
        if self.periodic:
            raise ValueError("A periodic axis has no boundary.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        return self.left if side == "left" else self.right

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
        return 1.0  # u_ghost = u_0 + h g
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

    @property
    def is_singular(self) -> bool:
        """True if the constants are in the kernel of the Laplacian: every
        axis periodic or with Neumann on both faces."""
        return all(axis.is_singular for axis in self.axes)

    def constant_mode_regularization(self, sigma: float = None) -> "tensap.TTOperator":
        """The rank-one term R = (sigma / N) 1 1^T for a singular Laplacian.

        -Delta_h + R is symmetric positive definite: writing u = u_0 + c 1
        with u_0 of zero mean, (-Delta_h + R) u = b gives
        -Delta_h u_0 = b - mean(b) 1 and c = mean(b) / sigma. For compatible
        data (mean(b) = 0, the discrete form of int f + int_boundary g = 0)
        the solution is the solution of zero mean; otherwise the least-squares
        solution, shifted by a constant. The eigenvalues of -Delta_h + R are
        those of -Delta_h with 0 replaced by sigma.

        Parameters
        ----------
        sigma : float, optional
            The default is the smallest non-zero eigenvalue of -Delta_h (the
            minimum over the axes of their smallest non-zero eigenvalue): the
            smallest value that does not degrade the conditioning.

        Returns
        -------
        tensap.TTOperator
            Of rank one, to be added to -1.0 * self.laplacian().
        """
        if not self.is_singular:
            raise ValueError("The Laplacian of this grid is not singular.")
        if sigma is None:
            sigma = min(axis.smallest_nonzero_eigenvalue() for axis in self.axes)
        N = int(np.prod(self.shape))
        return tensap.qtt_ones_matrix(int(self.tensorizer.levels.sum())) * (sigma / N)

    def boundary_source(self, data=None, tolerance: float = 1e-12) -> "tensap.TTTensor":
        """The source s of the boundary data: Delta u ~ Delta_h u + s.

        Parameters
        ----------
        data : dict, optional
            {(axis, side): g}, side "left" (x_axis = 0) or "right"
            (x_axis = L_axis), g a scalar, a function of the points of the
            face (array of shape (n, dim - 1), the coordinates of the other
            axes in their order), or a TTTensor on the face, in the layout of
            face_tensorizer(axis). The default takes the values of the
            conditions of the axes. Missing faces have no data (g = 0).
        tolerance : float, optional
            Precision of the quantization of sampled data and of the rounding
            of the sum of the faces.

        Returns
        -------
        tensap.TTTensor
            Zero away from the faces; on the cell next to a face, the weight of
            the closure times g (2 g / h^2 for Dirichlet, g / h for Neumann).
            Built in QTT, face by face.
        """
        if data is None:
            data = {}
            for k, axis in enumerate(self.axes):
                if not axis.periodic:
                    for side in ("left", "right"):
                        data[(k, side)] = axis._condition(side).value
        result = None
        for (k, side), g in data.items():
            axis = self.axes[k]
            weight = axis.boundary_weight(side)
            if not callable(g) and not isinstance(g, tensap.TTTensor) and g == 0:
                continue
            index = 0 if side == "left" else axis.size - 1
            vector = tensap.qtt_elementary_vector(axis.levels, index) * weight
            face = self._face_tensor(k, g, tolerance)
            term = tensap.qtt_insert_axis(face, k, vector, self.tensorizer)
            result = term if result is None else result + term
        if result is None:
            return tensap.TTTensor([np.zeros((1, 2, 1))] * int(self.tensorizer.levels.sum()))
        return tensap.Truncator(tolerance=tolerance).truncate(result)

    def face_tensorizer(self, axis: int) -> "tensap.Tensorizer":
        """Tensorizer of the face normal to axis: the other axes, same ordering."""
        others = [a for j, a in enumerate(self.axes) if j != axis]
        X = tensap.RandomVector([a.random_variable() for a in others])
        t = tensap.Tensorizer(2, [a.levels for a in others], len(others), X=X)
        t.ordering_type = self.tensorizer.ordering_type
        return t

    def _face_tensor(self, axis, g, tolerance):
        """The data g on the face normal to axis, as a QTT on the face."""
        if self.dim == 1:
            if callable(g) or isinstance(g, tensap.TTTensor):
                raise TypeError("In dimension 1, the data of a face is a scalar.")
            return float(g)
        t = self.face_tensorizer(axis)
        if isinstance(g, tensap.TTTensor):
            if g.order != int(t.levels.sum()):
                raise ValueError("The face tensor does not have the layout of the face.")
            return g
        if callable(g):
            return tensap.quantize_function(g, t, 0.5, tolerance)
        cores = [np.ones((1, 2, 1))] * int(t.levels.sum())
        return tensap.TTTensor(cores) * float(g)

    def boundary_source_array(self, data=None) -> np.ndarray:
        """The same source as a dense array (reference for the tests)."""
        if data is None:
            data = {(k, side): a._condition(side).value for k, a in enumerate(self.axes)
                    if not a.periodic for side in ("left", "right")}
        s = np.zeros(self.shape)
        x = self.cell_centers()
        for (k, side), g in data.items():
            axis = self.axes[k]
            index = 0 if side == "left" else axis.size - 1
            face_points = np.take(x, index, axis=k)
            face_points = np.delete(face_points, k, axis=-1)
            if isinstance(g, tensap.TTTensor):
                values = tensap.dequantize(g, self.face_tensorizer(k))
            elif callable(g):
                values = g(face_points.reshape(-1, self.dim - 1)).reshape(face_points.shape[:-1])
            else:
                values = float(g)
            slicer = [slice(None)] * self.dim
            slicer[k] = index
            s[tuple(slicer)] += axis.boundary_weight(side) * values
        return s

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
