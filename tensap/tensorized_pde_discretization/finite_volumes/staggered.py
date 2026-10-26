# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module staggered: finite volumes on a staggered (MAC) grid, in QTT format.

An axis (0, L) has N = 2^l cells of size h = L / N. Two locations:

- cells: the centres x_i = (i + 1/2) h, i = 0, ..., N - 1 (pressure, and the
  velocity components along the other axes);
- faces: x_i = i h, i = 0, ..., N - 1, the left face of each cell (the
  velocity component normal to the axis).

An axis has N + 1 faces: the N left faces are stored, so that every field has
2^l values per axis. On a periodic axis, face N is face 0. On a bounded axis,
face 0 is a boundary face whose value is known (wall, inflow): it stays in
the vector, and the operators leave it unchanged (zero rows); face N is not
stored, its value comes from the condition of the field at x = L: Dirichlet
u_N = g (wall, given velocity) or Neumann u_N = u_{N-1} + h g (outflow,
du/dn = g).

The 1D blocks (each a tensap.TTOperator built from qtt_banded and qtt_corner,
with a scipy-free numpy reference in the tests):

- gradient, cells -> faces: (c_i - c_{i-1}) / h on the face i (zero row on
  face 0 of a bounded axis);
- divergence, faces -> cells: (f_{i+1} - f_i) / h on the cell i;
- face_laplacian: (f_{i+1} - 2 f_i + f_{i-1}) / h^2 on the faces (zero row on
  face 0 of a bounded axis);
- cell_laplacian: the cell-centred second difference of finite_volumes.Axis;
- face_to_cell and cell_to_face: the averages of the two neighbours.

The data g of a condition enter a source, boundary_weight(block, condition)
times g on the last cell or face (first face for cell_to_face).

With these closures, the divergence of the gradient is exactly the
cell-centred Laplacian with Neumann conditions (or the circulant one):
D G = L_p, and G = -D^T on the interior faces.
"""

import numpy as np
import tensap
from ..boundary_conditions import Dirichlet, Neumann
from .grid import Axis


class StaggeredAxis:
    """One axis (0, length) of a staggered grid, with 2^levels cells.

    Attributes
    ----------
    length : float
    levels : int
        N = 2^levels cells (and N stored faces).
    periodic : bool
    """

    def __init__(self, length: float, levels: int, periodic: bool = False):
        if levels < 1:
            raise ValueError("levels must be at least 1.")
        self.length = float(length)
        self.levels = int(levels)
        self.periodic = bool(periodic)

    @property
    def size(self) -> int:
        """Number of cells N (and of stored faces)."""
        return 2**self.levels

    @property
    def h(self) -> float:
        """Size of the cells."""
        return self.length / self.size

    def positions(self, location: str) -> np.ndarray:
        """Coordinates of the cell centres ('cell') or of the stored faces
        ('face')."""
        if location == "cell":
            return (np.arange(self.size) + 0.5) * self.h
        if location == "face":
            return np.arange(self.size) * self.h
        raise ValueError("location must be 'cell' or 'face'.")

    # --- 1D blocks ---

    def _banded(self, diagonals, corners=()):
        operator = tensap.qtt_banded(diagonals, self.levels, periodic=self.periodic)
        for block, corner in corners:
            operator = operator + tensap.qtt_corner(np.atleast_2d(block), self.levels, corner)
        if corners:
            operator = tensap.Truncator(tolerance=1e-14).truncate(operator)
        return operator

    def gradient(self) -> "tensap.TTOperator":
        """Cells -> faces, (c_i - c_{i-1}) / h; zero row on face 0 of a
        bounded axis (a boundary face, where the normal velocity is known)."""
        corners = () if self.periodic else (([[-1.0]], "top-left"),)
        return self._banded({0: 1.0, -1: -1.0}, corners) * (1.0 / self.h)

    def divergence(self, right=None) -> "tensap.TTOperator":
        """Faces -> cells, (f_{i+1} - f_i) / h; f_N from the condition right of
        the face field at x = L: Dirichlet (data in the source), Neumann
        (f_N = f_{N-1} + h g)."""
        corners = self._right_corners(right, {Dirichlet: 0.0, Neumann: 1.0})
        return self._banded({0: -1.0, 1: 1.0}, corners) * (1.0 / self.h)

    def face_laplacian(self, right=None) -> "tensap.TTOperator":
        """Second difference on the faces; zero row on face 0 of a bounded axis
        (known value), f_N from the condition right."""
        corners = self._right_corners(right, {Dirichlet: 0.0, Neumann: 1.0})
        if not self.periodic:
            corners = corners + (([[2.0, -1.0]], "top-left"),)
        return self._banded({-1: 1.0, 0: -2.0, 1: 1.0}, corners) * (1.0 / self.h**2)

    def cell_laplacian(self, left=None, right=None) -> "tensap.TTOperator":
        """Cell-centred second difference with the finite volume closures
        (finite_volumes.Axis.second_difference)."""
        if self.periodic:
            return Axis(self.length, self.levels, periodic=True).second_difference()
        return Axis(self.length, self.levels, left, right).second_difference()

    def face_to_cell(self, right=None) -> "tensap.TTOperator":
        """Average (f_i + f_{i+1}) / 2 on the cell i; f_N from the condition
        right."""
        corners = self._right_corners(right, {Dirichlet: 0.0, Neumann: 0.5})
        return self._banded({0: 0.5, 1: 0.5}, corners)

    def cell_to_face(self, left=None) -> "tensap.TTOperator":
        """Average (c_{i-1} + c_i) / 2 on the face i; on face 0 of a bounded
        axis, the value of the cell field at x = 0 from the condition left:
        Dirichlet g (zero row, data in the source) or Neumann
        (c_{-1} = c_0 + h g, face value c_0 + h g / 2)."""
        corners = ()
        if not self.periodic:
            corner = _closure(left, {Dirichlet: -0.5, Neumann: 0.5})
            corners = (([[corner]], "top-left"),)
        return self._banded({0: 0.5, -1: 0.5}, corners)

    def _right_corners(self, right, table):
        if self.periodic:
            return ()
        value = _closure(right, table)
        return (([[value]], "bottom-right"),) if value else ()

    _WEIGHTS = {
        "divergence": {Dirichlet: lambda h: 1.0 / h, Neumann: lambda h: 1.0},
        "face_laplacian": {Dirichlet: lambda h: 1.0 / h**2, Neumann: lambda h: 1.0 / h},
        "face_to_cell": {Dirichlet: lambda h: 0.5, Neumann: lambda h: h / 2},
        "cell_to_face": {Dirichlet: lambda h: 1.0, Neumann: lambda h: h / 2},
    }

    def boundary_weight(self, block: str, condition) -> float:
        """Weight w of the data g of condition in the source of block: w g on
        the last cell or face (divergence, face_laplacian, face_to_cell, at
        x = L) or on face 0 (cell_to_face, at x = 0)."""
        if block not in self._WEIGHTS:
            raise ValueError(f"block must be one of {list(self._WEIGHTS)}.")
        return _closure(condition, self._WEIGHTS[block])(self.h)

    def __repr__(self):
        ends = "periodic" if self.periodic else "bounded"
        return f"StaggeredAxis(length={self.length}, levels={self.levels}, {ends})"


def _closure(condition, table):
    for kind, value in table.items():
        if isinstance(condition, kind):
            return value
    raise NotImplementedError(f"No staggered closure for {condition!r}.")


class StaggeredGrid:
    """Tensor product of staggered axes, in the layout of a tensap.Tensorizer
    (every location has the same layout: 2^l values per axis).

    Locations of the fields, per axis: 'p' (cells), the velocity component k
    (faces along the axis k, cells along the others; 'u', 'v', 'w' for k = 0,
    1, 2) and 'corner' (faces along every axis).

    Attributes
    ----------
    axes : list of StaggeredAxis
    ordering : str
        "blocks" or "interleaved".
    tensorizer : tensap.Tensorizer
        The layout (base 2, the levels of the axes).
    """

    _ORDERINGS = {"blocks": 1, "interleaved": 2}
    _COMPONENTS = ("u", "v", "w")

    def __init__(self, axes, ordering: str = "blocks"):
        if ordering not in self._ORDERINGS:
            raise ValueError(f"ordering must be one of {list(self._ORDERINGS)}.")
        self.axes = list(axes)
        self.ordering = ordering
        self.tensorizer = tensap.Tensorizer(2, [axis.levels for axis in self.axes],
                                            len(self.axes))
        self.tensorizer.ordering_type = self._ORDERINGS[ordering]

    @property
    def dim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple:
        return tuple(axis.size for axis in self.axes)

    def locations(self, field: str) -> tuple:
        """'cell' or 'face' along each axis for the field 'p', 'u', 'v', 'w'
        or 'corner'."""
        if field == "p":
            return ("cell",) * self.dim
        if field == "corner":
            return ("face",) * self.dim
        if field in self._COMPONENTS[: self.dim]:
            k = self._COMPONENTS.index(field)
            return tuple("face" if j == k else "cell" for j in range(self.dim))
        raise ValueError(f"Unknown field {field!r}.")

    def positions(self, field: str):
        """Coordinates of the values of the field, as arrays of the grid shape
        (indexing 'ij')."""
        coordinates = [axis.positions(location)
                       for axis, location in zip(self.axes, self.locations(field))]
        return np.meshgrid(*coordinates, indexing="ij")

    def quantize(self, values, tolerance: float = 1e-12) -> "tensap.TTTensor":
        """QTT of an array of the grid shape (any location)."""
        return tensap.quantize(np.asarray(values, dtype=float), self.tensorizer, tolerance)

    def dequantize(self, tensor) -> np.ndarray:
        """Array of the grid shape of a QTT."""
        return tensap.dequantize(tensor, self.tensorizer)

    def embed(self, operator, axis: int) -> "tensap.TTOperator":
        """A 1D operator acting on one axis of the layout."""
        return tensap.qtt_embed(operator, axis, self.tensorizer)

    # --- assembled operators ---

    def gradient(self) -> list:
        """The d components of the pressure gradient, cells -> faces: the
        operator k maps 'p' to the location of the component k."""
        return [self.embed(axis.gradient(), k) for k, axis in enumerate(self.axes)]

    def divergence(self, rights) -> list:
        """The d terms of the divergence, faces -> cells: the operator k maps
        the component k to 'p'; rights[k] is the condition of the component k
        at the end x_k = L_k (None on a periodic axis)."""
        return [self.embed(axis.divergence(right), k)
                for k, (axis, right) in enumerate(zip(self.axes, rights))]

    def pressure_laplacian(self, tolerance: float = 1e-14) -> "tensap.TTOperator":
        """sum_k D_k G_k: the cell-centred Laplacian with Neumann conditions
        on the bounded axes (circulant on the periodic ones)."""
        terms = [self.embed(axis.cell_laplacian(*(() if axis.periodic
                                                  else (Neumann(), Neumann()))), k)
                 for k, axis in enumerate(self.axes)]
        return _sum(terms, tolerance)

    def laplacian(self, field: str, conditions, tolerance: float = 1e-14) -> "tensap.TTOperator":
        """Laplacian of a field: the second difference of its location along
        each axis, conditions[k] = (left, right) the conditions of the field at
        the two ends of the axis k (the left one is not used along the axis
        where the field is on the faces: face 0 is stored)."""
        locations = self.locations(field)
        if field == "corner":
            raise ValueError("No Laplacian at the corners.")
        terms = []
        for k, (axis, location, (left, right)) in enumerate(
                zip(self.axes, locations, conditions)):
            block = (axis.face_laplacian(right) if location == "face"
                     else axis.cell_laplacian(left, right))
            terms.append(self.embed(block, k))
        return _sum(terms, tolerance)

    def __repr__(self):
        return f"StaggeredGrid(shape={self.shape}, ordering={self.ordering!r})"


def _sum(terms, tolerance):
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return tensap.Truncator(tolerance=tolerance).truncate(result)
