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

    def pressure_laplacian(self, conditions=None,
                           tolerance: float = 1e-14) -> "tensap.TTOperator":
        """The pressure Laplacian L_p = sum_k D_k G_k: the cell-centred
        Laplacian with the conditions of the pressure, conditions[k] = (left,
        right) on the axis k (default: homogeneous Neumann on the bounded axes,
        the walls; circulant on the periodic ones). A Dirichlet condition (p
        given at an outlet face x_k = L_k) comes from the gradient at that face,
        (p_ghost - p_{N-1}) / h with p_ghost = 2 g - p_{N-1}, in the divergence
        of the corrected velocity."""
        conditions = self._pressure_conditions(conditions)
        terms = [self.embed(axis.cell_laplacian(*condition), k)
                 for k, (axis, condition) in enumerate(zip(self.axes, conditions))]
        return _sum(terms, tolerance)

    def _pressure_conditions(self, conditions):
        if conditions is None:
            conditions = [(None, None) if axis.periodic else (Neumann(), Neumann())
                          for axis in self.axes]
        return [(None, None) if axis.periodic else tuple(condition)
                for axis, condition in zip(self.axes, conditions)]

    def pressure_is_singular(self, conditions=None) -> bool:
        """True if the constants are in the kernel of the pressure Laplacian
        (no Dirichlet condition on the pressure)."""
        conditions = self._pressure_conditions(conditions)
        return not any(isinstance(c, Dirichlet) for pair in conditions for c in pair)

    def interior_mask(self, field: str) -> "tensap.TTOperator":
        """Diagonal operator that zeroes the stored boundary values of a field:
        face 0 along every bounded axis where the field is on the faces (the
        known normal velocity on a wall or an inflow). Left-multiplying an
        operator by it keeps those values unchanged in a time step."""
        mask = None
        for k, (axis, location) in enumerate(zip(self.axes, self.locations(field))):
            if location != "face" or axis.periodic:
                continue
            block = tensap.qtt_identity(axis.levels) + tensap.qtt_corner(
                np.array([[-1.0]]), axis.levels, "top-left")
            block = self.embed(tensap.Truncator(tolerance=1e-14).truncate(block), k)
            mask = block if mask is None else mask @ block
        if mask is None:
            return self.embed(tensap.qtt_identity(self.axes[0].levels), 0)
        return mask

    def laplacian(self, field: str, conditions, tolerance: float = 1e-14) -> "tensap.TTOperator":
        """Laplacian of a field: the second difference of its location along
        each axis, conditions[k] = (left, right) the conditions of the field at
        the two ends of the axis k (the left one is not used along the axis
        where the field is on the faces: face 0 is stored). The rows of the
        stored boundary faces are zero (interior_mask): the second differences
        along the other axes do not act on the known values there."""
        locations = self.locations(field)
        if field == "corner":
            raise ValueError("No Laplacian at the corners.")
        terms = []
        for k, (axis, location, (left, right)) in enumerate(
                zip(self.axes, locations, conditions)):
            block = (axis.face_laplacian(right) if location == "face"
                     else axis.cell_laplacian(left, right))
            terms.append(self.embed(block, k))
        operator = _sum(terms, tolerance)
        if any(location == "face" and not axis.periodic
               for axis, location in zip(self.axes, locations)):
            operator = tensap.Truncator(tolerance=tolerance).truncate(
                self.interior_mask(field) @ operator)
        return operator

    def laplacian_source(self, field: str, conditions,
                         tolerance: float = 1e-12) -> "tensap.TTTensor":
        """The source s of the boundary data of the Laplacian of a field:
        Delta f ~ laplacian(field, conditions) f + s.

        conditions[k] = (left, right) as in laplacian, with their data g: a
        scalar, or a function of the points of the face (array of shape
        (n, dim - 1), the coordinates of the other axes at the locations of the
        field). Along an axis where the field is on the cells, the data of both
        ends enter the first and last cells (weights 2 / h^2 Dirichlet, 1 / h
        Neumann); along the axis where it is on the faces, only the right end
        (face 0 is stored), on the last face (boundary_weight of
        face_laplacian). Built in QTT, face by face: the data are sampled on
        the faces only; the stored boundary faces are masked.
        """
        locations = self.locations(field)
        result = None
        for k, (axis, location, pair) in enumerate(zip(self.axes, locations, conditions)):
            if axis.periodic:
                continue
            for side, condition in zip(("left", "right"), pair):
                if condition is None or condition.is_homogeneous:
                    continue
                if location == "face":
                    if side == "left":
                        continue
                    weight = axis.boundary_weight("face_laplacian", condition)
                else:
                    cells = Axis(axis.length, axis.levels, *(
                        (condition, Neumann()) if side == "left" else (Neumann(), condition)))
                    weight = cells.boundary_weight(side)
                index = 0 if side == "left" else axis.size - 1
                vector = tensap.qtt_elementary_vector(axis.levels, index) * weight
                face = self._face_data(field, k, condition.value, tolerance)
                term = tensap.qtt_insert_axis(face, k, vector, self.tensorizer)
                result = term if result is None else result + term
        if result is None:
            return tensap.TTTensor([np.zeros((1, 2, 1))] * int(self.tensorizer.levels.sum()))
        result = tensap.Truncator(tolerance=tolerance).truncate(result)
        return tensap.Truncator(tolerance=tolerance).truncate(self.interior_mask(field) @ result)

    def _face_data(self, field, axis, g, tolerance):
        """The data g on the face normal to axis, at the locations of the field
        along the other axes, as a QTT on the face (a float in 1D)."""
        if self.dim == 1:
            if callable(g):
                raise TypeError("In dimension 1, the data of a face is a scalar.")
            return float(g)
        others = [j for j in range(self.dim) if j != axis]
        layout = tensap.Tensorizer(2, [self.axes[j].levels for j in others], len(others))
        layout.ordering_type = self.tensorizer.ordering_type
        if not callable(g):
            return tensap.TTTensor([np.ones((1, 2, 1))] * int(layout.levels.sum())) * float(g)
        locations = self.locations(field)
        coordinates = [self.axes[j].positions(locations[j]) for j in others]
        points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
        values = np.asarray(g(points.reshape(-1, len(others))), dtype=float)
        return tensap.quantize(values.reshape(points.shape[:-1]), layout, tolerance)

    def convection(self, velocity, conditions=None, truncator=None) -> list:
        """Convective term in conservative (divergence) form, N_j = sum_k
        d_k (u_j u_k) at the location of the component j (Harlow and Welch):

        - k = j: the component averaged to the cells along k, squared, and
          differenced back to the faces (the gradient block, cells -> faces);
        - k != j: u_j averaged along k and u_k averaged along j to the corners
          of the plane (j, k) (cell_to_face), their product, and the
          divergence block along k (faces -> cells).

        Parameters
        ----------
        velocity : list of tensap.TTTensor
            The d components, at their faces.
        conditions : list, optional
            conditions[j][k] = (left, right), the conditions of the component j
            at the ends of the axis k (as in laplacian; default homogeneous
            Dirichlet, walls; ignored on the periodic axes). Only homogeneous
            data are supported here (a wall, an outflow); the data of a moving
            wall enter as a source (later).
        truncator : tensap.Truncator, optional
            Rounding of the products and of the result (default tolerance
            1e-10, maximal rank 100).

        Returns
        -------
        list of tensap.TTTensor
            The d components of N, at the faces; zero on the stored boundary
            faces (interior_mask).
        """
        d = self.dim
        if conditions is None:
            conditions = [[(Dirichlet(), Dirichlet())] * d for _ in range(d)]
        for row, axis_k in zip(zip(*conditions), self.axes):
            if axis_k.periodic:
                continue
            for pair in row:
                if any(not c.is_homogeneous for c in pair if c is not None):
                    raise NotImplementedError("convection: only homogeneous data.")
        if truncator is None:
            truncator = tensap.Truncator(tolerance=1e-10, max_rank=100)

        def condition(j, k, side):
            return None if self.axes[k].periodic else conditions[j][k][side]

        block = self._embedded_block

        terms = [[] for _ in range(d)]
        for k, axis_k in enumerate(self.axes):
            uk = velocity[k]
            # k = j: d_k (u_k u_k), at the cells along k
            mean = block("face_to_cell", k, condition(k, k, 1)) @ uk
            square = truncator.hadamard_product(mean, mean)
            terms[k].append(block("gradient", k, None) @ square)
            for j in range(d):
                if j == k:
                    continue
                # k != j: d_k (u_j u_k), at the corners of the plane (j, k)
                uj_mean = block("cell_to_face", k, condition(j, k, 0)) @ velocity[j]
                uk_mean = block("cell_to_face", j, condition(k, j, 0)) @ uk
                product = truncator.hadamard_product(uj_mean, uk_mean)
                terms[j].append(block("divergence", k, condition(j, k, 1)) @ product)
        result = []
        for j, field in enumerate(self._COMPONENTS[:d]):
            total = terms[j][0]
            for term in terms[j][1:]:
                total = total + term
            total = truncator.truncate(total)
            if any(location == "face" and not axis.periodic
                   for axis, location in zip(self.axes, self.locations(field))):
                total = truncator.truncate(self.interior_mask(field) @ total)
            result.append(total)
        return result

    def _embedded_block(self, name: str, axis: int, condition):
        """A 1D block embedded on an axis, cached (the convective term is
        evaluated at every time step; the blocks only depend on the kind of
        the condition, the data being homogeneous)."""
        key = (name, axis, type(condition).__name__)
        cache = self.__dict__.setdefault("_blocks", {})
        if key not in cache:
            method = getattr(self.axes[axis], name)
            cache[key] = self.embed(method() if name == "gradient" else method(condition), axis)
        return cache[key]

    def curl(self) -> list:
        """Discrete curl in 2D, corners -> faces: u = d psi / dy, v = -d psi / dx
        for a stream function psi at the corners, by the divergence blocks
        (faces -> cells) along the other axis; psi vanishes at the ends of the
        bounded axes. Then sum_k D_k (curl psi)_k = 0 exactly (the blocks of
        different axes commute)."""
        if self.dim != 2:
            raise ValueError("The curl of a stream function is defined in 2D.")
        dx, dy = (axis.divergence(None if axis.periodic else Dirichlet())
                  for axis in self.axes)
        return [self.embed(dy, 1), self.embed(dx, 0) * -1.0]

    def constant_mode_regularization(self, sigma: float = None) -> "tensap.TTOperator":
        """The rank-one term R = (sigma / N) 1 1^T that makes minus the pressure
        Laplacian (singular: Neumann or periodic on every axis) symmetric
        positive definite: -L_p + R has the eigenvalues of -L_p with 0 replaced
        by sigma. sigma is by default the smallest non-zero eigenvalue of -L_p
        (as for the cell-centred grids,
        finite_volumes.Grid.constant_mode_regularization). To be added to
        -1.0 * pressure_laplacian(), not to pressure_laplacian(): L_p + R is
        invertible but indefinite."""
        if sigma is None:
            sigma = min(
                Axis(axis.length, axis.levels, periodic=True).smallest_nonzero_eigenvalue()
                if axis.periodic else
                Axis(axis.length, axis.levels, Neumann(), Neumann()).smallest_nonzero_eigenvalue()
                for axis in self.axes)
        N = int(np.prod(self.shape))
        return tensap.qtt_ones_matrix(int(self.tensorizer.levels.sum())) * (sigma / N)

    def _ones(self) -> "tensap.TTTensor":
        return tensap.TTTensor([np.ones((1, 2, 1))] * int(self.tensorizer.levels.sum()))

    def _last_slice(self, axis: int) -> "tensap.TTOperator":
        """Diagonal operator keeping the index N - 1 along axis."""
        block = tensap.qtt_corner(np.array([[1.0]]), self.axes[axis].levels, "bottom-right")
        return self.embed(block, axis)

    def _check_projection_conditions(self, rights, pressure):
        """The supported pairs (velocity component k at x_k = L_k, pressure
        at x_k = 0 and x_k = L_k), all homogeneous: a wall or an inflow at
        x_k = 0 (the face 0 is stored) with a Neumann pressure; at x_k = L_k,
        a wall (Dirichlet velocity) with a Neumann pressure, or an outflow
        (Neumann velocity) with a Dirichlet pressure (p = 0, an outlet)."""
        for k, (axis, right, (p_left, p_right)) in enumerate(zip(self.axes, rights, pressure)):
            if axis.periodic:
                continue
            for c in (right, p_left, p_right):
                if c is None or not c.is_homogeneous:
                    raise NotImplementedError(
                        f"project: axis {k}: only homogeneous conditions are supported, "
                        f"got {right!r}, ({p_left!r}, {p_right!r}).")
            if not isinstance(p_left, Neumann):
                raise NotImplementedError(
                    f"project: axis {k}: the pressure at x = 0 must be Neumann (the "
                    f"face 0 is stored, its velocity known), got {p_left!r}.")
            outlet = isinstance(right, Neumann)
            if outlet != isinstance(p_right, Dirichlet):
                raise ValueError(
                    f"project: axis {k}: an outflow velocity (Neumann) needs a Dirichlet "
                    f"pressure and a wall (Dirichlet) a Neumann one, got {right!r} and "
                    f"{p_right!r}.")

    def divergence_with_outlet(self, velocity, rights, outlet=None) -> "tensap.TTTensor":
        """The divergence of a face velocity whose face N is given on the
        outlet axes: outlet[k] (a QTT on the grid, zero except at the index
        N - 1 along k, where it holds the value of the face N) replaces the
        closure of rights[k]. On the other axes, the closures of rights. This is
        the divergence of the corrected velocity of project, outlet included."""
        outlet = outlet or [None] * self.dim
        total = None
        for k, (axis, w, right, face) in enumerate(zip(self.axes, velocity, rights, outlet)):
            if face is None:
                term = self.embed(axis.divergence(right), k) @ w
            else:
                term = self.embed(axis.divergence(Dirichlet()), k) @ w + face * (1.0 / axis.h)
            total = term if total is None else total + term
        return tensap.Truncator(tolerance=1e-14).truncate(total)

    def project(self, velocity, rights=None, truncator=None, pressure=None,
                compatibility_tolerance: float = None, return_info: bool = False,
                **solver_options):
        """Helmholtz--Hodge decomposition of a face velocity w = u + G phi with
        D u = 0: solve -L_p phi = -D w (minus the pressure Laplacian, SPD;
        regularized by the rank-one term R when singular), by
        TTTensorALSLinearSolver (AMEn by default), then u = w - G phi.

        Parameters
        ----------
        velocity : list of tensap.TTTensor
            The d components of w, at their faces.
        rights : list, optional
            The condition of each component at the end x_k = L_k (default:
            homogeneous Dirichlet, walls; None on a periodic axis).
        truncator : tensap.Truncator, optional
            Truncation of the solver and of the results (default tolerance
            1e-10, maximal rank 100).
        pressure : list, optional
            Conditions of the pressure, (left, right) per axis
            (pressure_laplacian); default: homogeneous Neumann (walls).
            Supported pairs (_check_projection_conditions): homogeneous data,
            Neumann pressure at x_k = 0 and at a wall, Dirichlet pressure (p =
            0) at an outflow (Neumann velocity). Then the problem is not
            singular and the face N of the outlet is corrected,
            u_N = u*_N - (G phi)_N = w_{N-1} + 2 phi_{N-1} / h (returned in the
            info).
        compatibility_tolerance : float, optional
            Singular problem (no outlet): the mean of D w, scaled by h / rms(w),
            must not exceed it (the boundary fluxes of w must balance), else
            ValueError. Below it, the mean is removed: phi has zero mean and the
            divergence of u is the removed mean (reported). The default,
            max(1e-8, 10 * truncator.tolerance), accepts the defect left by the
            rounding of w (a truncation at 1e-2 leaves about 5e-6), and rejects
            a physical imbalance of the fluxes.
        return_info : bool, optional
            Also return a dict: 'outlet' (per axis, None or the QTT of the face
            N at the index N - 1, see divergence_with_outlet), 'divergence' (the
            QTT of the divergence of u, outlet included), 'compatibility_defect',
            'solver' (the output of the pressure solve).
        solver_options
            Passed to TTTensorALSLinearSolver (default algorithm 'amen',
            tolerance 1e-12, stagnation 1e-10, 30 iterations, no display).

        Returns
        -------
        u : list of tensap.TTTensor
            The divergence-free part.
        phi : tensap.TTTensor
            The potential (of zero mean when the problem is singular).
        info : dict
            If return_info.
        """
        if rights is None:
            rights = [None if axis.periodic else Dirichlet() for axis in self.axes]
        pressure = self._pressure_conditions(pressure)
        self._check_projection_conditions(rights, pressure)
        if truncator is None:
            truncator = tensap.Truncator(tolerance=1e-10, max_rank=100)
        if compatibility_tolerance is None:
            compatibility_tolerance = max(1e-8, 10 * truncator.tolerance)
        options = {"algorithm": "amen", "tolerance": 1e-12, "stagnation": 1e-10,
                   "max_iterations": 30, "display": False}
        options.update(solver_options)
        divergence = self.divergence(rights)
        rhs = truncator.truncate(_sum([D @ w for D, w in zip(divergence, velocity)], 1e-14))
        operator = self.pressure_laplacian(pressure) * -1.0
        defect = 0.0
        if self.pressure_is_singular(pressure):
            ones = self._ones()
            N = int(np.prod(self.shape))
            mean = rhs.dot(ones) / N
            rms = np.sqrt(sum(w.dot(w) for w in velocity) / (self.dim * N))
            scale = min(axis.h for axis in self.axes) / rms if rms > 0 else 0.0
            defect = abs(mean) * scale
            if defect > compatibility_tolerance:
                raise ValueError(
                    f"project: incompatible boundary fluxes, mean of D w = {mean:.3e} "
                    f"(scaled defect {defect:.1e} > {compatibility_tolerance:.1e}).")
            if mean != 0:
                rhs = truncator.truncate(rhs - ones * mean)
            operator = operator + self.constant_mode_regularization()
        phi, output = tensap.TTTensorALSLinearSolver(
            operator, rhs * -1.0, truncator=truncator, **options).solve()
        u = [truncator.truncate(w - G @ phi) for w, G in zip(velocity, self.gradient())]
        if not return_info:
            return u, phi
        outlet = [None] * self.dim
        for k, (axis, right) in enumerate(zip(self.axes, rights)):
            if not axis.periodic and isinstance(right, Neumann):
                outlet[k] = truncator.truncate(
                    self._last_slice(k) @ (velocity[k] + phi * (2.0 / axis.h)))
        info = {"outlet": outlet,
                "divergence": self.divergence_with_outlet(u, rights, outlet),
                "compatibility_defect": defect, "solver": output}
        return u, phi, info

    def __repr__(self):
        return f"StaggeredGrid(shape={self.shape}, ordering={self.ordering!r})"


def _sum(terms, tolerance):
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return tensap.Truncator(tolerance=tolerance).truncate(result)
