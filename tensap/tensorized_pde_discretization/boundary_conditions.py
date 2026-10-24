# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module boundary_conditions: boundary conditions of the continuous problem.

A boundary condition is a type and data on one face of the domain:

- Dirichlet(g): u = g on the face;
- Neumann(g): du/dn = g on the face, n the outward unit normal.

The data g is a scalar, a function of the points of the face, or (later)
a QTT tensor on the grid of the face. Nothing here depends on a grid or a
discretization: each discretization translates a condition into its own
closure. Periodicity is not a condition on one face but a property of an
axis (its two faces are identified): it is declared on the axis.
"""


class BoundaryCondition:
    """Base class: the data g of the condition on a face.

    Attributes
    ----------
    value : float or callable
        The data g: a scalar, or a function of the points of the face (an
        array of shape (n, dim - 1), returning n values). The default is 0
        (homogeneous condition).
    """

    def __init__(self, value=0.0):
        self.value = value

    @property
    def is_homogeneous(self) -> bool:
        """True if the data is the scalar 0."""
        return not callable(self.value) and self.value == 0

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


class Dirichlet(BoundaryCondition):
    """Dirichlet condition u = g on a face."""


class Neumann(BoundaryCondition):
    """Neumann condition du/dn = g on a face (n: outward unit normal)."""
