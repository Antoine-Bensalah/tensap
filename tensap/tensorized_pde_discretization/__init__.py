# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Discretizations of partial differential equations whose unknowns are
tensorized, to be handled by tensor methods (QTT).

- boundary_conditions: the boundary conditions of the continuous problem
  (type and data on a face), independent of the discretization;
- finite_volumes: cell-centred finite volumes on tensor-product grids.

Each discretization translates the boundary conditions into its own closures
(matrix rows and boundary sources). The operators and vectors it produces
are plain tensap.TTOperator and tensap.TTTensor objects.
"""

from .boundary_conditions import BoundaryCondition, Dirichlet, Neumann
from . import finite_volumes

__all__ = ["BoundaryCondition", "Dirichlet", "Neumann", "finite_volumes"]
