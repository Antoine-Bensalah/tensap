# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Cell-centred finite volumes on tensor-product grids, in QTT format.

Axis describes one axis of the domain (length, number of digits, boundary
conditions); Grid assembles the axes in the layout of a tensap.Tensorizer.
StaggeredAxis and StaggeredGrid: the staggered (MAC) grid, pressure at the
cells and velocity components at the faces.
"""

from .grid import Axis, Grid
from .staggered import StaggeredAxis, StaggeredGrid

__all__ = ["Axis", "Grid", "StaggeredAxis", "StaggeredGrid"]
