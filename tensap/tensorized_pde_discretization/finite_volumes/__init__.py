# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Cell-centred finite volumes on tensor-product grids, in QTT format.

Axis describes one axis of the domain (length, number of digits, boundary
conditions); Grid assembles the axes in the layout of a tensap.Tensorizer.
"""

from .grid import Axis, Grid

__all__ = ["Axis", "Grid"]
