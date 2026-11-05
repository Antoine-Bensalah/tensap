Discretization of PDEs
======================

.. currentmodule:: tensap.tensorized_pde_discretization

Discretizations of partial differential equations whose operators and
vectors are :class:`~tensap.TTOperator` and :class:`~tensap.TTTensor`
objects in QTT format.

Sub-package: ``tensap.tensorized_pde_discretization``.

Boundary conditions
-------------------

.. autosummary::
   :toctree: generated/
   :template: class

   BoundaryCondition
   Dirichlet
   Neumann

Finite volumes
--------------

.. currentmodule:: tensap.tensorized_pde_discretization.finite_volumes

Cell-centred finite volumes, and finite volumes on a staggered (MAC) grid.

.. autosummary::
   :toctree: generated/
   :template: class

   Axis
   Grid
   StaggeredAxis
   StaggeredGrid
