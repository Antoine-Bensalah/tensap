Linear solvers
==============

.. currentmodule:: tensap

Iterative solvers of linear systems ``A @ x = b`` whose operator, right-hand
side and solution are in a low-rank tensor format, and approximate inverses
used as preconditioners.

Sub-package: ``tensap.solvers``.

Iterations with truncation
--------------------------

.. autosummary::
   :toctree: generated/
   :template: class

   LinearSolver
   TruncatedRichardsonIterations
   TruncatedPCG

Greedy and alternating solvers
------------------------------

.. autosummary::
   :toctree: generated/
   :template: class

   RankOneALSLinearSolver
   GreedyLinearSolver
   TuckerTensorALSLinearSolver
   TTTensorALSLinearSolver

Approximate inverses
--------------------

.. autosummary::
   :toctree: generated/
   :template: class

   RankOneInverseApproximatorByALS
   GreedyInverseApproximator
