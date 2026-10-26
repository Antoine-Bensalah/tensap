# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module local_diagnostics: what the local solves of an alternating solver did.

The alternating solvers (TTTensorALSLinearSolver, TuckerTensorALSLinearSolver)
solve many small systems: directly, or iteratively (CG, GMRES) with a
tolerance and a maximal number of iterations. An iterative local solve can
stop before its tolerance; a preconditioner can be disabled when it is not
positive definite. These events are counted here and returned in the output
of the outer solver (key 'local'), so that a caller can tell a converged run
from one whose local updates were inexact.
"""

import numpy as np


class LocalDiagnostics:
    """Counters of the local solves of one outer solve.

    Attributes
    ----------
    solves : int
        Number of local solves.
    iterative : int
        Of which iterative (CG or GMRES).
    failures : int
        Iterative solves that stopped before their tolerance (info != 0).
    worst_residual : float
        Largest relative residual ||r - M s|| / ||r|| of the iterative solves.
    preconditioner_disabled : int
        Local preconditioners rejected because not positive definite.
    inner_flags : list
        Flags of nested solvers (e.g. the TT solver of a Tucker core).
    """

    def __init__(self):
        self.solves = 0
        self.iterative = 0
        self.failures = 0
        self.worst_residual = 0.0
        self.preconditioner_disabled = 0
        self.inner_flags = []

    def direct(self):
        self.solves += 1

    def iterative_solve(self, info, operator, rhs, solution):
        """Record an iterative solve: its scipy info and its true residual."""
        self.solves += 1
        self.iterative += 1
        if info != 0:
            self.failures += 1
        norm = np.linalg.norm(rhs)
        if norm > 0:
            residual = np.linalg.norm(rhs - operator @ solution) / norm
            self.worst_residual = max(self.worst_residual, float(residual))

    def as_dict(self) -> dict:
        return {"solves": self.solves, "iterative": self.iterative,
                "failures": self.failures, "worst_residual": self.worst_residual,
                "preconditioner_disabled": self.preconditioner_disabled,
                "inner_flags": list(self.inner_flags)}
