# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module linear_solvers.

Iterative linear solvers Ax=b for operators and right-hand sides in tensor
format. Port of ApproximationToolbox/Approximation/TensorApproximation/
EquationsSolvers/LinearSolver.m and TruncatedRichardsonIterations.m.
"""

import abc


class LinearSolver(abc.ABC):
    """
    Abstract base class for iterative solvers of a tensor-structured linear
    system A @ x = b.

    Attributes
    ----------
    A : object
        Operator in tensor format.
    b : object
        Right-hand side in tensor format.
    P : object, optional
        Preconditioner in tensor format.
    x0 : object, optional
        Initial guess.
    max_iterations : int
    tolerance : float
    stagnation : float
    display : bool
    error_criterion : callable, optional
        Function taking the current iterate x and returning the error.
    """

    def __init__(
        self,
        A,
        b,
        P=None,
        x0=None,
        max_iterations=30,
        tolerance=1e-6,
        stagnation=1e-6,
        display=True,
        error_criterion=None,
    ):
        self.A = A
        self.b = b
        self.P = P
        self.x0 = x0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.stagnation = stagnation
        self.display = display
        self.error_criterion = error_criterion

    @abc.abstractmethod
    def solve(self):
        """Solve the linear system.

        Returns
        -------
        x : object
            The computed solution, in tensor format.
        output : dict
            Dictionary with keys 'flag', 'iter', 'error', 'errvec'.
        """


class TruncatedRichardsonIterations(LinearSolver):
    """
    Solve A @ x = b using (preconditioned) Richardson iterations, truncating
    the iterate at each step to keep the tensor representation rank under
    control.

    Attributes
    ----------
    truncator : tensap.Truncator
        Used to truncate the iterate after each update.
    step_size : float
        Relaxation parameter (rho) of the Richardson iteration.
    """

    def __init__(self, A, b, truncator, step_size=1.0, **kwargs):
        super().__init__(A, b, **kwargs)
        self.truncator = truncator
        self.step_size = step_size

    def solve(self):
        x = self.b * 0.0 if self.x0 is None else self.x0

        if self.error_criterion is None:
            b_norm = self.b.norm()

            def error_criterion(x):
                return (self.b - self.A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        flag = 1
        errvec = []
        k = 0
        for k in range(1, self.max_iterations + 1):
            x_old = x
            residual = self.b - self.A @ x
            direction = self.P @ residual if self.P is not None else residual

            x = self.truncator.truncate(x + self.step_size * direction)

            stagnation = (x - x_old).norm() / x.norm()
            error = error_criterion(x)
            errvec.append(error)

            if self.display:
                print(
                    f"Iteration {k:3d} - Stagnation {stagnation:.2e} "
                    f"- Error {error:.2e}"
                )

            if stagnation < self.stagnation:
                flag = 2
                break
            if error < self.tolerance:
                flag = 0
                break

        output = {
            "flag": flag,
            "iter": k,
            "error": errvec[-1],
            "errvec": errvec,
        }
        return x, output
