# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module linear_solvers.

Iterative linear solvers Ax=b for operators and right-hand sides in tensor
format. Port of ApproximationToolbox/Approximation/TensorApproximation/
EquationsSolvers/LinearSolver.m, TruncatedRichardsonIterations.m and
TruncatedPCG.m.
"""

import abc

import numpy as np


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

    # --- shared conventions of the iterative solvers ---
    # flag 0: tolerance reached (also at iteration 0, for an exact initial
    # guess or b = 0); 1: maximal number of iterations; 2: stagnation
    # (tested after the tolerance); 3: breakdown (non-positive curvature).

    def _default_error_criterion(self):
        """The relative residual ||b - A x|| / ||b||; the absolute residual
        when b = 0."""
        if self.error_criterion is not None:
            return self.error_criterion
        b_norm = self.b.norm()
        scale = b_norm if b_norm > 0 else 1.0

        def error_criterion(x):
            return (self.b - self.A @ x).norm() / scale

        return error_criterion

    @staticmethod
    def _stagnation(x, x_old):
        """||x - x_old|| / ||x||, 0 when both vanish."""
        difference = (x - x_old).norm()
        norm = x.norm()
        if norm > 0:
            return difference / norm
        return 0.0 if difference == 0 else np.inf

    @staticmethod
    def _output(flag, k, errvec, initial_error):
        """errvec: the errors of the iterations (empty if none was needed);
        'error' the last one, or the initial error."""
        error = errvec[-1] if errvec else initial_error
        return {"flag": flag, "iter": k, "error": error, "errvec": errvec}


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
        error_criterion = self._default_error_criterion()
        initial_error = error_criterion(x)
        errvec = []
        if initial_error < self.tolerance:
            return x, self._output(0, 0, errvec, initial_error)

        flag = 1
        k = 0
        for k in range(1, self.max_iterations + 1):
            x_old = x
            residual = self.b - self.A @ x
            direction = self.P @ residual if self.P is not None else residual

            x = self.truncator.truncate(x + self.step_size * direction)

            stagnation = self._stagnation(x, x_old)
            error = error_criterion(x)
            errvec.append(error)

            if self.display:
                print(
                    f"Iteration {k:3d} - Stagnation {stagnation:.2e} "
                    f"- Error {error:.2e}"
                )

            if error < self.tolerance:
                flag = 0
                break
            if stagnation < self.stagnation:
                flag = 2
                break

        return x, self._output(flag, k, errvec, initial_error)


class TruncatedPCG(LinearSolver):
    """
    Solve A @ x = b by the (preconditioned) conjugate gradient method, the
    iterate, the residual and the search direction being truncated at each
    step to keep the tensor representation rank under control. Port of
    TruncatedPCG.m.

    The residual is recomputed at each step as truncate(b - A @ x), and the
    new direction is A-orthogonalized against the previous one only.

    Attributes
    ----------
    truncator : tensap.Truncator
        Used to truncate the iterate, the residual and the direction.
    """

    def __init__(self, A, b, truncator, **kwargs):
        super().__init__(A, b, **kwargs)
        self.truncator = truncator

    def solve(self):
        truncate = self.truncator.truncate
        A, b, P = self.A, self.b, self.P
        x = b * 0.0 if self.x0 is None else self.x0
        error_criterion = self._default_error_criterion()
        initial_error = error_criterion(x)
        errvec = []
        if initial_error < self.tolerance:
            return x, self._output(0, 0, errvec, initial_error)

        r = b - A @ x
        z = P @ r if P is not None else r
        p = z
        Ap = A @ p
        pAp = p.dot(Ap)

        flag = 1
        k = 0
        for k in range(1, self.max_iterations + 1):
            if not pAp > 0:
                # A (or P) is not positive definite on the direction: the CG
                # step is undefined
                flag = 3
                break
            x_old = x
            omega = r.dot(p) / pAp
            x = truncate(x + omega * p)
            stagnation = self._stagnation(x, x_old)
            r = truncate(b - A @ x)
            errvec.append(error_criterion(x))

            if self.display:
                print(
                    f"Iteration {k:3d} - Stagnation {stagnation:.2e} "
                    f"- Error {errvec[-1]:.2e}"
                )

            if errvec[-1] < self.tolerance:
                flag = 0
                break
            if stagnation < self.stagnation:
                flag = 2
                break

            z = P @ r if P is not None else r
            beta = -z.dot(Ap) / pAp
            p = truncate(z + beta * p)
            Ap = A @ p
            pAp = p.dot(Ap)

        return x, self._output(flag, k, errvec, initial_error)


def _contract_except_dim(core, other, matrices, mu):
    """core.tensordot_matrix_product_except_dim as a numpy array.

    FullTensor returns a FullTensor and DiagonalTensor a numpy array.
    """
    result = core.tensordot_matrix_product_except_dim(other, matrices, mu)
    return result.numpy() if hasattr(result, "numpy") else np.asarray(result)
