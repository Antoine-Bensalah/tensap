# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module linear_solvers.

Iterative linear solvers Ax=b for operators and right-hand sides in tensor
format. Port of ApproximationToolbox/Approximation/TensorApproximation/
EquationsSolvers/LinearSolver.m, TruncatedRichardsonIterations.m,
RankOneALSLinearSolver.m and GreedyLinearSolver.m.
"""

import abc
import time

import numpy as np
import tensap


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


def _direct_local_solver(M, rhs, mu):
    """Default local solver of RankOneALSLinearSolver: dense direct solve."""
    return np.linalg.solve(M, rhs)


class RankOneALSLinearSolver(LinearSolver):
    """
    Compute a rank-one approximation x = lambda w^(1) x ... x w^(d) of the
    solution of A @ x = b by alternating least squares (ALS): for each
    dimension mu, the other factors are fixed and the Galerkin condition gives
    a small linear system M_mu w^(mu) = f_mu of size n_mu.

    The operator A is a TuckerTensorOperator and b a TuckerTensorVector, with
    any core format supported by convert_tensors.

    Attributes
    ----------
    local_solver : callable
        Function (M, rhs, mu) -> solution of the local system M w = rhs.
        Default is a dense direct solve.
    adjoint : bool
        If True, use test functions M_mu^{-T} w^(mu) (Petrov-Galerkin) on
        the dimensions adjoint_dimensions, instead of w^(mu) (Galerkin).
    adjoint_dimensions : list of int
        Dimensions on which the adjoint test functions are used.
    """

    def __init__(
        self,
        A,
        b,
        local_solver=_direct_local_solver,
        adjoint=False,
        adjoint_dimensions=None,
        stagnation=5e-2,
        max_iterations=10,
        **kwargs,
    ):
        super().__init__(
            A, b, stagnation=stagnation, max_iterations=max_iterations, **kwargs
        )
        self.local_solver = local_solver
        self.adjoint = adjoint
        self.adjoint_dimensions = (
            list(range(A.order)) if adjoint_dimensions is None else adjoint_dimensions
        )

    def solve(self):
        order = self.b.order
        A_core, A_frames = self.A.core, self.A.tensor_frame
        b_core, b_frames = self.b.core, self.b.tensor_frame

        # initialization: random normalized factors and unit core
        if self.x0 is None:
            sizes = [shape[0] for shape in self.b.shape]
            x_frames = tensap.TFrameVector.randn(sizes, [1] * order)
            x_core = tensap.DiagonalTensor(np.array([1.0]), order)
        else:
            x_frames = self.x0.tensor_frame
            x_core = self.x0.core
        test_frames = x_frames

        # reduced quantities w^T A_k w (R_A x 1) and b_l^T w (R_b x 1)
        xtAx = [
            m[:, :, 0] for m in test_frames.dot_with_metrics(x_frames, A_frames)
        ]
        xtb = b_frames.dot(test_frames)

        lambda_old = 1.0
        flag = 1
        k = 0
        for k in range(1, self.max_iterations + 1):
            for mu in range(order):
                # reduced operator M_mu = sum_k alpha_k A^(mu)_k
                core_a, core_x = tensap.convert_tensors(A_core, x_core)
                coef = core_a.tensordot_matrix_product_except_dim(core_x, xtAx, mu)
                M = A_frames.eval_in_space(mu, np.ravel(coef))

                # reduced right-hand side f_mu = sum_l beta_l b^(mu)_l
                core_b, core_x = tensap.convert_tensors(b_core, x_core)
                coef = core_b.tensordot_matrix_product_except_dim(core_x, xtb, mu)
                rhs = b_frames.eval_in_space(mu, np.ravel(coef))

                # local solve, normalization of the factor into the core
                sol = self.local_solver(M, rhs, mu)
                lambda_ = np.linalg.norm(sol)
                w = tensap.FrameVector((sol / lambda_).reshape(1, -1, 1))
                frames = list(x_frames.frames)
                frames[mu] = w
                x_frames = tensap.TFrameVector(frames)
                x_core = tensap.DiagonalTensor(np.array([lambda_]), order)

                if self.adjoint and mu in self.adjoint_dimensions:
                    w_test = self.local_solver(M.T, w.data[0], mu)
                    w_test = tensap.FrameVector(w_test.reshape(1, -1, 1))
                else:
                    w_test = w
                frames = list(test_frames.frames)
                frames[mu] = w_test
                test_frames = tensap.TFrameVector(frames)

                # update the reduced quantities of dimension mu only
                xtAx[mu] = w_test.dot_with_metric(w, A_frames.frames[mu])[:, :, 0]
                xtb[mu] = b_frames.frames[mu].dot(w_test)

            stagnation = abs(lambda_old - lambda_) / (lambda_old + lambda_)
            lambda_old = lambda_

            if self.display:
                print(f"Iteration {k:3d} - Stagnation {stagnation:.2e}")

            if k > 1 and stagnation < self.stagnation:
                flag = 2
                break

        output = {"flag": flag, "iter": k}
        return tensap.TuckerTensorVector(x_core, x_frames), output


class GreedyLinearSolver(LinearSolver):
    """
    Greedy (PGD-like) solver of A @ x = b: the solution is built by successive
    rank-one corrections,

        x_m = update(x_{m-1} + c_m),   c_m ~ solution of A @ c = b - A @ x_{m-1},

    where each correction c_m is computed by local_solver.

    Attributes
    ----------
    local_solver : LinearSolver
        Solver computing the correction from the current residual (its
        attribute b is overwritten). Default is a RankOneALSLinearSolver.
    update : callable
        Function x -> updated x applied after each correction (from the
        second iteration on). Default is the identity.
    minimize_residual : bool
        If True, apply the algorithm to the normal equations
        A^T A x = A^T b.
    check_residual : int
        The error is computed every check_residual iterations.
    initial_update : bool
        If True and x0 is given, apply update to x0 before the iterations.
    """

    def __init__(
        self,
        A,
        b,
        local_solver=None,
        update=None,
        minimize_residual=False,
        check_residual=1,
        initial_update=False,
        **kwargs,
    ):
        super().__init__(A, b, **kwargs)
        self.local_solver = (
            RankOneALSLinearSolver(A, b, display=False)
            if local_solver is None
            else local_solver
        )
        self.update = (lambda x: x) if update is None else update
        self.minimize_residual = minimize_residual
        self.check_residual = check_residual
        self.initial_update = initial_update

    def solve(self):
        flag = 1
        clock0 = time.perf_counter()

        if not self.minimize_residual:
            A, b = self.A, self.b
        else:
            A = self.A.T @ self.A
            b = self.A.T @ self.b
            self.local_solver.A = A
            self.local_solver.b = b

        if self.error_criterion is None:
            b_norm = self.b.norm()

            def error_criterion(x):
                return (self.b - self.A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        if self.x0 is None:
            x = None
            residual = b
        else:
            x = self.x0
            if self.initial_update:
                x = self.update(x)
                initial_error = error_criterion(x)
                # no stagnation criterion for the initial update
                if initial_error < self.tolerance:
                    if self.display:
                        print(
                            "Greedy: Converged on initialization "
                            f"- Error {initial_error:.2e}"
                        )
                    elapsed = time.perf_counter() - clock0
                    output = {
                        "flag": 0,
                        "iter": 0,
                        "error": initial_error,
                        "errvec": [initial_error],
                        "times": [elapsed],
                        "time": elapsed,
                    }
                    return x, output
            residual = b - A @ x

        errvec = []
        times = []
        error = None
        k = 0
        for k in range(1, self.max_iterations + 1):
            self.local_solver.b = residual
            correction, _ = self.local_solver.solve()

            x_old = x
            x = correction if x is None else x + correction
            if k > 1 or (self.initial_update and self.x0 is not None):
                x = self.update(x)

            stagnation = 1.0 if x_old is None else (x - x_old).norm() / x.norm()
            error = error_criterion(x) if k % self.check_residual == 0 else None
            errvec.append(error)
            times.append(time.perf_counter() - clock0)

            if self.display:
                message = f"Greedy: Iteration {k:3d} - Stagnation {stagnation:.2e}"
                if error is not None:
                    message += f" - Error {error:.2e}"
                print(message)

            if stagnation < self.stagnation:
                flag = 2
                break
            if error is not None and error < self.tolerance:
                flag = 0
                break

            residual = b - A @ x

        # compute the error of the last iteration if needed
        if error is None:
            error = error_criterion(x)
            errvec[-1] = error

        output = {
            "flag": flag,
            "iter": k,
            "error": error,
            "errvec": errvec,
            "times": times,
            "time": time.perf_counter() - clock0,
        }
        return x, output
