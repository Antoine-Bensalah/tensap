# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module linear_solvers.

Iterative linear solvers Ax=b for operators and right-hand sides in tensor
format. Port of ApproximationToolbox/Approximation/TensorApproximation/
EquationsSolvers/LinearSolver.m, TruncatedRichardsonIterations.m,
TruncatedPCG.m, RankOneALSLinearSolver.m, GreedyLinearSolver.m and
TuckerLikeTensorALSLinearSolver.m.
"""

import abc
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tensap
from tensap.solvers.local_diagnostics import LocalDiagnostics


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
                coef = _contract_except_dim(core_a, core_x, xtAx, mu)
                M = A_frames.eval_in_space(mu, np.ravel(coef))

                # reduced right-hand side f_mu = sum_l beta_l b^(mu)_l
                core_b, core_x = tensap.convert_tensors(b_core, x_core)
                coef = _contract_except_dim(core_b, core_x, xtb, mu)
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


def _terms_are_dependent(x, mu):
    """Canonical format: True if the rank-one terms of x, without their
    dimension mu, are linearly dependent up to the square root of the machine
    precision. Their Gram matrix is the Hadamard product of the Gram matrices
    of the frames of the other dimensions (m x m, cheap); the Galerkin system
    of the frame of dimension mu is then singular."""
    frames = x.tensor_frame.frames
    gram = np.ones((frames[0].dim, frames[0].dim))
    for nu, frame in enumerate(frames):
        if nu != mu:
            vectors = frame.data[:, :, 0]
            gram = gram * (vectors @ vectors.T)
    values = np.linalg.eigvalsh(gram)
    return values[0] < np.sqrt(np.finfo(float).eps) * values[-1]


def _warn_dependent_frames(mu):
    warnings.warn(
        f"update_frame: the frame of dimension {mu} is not updated, because the "
        "rank-one terms without this dimension are numerically linearly "
        "dependent and its Galerkin system is singular. With a diagonal core "
        "(canonical format), the frames cannot be orthonormalized; this happens "
        "when a greedy solver adds corrections beyond the rank needed: stop it "
        "earlier, or use a Tucker core.",
        RuntimeWarning,
    )


def _with_unit_diagonal_core(x):
    """x with its diagonal core replaced by ones (same frames)."""
    return tensap.TuckerTensorVector(
        tensap.DiagonalTensor(np.ones(x.core.data.size), x.core.order), x.tensor_frame
    )


def _times_diag_core(core, diag, mu):
    """Multiply a core tensor by diag(diag) along dimension mu."""
    if isinstance(core, tensap.DiagonalTensor):
        return tensap.DiagonalTensor(core.data * diag, core.order)
    return core.tensor_matrix_product([np.diag(diag)], [mu])


class TuckerTensorALSLinearSolver(LinearSolver):
    """
    Galerkin updates of a Tucker tensor x for A @ x = b by alternating least
    squares: update of the core with fixed frames (update_core), update of
    the frames with a fixed core (update_frames_by_als), or both alternately
    (solve). Port of TuckerLikeTensorALSLinearSolver.m.

    The methods update_core and update_frames_by_als can be used as the
    update of a GreedyLinearSolver.

    Attributes
    ----------
    max_iterations_frames : int
        Maximum number of sweeps over the dimensions in update_frames_by_als.
    stagnation_frames : float
        Stagnation criterion of update_frames_by_als.
    dense_max_size : int
        Full cores: Galerkin systems on the core up to this size are
        assembled and solved directly, larger ones by conjugate gradient
        without assembling the matrix.
    """

    def __init__(
        self,
        A,
        b,
        max_iterations_frames=10,
        stagnation_frames=1e-6,
        dense_max_size=2000,
        **kwargs,
    ):
        super().__init__(A, b, **kwargs)
        if type(A.core) is not type(b.core):
            raise TypeError("A and b must have the same type of core.")
        self.max_iterations_frames = max_iterations_frames
        self.stagnation_frames = stagnation_frames
        self.dense_max_size = dense_max_size
        # diagnostics of the local solves (reset by solve; also filled when
        # update_core is called directly, e.g. by a greedy solver)
        self._local = LocalDiagnostics()

    def solve(self, x=None):
        """Alternate core and frames updates, starting from x (or x0).

        Returns
        -------
        x : TuckerTensorVector
        output : dict
            Dictionary with keys 'flag', 'iter', 'error', 'errvec',
            'stagnation', 'times'.
        """
        if x is None:
            if self.x0 is None:
                raise ValueError("An initial guess must be provided.")
            x = self.x0

        clock0 = time.perf_counter()
        self._local = LocalDiagnostics()
        error_criterion = self._default_error_criterion()

        flag = 1
        errvec, stagnations, times = [], [], []
        k = 0
        for k in range(1, self.max_iterations + 1):
            x_old = x
            x = self.update_core(x)
            x = self.update_frames_by_als(x)

            times.append(time.perf_counter() - clock0)
            errvec.append(error_criterion(x))
            stagnations.append(self._stagnation(x, x_old))

            if self.display:
                print(
                    f"ALS: Iteration {k:3d} - Stagnation {stagnations[-1]:.2e} "
                    f"- Error {errvec[-1]:.2e}"
                )

            if errvec[-1] < self.tolerance:
                flag = 0
                break
            if stagnations[-1] < self.stagnation:
                flag = 2
                break

        output = {
            "flag": flag,
            "iter": k,
            "error": errvec[-1] if errvec else np.inf,
            "errvec": errvec,
            "stagnation": stagnations,
            "times": times,
            "local": self._local.as_dict(),
        }
        return x, output

    # ---- update of the core ----

    def update_core(self, x):
        """Galerkin update of the core of x, the frames being fixed."""
        if isinstance(self.A.core, tensap.DiagonalTensor):
            return self._update_diagonal_core(x)
        if isinstance(self.A.core, tensap.FullTensor):
            if int(np.prod(x.representation_rank())) > self.dense_max_size:
                return self._update_full_core_matrix_free(x)
            return self._update_full_core(x)
        raise NotImplementedError(
            f"update_core is not implemented for a {type(self.A.core).__name__} "
            "core."
        )

    def _update_diagonal_core(self, x):
        """Canonical format: solve G sigma = g, G(j, i) = <A x_i, x_j>."""
        if not isinstance(x.core, tensap.DiagonalTensor):
            raise TypeError("x must have a DiagonalTensor core.")
        x_frames = x.tensor_frame

        # products over the dimensions of w_j^T A_k w_i: (R_A, m, m)
        xAx = x_frames.dot_with_metrics(x_frames, self.A.tensor_frame)
        G = np.tensordot(self.A.core.data, np.prod(np.stack(xAx), axis=0), axes=1)

        # products over the dimensions of b_l^T w_j: (R_b, m)
        xb = self.b.tensor_frame.dot(x_frames)
        g = self.b.core.data @ np.prod(np.stack(xb), axis=0)

        sigma = np.linalg.solve(G, g)
        return tensap.TuckerTensorVector(
            tensap.DiagonalTensor(sigma, x.order), x_frames
        )

    def _update_full_core(self, x):
        """Tucker format: Galerkin system on the whole core, of size prod(R)."""
        x = x.orth()
        x_frames = x.tensor_frame
        order = x.order
        ranks = x.representation_rank()

        # operator on the core space: sum_k C_A(k) M^(1)_k x ... x M^(d)_k
        xAx = x_frames.dot_with_metrics(x_frames, self.A.tensor_frame)
        matrices = [m.reshape(m.shape[0], -1).T for m in xAx]
        core_A = tensap.FullTensor(self.A.core.full())
        MA = core_A.tensor_matrix_product(matrices).numpy()
        MA = MA.reshape([r for r in ranks for _ in range(2)])
        MA = MA.transpose(list(range(0, 2 * order, 2)) + list(range(1, 2 * order, 2)))
        size = int(np.prod(ranks))
        MA = MA.reshape(size, size)

        xb = self.b.tensor_frame.dot(x_frames)
        core_b = tensap.FullTensor(self.b.core.full())
        MB = core_b.tensor_matrix_product([m.T for m in xb]).numpy().reshape(size)

        core = np.linalg.solve(MA, MB).reshape(ranks)
        return tensap.TuckerTensorVector(tensap.FullTensor(core), x_frames)

    def _projected_problem(self, x):
        """A and b projected on the (orthonormal) frames of x, on full cores:
        a function applying the projected operator to arrays of shape
        ranks + extra axes, and the projected right-hand side, of shape ranks.

        The projected operator is sum_k C_A(k) P^(1)_{k_1} x ... x P^(d)_{k_d},
        P^(nu)_k(i, j) = (A^(nu)_k w_j, w_i), applied by mode products over the
        non-zero entries of the core of A, without assembling its matrix."""
        x_frames = x.tensor_frame
        order = x.order
        xAx = x_frames.dot_with_metrics(x_frames, self.A.tensor_frame)
        xb = self.b.tensor_frame.dot(x_frames)
        core_A = np.asarray(self.A.core.full().numpy())
        nonzeros = np.argwhere(core_A != 0)

        def apply(c):
            result = np.zeros_like(c)
            for k in nonzeros:
                t = c
                for nu in range(order):
                    t = np.moveaxis(np.tensordot(xAx[nu][k[nu]], t, axes=(1, nu)), 0, nu)
                result += core_A[tuple(k)] * t
            return result

        core_b = tensap.FullTensor(np.asarray(self.b.core.full().numpy()))
        rhs = core_b.tensor_matrix_product([m.T for m in xb]).numpy()
        return apply, rhs

    def _update_full_core_matrix_free(self, x):
        """Full core, large Galerkin system: conjugate gradient on the core,
        the projected operator being applied by mode products."""
        x = x.orth()
        ranks = tuple(x.representation_rank())
        apply, rhs = self._projected_problem(x)
        size = rhs.size
        operator = spla.LinearOperator(
            (size, size), matvec=lambda v: apply(v.reshape(ranks)).ravel(), dtype=float
        )
        guess = np.asarray(x.core.full().numpy()).ravel()
        core, info = spla.cg(operator, rhs.ravel(), x0=guess, rtol=1e-12, atol=0.0,
                             maxiter=10 * size)
        self._local.iterative_solve(info, operator, rhs.ravel(), core)
        return tensap.TuckerTensorVector(tensap.FullTensor(core.reshape(ranks)),
                                         x.tensor_frame)

    # ---- update of the frames ----

    def update_frames_by_als(self, x):
        """Galerkin update of the frames of x, one dimension at a time."""
        A_frames = self.A.tensor_frame
        b_frames = self.b.tensor_frame
        xAx = [
            m.reshape(m.shape[0], -1)
            for m in x.tensor_frame.dot_with_metrics(x.tensor_frame, A_frames)
        ]
        xb = b_frames.dot(x.tensor_frame)

        for k in range(1, self.max_iterations_frames + 1):
            x_old = x
            for mu in range(self.A.order):
                x = self.update_frame(x, mu, xAx, xb)
                frame = x.tensor_frame.frames[mu]
                m = frame.dot_with_metric(frame, A_frames.frames[mu])
                xAx[mu] = m.reshape(m.shape[0], -1)
                xb[mu] = b_frames.frames[mu].dot(frame)

            stagnation = (x - x_old).norm() / x_old.norm()
            if self.display:
                print(f"update_frames: iteration {k:2d} - stagnation {stagnation:.2e}")
            if k > 1 and stagnation < self.stagnation_frames:
                break
        return x

    def update_frame(self, x, mu, xAx=None, xb=None):
        """Galerkin update of all the elements of the frame of dimension mu.

        The unknown W = [w_1, ..., w_m] solves the system of size n*m
        (sum_k C_k x A^(mu)_k) vec(W) = vec(F), then its columns are
        normalized and their norms moved into the core.

        With a diagonal core (canonical format), the unknowns are the terms
        sigma_i w_i, so that the system does not depend on the spread of the
        coefficients sigma_i. The frames cannot be orthonormalized in this
        format: if the rank-one terms without dimension mu are numerically
        linearly dependent (which happens when a greedy solver adds
        corrections beyond the rank needed), the system is singular; the
        frame is then not updated and a RuntimeWarning is issued.
        """
        A_frames = self.A.tensor_frame
        b_frames = self.b.tensor_frame
        if xAx is None:
            xAx = [
                m.reshape(m.shape[0], -1)
                for m in x.tensor_frame.dot_with_metrics(x.tensor_frame, A_frames)
            ]
        if xb is None:
            xb = b_frames.dot(x.tensor_frame)

        x_input = x
        if isinstance(x.core, tensap.DiagonalTensor):
            # canonical format: the unknowns are the terms sigma_i w_i of
            # dimension mu, not the frame vectors w_i, so that the system does
            # not depend on the spread of the coefficients sigma_i (they would
            # enter its matrix squared); same Galerkin solution
            if _terms_are_dependent(x, mu):
                _warn_dependent_frames(mu)
                return x_input
            x = _with_unit_diagonal_core(x)
        n, rank = x.tensor_frame.frames[mu].shape[0], x.tensor_frame.frames[mu].dim

        # C_k(p, q): contraction of the cores with all the dimensions but mu
        core_A, core_x = tensap.convert_tensors(self.A.core, x.core)
        coef = _contract_except_dim(
            core_A, core_x.kron(core_x), xAx, mu
        ).reshape(-1, rank, rank)
        K = sp.csr_matrix((n * rank, n * rank))
        for C_k, A_k in zip(coef, A_frames.frames[mu].data):
            K = K + sp.kron(C_k, sp.csr_matrix(A_k), format="csr")

        # right-hand side: F = sum_l b^(mu)_l c_l^T, block p = column p of F
        core_b, core_x = tensap.convert_tensors(self.b.core, x.core)
        coef = _contract_except_dim(core_b, core_x, xb, mu)
        F = b_frames.frames[mu].data[:, :, 0].T @ coef
        rhs = F.T.ravel()

        # a singular system: the frame is kept as it is
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            try:
                W = spla.spsolve(K.tocsc(), rhs).reshape(rank, n)
            except spla.MatrixRankWarning:
                W = None
        norms = None if W is None else np.linalg.norm(W, axis=1)
        if W is None or not np.all(np.isfinite(W)) or not np.all(norms > 0):
            _warn_dependent_frames(mu)
            return x_input

        frames = list(x.tensor_frame.frames)
        frames[mu] = tensap.FrameVector((W / norms[:, None]).reshape(rank, n, 1))
        return tensap.TuckerTensorVector(
            _times_diag_core(x.core, norms, mu), tensap.TFrameVector(frames)
        )
