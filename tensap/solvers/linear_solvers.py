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

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tensap
from tensap.tensor_algebra.tensors.tt_tensor import as_tt_tensor


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

        if self.error_criterion is None:
            b_norm = b.norm()

            def error_criterion(x):
                return (b - A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        r = b - A @ x
        z = P @ r if P is not None else r
        p = z
        Ap = A @ p
        pAp = p.dot(Ap)

        flag = 1
        errvec = []
        k = 0
        for k in range(1, self.max_iterations + 1):
            x_old = x
            omega = r.dot(p) / pAp
            x = truncate(x + omega * p)
            stagnation = (x - x_old).norm() / x.norm()
            r = truncate(b - A @ x)
            errvec.append(error_criterion(x))

            if self.display:
                print(
                    f"Iteration {k:3d} - Stagnation {stagnation:.2e} "
                    f"- Error {errvec[-1]:.2e}"
                )

            if stagnation < self.stagnation:
                flag = 2
                break
            if errvec[-1] < self.tolerance:
                flag = 0
                break

            z = P @ r if P is not None else r
            beta = -z.dot(Ap) / pAp
            p = truncate(z + beta * p)
            Ap = A @ p
            pAp = p.dot(Ap)

        output = {
            "flag": flag,
            "iter": k,
            "error": errvec[-1],
            "errvec": errvec,
        }
        return x, output


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


def _kronecker_terms_apply(node, interfaces, c):
    """y(i_0, ..., i_{n-1}) = sum_k node(k_0, ..., k_{n-1}) prod_t
    L_t(i_t, k_t, j_t) c(j_0, ..., j_{n-1}): a sum of Kronecker products of
    the matrices L_t(:, k_t, :), applied by one tensordot per axis (BLAS),
    then contracted with the node tensor."""
    n = c.ndim
    labels = [("j", t) for t in range(n)]
    for t, interface in enumerate(interfaces):
        position = labels.index(("j", t))
        c = np.tensordot(interface, c, axes=(2, position))
        labels = [("i", t), ("k", t)] + labels[:position] + labels[position + 1:]
    k_axes = [labels.index(("k", t)) for t in range(n)]
    c = np.tensordot(c, node, axes=(k_axes, list(range(n))))
    labels = [label for label in labels if label[0] != "k"]
    return c.transpose([labels.index(("i", t)) for t in range(n)])


def _kronecker_sum_preconditioner(node, interfaces, shape):
    """Inverse of the Kronecker sum closest to the local operator
    M = sum_k node(k) L_0^{k_0} x ... x L_{n-1}^{k_{n-1}}, applied by fast
    diagonalization.

    The orthogonal projection of M on the Kronecker sums
    S_0 x I x ... x I + ... + I x ... x I x S_{n-1} (Frobenius inner product)
    is given by the partial traces T_t = tr_{s != t}(M) / prod_{s != t} n_s and
    the mean tau = tr(M) / prod_s n_s: S_t = T_t, with the identity counted
    n - 1 times too many, removed from the eigenvalues. With the
    eigendecompositions T_t = Q_t diag(lambda_t) Q_t^T, the inverse is
    (Q_0 x ... x Q_{n-1}) diag(1 / (sum_t lambda_t - (n - 1) tau))
    (Q_0 x ... x Q_{n-1})^T. It is exact when M is a Kronecker sum, which is
    the case of the Laplacian with orthonormal interfaces (the interfaces of
    the subtrees without the operator K are identities)."""
    n = len(shape)
    traces = [np.einsum("iki->k", interface) for interface in interfaces]
    total = np.einsum(node, list(range(n)), *[x for t in range(n) for x in (traces[t], [t])], [])
    size = int(np.prod(shape))
    tau = total / size
    eigenvalues, bases = [], []
    for t, interface in enumerate(interfaces):
        weights = np.einsum(node, list(range(n)),
                            *[x for s in range(n) if s != t for x in (traces[s], [s])], [t])
        partial = np.tensordot(interface, weights, axes=(1, 0)) * shape[t] / size
        partial = 0.5 * (partial + partial.T)
        values, vectors = np.linalg.eigh(partial)
        eigenvalues.append(values)
        bases.append(vectors)
    denominator = -(n - 1) * tau
    for t in range(n):
        denominator = np.add.outer(denominator, eigenvalues[t]) if t else (
            eigenvalues[0] + denominator)
    denominator = np.where(np.abs(denominator) > 1e-14 * np.abs(denominator).max(),
                           denominator, 1e-14 * np.abs(denominator).max())

    def apply(vector):
        v = vector.reshape(shape)
        for t in range(n):
            v = np.moveaxis(np.tensordot(bases[t].T, v, axes=(1, t)), 0, t)
        v = v / denominator
        for t in range(n):
            v = np.moveaxis(np.tensordot(bases[t], v, axes=(1, t)), 0, t)
        return v.ravel()

    return spla.LinearOperator((size, size), matvec=apply, dtype=float)


class AnisotropicGreedySpaceLinearSolver(LinearSolver):
    """
    Greedy construction of tensor subspaces U_1 x ... x U_d with anisotropic
    enrichment, the approximation being sought in the Tucker format with a TT
    core. Port of AnisotropicGreedySpaceLinearSolver.m.

    At each iteration: a rank-one correction c = c_1 x ... x c_d of the
    residual (local_solver); the spaces U_mu are enriched by c_mu only in the
    dimensions mu where the smallest mu-singular value of x + c is at least
    theta times the largest of these values over the dimensions; then the
    core is updated by the Galerkin problem on U_1 x ... x U_d
    (TuckerTensorALSLinearSolver.update_core, TT core solved by
    core_algorithm).

    Attributes
    ----------
    local_solver : LinearSolver
        Solver of the rank-one corrections (its attribute b is overwritten).
        Default is a RankOneALSLinearSolver.
    theta : float
        Bulk parameter of the enrichment, in [0, 1]: 0 enriches every
        dimension, 1 only the dimension of largest estimated error. The
        default is 0.8, as in Matlab.
    minimize_residual : bool
        If True, apply the algorithm to the normal equations.
    check_residual : int
        The error is computed every check_residual iterations.
    core_algorithm : str
        Algorithm of the core update, 'dmrg' (default, Matlab useDMRG),
        'amen' or 'als' (the TT ranks of the core then stay those of the first
        correction).
    core_truncator : tensap.Truncator, optional
        Truncator of the core update (DMRG and AMEn).
    max_iterations_core, stagnation_core : int, float
        Stopping criteria of the core update.
    """

    def __init__(
        self,
        A,
        b,
        local_solver=None,
        theta=0.8,
        minimize_residual=False,
        check_residual=1,
        core_algorithm="dmrg",
        core_truncator=None,
        max_iterations_core=10,
        stagnation_core=1e-6,
        **kwargs,
    ):
        super().__init__(A, b, **kwargs)
        self.local_solver = (
            RankOneALSLinearSolver(A, b, display=False) if local_solver is None else local_solver
        )
        self.theta = theta
        self.minimize_residual = minimize_residual
        self.check_residual = check_residual
        self.core_algorithm = core_algorithm
        self.core_truncator = core_truncator
        self.max_iterations_core = max_iterations_core
        self.stagnation_core = stagnation_core

    def solve(self):
        """Greedy enrichment of the spaces and Galerkin update of the core.

        Returns
        -------
        x : TuckerTensorVector
            With a TTTensor core.
        output : dict
            Keys 'flag', 'iter', 'error', 'errvec', and for each iteration
            'enriched' (the enriched dimensions), 'dims' (dimensions of the
            spaces), 'ranks' (TT ranks of the core) and 'storage'.
        """
        if not self.minimize_residual:
            A, b = self.A, self.b
        else:
            A, b = self.A.T @ self.A, self.A.T @ self.b
            self.local_solver.A = A
        b_norm = self.b.norm()
        if self.error_criterion is None:

            def error_criterion(x):
                return (self.b - self.A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        updater = TuckerTensorALSLinearSolver(
            _with_tt_core(A),
            _with_tt_core(b),
            core_algorithm=self.core_algorithm,
            core_truncator=self.core_truncator,
            max_iterations_core=self.max_iterations_core,
            stagnation_core=self.stagnation_core,
            display=False,
        )
        x = None if self.x0 is None else _with_tt_core(self.x0)
        residual = b if x is None else b - A @ x
        output = {"enriched": [], "dims": [], "ranks": [], "storage": []}
        flag, errvec, error, k = 1, [], None, 0
        for k in range(1, self.max_iterations + 1):
            self.local_solver.b = residual
            correction, _ = self.local_solver.solve()
            x_old = x
            if x is None:
                x = _with_tt_core(correction)
                enriched = list(range(x.order))
            else:
                values = (x + correction).singular_values()
                smallest = np.array([v.min() for v in values])
                enriched = [int(m) for m in np.nonzero(smallest >= self.theta * smallest.max())[0]]
                x = _enrich(x, correction, enriched)
                x = updater.update_core(x)
            output["enriched"].append(enriched)
            output["dims"].append(tuple(f.dim for f in x.tensor_frame.frames))
            output["ranks"].append(x.core.ranks)
            output["storage"].append(x.storage())

            stagnation = 1.0 if x_old is None else (x - x_old).norm() / x.norm()
            error = error_criterion(x) if k % self.check_residual == 0 else None
            errvec.append(error)
            if self.display:
                message = (f"AnisotropicGreedy: Iteration {k:3d} - Stagnation "
                           f"{stagnation:.2e} - Enriched {enriched}")
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

        if error is None:
            error = error_criterion(x)
            errvec[-1] = error
        output.update({"flag": flag, "iter": k, "error": error, "errvec": errvec})
        return x, output


def _with_tt_core(tensor):
    """The same Tucker tensor (vector or operator) with a TTTensor core."""
    return tensor.__class__(as_tt_tensor(tensor.core), tensor.tensor_frame)


def _enrich(x, correction, dims):
    """Add the frame vectors of the rank-one correction to the frames of x in
    the dimensions dims, the TT core being padded with zeros (the tensor is
    unchanged)."""
    frames = list(x.tensor_frame.frames)
    cores = list(x.core.cores)
    for mu in dims:
        data = np.concatenate([frames[mu].data, correction.tensor_frame.frames[mu].data])
        frames[mu] = tensap.FrameVector(data)
        r0, n, r1 = cores[mu].shape
        cores[mu] = np.concatenate([cores[mu], np.zeros((r0, 1, r1))], axis=1)
    return tensap.TuckerTensorVector(tensap.TTTensor(cores), tensap.TFrameVector(frames))


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
    max_iterations_core : int
        Maximum number of iterations of the core update (iterative core
        formats only).
    max_iterations_frames : int
        Maximum number of sweeps over the dimensions in update_frames_by_als.
    stagnation_core : float
        Stagnation criterion of the core update (iterative core formats only).
    stagnation_frames : float
        Stagnation criterion of update_frames_by_als.
    core_algorithm : str
        TT cores: algorithm of the TTTensorALSLinearSolver used for the core
        update, 'als' (fixed ranks, default), 'dmrg' or 'amen' (Matlab:
        useDMRG).
    core_truncator : tensap.Truncator, optional
        TT cores: truncator of DMRG and AMEn.
    dense_max_size : int
        Full and tree-based cores: Galerkin systems on the core (or on a node)
        up to this size are assembled and solved directly, larger ones by
        conjugate gradient without assembling the matrix.
    local_preconditioner : str or None
        Tree-based cores, matrix-free local systems: 'kronecker_sum'
        (default; inverse of the Kronecker sum closest to the local operator,
        by fast diagonalization) or None.
    local_tolerance : float
        Relative tolerance of the conjugate gradient of the matrix-free local
        systems of the tree-based cores.
    """

    def __init__(
        self,
        A,
        b,
        max_iterations_core=10,
        max_iterations_frames=10,
        stagnation_core=1e-6,
        stagnation_frames=1e-6,
        core_algorithm="als",
        core_truncator=None,
        dense_max_size=2000,
        local_preconditioner="kronecker_sum",
        local_tolerance=1e-10,
        **kwargs,
    ):
        super().__init__(A, b, **kwargs)
        if type(A.core) is not type(b.core):
            raise TypeError("A and b must have the same type of core.")
        self.max_iterations_core = max_iterations_core
        self.max_iterations_frames = max_iterations_frames
        self.stagnation_core = stagnation_core
        self.stagnation_frames = stagnation_frames
        self.core_algorithm = core_algorithm
        self.core_truncator = core_truncator
        self.dense_max_size = dense_max_size
        if local_preconditioner not in (None, "kronecker_sum"):
            raise ValueError("local_preconditioner must be None or 'kronecker_sum'.")
        self.local_preconditioner = local_preconditioner
        self.local_tolerance = local_tolerance

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
        if self.error_criterion is None:
            b_norm = self.b.norm()

            def error_criterion(x):
                return (self.b - self.A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        flag = 1
        errvec, stagnations, times = [], [], []
        k = 0
        for k in range(1, self.max_iterations + 1):
            x_old = x
            x = self.update_core(x)
            x = self.update_frames_by_als(x)

            times.append(time.perf_counter() - clock0)
            errvec.append(error_criterion(x))
            stagnations.append((x - x_old).norm() / x.norm())

            if self.display:
                print(
                    f"ALS: Iteration {k:3d} - Stagnation {stagnations[-1]:.2e} "
                    f"- Error {errvec[-1]:.2e}"
                )

            if stagnations[-1] < self.stagnation:
                flag = 2
                break
            if errvec[-1] < self.tolerance:
                flag = 0
                break

        output = {
            "flag": flag,
            "iter": k,
            "error": errvec[-1],
            "errvec": errvec,
            "stagnation": stagnations,
            "times": times,
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
        if isinstance(self.A.core, tensap.TTTensor):
            return self._update_tt_core(x)
        if isinstance(self.A.core, tensap.TreeBasedTensor):
            return self._update_tree_core(x)
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
        core, _ = spla.cg(operator, rhs.ravel(), x0=guess, rtol=1e-12, atol=0.0,
                          maxiter=10 * size)
        return tensap.TuckerTensorVector(tensap.FullTensor(core.reshape(ranks)),
                                         x.tensor_frame)

    def _update_tree_core(self, x):
        """Tree-based core: Galerkin update of the active nodes one after the
        other, the representation being orthogonal with respect to the
        updated node (Matlab updateTreeBasedCore and
        updateInternalNodeTreeBasedCore).

        With inactive leaves (as in Matlab), the local system of a node is
        built by the tree contraction (TreeBasedTensor.
        reduce_dot_with_rank_one_metric_at_node) of the projected problem
        around the node, without forming the full core (_tree_node_problem).
        With active leaves, the map from the node tensor to the full core is
        formed explicitly (_tree_node_problem_explicit, moderate sizes
        only)."""
        tree = self.A.core.tree
        if not isinstance(x.core, tensap.TreeBasedTensor):
            x = tensap.TuckerTensorVector(x.core.tree_based_tensor(tree), x.tensor_frame)
        x = x.orth()
        core = x.core
        leaves = tree.is_leaf
        contracted = not any(
            np.any(t.is_active_node[leaves]) for t in (core, self.A.core, self.b.core)
        )
        if contracted:
            x_frames = x.tensor_frame
            xAx = x_frames.dot_with_metrics(x_frames, self.A.tensor_frame)
            # rows (k, trial i), columns test j: (A_k w_i, w_j), as Matlab
            # dot(A.space * x.space, x.space), for the Kronecker product of the
            # cores of A and x
            metric_A = [m.transpose(0, 2, 1).reshape(-1, m.shape[1]) for m in xAx]
            metric_b = list(self.b.tensor_frame.dot(x_frames))

            def local_problem(core, nod):
                return self._tree_node_problem(core, nod, metric_A, metric_b)

        else:
            apply, rhs = self._projected_problem(x)

            def local_problem(core, nod):
                return self._tree_node_problem_explicit(core, nod, apply, rhs)

        nodes = [nod for nod in range(1, tree.nb_nodes + 1) if core.is_active_node[nod - 1]]
        for _ in range(self.max_iterations_core):
            core_old = tensap.TreeBasedTensor(list(core.tensors), tree)
            for nod in nodes:
                core = core.orth_at_node(nod)
                shape = core.tensors[nod - 1].shape
                solution = local_problem(core, nod)
                core.tensors[nod - 1] = tensap.FullTensor(solution.reshape(shape))
                core.is_orth = False
            stagnation = (core - core_old).norm() / core.norm()
            if self.display:
                print(f"update_tree_core: stagnation {stagnation:.2e}")
            if stagnation < self.stagnation_core:
                break
        return tensap.TuckerTensorVector(core, x.tensor_frame)

    def _tree_node_problem(self, core, nod, metric_A, metric_b):
        """Galerkin system of the node nod of a tree-based core with inactive
        leaves, built by the tree contraction and solved directly (up to
        dense_max_size unknowns) or by the conjugate gradient without
        assembling its matrix, preconditioned by local_preconditioner.

        The interfaces of the children and of the parent of the node,
        L_t(i, k, j) for the test index i, the index k of A and the trial
        index j, come from the contraction of kron(C_A, C_x) with C_x; the
        local operator is
        (M c)(i_1, ..., i_p, i_0) = sum C_A^nod(k_1, ..., k_p, k_0)
        prod_t L_t(i_t, k_t, j_t) c(j_1, ..., j_p, j_0) (t = 0: the parent,
        absent at the root), and the right-hand side is built the same way
        with C_b."""
        A_core, b_core = self.A.core, self.b.core
        interfaces = A_core.kron(core).reduce_dot_with_rank_one_metric_at_node(
            core, metric_A, nod
        )
        # (k * r + trial, test) -> (test, k, trial)
        interfaces = [
            m.reshape(m.shape[0] // m.shape[1], m.shape[1], m.shape[1]).transpose(2, 0, 1)
            for m in interfaces
        ]
        interfaces_b = [m.T for m in b_core.reduce_dot_with_rank_one_metric_at_node(
            core, metric_b, nod)]
        node_A = A_core.tensors[nod - 1].numpy()
        node_b = b_core.tensors[nod - 1].numpy()
        shape = core.tensors[nod - 1].shape
        n = len(shape)
        test, trial, op = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n))

        rhs_operands = [node_b, op]
        for t, interface in enumerate(interfaces_b):
            rhs_operands += [interface, [test[t], op[t]]]
        rhs = np.einsum(*rhs_operands, test, optimize=True).ravel()
        size = rhs.size

        if size <= self.dense_max_size:
            operands = [node_A, op]
            for t, interface in enumerate(interfaces):
                operands += [interface, [test[t], op[t], trial[t]]]
            matrix = np.einsum(*operands, test + trial, optimize=True).reshape(size, size)
            return np.linalg.solve(matrix, rhs)

        def matvec(vector):
            return _kronecker_terms_apply(node_A, interfaces, vector.reshape(shape)).ravel()

        operator = spla.LinearOperator((size, size), matvec=matvec, dtype=float)
        preconditioner = None
        if self.local_preconditioner == "kronecker_sum":
            preconditioner = _kronecker_sum_preconditioner(node_A, interfaces, shape)
        guess = core.tensors[nod - 1].numpy().ravel()
        solution, _ = spla.cg(operator, rhs, x0=guess, rtol=self.local_tolerance, atol=0.0,
                              maxiter=10 * size, M=preconditioner)
        return solution

    @staticmethod
    def _tree_node_problem_explicit(core, nod, apply, rhs):
        """Galerkin system of the node nod from the explicit map Psi from the
        node tensor to the full core (its columns: the full core with the node
        tensor set to each canonical vector): Psi^T A_c Psi c = Psi^T b_c."""
        ranks = tuple(core.shape)
        shape = core.tensors[nod - 1].shape
        size = int(np.prod(shape))
        psi = np.empty(ranks + (size,))
        for n in range(size):
            unit = np.zeros(size)
            unit[n] = 1.0
            core.tensors[nod - 1] = tensap.FullTensor(unit.reshape(shape))
            psi[..., n] = core.full().numpy()
        psi_matrix = psi.reshape(-1, size)
        M = psi_matrix.T @ apply(psi).reshape(-1, size)
        return np.linalg.solve(M, psi_matrix.T @ rhs.ravel())

    def _update_tt_core(self, x):
        """TT core: Galerkin problem on the core alone, A and b being projected
        on the orthonormalized frames of x (a TTOperator with cores of size
        R_nu x R_nu and a TTTensor), solved by TTTensorALSLinearSolver from
        the current core (Matlab reduceProblemToTTCore)."""
        x = tensap.TuckerTensorVector(as_tt_tensor(x.core), x.tensor_frame).orth()
        x_frames = x.tensor_frame
        xAx = x_frames.dot_with_metrics(x_frames, self.A.tensor_frame)
        xb = self.b.tensor_frame.dot(x_frames)
        core_A, core_b = as_tt_tensor(self.A.core), as_tt_tensor(self.b.core)
        A_core = tensap.TTOperator(
            [np.einsum("alb,lij->aijb", g, m) for g, m in zip(core_A.cores, xAx)]
        )
        b_core = tensap.TTTensor(
            [np.einsum("alb,lj->ajb", g, m) for g, m in zip(core_b.cores, xb)]
        )
        solver = tensap.TTTensorALSLinearSolver(
            A_core,
            b_core,
            algorithm=self.core_algorithm,
            truncator=self.core_truncator,
            x0=as_tt_tensor(x.core),
            max_iterations=self.max_iterations_core,
            stagnation=self.stagnation_core,
            tolerance=0.0,
            display=False,
        )
        core, _ = solver.solve()
        return tensap.TuckerTensorVector(core, x_frames)

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

        W = spla.spsolve(K.tocsc(), rhs).reshape(rank, n)
        norms = np.linalg.norm(W, axis=1)

        frames = list(x.tensor_frame.frames)
        frames[mu] = tensap.FrameVector((W / norms[:, None]).reshape(rank, n, 1))
        return tensap.TuckerTensorVector(
            _times_diag_core(x.core, norms, mu), tensap.TFrameVector(frames)
        )
