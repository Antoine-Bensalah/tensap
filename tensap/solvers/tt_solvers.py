# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tt_solvers.

Alternating solvers of A @ x = b in tensor-train format: ALS (one core at a
time, fixed ranks), DMRG (two cores at a time, ranks adapted by a truncated
SVD) and AMEn (one core at a time, ranks adapted by a truncated SVD and an
enrichment with the residual, Dolgov and Savostyanov 2014). ALS and DMRG are
a port of ApproximationToolbox/Approximation/TensorApproximation/
EquationsSolvers/TTTensorALSLinearSolver.m; AMEn is not in Matlab.

Differences with the Matlab code: the interfaces of the operator and of the
right-hand side are kept in memory and updated from one core to the next
(O(d) contractions per sweep instead of O(d^2)), the sweeps go back and forth,
and the local systems are solved without assembling their matrix (conjugate
gradient, or GMRES for a non-symmetric operator) when they are large.
"""

import numpy as np
import scipy.sparse.linalg as spla
import tensap
from tensap.solvers.linear_solvers import LinearSolver
from tensap.solvers.local_diagnostics import LocalDiagnostics
from tensap.tensor_algebra.tensors.tt_tensor import _truncated_svd


class TTTensorALSLinearSolver(LinearSolver):
    """
    Solve A @ x = b, with A a TTOperator and b a TTTensor, by alternating
    sweeps over the cores of x.

    At each step the cores other than the updated ones are orthogonal, and
    the updated core (ALS) or pair of cores (DMRG) solves the Galerkin
    projection of the system on the subspace spanned by the other cores: for
    a symmetric positive definite A, it minimizes the energy
    J(x) = 1/2 (A x, x) - (b, x) over these cores.

    Attributes
    ----------
    algorithm : str
        'als' (fixed ranks), 'dmrg' (ranks adapted at each bond by a
        truncated SVD of the supercore) or 'amen' (truncated SVD of the
        updated core, then enrichment by kick_rank directions of the
        residual).
    truncator : tensap.Truncator
        Truncation of the supercores in DMRG and of the cores in AMEn
        (relative tolerance and maximal rank). The default is
        Truncator(tolerance=1e-12, max_rank=100).
    kick_rank : int
        AMEn: rank of the approximation of the residual used to enrich the
        cores. The default is 4.
    minimize_residual : bool
        If True, solve the normal equations A^T A x = A^T b.
    ranks : int or list of int, optional
        Ranks of the random initial guess when x0 is None. The default is 5
        for ALS and 2 for DMRG.
    dense_max_size : int
        Local systems up to this size are assembled and solved by a dense
        direct solver; larger ones are solved matrix-free.
    local_tolerance : float
        Relative tolerance of the matrix-free local solver.
    local_max_iterations : int, optional
        Maximal number of iterations of the matrix-free local solver.
    check_residual : int
        The error criterion is evaluated every check_residual iterations.
    local_preconditioner : str or None
        Preconditioner of the matrix-free local solver: 'block_jacobi'
        (default; inverse of the blocks of the physical indices for each pair
        of rank indices) or None.
    local_method : str
        Matrix-free local solver: 'auto' (default: CG if the operator is
        symmetric, GMRES otherwise; CG assumes that a symmetric operator is
        positive definite, as for the energy interpretation of the sweeps),
        'cg' or 'gmres' (e.g. for a symmetric indefinite operator).

    The output of solve has the key 'local' (LocalDiagnostics.as_dict): the
    number of local solves, of iterative ones that stopped before their
    tolerance, the worst local relative residual.
    """

    def __init__(
        self,
        A,
        b,
        algorithm="als",
        truncator=None,
        minimize_residual=False,
        ranks=None,
        dense_max_size=1000,
        local_tolerance=1e-12,
        local_max_iterations=None,
        check_residual=1,
        kick_rank=4,
        local_preconditioner="block_jacobi",
        local_method="auto",
        **kwargs,
    ):
        super().__init__(A, b, **kwargs)
        if algorithm not in ("als", "dmrg", "amen"):
            raise ValueError("algorithm must be 'als', 'dmrg' or 'amen'.")
        if A.order != b.order or any(m != n for (_, m), n in zip(A.shape, b.shape)):
            raise ValueError("The operator and the right-hand side are not compatible.")
        self.algorithm = algorithm
        if truncator is None:
            truncator = tensap.Truncator(tolerance=1e-12, max_rank=100)
        self.truncator = truncator
        self.minimize_residual = minimize_residual
        self.ranks = ranks
        self.dense_max_size = dense_max_size
        self.local_tolerance = local_tolerance
        self.local_max_iterations = local_max_iterations
        self.check_residual = check_residual
        self.kick_rank = kick_rank
        if local_preconditioner not in (None, "block_jacobi"):
            raise ValueError("local_preconditioner must be None or 'block_jacobi'.")
        self.local_preconditioner = local_preconditioner
        if local_method not in ("auto", "cg", "gmres"):
            raise ValueError("local_method must be 'auto', 'cg' or 'gmres'.")
        self.local_method = local_method
        if check_residual < 1 or self.max_iterations < 0:
            raise ValueError("check_residual must be >= 1 and max_iterations >= 0.")

    def is_symmetric(self, tolerance=1e-12):
        """True if the operator of the solved system is symmetric."""
        if self.minimize_residual:
            return True
        A = self.A
        return (A - A.T).norm() <= tolerance * A.norm()

    def solve(self):
        """Alternating sweeps (back and forth for ALS and DMRG, from left to
        right for AMEn) until the error criterion is below the tolerance or
        the iterate stagnates.

        Returns
        -------
        x : tensap.TTTensor
            The computed solution.
        output : dict
            Keys 'flag' (0: tolerance reached, 1: maximal number of
            iterations, 2: stagnation), 'iter', 'error', 'errvec', 'ranks'.
        """
        A, b = self.A, self.b
        if self.b.norm() == 0:
            # x = 0 is the solution; the sweeps would divide by its norm
            x = tensap.TTTensor.zeros(self.b.shape)
            return x, {"flag": 0, "iter": 0, "error": 0.0, "errvec": [], "ranks": x.ranks,
                       "local": LocalDiagnostics().as_dict()}
        if self.minimize_residual:
            A, b = A.T @ A, A.T @ b
        if self.error_criterion is None:
            b_norm = self.b.norm()

            def error_criterion(x):
                return (self.b - self.A @ x).norm() / b_norm

        else:
            error_criterion = self.error_criterion

        x = self.x0
        if x is None:
            ranks = self.ranks
            if ranks is None:
                ranks = 5 if self.algorithm == "als" else 2
            ranks = np.broadcast_to(ranks, (b.order - 1,))
            x = tensap.TTTensor.randn(b.shape, ranks)
        sweep = _Sweep(A, b, x, self.is_symmetric(), self)
        algorithm = self.algorithm if b.order > 1 else "als"

        flag, errvec, error, n = 1, [], np.inf, 0
        for n in range(1, self.max_iterations + 1):
            x_old = sweep.tensor()
            forward = n % 2 == 1
            if algorithm == "dmrg":
                sweep.dmrg_half_sweep(forward)
            elif algorithm == "amen":
                sweep.amen_sweep()
            else:
                sweep.als_half_sweep(forward)
            x = sweep.tensor()
            stagnation = (x - x_old).norm() / x.norm()
            checked = n % self.check_residual == 0 or n == self.max_iterations
            if checked:
                error = error_criterion(x)
                errvec.append(error)
            if self.display:
                print(
                    f"{self.algorithm.upper()}: Iteration {n:3d} - Stagnation "
                    f"{stagnation:.2e} - Error {error:.2e} - Ranks {x.ranks}"
                )
            if checked and error < self.tolerance:
                flag = 0
                break
            if stagnation < self.stagnation:
                flag = 2
                break

        if self.max_iterations > 0 and not checked:
            # stopped (stagnation) between two scheduled checks: the error
            # reported is that of the returned iterate
            error = error_criterion(x)
            errvec.append(error)
        output = {
            "flag": flag,
            "iter": n,
            "error": error,
            "errvec": errvec,
            "ranks": x.ranks,
            "local": sweep.diagnostics.as_dict(),
        }
        return x, output


class _Sweep:
    """Cores of the iterate with the interfaces of the operator and of the
    right-hand side, kept up to date while the orthogonality center moves.

    For the core k, left_a[k] (shape (r_{k-1}, rA_{k-1}, r_{k-1})) contracts
    the cores 0, ..., k-1 of x, A and x, and right_a[k] (shape (r_k, rA_k,
    r_k)) the cores k+1, ..., d-1; left_b[k] and right_b[k] do the same with
    x and b. They are valid when the cores on the left of k are
    left-orthogonal and those on the right right-orthogonal.
    """

    def __init__(self, A, b, x, symmetric, options):
        self.a_cores = list(A.cores)
        self.b_cores = list(b.cores)
        self.cores = [np.array(c) for c in x.orth(0).cores]
        self.symmetric = symmetric
        self.options = options
        self.diagnostics = LocalDiagnostics()
        d = len(self.cores)
        self.left_a = [np.ones((1, 1, 1))] + [None] * (d - 1)
        self.right_a = [None] * (d - 1) + [np.ones((1, 1, 1))]
        self.left_b = [np.ones((1, 1))] + [None] * (d - 1)
        self.right_b = [None] * (d - 1) + [np.ones((1, 1))]
        for k in range(d - 1, 0, -1):
            self._update_right(k)

    def tensor(self):
        return tensap.TTTensor(self.cores)

    # --- interfaces ---

    @staticmethod
    def _left_step(left_a, left_b, test, trial, a, b):
        """Interfaces of the next core: one more core of (test, A, trial) and
        of (test, b) contracted from the left."""
        t = np.tensordot(left_a, test, axes=(0, 0))
        t = np.tensordot(t, a, axes=([0, 2], [0, 1]))
        t_b = np.tensordot(left_b, test, axes=(0, 0))
        return (
            np.tensordot(t, trial, axes=([0, 2], [0, 1])),
            np.tensordot(t_b, b, axes=([0, 1], [0, 1])),
        )

    @staticmethod
    def _right_step(right_a, right_b, test, trial, a, b):
        """Interfaces of the previous core, contracted from the right."""
        t = np.tensordot(test, right_a, axes=(2, 0))
        t = np.tensordot(t, a, axes=([1, 2], [1, 3]))
        t_b = np.tensordot(test, right_b, axes=(2, 0))
        return (
            np.tensordot(t, trial, axes=([1, 3], [2, 1])),
            np.tensordot(t_b, b, axes=([1, 2], [1, 2])),
        )

    def _update_left(self, k):
        """Interfaces of the core k + 1 from those of the core k."""
        x = self.cores[k]
        self.left_a[k + 1], self.left_b[k + 1] = self._left_step(
            self.left_a[k], self.left_b[k], x, x, self.a_cores[k], self.b_cores[k]
        )

    def _update_right(self, k):
        """Interfaces of the core k - 1 from those of the core k."""
        x = self.cores[k]
        self.right_a[k - 1], self.right_b[k - 1] = self._right_step(
            self.right_a[k], self.right_b[k], x, x, self.a_cores[k], self.b_cores[k]
        )

    def _residual(self, k, core, left_a, left_b, right_a, right_b):
        """Projection of b - A x, x with the core k replaced by core, on the
        test bases of the given interfaces."""
        t_b = np.tensordot(left_b, self.b_cores[k], axes=(1, 0))
        t_b = np.tensordot(t_b, right_b, axes=(2, 1))
        t = np.tensordot(left_a, core, axes=(2, 0))
        t = np.tensordot(t, self.a_cores[k], axes=([1, 2], [0, 2]))
        t = np.tensordot(t, right_a, axes=([1, 3], [2, 1]))
        return t_b - t

    # --- local problem on the cores k, ..., k + p - 1 ---

    def _local_solve(self, k, p, guess):
        """Solution of the Galerkin system on the cores k, ..., k + p - 1, an
        array of shape (r_{k-1}, n_k, ..., n_{k+p-1}, r_{k+p-1})."""
        left, right = self.left_a[k], self.right_a[k + p - 1]
        a_cores = self.a_cores[k:k + p]
        rhs = np.tensordot(self.left_b[k], self.b_cores[k], axes=(1, 0))
        for b in self.b_cores[k + 1:k + p]:
            rhs = np.tensordot(rhs, b, axes=(-1, 0))
        rhs = np.tensordot(rhs, self.right_b[k + p - 1], axes=(-1, 1))
        shape = rhs.shape
        size = rhs.size
        options = self.options

        if size <= options.dense_max_size:
            if p == 1:
                matrix = np.einsum("xay,aijb,cbd->xicyjd", left, a_cores[0], right, optimize=True)
            else:
                matrix = np.einsum(
                    "xay,aijb,bklg,cgd->xikcyjld",
                    left,
                    a_cores[0],
                    a_cores[1],
                    right,
                    optimize=True,
                )
            solution = np.linalg.solve(matrix.reshape(size, size), rhs.ravel())
            self.diagnostics.direct()
            return solution.reshape(shape)

        def matvec(vector):
            t = np.tensordot(left, vector.reshape(shape), axes=(2, 0))
            t = np.tensordot(t, a_cores[0], axes=([1, 2], [0, 2]))
            for a in a_cores[1:]:
                t = np.tensordot(t, a, axes=([t.ndim - 1, 1], [0, 2]))
            t = np.tensordot(t, right, axes=([1, t.ndim - 1], [2, 1]))
            return t.ravel()

        operator = spla.LinearOperator((size, size), matvec=matvec, dtype=float)
        preconditioner = None
        if options.local_preconditioner == "block_jacobi":
            preconditioner = self._block_jacobi(left, a_cores, right, shape)
        method = self._local_method()
        solution, info = method(
            operator,
            rhs.ravel(),
            x0=guess.ravel(),
            rtol=options.local_tolerance,
            atol=0.0,
            maxiter=options.local_max_iterations,
            M=preconditioner,
        )
        self.diagnostics.iterative_solve(info, operator, rhs.ravel(), solution)
        return solution.reshape(shape)

    def _local_method(self):
        """CG for an operator declared or detected symmetric (assumed positive
        definite: symmetry alone does not make CG applicable), else GMRES."""
        choice = self.options.local_method
        if choice == "cg" or (choice == "auto" and self.symmetric):
            return spla.cg
        return spla.gmres

    @staticmethod
    def _block_jacobi(left, a_cores, right, shape):
        """Block Jacobi preconditioner of the local operator: the blocks of
        the physical indices for each pair (k, k') of rank indices,
        B_{k k'} = sum L(k, a, k) A(a, :, :, b) R(k', b, k'), inverted once
        (the preconditioner 'c' of torchTT). A principal submatrix of an SPD
        matrix is SPD, so the preconditioner is SPD for CG."""
        left_diag = np.einsum("xax->xa", left)
        right_diag = np.einsum("cbc->cb", right)
        a = a_cores[0]
        for other in a_cores[1:]:
            a = np.einsum("aijg,gklb->aikjlb", a, other).reshape(
                a.shape[0], a.shape[1] * other.shape[1], a.shape[2] * other.shape[2], -1
            )
        blocks = np.einsum("xa,aijb,cb->xcij", left_diag, a, right_diag)
        inverses = np.linalg.inv(blocks)  # (r0, r1, n, n)
        r0, r1 = shape[0], shape[-1]
        n = int(np.prod(shape[1:-1]))

        def apply(vector):
            v = vector.reshape(r0, n, r1)
            return np.einsum("xcij,xjc->xic", inverses, v).ravel()

        size = r0 * n * r1
        return spla.LinearOperator((size, size), matvec=apply, dtype=float)

    # --- half sweeps ---

    def als_half_sweep(self, forward):
        """Update every core once, from left to right or from right to left,
        the orthogonality center following the updated core."""
        d = len(self.cores)
        for k in range(d) if forward else range(d - 1, -1, -1):
            self.cores[k] = self._local_solve(k, 1, self.cores[k])
            r0, n, r1 = self.cores[k].shape
            if forward and k < d - 1:
                q, r = np.linalg.qr(self.cores[k].reshape(r0 * n, r1))
                self.cores[k] = q.reshape(r0, n, -1)
                self.cores[k + 1] = np.tensordot(r, self.cores[k + 1], axes=(1, 0))
                self._update_left(k)
            elif not forward and k > 0:
                q, r = np.linalg.qr(self.cores[k].reshape(r0, n * r1).T)
                self.cores[k] = q.T.reshape(-1, n, r1)
                self.cores[k - 1] = np.tensordot(self.cores[k - 1], r.T, axes=(-1, 0))
                self._update_right(k)

    def dmrg_half_sweep(self, forward):
        """Update every pair of neighbouring cores once; the supercore is
        split by a truncated SVD, which sets the rank of the bond."""
        d = len(self.cores)
        truncator = self.options.truncator
        for k in range(d - 1) if forward else range(d - 2, -1, -1):
            guess = np.tensordot(self.cores[k], self.cores[k + 1], axes=(2, 0))
            supercore = self._local_solve(k, 2, guess)
            r0, n0, n1, r1 = supercore.shape
            matrix = supercore.reshape(r0 * n0, n1 * r1)
            if forward:
                left, right = _truncated_svd(truncator, matrix, truncator.tolerance)
                self.cores[k] = left.reshape(r0, n0, -1)
                self.cores[k + 1] = right.reshape(-1, n1, r1)
                self._update_left(k)
            else:
                left, right = _truncated_svd(truncator, matrix.T, truncator.tolerance)
                self.cores[k] = right.T.reshape(r0, n0, -1)
                self.cores[k + 1] = left.T.reshape(-1, n1, r1)
                self._update_right(k + 1)

    def amen_sweep(self):
        """One AMEn sweep from left to right: each core is updated by the
        local Galerkin problem, truncated by an SVD, then enriched by
        kick_rank directions of the residual b - A x; the residual itself is
        approximated by a TT tensor z of rank kick_rank, updated along the
        sweep (Dolgov and Savostyanov 2014)."""
        d = len(self.cores)
        options = self.options
        truncator = options.truncator
        shape = [c.shape[1] for c in self.cores]
        if getattr(self, "z_cores", None) is None:
            z = tensap.TTTensor.randn(shape, [options.kick_rank] * (d - 1))
            self.z_left_a = [np.ones((1, 1, 1))] + [None] * (d - 1)
            self.z_left_b = [np.ones((1, 1))] + [None] * (d - 1)
            self.z_right_a = [None] * (d - 1) + [np.ones((1, 1, 1))]
            self.z_right_b = [None] * (d - 1) + [np.ones((1, 1))]
        else:
            z = tensap.TTTensor(self.z_cores)
            self.cores = [np.array(c) for c in tensap.TTTensor(self.cores).orth(0).cores]
            for k in range(d - 1, 0, -1):
                self._update_right(k)
        self.z_cores = [np.array(c) for c in z.orth(0).cores]
        for k in range(d - 1, 0, -1):
            self.z_right_a[k - 1], self.z_right_b[k - 1] = self._right_step(
                self.z_right_a[k], self.z_right_b[k], self.z_cores[k], self.cores[k],
                self.a_cores[k], self.b_cores[k],
            )

        for k in range(d - 1):
            core = self._local_solve(k, 1, self.cores[k])
            r0, n, r1 = core.shape
            u, v = _truncated_svd(truncator, core.reshape(r0 * n, r1), truncator.tolerance)
            core = (u @ v).reshape(r0, n, r1)
            z_core = self._residual(k, core, self.z_left_a[k], self.z_left_b[k],
                                    self.z_right_a[k], self.z_right_b[k])
            enrichment = self._residual(k, core, self.left_a[k], self.left_b[k],
                                        self.z_right_a[k], self.z_right_b[k])
            enrichment = enrichment.reshape(r0 * n, -1)
            kick = int(max(0, min(enrichment.shape[1], truncator.max_rank - u.shape[1])))
            q, r = np.linalg.qr(np.hstack([u, enrichment[:, :kick]]))
            v = np.vstack([v, np.zeros((kick, r1))])
            self.cores[k] = q.reshape(r0, n, -1)
            self.cores[k + 1] = np.tensordot(r @ v, self.cores[k + 1], axes=(1, 0))
            self._update_left(k)
            rz0, _, rz1 = z_core.shape
            z_q, _ = np.linalg.qr(z_core.reshape(rz0 * n, rz1))
            self.z_cores[k] = z_q.reshape(rz0, n, -1)
            self.z_left_a[k + 1], self.z_left_b[k + 1] = self._left_step(
                self.z_left_a[k], self.z_left_b[k], self.z_cores[k], self.cores[k],
                self.a_cores[k], self.b_cores[k],
            )
        k = d - 1
        self.cores[k] = self._local_solve(k, 1, self.cores[k])
        self.z_cores[k] = self._residual(k, self.cores[k], self.z_left_a[k], self.z_left_b[k],
                                         self.z_right_a[k], self.z_right_b[k])
