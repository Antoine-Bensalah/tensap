# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module inverse_approximators.

Low-rank approximation of the inverse of an operator in tensor format, used
as a preconditioner. Port of ApproximationToolbox/Approximation/
TensorApproximation/EquationsSolvers/RankOneInverseApproximatorByALS.m and
GreedyInverseApproximator.m.

For A symmetric positive definite, both minimize over operators P the
functional J(P) = 1/2 <P A, P> - <B, P>, whose minimizer is P = B A^{-1}.
"""

import time
import warnings

import numpy as np
import tensap
from .linear_solvers import _contract_except_dim


def _frobenius(M1, M2):
    """Frobenius inner product of two matrices (dense or sparse)."""
    return float(np.sum(np.asarray(M1) * np.asarray(M2)))


class RankOneInverseApproximatorByALS:
    """
    Rank-one approximation W = a v^(1) x ... x v^(d) of the solution of
    W A = B, by alternating least squares: for each dimension, the other
    factors being fixed, the factor v solves the matrix equation v Q = Z.

    Attributes
    ----------
    A : TuckerTensorOperator
        Operator to approximately invert.
    B : TuckerTensorOperator
        Right-hand side (the identity, or a residual B - P A).
    P0 : TuckerTensorOperator, optional
        Kept for compatibility with the MATLAB class (not used by solve).
    W0 : TuckerTensorOperator, optional
        Rank-one initial guess (default: random factors).
    max_iterations : int
        Maximum number of sweeps over the dimensions.
    stagnation : float
        Stagnation criterion on the scaling factor a.
    display : bool
    symmetric : bool or list of bool
        Dimensions with a symmetric factor v, solution of the Lyapunov
        equation v Q + Q v = Z + Z^T.
    sparse : bool or list of bool
        Dimensions with a sparse factor v, computed by gspai.
    sparse_options : tuple
        Arguments of gspai after (Q, Z), default ('auto', 100, 1e-5) as in
        MATLAB.
    """

    def __init__(
        self,
        A,
        B,
        P0=None,
        W0=None,
        max_iterations=5,
        stagnation=1e-6,
        display=True,
        symmetric=False,
        sparse=False,
        sparse_options=("auto", 100, 1e-5),
    ):
        self.A = A
        self.B = B
        self.P0 = P0
        self.W0 = W0
        self.max_iterations = max_iterations
        self.stagnation = stagnation
        self.display = display
        self.sparse_options = tuple(sparse_options)

        order = A.order
        self.symmetric = np.broadcast_to(np.atleast_1d(symmetric), (order,)).copy()
        self.sparse = np.broadcast_to(np.atleast_1d(sparse), (order,)).copy()
        if np.any(self.symmetric & self.sparse):
            raise ValueError("Approximation can not be sparse AND symmetric.")

    def _local_solve(self, Q, Z, mu):
        """Solve v Q = Z in dimension mu, with the constraint of that dimension."""
        if self.symmetric[mu]:
            if Q.shape[0] < 1500:
                return tensap.lyapsym(Q, Z + Z.T)
            warnings.warn("Symmetric constraint: too large for lyapsym.")
            v = np.linalg.solve(Q.T, Z.T).T
            return (v + v.T) / 2
        if self.sparse[mu]:
            return tensap.gspai(Q, Z, *self.sparse_options).toarray()
        return np.linalg.solve(Q.T, Z.T).T

    def solve(self):
        """Compute the rank-one approximation.

        Returns
        -------
        W : TuckerTensorOperator
            Rank-one operator, with a DiagonalTensor core (a).
        output : dict
            Dictionary with keys 'flag' (1: maximum number of iterations,
            2: stagnation), 'stagnation', 'iter'.
        """
        A, B = self.A, self.B
        order = A.order

        if self.W0 is None:
            dims1 = [shape[0] for shape in B.shape]
            dims2 = [shape[1] for shape in B.shape]
            W_frames = tensap.TFrameOperator.randn(dims1, dims2, [1] * order)
        else:
            W_frames = self.W0.tensor_frame
        W_core = tensap.DiagonalTensor(np.array([1.0]), order)

        # reduced quantities <A_i, v^T v> (R_A x 1) and <B_l, v> (R_B x 1)
        def reduced(mu, v):
            WAW = np.array([[_frobenius(Ai, v.T @ v)] for Ai in A.tensor_frame.frames[mu].data])
            BW = np.array([[_frobenius(Bl, v)] for Bl in B.tensor_frame.frames[mu].data])
            return WAW, BW

        WAW, BW = map(list, zip(*(reduced(mu, W_frames.frames[mu].data[0]) for mu in range(order))))

        a = 1.0
        flag = 1
        stagnation = None
        k = 0
        for k in range(1, self.max_iterations + 1):
            a_old = a
            for mu in range(order):
                core_B, core_W = tensap.convert_tensors(B.core, W_core)
                coef = _contract_except_dim(core_B, core_W, BW, mu)
                Z = B.tensor_frame.eval_in_space(mu, np.ravel(coef))

                core_A, core_W = tensap.convert_tensors(A.core, W_core)
                coef = _contract_except_dim(core_A, core_W, WAW, mu)
                Q = A.tensor_frame.eval_in_space(mu, np.ravel(coef))

                v = self._local_solve(Q, Z, mu)
                a = np.linalg.norm(v)
                v = v / a

                frames = list(W_frames.frames)
                frames[mu] = tensap.FrameOperator(v[None, :, :])
                W_frames = tensap.TFrameOperator(frames)
                W_core = tensap.DiagonalTensor(np.array([a]), order)
                WAW[mu], BW[mu] = reduced(mu, v)

            stagnation = abs(a_old - a) / abs(a_old + a)
            if self.display:
                print(f"Iteration {k:3d} - Stagnation {stagnation:.2e}")
            if k > 1 and stagnation < self.stagnation:
                flag = 2
                break

        W = tensap.TuckerTensorOperator(W_core, W_frames)
        output = {"flag": flag, "stagnation": stagnation, "iter": k}
        return W, output


class GreedyInverseApproximator:
    """
    Greedy approximation of the inverse of an operator: P = sum_k W_k, each
    W_k being a rank-one approximation of the solution of W A = B - P_{k-1} A.

    Attributes
    ----------
    A : TuckerTensorOperator
        Operator to approximately invert.
    I : TuckerTensorOperator
        Identity (default: tensap.BaseTuckerTensor.eye).
    P0 : TuckerTensorOperator, optional
        Initial approximation.
    minimize_residual : bool
        If True, work with A A^T and I A^T, i.e. minimize ||I - P A||.
    local_solver : callable
        Function (A, B) -> solver with attribute B and method solve()
        returning (W, output). Default: RankOneInverseApproximatorByALS.
    update : callable, optional
        Function (A, B, P) -> updated P, applied after each correction.
    max_iterations : int
    tolerance : float
        Tolerance on the relative residual ||I - P A|| / ||I||.
    stagnation : float
    display : bool
    check_residual : int
        The residual is computed every check_residual iterations.
    """

    def __init__(
        self,
        A,
        I=None,
        P0=None,
        minimize_residual=False,
        local_solver=None,
        update=None,
        max_iterations=5,
        tolerance=1e-6,
        stagnation=1e-6,
        display=True,
        check_residual=1,
    ):
        self.A = A
        self.I = (  # noqa: E741 (the target I of ||I - P A||)
            tensap.BaseTuckerTensor.eye([shape[0] for shape in A.shape])
            if I is None
            else I
        )
        self.P0 = P0
        self.minimize_residual = minimize_residual
        self.local_solver = (
            (lambda A, B: RankOneInverseApproximatorByALS(A, B, display=False))
            if local_solver is None
            else local_solver
        )
        self.update = update
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.stagnation = stagnation
        self.display = display
        self.check_residual = check_residual

    def solve(self):
        """Compute the approximation of the inverse.

        Returns
        -------
        P : TuckerTensorOperator
        output : dict
            Dictionary with keys 'iter', 'flag' (0: tolerance, 1: maximum
            number of iterations, 2: stagnation), 'stagnation', 'relres'
            (last relative residual ||I - P A|| / ||I||), 'resvec' (relative
            residuals, nan where not computed), 'time'.
        """
        clock0 = time.perf_counter()
        Id = self.I
        if not self.minimize_residual:
            A, B = self.A, Id
        else:
            A = self.A @ self.A.T
            B = Id @ self.A.T

        norm_I = Id.norm()

        def relative_residual(P):
            return (Id - P @ self.A).norm() / norm_I

        P = self.P0
        R = B if P is None else B - P @ A
        local_solver = self.local_solver(A, R)

        flag = 1
        resvec = []
        stagnation = None
        k = 0
        for k in range(1, self.max_iterations + 1):
            local_solver.B = R
            W, _ = local_solver.solve()

            P_old = P
            P = W if P is None else P + W
            if self.update is not None:
                P = self.update(A, B, P)

            stagnation = 1.0 if P_old is None else (P - P_old).norm() / P.norm()
            residual = (
                relative_residual(P) if k % self.check_residual == 0 else np.nan
            )
            resvec.append(residual)

            if self.display:
                message = f"Iteration {k:3d} - Stagnation {stagnation:.2e}"
                if not np.isnan(residual):
                    message += f" - Residual {residual:.2e}"
                print(message)

            if stagnation < self.stagnation:
                flag = 2
                break
            if not np.isnan(residual) and residual < self.tolerance:
                flag = 0
                break

            R = B - P @ A

        if np.isnan(resvec[-1]):
            resvec[-1] = relative_residual(P)

        output = {
            "iter": k,
            "flag": flag,
            "stagnation": stagnation,
            "relres": resvec[-1],
            "resvec": resvec,
            "time": time.perf_counter() - clock0,
        }
        return P, output
