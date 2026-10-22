# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module gspai.

Port of ApproximationToolbox/LinearAlgebra/gspai.m: generalized sparse
approximate inverse.
"""

import numpy as np
import scipy.sparse as sp


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)


def _solve_row(A, B_row, J):
    """Least-squares solution p of p @ A[J, :] = B_row (Matlab B(k,:)/A(J,:))."""
    return np.linalg.lstsq(A[J, :].T, B_row, rcond=None)[0]


def _next_index(P_row, A, B_row, zeros, nrA):
    """Index, among the zero entries of the row, whose inclusion reduces the
    residual ||p A - b|| the most."""
    r = P_row @ A - B_row
    candidates = np.flatnonzero(zeros)
    Ar = A[candidates, :] @ r
    reduced = np.linalg.norm(r) ** 2 - Ar**2 / nrA[candidates]
    return candidates[np.argmin(reduced)]


def gspai(A, B, pattern, *args):
    """
    Sparse approximation P of the solution of P A = B, computed row by row by
    least squares on a sparsity pattern.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
        Square matrix of size n.
    B : numpy.ndarray or scipy.sparse matrix
        Matrix of size n x n.
    pattern : str
        - 'band', bw: band of half-width bw - 1 (bw = 1 is diagonal);
        - 'diag': same as 'band' with bw = 1;
        - 'tridiag': same as 'band' with bw = 2;
        - 'auto', maxnz, tol: starting from 'diag', add to each row the entry
          that reduces the residual the most, up to maxnz entries per row or
          until ||P A - B|| / ||B|| < tol;
        - 'imposed', mask: the nonzero entries of the boolean n x n array
          mask;
        - 'power', powermax, tol: pattern of B, then of B A, B A^2, ...,
          until ||P A - B|| < tol or powermax is reached.
    *args
        The parameters of the pattern, as listed above.

    Returns
    -------
    scipy.sparse.csr_matrix
        The approximation P.

    """
    A = _dense(A)
    B = _dense(B)
    n = A.shape[0]

    if pattern == "band":
        bw = args[0]
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            mask[i, max(0, i - (bw - 1)) : min(n, i + bw)] = True
        return gspai(A, B, "imposed", mask)

    if pattern == "diag":
        return gspai(A, B, "band", 1)

    if pattern == "tridiag":
        return gspai(A, B, "band", 2)

    if pattern == "imposed":
        mask = np.asarray(args[0].toarray() if sp.issparse(args[0]) else args[0])
        mask = mask != 0
        P = np.zeros((n, n))
        for k in range(n):
            J = np.flatnonzero(mask[k])
            if J.size:
                P[k, J] = _solve_row(A, B[k], J)
        return sp.csr_matrix(P)

    if pattern == "auto":
        maxnz, tol = args[0], args[1]
        P = gspai(A, B, "diag").toarray()
        nB = np.linalg.norm(B, 2)
        nrA = np.sum(A**2, axis=1)
        for _ in range(maxnz - 1):
            for k in range(n):
                zeros = P[k] == 0
                if not zeros.any():
                    continue
                j = _next_index(P[k], A, B[k], zeros, nrA)
                J = ~zeros
                J[j] = True
                J = np.flatnonzero(J)
                P[k, J] = _solve_row(A, B[k], J)
            if np.linalg.norm(P @ A - B, 2) / nB < tol:
                break
        return sp.csr_matrix(P)

    if pattern == "power":
        powermax, tol = args[0], args[1]
        BA = B.copy()
        mask = BA != 0
        for power in range(powermax + 1):
            P = gspai(A, B, "imposed", mask)
            if np.linalg.norm(P @ A - B, 2) < tol or power == powermax:
                return P
            BA = BA @ A
            mask = mask | (BA != 0)

    raise ValueError(f"Unknown pattern {pattern!r}.")
