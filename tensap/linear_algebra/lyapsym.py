# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module lyapsym.

Port of ApproximationToolbox/LinearAlgebra/lyapsym.m.
"""

import numpy as np
import scipy.sparse as sp


def lyapsym(A, C):
    """
    Solve the Lyapunov equation X A + A X = C for a symmetric matrix A.

    With the eigendecomposition A = U diag(r) U^T, the solution is
    X = U Y U^T with Y(i, j) = (U^T C U)(i, j) / (r_i + r_j). X is symmetric
    when C is.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
        Symmetric matrix, with r_i + r_j != 0 for all eigenvalues r_i, r_j
        (e.g. A positive definite).
    C : numpy.ndarray or scipy.sparse matrix
        Right-hand side, same shape as A.

    Returns
    -------
    numpy.ndarray
        The solution X.

    """
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    C = C.toarray() if sp.issparse(C) else np.asarray(C)

    r, U = np.linalg.eigh(A)
    Y = (U.T @ C @ U) / (r[:, None] + r[None, :])
    return U @ Y @ U.T
