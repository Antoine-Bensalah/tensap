# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tt_products: the entrywise (Hadamard) product of two TT tensors,
rounded on the fly.

The exact product z = x * y has the cores x_k(i) (x) y_k(i), of ranks r_x r_y;
rounding it (TT-SVD) orthogonalizes these cores, at a cost O(d n r^6) for
r_x = r_y = r. Here z is computed directly at the requested precision by a
two-site DMRG fit of the projection: with the cores of z orthonormal on the
left of the bond (k, k + 1) and on its right, the best supercore is the
projection of x * y on these bases,

    S(s, i, j, s') = sum L_k(s, a, b) x_k(a, i, a') y_k(b, i, b')
                         x_{k+1}(a', j, a'') y_{k+1}(b', j, b'') R_{k+1}(s', a'', b''),

where L_k and R_{k+1} contract the left and right parts of z with those of x
and y. Its truncated SVD gives the two cores and the rank of the bond. Every
contraction involves one core of z, x and y at a time: O(d n^2 r^4) per sweep.
The sweeps alternate (left to right, right to left) until the norm of the
supercores stagnates.

The initial guess is a rounding of x (or a given tensor): its ranks are a
first estimate of those of z, which the SVDs then adapt (two-site DMRG can
increase a rank by a factor n per sweep).
"""

import numpy as np
import tensap


def _left_interface(L, z, x, y):
    """L'(s', a', b') = sum L(s, a, b) z(s, n, s') x(a, n, a') y(b, n, b')."""
    t = np.tensordot(L, z, axes=(0, 0))                     # (a, b, n, s')
    t = np.einsum("abnt,anc->bntc", t, x)                   # (b, n, s', a')
    return np.einsum("bntc,bnd->tcd", t, y)                 # (s', a', b')


def _right_interface(R, z, x, y):
    """R'(s, a, b) = sum z(s, n, s') x(a, n, a') y(b, n, b') R(s', a', b')."""
    t = np.tensordot(z, R, axes=(2, 0))                     # (s, n, a', b')
    t = np.einsum("sncd,anc->sadn", t, x)                   # (s, a, b', n)
    return np.einsum("sadn,bnd->sab", t, y)                 # (s, a, b)


def _supercore(L, R, xk, yk, xk1, yk1):
    t = np.tensordot(L, xk, axes=(1, 0))                    # (s, b, n, a')
    t = np.einsum("sbnc,bnd->sncd", t, yk)                  # (s, n, a', b')
    t = np.einsum("sncd,cme->snmde", t, xk1)                # (s, n, m, b', a'')
    t = np.einsum("snmde,dmf->snmef", t, yk1)               # (s, n, m, a'', b'')
    return np.tensordot(t, R, axes=([3, 4], [1, 2]))        # (s, n, m, s')


def hadamard_round(x, y, truncator, x0=None, max_sweeps=12, stagnation=None):
    """The entrywise product of two TTTensors at the precision of truncator.

    Parameters
    ----------
    x, y : tensap.TTTensor
        Of the same shape.
    truncator : tensap.Truncator
        Relative tolerance (each SVD at tolerance / sqrt(d - 1)) and maximal
        rank.
    x0 : tensap.TTTensor, optional
        Initial guess; default truncator.truncate(x).
    max_sweeps : int
        Maximal number of half sweeps.
    stagnation : float, optional
        Stop when the relative change ||z_n - z_{n-1}|| / ||z_n|| between two
        half sweeps is below it (default truncator.tolerance / 2). (A
        criterion on the norm alone stopped too early: the norm converges
        faster than the bases, errors 5 to 30 times the tolerance.)

    Returns
    -------
    z : tensap.TTTensor
        Orthogonality center on the first or the last core.
    info : dict
        'sweeps', 'norms' (norm of z after each half sweep), 'changes'.
    """
    x._check_compatible(y)
    d = x.order
    if d == 1:
        return truncator.truncate(x.hadamard_product(y)), {"sweeps": 0, "norms": []}
    if stagnation is None:
        stagnation = truncator.tolerance / 2
    z = truncator.truncate(x) if x0 is None else x0
    if z.norm() == 0 or x.norm() == 0 or y.norm() == 0:
        return x.hadamard_product(y) * 0.0, {"sweeps": 0, "norms": [0.0]}
    xs, ys = list(x.cores), list(y.cores)
    cores = [np.array(c) for c in z.orth(0).cores]
    local = truncator.tolerance / np.sqrt(d - 1)
    max_rank = np.broadcast_to(np.atleast_1d(truncator.max_rank), (d - 1,))
    one = np.ones((1, 1, 1))
    left = [one] + [None] * (d - 1)
    right = [None] * (d - 1) + [one]
    for k in range(d - 1, 0, -1):
        right[k - 1] = _right_interface(right[k], cores[k], xs[k], ys[k])

    svd_truncator = tensap.Truncator(tolerance=local)
    norms, changes, previous, sweep = [], [], None, 0
    for sweep in range(1, max_sweeps + 1):
        forward = sweep % 2 == 1
        bonds = range(d - 1) if forward else range(d - 2, -1, -1)
        for k in bonds:
            S = _supercore(left[k], right[k + 1], xs[k], ys[k], xs[k + 1], ys[k + 1])
            s0, n0, n1, s1 = S.shape
            svd_truncator.max_rank = max_rank[k]
            svd = svd_truncator.trunc_svd(S.reshape(s0 * n0, n1 * s1), local)
            u, v = svd.space
            sigma = svd.core.data
            if forward:
                cores[k] = u.reshape(s0, n0, -1)
                cores[k + 1] = (sigma[:, None] * v.T).reshape(-1, n1, s1)
                left[k + 1] = _left_interface(left[k], cores[k], xs[k], ys[k])
            else:
                cores[k] = (u * sigma[None, :]).reshape(s0, n0, -1)
                cores[k + 1] = v.T.reshape(-1, n1, s1)
                right[k] = _right_interface(right[k + 1], cores[k + 1], xs[k + 1], ys[k + 1])
        norms.append(float(np.linalg.norm(sigma)))
        current = tensap.TTTensor(cores)
        if previous is not None:
            changes.append((current - previous).norm() / norms[-1])
            if changes[-1] <= stagnation:
                break
        previous = current
    center = d - 1 if forward else 0
    info = {"sweeps": sweep, "norms": norms, "changes": changes}
    return tensap.TTTensor(cores, orth_center=center), info
