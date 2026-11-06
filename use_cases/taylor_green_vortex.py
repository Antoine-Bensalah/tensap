"""
Decay of a Taylor-Green vortex
==============================

The Taylor-Green vortex is a classical test of incompressible flow
solvers: an array of counter-rotating vortices that decays under the
effect of viscosity, for which the Navier-Stokes equations have an exact
solution. We solve it with finite volumes on a staggered grid whose
fields and operators are tensor trains in QTT format, check the accuracy
and the conservation properties of the scheme, and look at what limits
the compression for a flow that is not separable.

This use case comes from the script
``test/non_regression_scripts/solvers/tutorial_staggered_taylor_green.py``,
on coarser grids and fewer time steps.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensap
from tensap.tensorized_pde_discretization.finite_volumes import (
    StaggeredAxis,
    StaggeredGrid,
)

# %%
# The model
# ---------
#
# The velocity :math:`u = (u_1, u_2)` and the pressure :math:`p` of an
# incompressible fluid of kinematic viscosity :math:`\nu` solve
#
# .. math::
#
#    \partial_t u + \nabla \cdot (u \otimes u) + \nabla p = \nu \Delta u,
#    \qquad \nabla \cdot u = 0,
#
# here on the periodic square :math:`(0, 2\pi)^2`. The Taylor-Green
# vortex
#
# .. math::
#
#    \begin{aligned}
#    u_1 &= \sin x \cos y \, F(t), & u_2 &= -\cos x \sin y \, F(t), \\
#    p &= \frac{\cos 2x + \cos 2y}{4} F(t)^2, & F(t) &= e^{-2 \nu t},
#    \end{aligned}
#
# is an exact solution: the convective term is balanced by the pressure
# gradient, and the vortices decay at the rate :math:`2 \nu`. We take
# :math:`\nu = 0.01` up to :math:`t = 1`.
NU, T_END = 0.01, 1.0


def exact(grid, t):
    """The exact velocity at the faces and pressure at the cells."""
    xu, yu = grid.positions("u")
    xv, yv = grid.positions("v")
    xp, yp = grid.positions("p")
    F = np.exp(-2 * NU * t)
    return (
        np.sin(xu) * np.cos(yu) * F,
        -np.cos(xv) * np.sin(yv) * F,
        (np.cos(2 * xp) + np.cos(2 * yp)) * F**2 / 4,
    )


# %%
# The discretization
# ------------------
#
# On a staggered (MAC) grid of :math:`2^l \times 2^l` cells, the pressure
# lives at the centres of the cells and each velocity component at the
# faces normal to its direction
# (:class:`~tensap.tensorized_pde_discretization.finite_volumes.StaggeredGrid`).
# Every field has :math:`2^l` values per direction and is stored as a QTT
# tensor of order :math:`2l`; the discrete operators (gradient,
# divergence, Laplacians) are :class:`~tensap.TTOperator` objects of
# small ranks, independent of the resolution.
#
# The scheme has three ingredients:
#
# - the convective term :math:`\nabla \cdot (u \otimes u)` in conservative
#   form (``convection``): the products of the components are formed by
#   averages at the cells and at the corners, at the precision of a
#   :class:`~tensap.Truncator` (entrywise products of QTT tensors), and
#   differenced back to the faces. It conserves the momentum and, for a
#   divergence-free field, the kinetic energy;
# - the Helmholtz-Hodge projection (``project``) onto the
#   divergence-free fields, which solves a pressure Poisson equation by
#   AMEn (:class:`~tensap.TTTensorALSLinearSolver`);
# - the time integration by Heun's method (explicit Runge-Kutta of order
#   2) on the projected equations
#   :math:`\partial_t u = P(-N(u) + \nu L u)`.
#
# All the iterates are rounded at the precision :math:`10^{-10}`.
TRUNCATOR = tensap.Truncator(tolerance=1e-10, max_rank=50)
SOLVER = {"tolerance": 1e-9, "stagnation": 1e-9}


def rhs(grid, velocity, laplacian):
    """F(u) = -N(u) + nu L u, rounded."""
    N = grid.convection(velocity, truncator=TRUNCATOR)
    return [
        TRUNCATOR.truncate(L @ w * NU - n)
        for L, w, n in zip(laplacian, velocity, N)
    ]


def step(grid, velocity, laplacian, dt):
    """One step of Heun's method, each stage projected."""
    f0 = rhs(grid, velocity, laplacian)
    stage = [
        TRUNCATOR.truncate(w + f * dt) for w, f in zip(velocity, f0)
    ]
    u1, _ = grid.project(stage, truncator=TRUNCATOR, **SOLVER)
    f1 = rhs(grid, u1, laplacian)
    stage = [
        TRUNCATOR.truncate(w + (a + b) * (dt / 2))
        for w, a, b in zip(velocity, f0, f1)
    ]
    u2, _ = grid.project(stage, truncator=TRUNCATOR, **SOLVER)
    return u2


def make_grid(levels):
    grid = StaggeredGrid(
        [StaggeredAxis(2 * np.pi, levels, periodic=True)] * 2
    )
    laplacian = [
        grid.laplacian(c, [(None, None)] * 2) for c in ("u", "v")
    ]
    return grid, laplacian


# %%
# The vortex up to :math:`t = 1`
# ------------------------------
#
# On :math:`32^2` and :math:`64^2` cells, with a time step proportional
# to the cell size (CFL number 0.3), the error on the velocity at the
# final time decreases by a factor 4 when the grid is refined: the
# scheme is of order 2. The kinetic energy decays as
# :math:`E(t) = E(0) F(t)^2`, up to an error of order 2 as well, and the
# divergence of the computed velocity
# stays at the precision of the projection.
results = {}
for levels, steps in ((5, 16), (6, 32)):
    grid, laplacian = make_grid(levels)
    u, v, _ = exact(grid, 0.0)
    velocity = [grid.quantize(u), grid.quantize(v)]
    energy0 = sum(w.dot(w) for w in velocity)
    start = time()
    for _ in range(steps):
        velocity = step(grid, velocity, laplacian, T_END / steps)
    elapsed = time() - start
    u_ex, v_ex, _ = exact(grid, T_END)
    values = [grid.dequantize(w) for w in velocity]
    error = max(
        np.abs(values[0] - u_ex).max(), np.abs(values[1] - v_ex).max()
    )
    energy = sum(w.dot(w) for w in velocity) / energy0
    D = grid.divergence([None, None])
    divergence = np.abs(
        sum(grid.dequantize(Dk @ w) for Dk, w in zip(D, velocity))
    ).max()
    results[levels] = (grid, values)
    print(
        f"{grid.shape[0]}^2 cells, {steps} steps: {elapsed:.1f} s, "
        f"max error on u {error:.1e}, "
        f"E(1) / E(0) - F(1)^2 {energy - np.exp(-4 * NU * T_END):.1e}, "
        f"max |div u| {divergence:.1e}, "
        f"QTT ranks of u {max(velocity[0].ranks)}"
    )

# %%
# The vorticity :math:`\omega = \partial_x u_2 - \partial_y u_1` at the
# final time, by differences of the computed velocity on the finest grid.
grid, (u1, u2) = results[6]
h = 2 * np.pi / grid.shape[0]
vorticity = (np.roll(u2, -1, axis=0) - u2) / h - (
    np.roll(u1, -1, axis=1) - u1
) / h
plt.figure()
plt.imshow(vorticity.T, origin="lower", extent=[0, 2 * np.pi] * 2)
plt.colorbar(label="vorticity")
plt.title("Taylor-Green vortex at t = 1")
plt.show()

# %%
# Why so cheap?
# -------------
#
# The QTT ranks of the velocity are 2 at the final time, as at the
# initial time: each component is a
# product of a function of :math:`x` and a function of :math:`y`, and a
# sine sampled on :math:`2^l` points has QTT rank 2. The cost of a step
# grows like the logarithm of the number of cells, not like the number
# of cells.
#
# A general flow is not separable. We perturb the vortex by a smooth
# divergence-free field (the projection of a sum of products of sines),
# and follow the QTT ranks of the velocity on :math:`32^2` cells: the
# nonlinear term transfers energy between the modes and the ranks grow.
grid, laplacian = make_grid(5)
u, v, _ = exact(grid, 0.0)
xu, yu = grid.positions("u")
xv, yv = grid.positions("v")
u = u + 0.3 * np.sin(3 * xu + 1) * np.cos(2 * yu - 0.5) * np.exp(
    -np.cos(yu)
)
v = v + 0.3 * np.cos(xv) * np.sin(4 * yv)
velocity, _ = grid.project(
    [grid.quantize(u), grid.quantize(v)], truncator=TRUNCATOR, **SOLVER
)
ranks = [max(max(w.ranks) for w in velocity)]
for _ in range(16):
    velocity = step(grid, velocity, laplacian, T_END / 16)
    ranks.append(max(max(w.ranks) for w in velocity))
plt.figure()
plt.plot(
    np.linspace(0, T_END, 17), ranks, "o-", label="perturbed vortex"
)
plt.plot([0, T_END], [2, 2], "k--", label="Taylor-Green vortex")
plt.axhline(grid.shape[0], color="gray", label="largest possible rank")
plt.xlabel("t")
plt.ylabel("max QTT rank of the velocity")
plt.legend()
plt.show()
print(
    f"QTT ranks of the perturbed vortex: from {ranks[0]} to {ranks[-1]}"
)

# %%
# Discussion
# ----------
#
# - The staggered scheme in QTT format keeps the properties of the
#   classical one: second order accuracy, exactly divergence-free
#   velocity up to the precision of the solver, conservation of momentum
#   and of kinetic energy by the convective term.
# - Its cost depends on the QTT ranks of the fields, not on the number of
#   cells. For a separable flow such as the Taylor-Green vortex, the
#   ranks are tiny and the grid can be refined almost for free.
# - For a general flow at the precision :math:`10^{-10}`, the ranks grow
#   towards the largest possible value on a coarse grid, and the QTT
#   format no longer compresses. It pays off on fine grids, for smooth
#   fields, or with a coarser truncation precision, which then sets the
#   accuracy of the simulation.
