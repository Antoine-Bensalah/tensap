# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: time stepping on a staggered (MAC) grid in QTT format, the heat
equation and the unsteady Stokes equations in a channel (plan P9, level 2).

No MATLAB counterpart.

Part 1, heat equation (T3)
--------------------------
d f / dt = nu Delta f on (0, 1)^2, f = 0 on the boundary, for a velocity
component on its faces (u: faces along x, cells along y) and for a cell field.
The sine modes are eigenvectors of the discrete Laplacian, closures included:

    f = sin(k pi x) sin(m pi y),   L f = -(lambda_k + lambda_m) f,
    lambda_k = (4 / h^2) sin^2(k pi h / 2),

on the faces (f_0 = f_N = 0) as on the cells (ghost values -f_0, -f_{N-1}).
Explicit Euler multiplies the mode by 1 - dt nu lambda at each step, implicit
Euler by 1 / (1 + dt nu lambda): the QTT computation must reproduce these
factors to the precision of the truncation, and the continuous solution
exp(-nu (k^2 + m^2) pi^2 t) to O(h^2 + dt).

Part 2, unsteady Stokes in a channel (T4)
-----------------------------------------
du/dt + grad p = nu Delta u, div u = 0 on (0, 2) x (0, 1); no-slip walls at
y = 0 and y = 1; inflow at x = 0, u = U(y), v = 0; outflow at x = 2,
du/dx = 0, dv/dx = 0 and p = 0. Starting from rest, the flow converges to the
Poiseuille flow u = U(y), v = 0, p linear in x.

The inflow profile is stored on face 0 of u (a boundary face, kept unchanged by
the operators: interior_mask), and is the discrete Poiseuille profile U_h,
-L_y U_h = 8 with the cell closures of the walls (the parabola 4 y (1 - y) up to
O(h^2)): the discrete steady state is then exactly u = U_h(y), v = 0,
dp/dx = -8 nu.

Incremental pressure correction (Goda), implicit viscous step:

    (I - dt nu L) u* = u^n - dt G p^n         (AMEn, one solve per component)
    L_p phi = D u*,  u^{n+1} = u* - G phi,    (StaggeredGrid.project)
    p^{n+1} = p^n + phi / dt,

with the pressure conditions of the channel: homogeneous Neumann on the walls
and the inflow, homogeneous Dirichlet at the outlet (the pressure problem is
then not singular). The outlet face N of u is not stored: u*_N = u*_{N-1}
(outflow), and after the correction u_N = u*_N - (G phi)_N, the gradient at
that face using the ghost value -phi_{N-1}; the cells of the last column are
then exactly divergence-free.

Steps: the dense computation (matrices of the same operators) and the QTT one
on 32 x 16 cells; the QTT computation on 64 x 32 cells, against U_h and the
parabola (about 1.5 min: a slow tutorial).
"""

# %% Imports
import time

import numpy as np
import tensap
from tensap.tensorized_pde_discretization import Dirichlet, Neumann
from tensap.tensorized_pde_discretization.finite_volumes import StaggeredAxis, StaggeredGrid

# the criteria of the solvers are matched to the truncation (1e-9): a tighter
# tolerance is never reached, and AMEn then runs all its sweeps
OPTIONS = {"algorithm": "amen", "tolerance": 1e-8, "stagnation": 1e-8,
           "max_iterations": 20, "display": False}


def matrix(operator):
    """Dense matrix of a QTT operator, in the layout of the grid."""
    full = operator.full().numpy()
    d = operator.order
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    n = int(np.prod([s[0] for s in operator.shape]))
    return full.transpose(perm).reshape(n, n)


# %% Part 1: heat equation, decay of a sine mode
print("Part 1. heat equation on (0, 1)^2, f = 0 on the boundary")
nu, k, m = 1.0, 1, 2
walls = [(Dirichlet(), Dirichlet())] * 2
grid = StaggeredGrid([StaggeredAxis(1.0, 7)] * 2)
h = grid.axes[0].h
lam = sum(4 / h**2 * np.sin(j * np.pi * h / 2) ** 2 for j in (k, m))
identity = grid.interior_mask("p")
truncator = tensap.Truncator(tolerance=1e-12, max_rank=20)
for field in ("u", "p"):
    x, y = grid.positions(field)
    mode = np.sin(k * np.pi * x) * np.sin(m * np.pi * y)
    L = grid.laplacian(field, walls)
    for scheme, dt, steps in (("explicit", 0.2 * h**2, 200), ("implicit", 1e-3, 20)):
        f = grid.quantize(mode)
        start = time.time()
        if scheme == "explicit":
            A = truncator.truncate(identity + L * (dt * nu))
            for _ in range(steps):
                f = truncator.truncate(A @ f)
            factor = (1 - dt * nu * lam) ** steps
        else:
            A = truncator.truncate(identity + L * (-dt * nu))
            for _ in range(steps):
                f, _ = tensap.TTTensorALSLinearSolver(A, f, truncator=truncator, x0=f,
                                                      **OPTIONS).solve()
            factor = (1 + dt * nu * lam) ** -steps
        t = dt * steps
        values = grid.dequantize(f)
        discrete = np.abs(values - factor * mode).max()
        continuous = np.abs(values - np.exp(-nu * (k**2 + m**2) * np.pi**2 * t) * mode).max()
        print(f"  {field} ({' x '.join(grid.locations(field))}), {scheme} Euler, "
              f"{steps} steps to t = {t:.2e}: error against the discrete factor {discrete:.1e}, "
              f"against the continuous solution {continuous:.1e}, ranks {max(f.ranks)}, "
              f"{time.time() - start:.2f} s")
        assert discrete < 1e-9


# %% Part 2: unsteady Stokes in a channel
def channel(levels):
    """Grid (0, 2) x (0, 1) of 2^(levels + 1) x 2^levels cells, the discrete
    Poiseuille profile U_h, the operators of the channel."""
    grid = StaggeredGrid([StaggeredAxis(2.0, levels + 1), StaggeredAxis(1.0, levels)])
    ay = grid.axes[1]
    Ly = matrix(ay.cell_laplacian(Dirichlet(), Dirichlet()))
    profile = np.linalg.solve(-Ly, 8 * np.ones(ay.size))
    conditions = {"u": [(None, Neumann()), (Dirichlet(), Dirichlet())],
                  "v": [(Dirichlet(), Neumann()), (None, Dirichlet())]}
    return grid, profile, conditions


RIGHTS = [Neumann(), Dirichlet()]                         # u: outflow, v: wall
PRESSURE = [(Neumann(), Dirichlet()), (Neumann(), Neumann())]
nu, dt = 1.0, 0.01


def stokes_qtt(levels, steps, truncator):
    grid, profile, conditions = channel(levels)
    identity = grid.interior_mask("p")
    A = [truncator.truncate(identity + grid.laplacian(c, conditions[c]) * (-dt * nu))
         for c in ("u", "v")]
    G = grid.gradient()
    u0 = np.zeros(grid.shape)
    u0[0, :] = profile
    velocity = [grid.quantize(u0), grid.quantize(np.zeros(grid.shape))]
    p = grid.quantize(np.zeros(grid.shape))
    for _ in range(steps):
        star = [tensap.TTTensorALSLinearSolver(
            Ak, truncator.truncate(w - (G[k] @ p) * dt), truncator=truncator, x0=w,
            **OPTIONS).solve()[0] for k, (Ak, w) in enumerate(zip(A, velocity))]
        velocity, phi = grid.project(star, rights=RIGHTS, truncator=truncator,
                                     pressure=PRESSURE, tolerance=OPTIONS["tolerance"],
                                     stagnation=OPTIONS["stagnation"])
        p = truncator.truncate(p + phi * (1 / dt))
    return grid, profile, velocity, p, star, phi


def stokes_dense(levels, steps):
    grid, profile, conditions = channel(levels)
    n = int(np.prod(grid.shape))
    A = [np.eye(n) - dt * nu * matrix(grid.laplacian(c, conditions[c])) for c in ("u", "v")]
    G = [matrix(Gk) for Gk in grid.gradient()]
    D = [matrix(Dk) for Dk in grid.divergence(RIGHTS)]
    Lp = matrix(grid.pressure_laplacian(PRESSURE))
    u0 = np.zeros(grid.shape)
    u0[0, :] = profile
    velocity, p = [u0.ravel(), np.zeros(n)], np.zeros(n)
    for _ in range(steps):
        star = [np.linalg.solve(Ak, w - dt * Gk @ p) for Ak, w, Gk in zip(A, velocity, G)]
        phi = np.linalg.solve(Lp, D[0] @ star[0] + D[1] @ star[1])
        velocity = [s - Gk @ phi for s, Gk in zip(star, G)]
        p = p + phi / dt
    return [w.reshape(grid.shape) for w in velocity], p.reshape(grid.shape)


def outlet_divergence(grid, star, phi, velocity):
    """Divergence of the last column of cells with the outlet face
    u_N = u*_{N-1} - (G phi)_N, (G phi)_N = (-phi_{N-1} - phi_{N-1}) / h."""
    ax, ay = grid.axes
    u_star, phi, u, v = (grid.dequantize(a) for a in (star[0], phi, *velocity))
    u_outlet = u_star[-1] + 2 * phi[-1] / ax.h
    v_top = np.zeros(ay.size)                              # wall at y = 1
    v_faces = np.concatenate([v[-1], [v_top[0]]])
    return (u_outlet - u[-1]) / ax.h + (v_faces[1:] - v_faces[:-1]) / ay.h, u_outlet


print("Part 2. unsteady Stokes in the channel (0, 2) x (0, 1), from rest")
truncator = tensap.Truncator(tolerance=1e-9, max_rank=60)

# dense against QTT, 32 x 16 cells
start = time.time()
grid, profile, velocity, p, star, phi = stokes_qtt(4, 40, truncator)
elapsed = time.time() - start
dense_velocity, dense_p = stokes_dense(4, 40)
difference = max(np.abs(grid.dequantize(a) - b).max() for a, b in zip(velocity, dense_velocity))
print(f"  32 x 16 cells, 40 steps: QTT against dense, velocity {difference:.1e}, "
      f"pressure {np.abs(grid.dequantize(p) - dense_p).max():.1e} ({elapsed:.1f} s)")
assert difference < 1e-6

# QTT on 64 x 32 cells, to t = 1 (the ranks of v grow with the resolution
# during the start from rest: 31 here, 58 on 128 x 64 cells, about 11 min)
start = time.time()
grid, profile, velocity, p, star, phi = stokes_qtt(5, 100, truncator)
elapsed = time.time() - start
u, v = (grid.dequantize(w) for w in velocity)
pressure = grid.dequantize(p)
ax, ay = grid.axes
yc = ay.positions("cell")
xc = ax.positions("cell")
error_steady = max(np.abs(u - profile[None]).max(), np.abs(v).max())
error_parabola = np.abs(profile - 4 * yc * (1 - yc)).max()
slope = np.polyfit(xc, pressure.mean(axis=1), 1)[0]
div_outlet, u_outlet = outlet_divergence(grid, star, phi, velocity)
inflow, outflow = profile.sum() * ay.h, u_outlet.sum() * ay.h
print(f"  64 x 32 cells, 100 steps to t = 1 ({elapsed:.1f} s): "
      f"|u - U_h| and |v| {error_steady:.1e}, |U_h - 4 y (1 - y)| {error_parabola:.1e}, "
      f"dp/dx {slope:.4f} (-8), p at the outlet face "
      f"{np.abs(1.5 * pressure[-1] - 0.5 * pressure[-2]).max():.1e}")
print(f"  flow rate in {inflow:.6f}, out {outflow:.6f}; divergence of the last column "
      f"{ax.h * np.abs(div_outlet).max():.1e} (times h); QTT ranks u {max(velocity[0].ranks)}, "
      f"v {max(velocity[1].ranks)}, p {max(p.ranks)}")
assert error_steady < 1e-5 and abs(slope + 8) < 1e-3 and abs(inflow - outflow) < 1e-6
