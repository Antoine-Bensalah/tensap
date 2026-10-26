# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Tutorial: the Taylor-Green vortex, an exact solution of the incompressible
Navier-Stokes equations, on a staggered (MAC) grid in QTT format (plan P9,
level 3).

No MATLAB counterpart.

Problem
-------
du/dt + div(u u) + grad p = nu Delta u, div u = 0 on (0, 2 pi)^2, periodic:

    u = sin x cos y F(t),  v = -cos x sin y F(t),
    p = (cos 2x + cos 2y) F(t)^2 / 4,  F(t) = exp(-2 nu t).

The convective term is in conservative form (StaggeredGrid.convection): the
products u_j u_k are formed at the cells (j = k) or at the corners (j != k)
from averages, and differenced back to the faces. It conserves the momentum
exactly (periodic: the sum of N_j vanishes) and, for a divergence-free field,
the kinetic energy: (u, N(u)) = 0 (Morinishi et al. 1998).

Time stepping: Heun (explicit second-order Runge-Kutta) on the projected
equations du/dt = P(-N(u) + nu L u), P the Helmholtz-Hodge projection
(StaggeredGrid.project):

    u1 = P(u^n + dt F(u^n)),   u^{n+1} = P(u^n + dt (F(u^n) + F(u1)) / 2),

The pressure at the final time solves L_p p = -D N(u). Explicit Heun has two
stability limits, dt <~ h / |u| (advection) and dt <~ h^2 / (4 nu) (diffusion,
worst mode): the steps below satisfy both.

Steps
-----
1. the convective term of the exact field against (u . grad) u = (sin 2x / 2,
   sin 2y / 2): order 2, momentum and energy;
2. the vortex for nu = 0.01: the spatial order (32^2 to 128^2 cells, small
   fixed dt) and the temporal order (32^2 cells, dt refined) measured apart,
   errors on u and p; after the last rounding, the divergence, the mean of u
   (momentum) and the energy against E(0) F(t)^2 are asserted;
3. on 32^2 cells, the same scheme with dense matrices (numpy), as a reference.
"""

# %% Imports
import time

import numpy as np
import tensap
from tensap.tensorized_pde_discretization.finite_volumes import StaggeredAxis, StaggeredGrid

NU, T = 0.01, 1.0
TRUNCATOR = tensap.Truncator(tolerance=1e-10, max_rank=50)
SOLVER = {"tolerance": 1e-9, "stagnation": 1e-9}


def exact(grid, t):
    xu, yu = grid.positions("u")
    xv, yv = grid.positions("v")
    xp, yp = grid.positions("p")
    F = np.exp(-2 * NU * t)
    return (np.sin(xu) * np.cos(yu) * F, -np.cos(xv) * np.sin(yv) * F,
            (np.cos(2 * xp) + np.cos(2 * yp)) * F**2 / 4)


# %% 1. Convective term of the exact field
print("1. convective term of the Taylor-Green field")
orders = []
for levels in (5, 6, 7, 8):
    grid = StaggeredGrid([StaggeredAxis(2 * np.pi, levels, periodic=True)] * 2)
    u, v, _ = exact(grid, 0.0)
    velocity = [grid.quantize(u), grid.quantize(v)]
    N = grid.convection(velocity, truncator=TRUNCATOR)
    xu, _ = grid.positions("u")
    _, yv = grid.positions("v")
    error = max(np.abs(grid.dequantize(N[0]) - np.sin(2 * xu) / 2).max(),
                np.abs(grid.dequantize(N[1]) - np.sin(2 * yv) / 2).max())
    momentum = max(abs(grid.dequantize(Nk).sum()) for Nk in N) / u.size
    energy = abs(sum(a.dot(b) for a, b in zip(velocity, N)))
    orders.append(error)
    print(f"  {grid.shape[0]}^2 cells: error {error:.2e}, mean of N {momentum:.1e}, "
          f"(u, N(u)) {energy:.1e}, ranks of N {[max(Nk.ranks) for Nk in N]}")
rates = np.log2(np.array(orders[:-1]) / np.array(orders[1:]))
print(f"  observed orders {np.round(rates, 2)}")
assert min(rates) > 1.9
# exact conservation of the convective term, asserted (it was only printed)
assert momentum < 1e-12 and energy < 1e-10, (momentum, energy)


# %% 2. The vortex up to t = 1
def rhs(grid, velocity, laplacian):
    N = grid.convection(velocity, truncator=TRUNCATOR)
    return [TRUNCATOR.truncate(L @ w * NU - n) for L, w, n in zip(laplacian, velocity, N)]


def step(grid, velocity, laplacian, dt):
    f0 = rhs(grid, velocity, laplacian)
    u1, _ = grid.project([TRUNCATOR.truncate(w + f * dt) for w, f in zip(velocity, f0)],
                         truncator=TRUNCATOR, **SOLVER)
    f1 = rhs(grid, u1, laplacian)
    u2, _ = grid.project([TRUNCATOR.truncate(w + (a + b) * (dt / 2))
                          for w, a, b in zip(velocity, f0, f1)], truncator=TRUNCATOR, **SOLVER)
    return u2


def run(levels, steps, t_end):
    """The vortex on 2^levels x 2^levels cells, steps Heun steps to t_end:
    the velocity, the pressure (L_p p = -D N(u), solved as -L_p p = D N with
    the SPD regularized operator) and the diagnostics."""
    grid = StaggeredGrid([StaggeredAxis(2 * np.pi, levels, periodic=True)] * 2)
    laplacian = [grid.laplacian(c, [(None, None)] * 2) for c in ("u", "v")]
    dt = t_end / steps
    u, v, _ = exact(grid, 0.0)
    velocity = [grid.quantize(u), grid.quantize(v)]
    energy0 = sum(w.dot(w) for w in velocity)
    start = time.time()
    for _ in range(steps):
        velocity = step(grid, velocity, laplacian, dt)
    elapsed = time.time() - start
    N = grid.convection(velocity, truncator=TRUNCATOR)
    D = grid.divergence([None, None])
    operator = grid.pressure_laplacian() * -1.0 + grid.constant_mode_regularization()
    source = TRUNCATOR.truncate(D[0] @ N[0] + D[1] @ N[1])
    p, _ = tensap.TTTensorALSLinearSolver(operator, source, truncator=TRUNCATOR, algorithm="amen",
                                          display=False, **SOLVER).solve()
    u_ex, v_ex, p_ex = exact(grid, t_end)
    values = [grid.dequantize(w) for w in velocity]
    return {
        "grid": grid, "velocity": values, "elapsed": elapsed, "ranks": max(velocity[0].ranks),
        "error_u": max(np.abs(values[0] - u_ex).max(), np.abs(values[1] - v_ex).max()),
        "error_p": np.abs(grid.dequantize(p) - p_ex).max(),
        "energy": sum(w.dot(w) for w in velocity) / energy0 - np.exp(-4 * NU * t_end),
        # after the last rounding: divergence and mean (momentum) of the velocity
        "divergence": np.abs(sum(grid.dequantize(Dk @ w) for Dk, w in zip(D, velocity))).max(),
        "momentum": max(abs(w.mean()) for w in values),
    }


# %% 2. Spatial and temporal convergence, separately
# A refinement in h with dt proportional to h mixes the spatial and temporal
# errors of Heun (both of order 2): they are measured apart, up to t = 1/4.
# Space: dt fixed and small (64 steps; CFL 0.16 and 4 nu dt / h^2 = 0.03 on the
# finest grid), the temporal error negligible. Time: one grid, 32^2, against a
# reference with 128 steps on the same grid.
T_CONV = 0.25
print(f"2a. spatial convergence, nu = {NU}, t = {T_CONV}, 64 steps")
spatial = []
for levels in (5, 6, 7):
    r = run(levels, 64, T_CONV)
    spatial.append(r)
    print(f"  {r['grid'].shape[0]}^2 cells: error u {r['error_u']:.2e}, p {r['error_p']:.2e}, "
          f"E(t) / E(0) - F(t)^2 {r['energy']:.1e}, max |D u| {r['divergence']:.1e}, "
          f"mean of u {r['momentum']:.1e}, ranks {r['ranks']}, {r['elapsed']:.1f} s")
errors = np.array([(r["error_u"], r["error_p"]) for r in spatial])
rates = np.log2(errors[:-1] / errors[1:])
print(f"  observed orders in h (u, p): {np.round(rates, 2).tolist()}")
assert rates.min() > 1.8, rates
for r in spatial:
    # conservation after rounding: divergence-free, zero mean (momentum)
    assert r["divergence"] < 1e-8 and r["momentum"] < 1e-12, (r["divergence"], r["momentum"])
# the energy defect is O(h^2)
energies = np.abs([r["energy"] for r in spatial])
assert np.all(energies[1:] < energies[:-1] / 3), energies

print(f"2b. temporal convergence, 32^2 cells, t = {T_CONV}")
reference = run(5, 128, T_CONV)["velocity"]
temporal = []
for steps in (4, 8, 16):
    r = run(5, steps, T_CONV)
    difference = max(np.abs(a - b).max() for a, b in zip(r["velocity"], reference))
    temporal.append(difference)
    print(f"  dt = {T_CONV / steps:.4f}: difference with dt = {T_CONV / 128:.5f}: {difference:.2e}")
rates = np.log2(np.array(temporal[:-1]) / np.array(temporal[1:]))
print(f"  observed orders in dt: {np.round(rates, 2).tolist()}")
assert rates.min() > 1.8, rates


# %% 3. Dense reference on 32^2 cells
def matrix(operator):
    full = operator.full().numpy()
    d = operator.order
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    n = int(np.prod([s[0] for s in operator.shape]))
    return full.transpose(perm).reshape(n, n)


print("3. dense reference, 32^2 cells")
grid = StaggeredGrid([StaggeredAxis(2 * np.pi, 5, periodic=True)] * 2)
ax, ay = grid.axes
M = {name: [matrix(grid.embed(block, k)) for k, block in enumerate(blocks)]
     for name, blocks in (("to_cell", [a.face_to_cell() for a in grid.axes]),
                          ("to_face", [a.cell_to_face() for a in grid.axes]),
                          ("gradient", [a.gradient() for a in grid.axes]),
                          ("divergence", [a.divergence() for a in grid.axes]))}
L = [matrix(grid.laplacian(c, [(None, None)] * 2)) for c in ("u", "v")]
Lp = sum(Dk @ Gk for Dk, Gk in zip(M["divergence"], M["gradient"]))


def dense_convection(u):
    N = [np.zeros_like(u[0]), np.zeros_like(u[0])]
    for k in range(2):
        mean = M["to_cell"][k] @ u[k]
        N[k] += M["gradient"][k] @ (mean * mean)
        j = 1 - k
        N[j] += M["divergence"][k] @ ((M["to_face"][k] @ u[j]) * (M["to_face"][j] @ u[k]))
    return N


def dense_project(w):
    div = sum(Dk @ wk for Dk, wk in zip(M["divergence"], w))
    phi = np.linalg.lstsq(Lp, div, rcond=None)[0]
    return [wk - Gk @ phi for wk, Gk in zip(w, M["gradient"])]


def dense_rhs(u):
    return [NU * Lk @ uk - nk for Lk, uk, nk in zip(L, u, dense_convection(u))]


steps = int(np.ceil(T / (ax.h / 2)))
dt = T / steps
u, v, _ = exact(grid, 0.0)
dense = [u.ravel(), v.ravel()]
velocity = [grid.quantize(u), grid.quantize(v)]
laplacian = [grid.laplacian(c, [(None, None)] * 2) for c in ("u", "v")]
for _ in range(steps):
    f0 = dense_rhs(dense)
    u1 = dense_project([a + dt * b for a, b in zip(dense, f0)])
    f1 = dense_rhs(u1)
    dense = dense_project([a + dt / 2 * (b + c) for a, b, c in zip(dense, f0, f1)])
    velocity = step(grid, velocity, laplacian, dt)
difference = max(np.abs(grid.dequantize(a).ravel() - b).max() for a, b in zip(velocity, dense))
print(f"  {steps} steps: QTT against dense {difference:.1e}")
assert difference < 1e-8
