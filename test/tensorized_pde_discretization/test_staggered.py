"""
Unit tests of the staggered (MAC) finite volume grid in QTT format
(tensap.tensorized_pde_discretization.finite_volumes: StaggeredAxis,
StaggeredGrid): the 1D blocks between cells and faces with their closures,
their assembly in 2D, and the identity D G = L_p.

Oracles: the same matrices written here with numpy from the formulas of the
staggered scheme, and Kronecker products.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap.tensorized_pde_discretization import Dirichlet, Neumann
from tensap.tensorized_pde_discretization.finite_volumes import StaggeredAxis, StaggeredGrid


def _matrix(operator):
    full = operator.full().numpy()
    d = operator.order
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    n = int(np.prod([s[0] for s in operator.shape]))
    m = int(np.prod([s[1] for s in operator.shape]))
    return full.transpose(perm).reshape(n, m)


def _dequantize_1d(operator):
    return _matrix(operator)


CONDITIONS = [Dirichlet(), Neumann()]


# --- references written from the formulas ---------------------------------------------


def _reference_gradient(N, h, periodic):
    G = np.zeros((N, N))
    for i in range(1, N):  # face i between cells i - 1 and i
        G[i, i], G[i, i - 1] = 1.0, -1.0
    if periodic:
        G[0, 0], G[0, N - 1] = 1.0, -1.0
    return G / h


def _reference_divergence(N, h, right, periodic):
    D = np.zeros((N, N))
    for i in range(N):  # cell i between faces i and i + 1
        D[i, i] = -1.0
        if i + 1 < N:
            D[i, i + 1] = 1.0
    if periodic:
        D[N - 1, 0] = 1.0
    elif isinstance(right, Neumann):  # outflow: u_N = u_{N-1}
        D[N - 1, N - 1] = 0.0
    return D / h


def _reference_face_laplacian(N, h, right, periodic):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i + 1 < N:
            L[i, i + 1] = 1.0
    if periodic:
        L[0, N - 1] = L[N - 1, 0] = 1.0
    else:
        L[0, :] = 0.0  # face 0: known value
        if isinstance(right, Neumann):
            L[N - 1, N - 1] += 1.0
    return L / h**2


def _reference_face_to_cell(N, right, periodic):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.5
        if i + 1 < N:
            A[i, i + 1] = 0.5
    if periodic:
        A[N - 1, 0] = 0.5
    elif isinstance(right, Neumann):
        A[N - 1, N - 1] = 1.0
    return A


def _reference_cell_to_face(N, left, periodic):
    A = np.zeros((N, N))
    for i in range(1, N):
        A[i, i], A[i, i - 1] = 0.5, 0.5
    if periodic:
        A[0, 0], A[0, N - 1] = 0.5, 0.5
    elif isinstance(left, Neumann):
        A[0, 0] = 1.0
    return A


# --- one axis --------------------------------------------------------------------------


def test_axis_geometry():
    axis = StaggeredAxis(2.0, 3)
    assert axis.size == 8 and axis.h == 0.25
    npt.assert_allclose(axis.positions("cell"), (np.arange(8) + 0.5) * 0.25)
    npt.assert_allclose(axis.positions("face"), np.arange(8) * 0.25)
    with pytest.raises(ValueError):
        axis.positions("corner")
    assert "StaggeredAxis" in repr(axis)


@pytest.mark.parametrize("periodic", [False, True])
def test_gradient(periodic):
    axis = StaggeredAxis(1.5, 4, periodic=periodic)
    npt.assert_allclose(_dequantize_1d(axis.gradient()),
                        _reference_gradient(16, axis.h, periodic), atol=1e-12)


@pytest.mark.parametrize("right", CONDITIONS)
@pytest.mark.parametrize("periodic", [False, True])
def test_divergence_face_laplacian_and_averages(right, periodic):
    axis = StaggeredAxis(1.5, 4, periodic=periodic)
    right = None if periodic else right
    npt.assert_allclose(_dequantize_1d(axis.divergence(right)),
                        _reference_divergence(16, axis.h, right, periodic), atol=1e-12)
    npt.assert_allclose(_dequantize_1d(axis.face_laplacian(right)),
                        _reference_face_laplacian(16, axis.h, right, periodic), atol=1e-9)
    npt.assert_allclose(_dequantize_1d(axis.face_to_cell(right)),
                        _reference_face_to_cell(16, right, periodic), atol=1e-12)


@pytest.mark.parametrize("left", CONDITIONS)
@pytest.mark.parametrize("periodic", [False, True])
def test_cell_to_face(left, periodic):
    axis = StaggeredAxis(1.0, 3, periodic=periodic)
    left = None if periodic else left
    npt.assert_allclose(_dequantize_1d(axis.cell_to_face(left)),
                        _reference_cell_to_face(8, left, periodic), atol=1e-12)


def test_cell_laplacian_is_the_finite_volume_one():
    axis = StaggeredAxis(1.0, 3)
    reference = tensap.tensorized_pde_discretization.finite_volumes.Axis(
        1.0, 3, Dirichlet(), Neumann()).second_difference_matrix().toarray()
    npt.assert_allclose(_dequantize_1d(axis.cell_laplacian(Dirichlet(), Neumann())), reference,
                        atol=1e-9)


def test_boundary_weights():
    axis = StaggeredAxis(1.0, 3)
    h = axis.h
    assert axis.boundary_weight("divergence", Dirichlet()) == pytest.approx(1 / h)
    assert axis.boundary_weight("divergence", Neumann()) == pytest.approx(1.0)
    assert axis.boundary_weight("face_laplacian", Dirichlet()) == pytest.approx(1 / h**2)
    assert axis.boundary_weight("face_laplacian", Neumann()) == pytest.approx(1 / h)
    assert axis.boundary_weight("face_to_cell", Dirichlet()) == pytest.approx(0.5)
    assert axis.boundary_weight("face_to_cell", Neumann()) == pytest.approx(h / 2)
    assert axis.boundary_weight("cell_to_face", Dirichlet()) == pytest.approx(1.0)
    assert axis.boundary_weight("cell_to_face", Neumann()) == pytest.approx(h / 2)
    with pytest.raises(ValueError):
        axis.boundary_weight("gradient", Dirichlet())
    with pytest.raises(NotImplementedError):
        axis.boundary_weight("divergence", object())


def test_closures_reproduce_the_boundary_data():
    """The block with its source reproduces the formula of the scheme with the
    ghost value set by the condition: u_N = g (Dirichlet), u_N = u_{N-1} + h g
    (Neumann, outward normal derivative g at x = L)."""
    axis = StaggeredAxis(1.0, 3)
    h, N = axis.h, 8
    rng = np.random.default_rng(0)
    f, g = rng.standard_normal(N), 0.7
    e_last = np.eye(N)[-1]
    for condition, ghost in ((Dirichlet(g), g), (Neumann(g), f[-1] + h * g)):
        extended = np.append(f, ghost)
        div = _dequantize_1d(axis.divergence(condition)) @ f \
            + axis.boundary_weight("divergence", condition) * g * e_last
        npt.assert_allclose(div, np.diff(extended) / h)
        avg = _dequantize_1d(axis.face_to_cell(condition)) @ f \
            + axis.boundary_weight("face_to_cell", condition) * g * e_last
        npt.assert_allclose(avg, 0.5 * (extended[:-1] + extended[1:]))
        lap = _dequantize_1d(axis.face_laplacian(condition)) @ f \
            + axis.boundary_weight("face_laplacian", condition) * g * e_last
        expected = (extended[2:] - 2 * extended[1:-1] + extended[:-2]) / h**2
        npt.assert_allclose(lap[1:], expected)
    # cell to face at x = 0: face value g (Dirichlet), c_0 + h g / 2 (Neumann)
    c = rng.standard_normal(N)
    for condition, value in ((Dirichlet(g), g), (Neumann(g), c[0] + h * g / 2)):
        out = _dequantize_1d(axis.cell_to_face(condition)) @ c \
            + axis.boundary_weight("cell_to_face", condition) * g * np.eye(N)[0]
        assert out[0] == pytest.approx(value)
        npt.assert_allclose(out[1:], 0.5 * (c[:-1] + c[1:]))


@pytest.mark.parametrize("periodic", [False, True])
def test_divergence_of_gradient_is_the_cell_laplacian(periodic):
    """D G = L_p: Neumann on both ends for walls (u known at the faces 0 and
    N), circulant for a periodic axis."""
    axis = StaggeredAxis(1.3, 4, periodic=periodic)
    right = None if periodic else Dirichlet()
    DG = _dequantize_1d(axis.divergence(right)) @ _dequantize_1d(axis.gradient())
    conditions = (None, None) if periodic else (Neumann(), Neumann())
    npt.assert_allclose(DG, _dequantize_1d(axis.cell_laplacian(*conditions)), atol=1e-8)


def test_gradient_is_minus_the_adjoint_of_the_divergence():
    """G = -D^T on the interior faces (face 0, a wall face, excluded)."""
    axis = StaggeredAxis(1.0, 4)
    G = _dequantize_1d(axis.gradient())
    D = _dequantize_1d(axis.divergence(Dirichlet()))
    npt.assert_allclose(G[1:], -D.T[1:], atol=1e-12)


def test_exactness_on_polynomials():
    axis = StaggeredAxis(1.0, 5)
    xc, xf = axis.positions("cell"), axis.positions("face")
    G = _dequantize_1d(axis.gradient())
    npt.assert_allclose((G @ (3 * xc + 1))[1:], 3.0, rtol=1e-10)
    D = _dequantize_1d(axis.divergence(Dirichlet()))
    npt.assert_allclose((D @ (2 * xf))[:-1], 2.0, rtol=1e-10)
    L = _dequantize_1d(axis.face_laplacian(Dirichlet()))
    npt.assert_allclose((L @ xf**2)[1:-1], 2.0, rtol=1e-8)


# --- the grid ---------------------------------------------------------------------------


def _grid(ordering="blocks", periodic=False):
    return StaggeredGrid([StaggeredAxis(2.0, 3, periodic=periodic),
                          StaggeredAxis(1.0, 2, periodic=periodic)], ordering=ordering)


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
@pytest.mark.parametrize("periodic", [False, True])
def test_grid_divergence_of_gradient(ordering, periodic):
    grid = _grid(ordering, periodic)
    right = None if periodic else Dirichlet()
    G = grid.gradient()
    D = grid.divergence([right, right])
    DG = sum(_matrix(Dk @ Gk) for Dk, Gk in zip(D, G))
    npt.assert_allclose(DG, _matrix(grid.pressure_laplacian()), atol=1e-7)
    # the pressure Laplacian is the cell-centred one with Neumann walls
    axes = grid.axes
    L0 = _matrix(axes[0].cell_laplacian(*(() if periodic else (Neumann(), Neumann()))))
    L1 = _matrix(axes[1].cell_laplacian(*(() if periodic else (Neumann(), Neumann()))))
    expected = np.kron(L0, np.eye(4)) + np.kron(np.eye(8), L1)
    if ordering == "blocks":
        npt.assert_allclose(_matrix(grid.pressure_laplacian()), expected, atol=1e-7)


def test_grid_locations_and_quantization():
    grid = _grid()
    assert grid.shape == (8, 4)
    x, y = grid.positions("u")
    npt.assert_allclose(x[:, 0], np.arange(8) * 0.25)
    npt.assert_allclose(y[0], (np.arange(4) + 0.5) * 0.25)
    x, y = grid.positions("corner")
    npt.assert_allclose(y[0], np.arange(4) * 0.25)
    with pytest.raises(ValueError):
        grid.positions("w")
    values = np.random.default_rng(1).standard_normal((8, 4))
    npt.assert_allclose(grid.dequantize(grid.quantize(values)), values, atol=1e-12)
    for ordering in ("blocks", "interleaved"):
        g = _grid(ordering)
        npt.assert_allclose(g.dequantize(g.quantize(values)), values, atol=1e-12)
    with pytest.raises(ValueError):
        StaggeredGrid(grid.axes, ordering="random")


def test_grid_laplacians_of_the_velocity():
    """Viscous term of u: faces along x, cells along y; of v: the reverse. The
    rows of the stored boundary faces (face 0 along the face axis) are zero."""
    grid = _grid()
    ax, ay = grid.axes
    mask_x = np.diag(np.r_[0.0, np.ones(7)])
    mask_y = np.diag(np.r_[0.0, np.ones(3)])
    Lu = grid.laplacian("u", [(Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())])
    expected = (np.kron(_matrix(ax.face_laplacian(Dirichlet())), np.eye(4))
                + np.kron(mask_x, _matrix(ay.cell_laplacian(Dirichlet(), Dirichlet()))))
    npt.assert_allclose(_matrix(Lu), expected, atol=1e-7)
    npt.assert_allclose(_matrix(grid.interior_mask("u")), np.kron(mask_x, np.eye(4)), atol=1e-12)
    Lv = grid.laplacian("v", [(Dirichlet(), Neumann()), (Dirichlet(), Dirichlet())])
    expected = (np.kron(_matrix(ax.cell_laplacian(Dirichlet(), Neumann())), mask_y)
                + np.kron(np.eye(8), _matrix(ay.face_laplacian(Dirichlet()))))
    npt.assert_allclose(_matrix(Lv), expected, atol=1e-7)
    npt.assert_allclose(_matrix(grid.interior_mask("corner")), np.kron(mask_x, mask_y), atol=1e-12)
    npt.assert_allclose(_matrix(grid.interior_mask("p")), np.eye(32), atol=1e-12)
    Lp = grid.laplacian("p", [(Neumann(), Neumann()), (Neumann(), Neumann())])
    npt.assert_allclose(_matrix(Lp), _matrix(grid.pressure_laplacian()), atol=1e-7)
    with pytest.raises(ValueError):
        grid.laplacian("corner", [(None, None)] * 2)


def test_ranks_do_not_depend_on_the_resolution_and_errors():
    for levels in (4, 8, 12):
        axis = StaggeredAxis(1.0, levels)
        assert max(axis.gradient().ranks) <= 3
        assert max(axis.divergence(Neumann()).ranks) <= 4
        assert max(axis.face_laplacian(Neumann()).ranks) <= 5
    with pytest.raises(ValueError):
        StaggeredAxis(1.0, 0)
    assert "StaggeredGrid" in repr(_grid())


# --- level 1: curl, regularization, Helmholtz-Hodge decomposition ------------------


@pytest.mark.parametrize("periodic", [False, True])
def test_helmholtz_hodge_decomposition(periodic):
    L = 2 * np.pi if periodic else 1.0
    grid = StaggeredGrid([StaggeredAxis(L, 4, periodic), StaggeredAxis(L, 3, periodic)])
    rng = np.random.default_rng(3)
    psi = rng.standard_normal(grid.shape)
    if not periodic:
        psi[0, :] = psi[:, 0] = 0.0  # the stream function vanishes on the boundary
    phi = rng.standard_normal(grid.shape)
    Psi, Phi = grid.quantize(psi), grid.quantize(phi)
    u = [C @ Psi for C in grid.curl()]
    rights = [None if periodic else Dirichlet()] * 2
    divergence = grid.divergence(rights)
    npt.assert_allclose(sum(grid.dequantize(D @ uk) for D, uk in zip(divergence, u)), 0.0,
                        atol=1e-10)
    w = [uk + G @ Phi for uk, G in zip(u, grid.gradient())]
    u_rec, phi_rec = grid.project(w, rights)
    for a, b in zip(u_rec, u):
        npt.assert_allclose(grid.dequantize(a), grid.dequantize(b), atol=1e-8)
    npt.assert_allclose(grid.dequantize(phi_rec), phi - phi.mean(), atol=1e-8)
    # orthogonality: (u, G phi) = -(D u, phi) = 0
    gradient = [G @ phi_rec for G in grid.gradient()]
    assert abs(sum(a.dot(b) for a, b in zip(u_rec, gradient))) < 1e-8
    if periodic:
        u_default, _ = grid.project(w)
        npt.assert_allclose(grid.dequantize(u_default[0]), grid.dequantize(u[0]), atol=1e-8)


def test_curl_and_regularization_errors():
    grid3 = StaggeredGrid([StaggeredAxis(1.0, 2)] * 3)
    with pytest.raises(ValueError):
        grid3.curl()
    grid = _grid()
    R = _matrix(grid.constant_mode_regularization(sigma=2.0))
    npt.assert_allclose(R, np.full((32, 32), 2.0 / 32), atol=1e-12)
    # sigma by default: the smallest non-zero eigenvalue of -L_p
    eigenvalues = np.linalg.eigvalsh(-_matrix(grid.pressure_laplacian()))
    sigma = np.sort(eigenvalues)[1]
    npt.assert_allclose(_matrix(grid.constant_mode_regularization()),
                        np.full((32, 32), sigma / 32), rtol=1e-8)


# --- level 2: time stepping, outlet pressure, channel --------------------------------


@pytest.mark.parametrize("field", ["u", "p"])
def test_sine_modes_are_eigenvectors_of_the_laplacian(field):
    """sin(k pi x) sin(m pi y) on the faces (f_0 = f_N = 0) as on the cells
    (ghosts -f_0, -f_{N-1}): L f = -(lambda_k + lambda_m) f exactly."""
    grid = StaggeredGrid([StaggeredAxis(1.0, 4), StaggeredAxis(1.0, 3)])
    x, y = grid.positions(field)
    mode = np.sin(2 * np.pi * x) * np.sin(3 * np.pi * y)
    lam = sum(4 / a.h**2 * np.sin(j * np.pi * a.h / 2) ** 2 for a, j in zip(grid.axes, (2, 3)))
    L = grid.laplacian(field, [(Dirichlet(), Dirichlet())] * 2)
    npt.assert_allclose(grid.dequantize(L @ grid.quantize(mode)), -lam * mode, atol=1e-8)


def _channel():
    grid = StaggeredGrid([StaggeredAxis(2.0, 4), StaggeredAxis(1.0, 3)])
    pressure = [(Neumann(), Dirichlet()), (Neumann(), Neumann())]
    return grid, [Neumann(), Dirichlet()], pressure


def test_outlet_pressure_projection():
    """p = 0 at the outlet: L_p is not singular and not regularized; the
    corrected velocity, with the outlet face u_N = u*_{N-1} + 2 phi_{N-1} / h,
    is divergence-free in every cell."""
    grid, rights, pressure = _channel()
    assert grid.pressure_is_singular() and not grid.pressure_is_singular(pressure)
    ax, ay = grid.axes
    Lp = _matrix(grid.pressure_laplacian(pressure))
    expected = (np.kron(_matrix(ax.cell_laplacian(Neumann(), Dirichlet())), np.eye(8))
                + np.kron(np.eye(16), _matrix(ay.cell_laplacian(Neumann(), Neumann()))))
    npt.assert_allclose(Lp, expected, atol=1e-8)
    rng = np.random.default_rng(5)
    u_star = rng.standard_normal(grid.shape)
    v_star = rng.standard_normal(grid.shape)
    v_star[:, 0] = 0.0  # wall at y = 0
    star = [grid.quantize(u_star), grid.quantize(v_star)]
    (u, v), phi = grid.project(star, rights=rights, pressure=pressure)
    u, v, phi = (grid.dequantize(a) for a in (u, v, phi))
    u_faces = np.vstack([u, u_star[-1] + 2 * phi[-1] / ax.h])
    v_faces = np.hstack([v, np.zeros((16, 1))])
    divergence = np.diff(u_faces, axis=0) / ax.h + np.diff(v_faces, axis=1) / ay.h
    npt.assert_allclose(divergence, 0.0, atol=1e-6)
    npt.assert_allclose(u[0], u_star[0], atol=1e-10)  # inflow face unchanged


def test_poiseuille_is_a_fixed_point_of_the_pressure_correction():
    """u = U_h(y) (-L_y U_h = 8), v = 0, p = 8 (L - x): one step of the
    incremental pressure correction leaves it unchanged."""
    grid, rights, pressure = _channel()
    ax, ay = grid.axes
    profile = np.linalg.solve(-_matrix(ay.cell_laplacian(Dirichlet(), Dirichlet())),
                              8 * np.ones(ay.size))
    xc, _ = grid.positions("p")
    u = grid.quantize(np.tile(profile, (ax.size, 1)))
    v = grid.quantize(np.zeros(grid.shape))
    p = grid.quantize(8 * (2.0 - xc))
    dt = 0.01
    conditions = {"u": [(None, Neumann()), (Dirichlet(), Dirichlet())],
                  "v": [(Dirichlet(), Neumann()), (None, Dirichlet())]}
    identity = grid.interior_mask("p")
    star = []
    for k, (c, w) in enumerate(zip("uv", (u, v))):
        A = identity + grid.laplacian(c, conditions[c]) * (-dt)
        rhs = w - (grid.gradient()[k] @ p) * dt
        star.append(tensap.TTTensorALSLinearSolver(A, rhs, algorithm="amen", tolerance=1e-10,
                                                   stagnation=1e-10, display=False).solve()[0])
    (u1, v1), phi = grid.project(star, rights=rights, pressure=pressure, tolerance=1e-10)
    npt.assert_allclose(grid.dequantize(u1), grid.dequantize(u), atol=1e-8)
    npt.assert_allclose(grid.dequantize(v1), 0.0, atol=1e-8)
    npt.assert_allclose(grid.dequantize(phi), 0.0, atol=1e-8)


# --- level 3: convective term -------------------------------------------------------


def _shift(a, axis, offset, periodic):
    """a[i + offset] along axis, zero outside a bounded axis."""
    if periodic:
        return np.roll(a, -offset, axis=axis)
    out = np.zeros_like(a)
    n = a.shape[axis]
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if offset > 0:
        src[axis], dst[axis] = slice(offset, n), slice(0, n - offset)
    else:
        src[axis], dst[axis] = slice(0, n + offset), slice(-offset, n)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _reference_convection(velocity, grid):
    """N_j = sum_k d_k (u_j u_k) written with shifts, walls (zero velocity on
    the boundary) or periodic."""
    d = grid.dim
    N = [np.zeros(grid.shape) for _ in range(d)]
    for k, ak in enumerate(grid.axes):
        per_k = ak.periodic
        mean = (velocity[k] + _shift(velocity[k], k, 1, per_k)) / 2         # cells along k
        N[k] += (mean**2 - _shift(mean**2, k, -1, per_k)) / ak.h            # faces along k
        for j, aj in enumerate(grid.axes):
            if j == k:
                continue
            uj = (velocity[j] + _shift(velocity[j], k, -1, per_k)) / 2      # faces along k
            uk = (velocity[k] + _shift(velocity[k], j, -1, aj.periodic)) / 2  # faces along j
            if not per_k:
                uj = np.moveaxis(uj, k, 0).copy()
                uj[0] = 0.0                                                  # wall at x_k = 0
                uj = np.moveaxis(uj, 0, k)
            if not aj.periodic:
                uk = np.moveaxis(uk, j, 0).copy()
                uk[0] = 0.0
                uk = np.moveaxis(uk, 0, j)
            q = uj * uk
            N[j] += (_shift(q, k, 1, per_k) - q) / ak.h                      # cells along k
    for j, aj in enumerate(grid.axes):
        if not aj.periodic:
            N[j] = np.moveaxis(N[j], j, 0).copy()
            N[j][0] = 0.0
            N[j] = np.moveaxis(N[j], 0, j)
    return N


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("levels", [(4, 3), (2, 3, 2)])
def test_convection_against_reference(periodic, levels):
    grid = StaggeredGrid([StaggeredAxis(1.0 + k, l, periodic) for k, l in enumerate(levels)])
    rng = np.random.default_rng(len(levels))
    velocity = []
    for j in range(grid.dim):
        w = rng.standard_normal(grid.shape)
        if not periodic:
            w = np.moveaxis(w, j, 0).copy()
            w[0] = 0.0  # walls
            w = np.moveaxis(w, 0, j)
        velocity.append(w)
    N = grid.convection([grid.quantize(w) for w in velocity])
    for a, b in zip(N, _reference_convection(velocity, grid)):
        npt.assert_allclose(grid.dequantize(a), b, atol=1e-7)


@pytest.mark.parametrize("periodic", [False, True])
def test_convection_conserves_momentum_and_energy(periodic):
    """Periodic: sum N_j = 0 (momentum). Divergence-free (curl of psi): (u, N(u))
    = 0 (kinetic energy), also with walls."""
    grid = StaggeredGrid([StaggeredAxis(1.0, 4, periodic), StaggeredAxis(1.0, 3, periodic)])
    psi = np.random.default_rng(1).standard_normal(grid.shape)
    if not periodic:
        psi[0, :] = psi[:, 0] = 0.0
    P = grid.quantize(psi)
    velocity = [C @ P for C in grid.curl()]
    N = grid.convection(velocity)
    scale = np.sqrt(sum(a.dot(a) for a in N) * sum(a.dot(a) for a in velocity))
    assert abs(sum(a.dot(b) for a, b in zip(velocity, N))) < 1e-10 * scale
    if periodic:
        for Nk in N:
            assert abs(grid.dequantize(Nk).sum()) < 1e-9 * np.sqrt(Nk.dot(Nk))


def test_convection_of_taylor_green_is_second_order():
    errors = []
    for levels in (4, 5, 6):
        grid = StaggeredGrid([StaggeredAxis(2 * np.pi, levels, periodic=True)] * 2)
        xu, yu = grid.positions("u")
        xv, yv = grid.positions("v")
        velocity = [grid.quantize(np.sin(xu) * np.cos(yu)),
                    grid.quantize(-np.cos(xv) * np.sin(yv))]
        N = grid.convection(velocity, truncator=tensap.Truncator(tolerance=1e-12))
        errors.append(np.abs(grid.dequantize(N[0]) - np.sin(2 * xu) / 2).max())
    assert np.all(np.log2(np.array(errors[:-1]) / np.array(errors[1:])) > 1.9)


def test_convection_rejects_data():
    grid = _grid()
    zero = grid.quantize(np.zeros(grid.shape))
    conditions = [[(Dirichlet(), Dirichlet(1.0)), (Dirichlet(), Dirichlet())]] * 2
    with pytest.raises(NotImplementedError):
        grid.convection([zero, zero], conditions)
    outflow = [[(Dirichlet(), Neumann()), (Dirichlet(), Dirichlet())]] * 2
    assert all(n.norm() == 0 for n in grid.convection([zero, zero], outflow))


# --- level 4: boundary sources ------------------------------------------------------


def _second_difference_with_ghosts(f, h, axis, location, left, right, g_left, g_right):
    """Second difference of f along axis, with the closures written as ghost
    values (cells) or as the face N (faces)."""
    f = np.moveaxis(f, axis, 0)
    n = f.shape[0]
    if location == "face":
        last = g_right if isinstance(right, Dirichlet) else f[-1] + h * g_right
        ext = np.concatenate([f[:1] * 0, f, last[None]])  # face 0: masked below
        out = (ext[:-2] - 2 * ext[1:-1] + ext[2:]) / h**2
        out[0] = 0.0
    else:
        lo = 2 * g_left - f[0] if isinstance(left, Dirichlet) else f[0] + h * g_left
        hi = 2 * g_right - f[-1] if isinstance(right, Dirichlet) else f[-1] + h * g_right
        ext = np.concatenate([lo[None], f, hi[None]])
        out = (ext[:-2] - 2 * ext[1:-1] + ext[2:]) / h**2
    assert out.shape[0] == n
    return np.moveaxis(out, 0, axis)


def test_laplacian_source_of_a_lid_and_of_neumann_data():
    """u of a cavity: lid g(x) at y = 1 (Dirichlet on the cells along y),
    Neumann data on the right end of the faces along x; v: Dirichlet data on
    both cell ends along x."""
    grid = StaggeredGrid([StaggeredAxis(1.0, 4), StaggeredAxis(2.0, 3)])
    ax, ay = grid.axes
    rng = np.random.default_rng(7)

    def lid(points):
        return np.sin(np.pi * points[:, 0])

    for field, conditions in (
            ("u", [(None, Neumann(0.5)), (Dirichlet(), Dirichlet(lid))]),
            ("v", [(Dirichlet(-1.0), Dirichlet(lambda p: p[:, 0] ** 2)), (None, Dirichlet())])):
        f = rng.standard_normal(grid.shape)
        locations = grid.locations(field)
        if locations[0] == "face":
            f[0] = 0.0
        else:
            f[:, 0] = 0.0
        total = grid.laplacian(field, conditions) @ grid.quantize(f) \
            + grid.laplacian_source(field, conditions)
        expected = np.zeros(grid.shape)
        for k, (axis, location, (left, right)) in enumerate(zip(grid.axes, locations, conditions)):
            other = grid.axes[1 - k].positions(locations[1 - k])

            def data(c):
                if c is None:
                    return np.zeros(other.size)
                if callable(c.value):
                    return c.value(other[:, None])
                return np.full(other.size, float(c.value))
            expected += _second_difference_with_ghosts(f, axis.h, k, location, left, right,
                                                       data(left), data(right))
        if locations[0] == "face":
            expected[0] = 0.0
        else:
            expected[:, 0] = 0.0
        npt.assert_allclose(grid.dequantize(total), expected, atol=1e-8)
    homogeneous = grid.laplacian_source("p", [(Dirichlet(), Neumann())] * 2)
    assert homogeneous.norm() == 0
    line = StaggeredGrid([StaggeredAxis(1.0, 3)])
    s = line.laplacian_source("p", [(Dirichlet(1.0), Dirichlet())])
    npt.assert_allclose(line.dequantize(s)[0], 2 / line.axes[0].h ** 2)
    with pytest.raises(TypeError):
        line.laplacian_source("p", [(Dirichlet(lid), Dirichlet())])


# --- review R0: definite pressure, compatibility, outlet state ------------------------


def test_pressure_operator_of_project_is_positive_definite():
    """-L_p + R is SPD (L_p + R, solved before, was indefinite: review M1)."""
    grid = StaggeredGrid([StaggeredAxis(1.0, 2)])
    A = -_matrix(grid.pressure_laplacian()) + _matrix(grid.constant_mode_regularization())
    assert np.linalg.eigvalsh(A).min() > 0
    wrong = _matrix(grid.pressure_laplacian()) + _matrix(grid.constant_mode_regularization())
    assert np.linalg.eigvalsh(wrong).min() < 0 < np.linalg.eigvalsh(wrong).max()


def test_project_rejects_incompatible_fluxes():
    """A net flux through closed walls cannot be projected (review M3)."""
    grid = StaggeredGrid([StaggeredAxis(1.0, 2)])
    w = np.zeros(4)
    w[0] = 1.0  # an inflow at x = 0, a wall at x = 1
    with pytest.raises(ValueError, match="incompatible"):
        grid.project([grid.quantize(w)])
    # compatible up to round-off: accepted, reported, zero-mean potential
    grid2 = StaggeredGrid([StaggeredAxis(1.0, 3), StaggeredAxis(1.0, 3)])
    rng = np.random.default_rng(2)
    w2 = [rng.standard_normal(grid2.shape) for _ in range(2)]
    w2[0][0] = 0.0
    w2[1][:, 0] = 0.0
    u, phi, info = grid2.project([grid2.quantize(a) for a in w2], return_info=True)
    assert info["compatibility_defect"] < 1e-10
    assert abs(grid2.dequantize(phi).mean()) < 1e-10
    npt.assert_allclose(grid2.dequantize(info["divergence"]), 0.0, atol=1e-8)
    assert info["outlet"] == [None, None] and info["solver"]["flag"] in (0, 2)


def test_project_returns_the_outlet_face():
    """With p = 0 at an outlet, the returned outlet face and
    divergence_with_outlet give a divergence-free field from the public result
    alone (review M4: the stored faces with the outflow closure do not)."""
    grid, rights, pressure = _channel()
    rng = np.random.default_rng(11)
    w = [rng.standard_normal(grid.shape) for _ in range(2)]
    w[1][:, 0] = 0.0
    u, phi, info = grid.project([grid.quantize(a) for a in w], rights=rights,
                                pressure=pressure, return_info=True)
    full = grid.divergence_with_outlet(u, rights, info["outlet"])
    npt.assert_allclose(grid.dequantize(full), 0.0, atol=1e-6)
    npt.assert_allclose(grid.dequantize(info["divergence"]), grid.dequantize(full), atol=1e-10)
    ax = grid.axes[0]
    outlet = grid.dequantize(info["outlet"][0])
    npt.assert_allclose(outlet[-1], w[0][-1] + 2 * grid.dequantize(phi)[-1] / ax.h, atol=1e-8)
    npt.assert_allclose(outlet[:-1], 0.0, atol=1e-12)
    assert info["outlet"][1] is None
    stored = sum(grid.dequantize(D @ a) for D, a in zip(grid.divergence(rights), u))
    assert np.abs(stored).max() > 1e-3  # the closure alone misses the corrected face


def test_project_rejects_unsupported_conditions():
    grid, rights, pressure = _channel()
    zero = [grid.quantize(np.zeros(grid.shape))] * 2
    with pytest.raises(ValueError, match="outflow"):
        grid.project(zero, rights=rights)  # Neumann pressure at an outflow
    with pytest.raises(NotImplementedError, match="x = 0"):
        grid.project(zero, rights=rights,
                     pressure=[(Dirichlet(), Dirichlet()), (Neumann(), Neumann())])
    with pytest.raises(NotImplementedError, match="homogeneous"):
        grid.project(zero, rights=[Neumann(1.0), Dirichlet()], pressure=pressure)
