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
    """Viscous term of u: faces along x, cells along y; of v: the reverse."""
    grid = _grid()
    ax, ay = grid.axes
    Lu = grid.laplacian("u", [(Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet())])
    expected = (np.kron(_matrix(ax.face_laplacian(Dirichlet())), np.eye(4))
                + np.kron(np.eye(8), _matrix(ay.cell_laplacian(Dirichlet(), Dirichlet()))))
    npt.assert_allclose(_matrix(Lu), expected, atol=1e-7)
    Lv = grid.laplacian("v", [(Dirichlet(), Neumann()), (Dirichlet(), Dirichlet())])
    expected = (np.kron(_matrix(ax.cell_laplacian(Dirichlet(), Neumann())), np.eye(4))
                + np.kron(np.eye(8), _matrix(ay.face_laplacian(Dirichlet()))))
    npt.assert_allclose(_matrix(Lv), expected, atol=1e-7)
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
