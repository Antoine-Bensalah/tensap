"""
Unit tests of tensap.tensorized_pde_discretization: boundary conditions and
cell-centred finite volumes (Axis, Grid) in QTT format.

Oracles: the same matrices assembled with scipy.sparse, written here from the
formulas of the finite volume closures, and Kronecker products.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
import tensap
from tensap.tensorized_pde_discretization import BoundaryCondition, Dirichlet, Neumann
from tensap.tensorized_pde_discretization.finite_volumes import Axis, Grid


# --- boundary conditions -------------------------------------------------------------


def test_boundary_conditions():
    assert Dirichlet().is_homogeneous and Neumann(0.0).is_homogeneous
    assert not Dirichlet(1.0).is_homogeneous
    assert not Neumann(lambda y: y[:, 0]).is_homogeneous
    assert isinstance(Dirichlet(), BoundaryCondition)
    assert repr(Dirichlet(2.0)) == "Dirichlet(2.0)"


# --- one axis ------------------------------------------------------------------------


def _reference_dirichlet(N, h):
    """Cell centres, ghosts u_g = -u_0 and u_g = -u_{N-1}: diagonal -3 at both ends."""
    D2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N)).toarray()
    D2[0, 0] = D2[-1, -1] = -3.0
    return D2 / h**2


def test_axis_geometry():
    axis = Axis(2.0, 3)
    assert axis.size == 8 and np.isclose(axis.h, 0.25)
    npt.assert_allclose(axis.cell_centers(), 0.125 + 0.25 * np.arange(8))
    assert isinstance(axis.left, Dirichlet) and isinstance(axis.right, Dirichlet)
    assert "periodic" in repr(Axis(1.0, 2, periodic=True))


@pytest.mark.parametrize("levels", [1, 3, 6])
def test_axis_dirichlet_second_difference(levels):
    axis = Axis(1.5, levels, Dirichlet(), Dirichlet(4.0))  # data: only for the source
    D2 = axis.second_difference()
    expected = _reference_dirichlet(axis.size, axis.h)
    npt.assert_allclose(tensap.dequantize_operator(D2), expected, atol=1e-9)
    npt.assert_allclose(axis.second_difference_matrix().toarray(), expected)
    assert max(D2.ranks, default=1) <= 5  # independent of the number of cells
    # symmetric negative definite
    assert np.all(np.linalg.eigvalsh(expected) < 0)


def test_axis_periodic_second_difference():
    axis = Axis(1.0, 4, periodic=True)
    N, h = axis.size, axis.h
    expected = (np.roll(np.eye(N), 1, axis=1) + np.roll(np.eye(N), -1, axis=1)
                - 2 * np.eye(N)) / h**2
    npt.assert_allclose(tensap.dequantize_operator(axis.second_difference()), expected,
                        atol=1e-9)
    npt.assert_allclose(axis.second_difference_matrix().toarray(), expected)


def test_axis_errors():
    with pytest.raises(ValueError):
        Axis(0.0, 3)
    with pytest.raises(ValueError):
        Axis(1.0, 0)
    with pytest.raises(ValueError):
        Axis(1.0, 3, Dirichlet(), periodic=True)
    with pytest.raises(TypeError):
        Axis(1.0, 3, "dirichlet")

    class Robin(BoundaryCondition):
        pass

    with pytest.raises(NotImplementedError):
        Axis(1.0, 3, Robin()).second_difference_matrix()


# --- grids ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_grid_laplacian(ordering):
    axes = [Axis(2.0, 4), Axis(1.0, 3), Axis(0.5, 2, periodic=True)]
    grid = Grid(axes, ordering)
    assert grid.dim == 3 and grid.shape == (16, 8, 4)
    A = grid.laplacian()
    # oracle: Kronecker sum of the 1D reference matrices, unknowns in C order
    D = [_reference_dirichlet(16, 2.0 / 16), _reference_dirichlet(8, 1.0 / 8),
         axes[2].second_difference_matrix().toarray()]
    I = [np.eye(16), np.eye(8), np.eye(4)]  # noqa: E741 (identities)
    expected = (np.kron(np.kron(D[0], I[1]), I[2]) + np.kron(np.kron(I[0], D[1]), I[2])
                + np.kron(np.kron(I[0], I[1]), D[2]))
    npt.assert_allclose(tensap.dequantize_operator(A, grid.tensorizer), expected, atol=1e-8)
    npt.assert_allclose(grid.laplacian_matrix().toarray(), expected)
    # small ranks; the interleaved ordering couples the axes digit by digit
    assert max(A.ranks) <= (6 if ordering == "blocks" else 12)


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_grid_cell_centers_and_quantization(ordering, rng):
    grid = Grid([Axis(2.0, 3), Axis(1.0, 2)], ordering)
    x = grid.cell_centers()
    assert x.shape == (8, 4, 2)
    npt.assert_allclose(x[:, 0, 0], (np.arange(8) + 0.5) * 0.25)
    npt.assert_allclose(x[0, :, 1], (np.arange(4) + 0.5) * 0.25)
    a = rng.standard_normal((8, 4))
    npt.assert_allclose(grid.dequantize(grid.quantize(a)), a)

    def f(p):
        return np.sin(p[:, 0]) * p[:, 1]

    npt.assert_allclose(grid.dequantize(grid.quantize(f)), f(x.reshape(-1, 2)).reshape(8, 4))
    # the Laplacian acts on the quantized arrays as the sparse matrix
    A = grid.laplacian()
    npt.assert_allclose(grid.dequantize(A @ grid.quantize(a)).ravel(),
                        grid.laplacian_matrix() @ a.ravel(), atol=1e-8)


def test_grid_errors():
    with pytest.raises(ValueError):
        Grid([Axis(1.0, 2)], ordering="natural")
    assert "blocks" in repr(Grid([Axis(1.0, 2)]))


# --- tensor products in a layout (qtt.py) --------------------------------------------


@pytest.mark.parametrize("ordering_type", [1, 2])
def test_layout_product_and_embed(rng, ordering_type):
    t = tensap.Tensorizer(2, [3, 1, 2], 3)
    t.ordering_type = ordering_type
    vectors = [rng.standard_normal(2**L) for L in t.levels]
    v = tensap.qtt_layout_product([tensap.quantize(u) for u in vectors], t)
    npt.assert_allclose(tensap.dequantize(v, t), np.einsum("i,j,k->ijk", *vectors))
    matrices = [rng.standard_normal((2**L, 2**L)) for L in t.levels]
    A = tensap.qtt_layout_product([tensap.quantize_operator(m) for m in matrices], t)
    npt.assert_allclose(tensap.dequantize_operator(A, t),
                        np.kron(np.kron(matrices[0], matrices[1]), matrices[2]))
    E = tensap.qtt_embed(tensap.quantize_operator(matrices[2]), 2, t)
    npt.assert_allclose(tensap.dequantize_operator(E, t),
                        np.kron(np.eye(16), matrices[2]))


def test_layout_product_errors(rng):
    t = tensap.Tensorizer(2, [2, 2], 2)
    x = tensap.quantize(rng.standard_normal(4))
    with pytest.raises(ValueError):
        tensap.qtt_layout_product([x], t)
    with pytest.raises(TypeError):
        tensap.qtt_layout_product([x, tensap.qtt_identity(2)], t)
    with pytest.raises(ValueError):
        tensap.qtt_layout_product([x, tensap.quantize(rng.standard_normal(8))], t)


# --- boundary sources (non-homogeneous conditions) -----------------------------------


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_boundary_source_2d(ordering):
    left = Dirichlet(lambda y: np.sin(3 * y[:, 0]))
    grid = Grid([Axis(2.0, 4, left, Dirichlet(1.5)),
                 Axis(1.0, 3, Dirichlet(), Dirichlet(lambda x: x[:, 0] ** 2))], ordering)
    s = grid.boundary_source()  # data of the conditions
    expected = np.zeros((16, 8))
    x = grid.cell_centers()
    hx, hy = 2.0 / 16, 1.0 / 8
    expected[0, :] += 2 / hx**2 * np.sin(3 * x[0, :, 1])
    expected[-1, :] += 2 / hx**2 * 1.5
    expected[:, -1] += 2 / hy**2 * x[:, -1, 0] ** 2  # the corner cells get both faces
    npt.assert_allclose(grid.dequantize(s), expected, atol=1e-10)
    npt.assert_allclose(grid.boundary_source_array(), expected)


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_boundary_source_3d_and_data_forms(ordering, rng):
    grid = Grid([Axis(1.0, 2), Axis(1.0, 3), Axis(2.0, 2)], ordering)
    face = rng.standard_normal((4, 8))  # data on the face normal to axis 2
    face_qtt = tensap.quantize(face, grid.face_tensorizer(2))
    data = {(0, "left"): lambda p: p[:, 0] * p[:, 1],
            (2, "left"): 2.0,
            (2, "right"): face_qtt}
    s = grid.boundary_source(data)
    npt.assert_allclose(grid.dequantize(s), grid.boundary_source_array(data), atol=1e-10)
    # the face data land on the last cells of axis 2 (h = 0.5), away from the
    # face x_0 = 0 which also has data
    npt.assert_allclose(grid.dequantize(s)[1:, :, -1], 2 / 0.5**2 * face[1:], atol=1e-10)
    # a homogeneous problem has a zero source
    assert grid.boundary_source().norm() == 0


def test_boundary_source_1d():
    grid = Grid([Axis(1.0, 3, Dirichlet(2.0), Dirichlet(-1.0))])
    expected = np.zeros(8)
    expected[0], expected[-1] = 2 * 2.0 * 64, 2 * -1.0 * 64
    npt.assert_allclose(grid.dequantize(grid.boundary_source()), expected, atol=1e-10)
    with pytest.raises(TypeError):
        grid.boundary_source({(0, "left"): lambda p: p})


def test_boundary_source_errors(rng):
    grid = Grid([Axis(1.0, 2), Axis(1.0, 2, periodic=True)])
    with pytest.raises(ValueError):  # a periodic axis has no face
        grid.boundary_source({(1, "left"): 1.0})
    with pytest.raises(ValueError):
        grid.boundary_source({(0, "top"): 1.0})
    with pytest.raises(ValueError):  # wrong layout of the face data
        grid.boundary_source({(0, "left"): tensap.quantize(rng.standard_normal(8))})

    class Robin(BoundaryCondition):
        pass

    with pytest.raises(NotImplementedError):
        Axis(1.0, 2, Robin()).boundary_weight("left")


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_non_homogeneous_dirichlet_solution(ordering):
    # -Delta_h u = f + s solved in QTT equals the sparse solve with the same data
    from scipy.sparse.linalg import spsolve
    grid = Grid([Axis(1.0, 3, Dirichlet(1.0), Dirichlet(lambda y: y[:, 0])),
                 Axis(1.0, 3, Dirichlet(0.5), Dirichlet(0.0))], ordering)
    A = -1.0 * grid.laplacian()
    b = grid.quantize(lambda p: np.ones(p.shape[0])) + grid.boundary_source()
    u, _ = tensap.TruncatedPCG(A, b, tensap.Truncator(tolerance=1e-12), tolerance=1e-12,
                               max_iterations=200, stagnation=1e-15, display=False).solve()
    rhs = 1.0 + grid.boundary_source_array()
    expected = spsolve(-grid.laplacian_matrix().tocsc(), rhs.ravel()).reshape(grid.shape)
    npt.assert_allclose(grid.dequantize(u), expected, rtol=1e-8)


# --- QTT helpers for the sources (qtt.py) --------------------------------------------


def test_elementary_vector_and_insert_axis(rng):
    e = tensap.qtt_elementary_vector(4, 11)
    expected = np.zeros(16)
    expected[11] = 1.0
    npt.assert_allclose(tensap.dequantize(e), expected)
    for ordering_type in [1, 2]:
        t = tensap.Tensorizer(2, [2, 3, 2], 3)
        t.ordering_type = ordering_type
        face_t = tensap.Tensorizer(2, [2, 2], 2)
        face_t.ordering_type = ordering_type
        face = rng.standard_normal((4, 4))
        u = tensap.qtt_insert_axis(tensap.quantize(face, face_t), 1,
                                   tensap.qtt_elementary_vector(3, 5), t)
        dense = np.zeros((4, 8, 4))
        dense[:, 5, :] = face
        npt.assert_allclose(tensap.dequantize(u, t), dense, atol=1e-12)
    with pytest.raises(ValueError):  # rank one only
        tensap.qtt_insert_axis(tensap.quantize(face, face_t), 1,
                               tensap.quantize(rng.standard_normal(8)), t)
    with pytest.raises(ValueError):
        tensap.qtt_insert_axis(tensap.quantize(face, face_t), 1,
                               tensap.qtt_elementary_vector(2, 1), t)
    with pytest.raises(ValueError):
        tensap.qtt_insert_axis(tensap.quantize(rng.standard_normal(8)), 1,
                               tensap.qtt_elementary_vector(3, 5), t)


# --- Neumann conditions (step 3) -----------------------------------------------------


@pytest.mark.parametrize("left, right", [(Neumann(), Dirichlet()), (Dirichlet(), Neumann()),
                                         (Neumann(), Neumann())])
def test_axis_neumann_second_difference(left, right):
    axis = Axis(2.0, 4, left, right)
    N, h = axis.size, axis.h
    expected = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N)).toarray()
    # ghosts: u_g = 2 g - u_0 (Dirichlet, -1) or u_g = u_0 + h g (Neumann, +1)
    expected[0, 0] += 1.0 if isinstance(left, Neumann) else -1.0
    expected[-1, -1] += 1.0 if isinstance(right, Neumann) else -1.0
    expected /= h**2
    npt.assert_allclose(tensap.dequantize_operator(axis.second_difference()), expected,
                        atol=1e-9)
    npt.assert_allclose(axis.second_difference_matrix().toarray(), expected)
    eigenvalues = np.linalg.eigvalsh(expected)  # symmetric
    if isinstance(left, Neumann) and isinstance(right, Neumann):
        assert np.isclose(eigenvalues.max(), 0.0, atol=1e-8)  # constants: kernel
        npt.assert_allclose(expected @ np.ones(N), 0.0, atol=1e-9)
    else:
        assert eigenvalues.max() < 0
    assert np.isclose(axis.boundary_weight("left"),
                      1 / h if isinstance(left, Neumann) else 2 / h**2)


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_neumann_source_and_solution(ordering):
    # u = exp(x) cos(y) is harmonic: Neumann data on the x faces (outward normal
    # derivative), Dirichlet data on the y faces; QTT solve against spsolve
    from scipy.sparse.linalg import spsolve
    grid = Grid([Axis(2.0, 3, Neumann(lambda y: -np.cos(y[:, 0])),
                      Neumann(lambda y: np.exp(2.0) * np.cos(y[:, 0]))),
                 Axis(1.0, 3, Dirichlet(lambda x: np.exp(x[:, 0])),
                      Dirichlet(lambda x: np.exp(x[:, 0]) * np.cos(1.0)))], ordering)
    s = grid.boundary_source()
    npt.assert_allclose(grid.dequantize(s), grid.boundary_source_array(), atol=1e-9)
    A = -1.0 * grid.laplacian()
    u, _ = tensap.TruncatedPCG(A, s, tensap.Truncator(tolerance=1e-12), tolerance=1e-12,
                               max_iterations=300, stagnation=1e-15, display=False).solve()
    expected = spsolve(-grid.laplacian_matrix().tocsc(),
                       grid.boundary_source_array().ravel()).reshape(grid.shape)
    npt.assert_allclose(grid.dequantize(u), expected, rtol=1e-7)
    x = grid.cell_centers()
    exact = np.exp(x[..., 0]) * np.cos(x[..., 1])
    assert np.sqrt(np.mean((expected - exact) ** 2)) < 1e-2  # coarse grid, order 2


# --- singular problems: periodic and pure Neumann (step 4) ---------------------------


@pytest.mark.parametrize("axis", [Axis(2.0, 4, Neumann(), Neumann()), Axis(1.5, 3, periodic=True)])
def test_axis_smallest_nonzero_eigenvalue(axis):
    assert axis.is_singular
    eigenvalues = np.sort(np.linalg.eigvalsh(-axis.second_difference_matrix().toarray()))
    assert np.isclose(eigenvalues[0], 0.0, atol=1e-10)
    assert np.isclose(axis.smallest_nonzero_eigenvalue(), eigenvalues[1])
    with pytest.raises(ValueError):
        Axis(1.0, 3, Neumann(), Dirichlet()).smallest_nonzero_eigenvalue()


@pytest.mark.parametrize("ordering", ["blocks", "interleaved"])
def test_constant_mode_regularization(ordering):
    grid = Grid([Axis(2.0, 3, Neumann(), Neumann()), Axis(1.0, 2, periodic=True)], ordering)
    assert grid.is_singular
    R = grid.constant_mode_regularization()
    assert R.ranks == (1,) * 4
    M = -grid.laplacian_matrix().toarray()
    sigma = min(axis.smallest_nonzero_eigenvalue() for axis in grid.axes)
    npt.assert_allclose(tensap.dequantize_operator(R, grid.tensorizer),
                        sigma / M.shape[0] * np.ones(M.shape))
    # the zero eigenvalue is replaced by sigma: symmetric positive definite,
    # with the same smallest eigenvalue
    eigenvalues = np.linalg.eigvalsh(M + tensap.dequantize_operator(R, grid.tensorizer))
    assert np.isclose(eigenvalues.min(), sigma)
    assert not Grid([Axis(1.0, 2, Neumann(), Dirichlet())]).is_singular
    with pytest.raises(ValueError):
        Grid([Axis(1.0, 2)]).constant_mode_regularization()


def test_regularized_solution_of_singular_problems(rng):
    # compatible data: the solution of zero mean; incompatible: a constant shift
    grid = Grid([Axis(1.0, 3, Neumann(), Neumann()), Axis(1.0, 3, Neumann(), Neumann())])
    M = -grid.laplacian_matrix().toarray()
    A = -1.0 * grid.laplacian() + grid.constant_mode_regularization()
    sigma = grid.axes[0].smallest_nonzero_eigenvalue()
    f = rng.standard_normal(grid.shape)
    f_mean = f.mean()
    for case, (data, shift) in enumerate([(f - f_mean, 0.0), (f, f_mean / sigma)]):
        u, output = tensap.TruncatedPCG(A, grid.quantize(data), tensap.Truncator(tolerance=1e-13),
                                        tolerance=1e-11, max_iterations=300, stagnation=1e-15,
                                        display=False).solve()
        u = grid.dequantize(u)
        # minimum norm solution, by the eigenvectors of M without its zero
        # eigenvalue (about 1e-13 in floating point: np.linalg.pinv with its
        # default cut-off inverted it on the Linux and macOS arm64 runners of
        # the CI, and the reference was wrong by 1e-2)
        w, V = np.linalg.eigh(M)
        keep = w > 1e-8 * w.max()
        rhs = V.T @ (data - data.mean()).ravel()
        zero_mean = V[:, keep] @ (rhs[keep] / w[keep])
        info = (f"case {case}: flag {output['flag']}, iter {output['iter']}, "
                f"error {output['error']:.2e}, sigma {sigma:.6e}, mean {f_mean:.3e}")
        npt.assert_allclose(u.ravel(), zero_mean + shift, atol=1e-7, err_msg=info)
