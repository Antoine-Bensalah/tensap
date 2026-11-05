"""
Unit tests of tensap.TTTensorALSLinearSolver: ALS and DMRG sweeps on
TTOperator / TTTensor systems.

Oracles: dense solutions of the same systems (numpy.linalg.solve on the
matrices of the operators), and the energy J(v) = 1/2 (Av, v) - (b, v).
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap.tensorized_pde_discretization import Dirichlet
from tensap.tensorized_pde_discretization.finite_volumes import Axis, Grid


def _laplacian(n, d, shift=0.0):
    K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) * (n + 1) ** 2
    A = tensap.LaplacianLikeOperatorFactory(K, np.eye(n), d).make_tt_operator()
    if shift:
        A = A + shift * tensap.TTOperator.eye([n] * d)
    return A


def _matrix(A):
    """Matrix of a TTOperator acting on the C-order vectors."""
    full = A.full().numpy()
    d = A.order
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    N = int(np.prod([n for n, _ in A.shape]))
    return full.transpose(perm).reshape(N, N)


def _dense_solution(A, b):
    return np.linalg.solve(_matrix(A), b.full().numpy().ravel())


def _energy(A, b, x):
    return 0.5 * x.dot(A @ x) - b.dot(x)


@pytest.fixture
def system():
    np.random.seed(0)
    A = _laplacian(5, 3, shift=10.0)
    b = tensap.TTTensor.randn([5, 5, 5], [2, 2])
    return A, b


def _solver(A, b, **kwargs):
    kwargs.setdefault("display", False)
    return tensap.TTTensorALSLinearSolver(A, b, **kwargs)


# --- ALS -----------------------------------------------------------------------------


def test_als_full_ranks_is_exact(system):
    A, b = system
    x0 = tensap.TTTensor.randn([5, 5, 5], [5, 5])
    x, output = _solver(A, b, x0=x0, max_iterations=4, tolerance=1e-10).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-10)
    assert output["flag"] == 0
    assert x.ranks == (5, 5)


def test_als_energy_decreases_and_ranks_are_fixed(system):
    A, b = system
    x0 = tensap.TTTensor.randn([5, 5, 5], [2, 2])
    energies = []
    for iterations in (1, 2, 3, 4):
        x, _ = _solver(A, b, x0=x0, max_iterations=iterations, tolerance=0.0,
                       stagnation=0.0).solve()
        energies.append(_energy(A, b, x))
        assert x.ranks == (2, 2)
    assert all(e1 >= e2 - 1e-10 for e1, e2 in zip(energies, energies[1:]))


def test_als_default_initial_ranks(system):
    A, b = system
    np.random.seed(1)
    x, output = _solver(A, b, ranks=5, max_iterations=4, tolerance=1e-10).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-9)


# --- DMRG ----------------------------------------------------------------------------


def test_dmrg_adapts_the_ranks(system):
    A, b = system
    x0 = tensap.TTTensor.ones([5, 5, 5])
    truncator = tensap.Truncator(tolerance=1e-12, max_rank=50)
    x, output = _solver(A, b, x0=x0, algorithm="dmrg", truncator=truncator,
                        tolerance=1e-10, max_iterations=10).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-10)
    assert output["flag"] == 0
    assert max(x.ranks) > 1


def test_dmrg_max_rank_is_respected(system):
    A, b = system
    truncator = tensap.Truncator(tolerance=1e-14, max_rank=2)
    x, _ = _solver(A, b, algorithm="dmrg", truncator=truncator,
                   max_iterations=3).solve()
    assert max(x.ranks) <= 2


def test_dmrg_order_two():
    np.random.seed(2)
    A = _laplacian(6, 2, shift=1.0)
    b = tensap.TTTensor.randn([6, 6], [3])
    x, _ = _solver(A, b, algorithm="dmrg", tolerance=1e-12, max_iterations=4).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-10)


def test_order_one():
    A = _laplacian(7, 1)
    b = tensap.TTTensor([np.arange(1.0, 8.0)[None, :, None]])
    for algorithm in ("als", "dmrg"):
        x, output = _solver(A, b, algorithm=algorithm).solve()
        npt.assert_allclose(x.full().numpy(), _dense_solution(A, b), atol=1e-12)
        assert output["flag"] == 0


# --- local solvers -------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["als", "dmrg"])
@pytest.mark.parametrize("preconditioner", ["block_jacobi", None])
def test_matrix_free_local_solver_matches_dense(system, algorithm, preconditioner):
    A, b = system
    x0 = tensap.TTTensor.randn([5, 5, 5], [3, 3])
    dense, _ = _solver(A, b, x0=x0, algorithm=algorithm, max_iterations=3).solve()
    free, _ = _solver(A, b, x0=x0, algorithm=algorithm, max_iterations=3,
                      dense_max_size=0, local_tolerance=1e-14,
                      local_preconditioner=preconditioner).solve()
    npt.assert_allclose(free.full().numpy(), dense.full().numpy(), atol=1e-8)


def test_non_symmetric_operator_uses_gmres():
    np.random.seed(3)
    n, d = 5, 3
    A = _laplacian(n, d, shift=5.0)
    B = np.eye(n, k=1) * 3.0
    advection = tensap.TTOperator([B[None, :, :, None]] + [np.eye(n)[None, :, :, None]] * 2)
    A = A + advection
    b = tensap.TTTensor.randn([n] * d, [2, 2])
    solver = _solver(A, b, algorithm="dmrg", dense_max_size=0, tolerance=1e-9,
                     local_tolerance=1e-13, max_iterations=10)
    assert not solver.is_symmetric()
    x, _ = solver.solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-7)


def test_minimize_residual():
    np.random.seed(4)
    n, d = 4, 3
    A = _laplacian(n, d, shift=5.0) + tensap.TTOperator(
        [(np.eye(n, k=1) * 2.0)[None, :, :, None]] + [np.eye(n)[None, :, :, None]] * 2)
    b = tensap.TTTensor.randn([n] * d, [2, 2])
    x, output = _solver(A, b, algorithm="dmrg", minimize_residual=True,
                        tolerance=1e-9, max_iterations=10).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-7)
    assert output["error"] < 1e-9


# --- QTT Poisson problem -------------------------------------------------------------


def test_qtt_poisson_dirichlet_dmrg():
    grid = Grid([Axis(1.0, 4, Dirichlet(), Dirichlet()), Axis(1.0, 3, Dirichlet(), Dirichlet())])
    A = -1.0 * grid.laplacian()
    xc, yc = np.meshgrid(grid.axes[0].cell_centers(), grid.axes[1].cell_centers(),
                         indexing="ij")
    f = np.exp(-20 * ((xc - 0.5) ** 2 + (yc - 0.4) ** 2))
    b = grid.quantize(f)
    truncator = tensap.Truncator(tolerance=1e-10, max_rank=30)
    x, output = _solver(A, b, algorithm="dmrg", truncator=truncator, tolerance=1e-8,
                        max_iterations=20).solve()
    reference = np.linalg.solve(-grid.laplacian_matrix().toarray(), f.ravel())
    npt.assert_allclose(grid.dequantize(x).ravel(), reference,
                        atol=1e-7 * np.abs(reference).max())
    assert output["flag"] == 0


# --- options and errors --------------------------------------------------------------


def test_stagnation_and_output(system):
    A, b = system
    x0 = tensap.TTTensor.randn([5, 5, 5], [1, 1])
    x, output = _solver(A, b, x0=x0, tolerance=0.0, stagnation=1e-3,
                        max_iterations=50).solve()
    assert output["flag"] == 2
    assert len(output["errvec"]) == output["iter"]
    _, output = _solver(A, b, x0=x0, tolerance=0.0, stagnation=0.0,
                        max_iterations=2).solve()
    assert output["flag"] == 1 and output["iter"] == 2


def test_error_criterion_and_check_residual(system, capsys):
    A, b = system
    calls = []

    def criterion(x):
        calls.append(1)
        return 1.0

    _solver(A, b, error_criterion=criterion, max_iterations=4, tolerance=1e-3,
            stagnation=0.0, check_residual=2, display=True).solve()
    assert len(calls) == 2
    assert "Iteration" in capsys.readouterr().out


def test_wrong_arguments(system):
    A, b = system
    with pytest.raises(ValueError):
        _solver(A, b, algorithm="cg")
    with pytest.raises(ValueError):
        _solver(A, b, local_preconditioner="ilu")
    with pytest.raises(ValueError):
        _solver(A, tensap.TTTensor.randn([5, 5], [2]))


# --- AMEn ----------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["als", "dmrg", "amen"])
def test_zero_right_hand_side(system, algorithm):
    """b = 0: the solution x = 0, without sweeps (they normalize x)."""
    A, b = system
    x, output = _solver(A, b * 0.0, algorithm=algorithm).solve()
    assert x.norm() == 0 and output["flag"] == 0 and output["iter"] == 0
    assert list(x.shape) == list(b.shape)


def test_amen_from_rank_one(system):
    A, b = system
    x0 = tensap.TTTensor.ones([5, 5, 5])
    x, output = _solver(A, b, x0=x0, algorithm="amen", tolerance=1e-10,
                        max_iterations=20).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-10)
    assert output["flag"] == 0
    assert max(x.ranks) > 1


def test_amen_max_rank_and_kick_rank(system):
    A, b = system
    truncator = tensap.Truncator(tolerance=1e-14, max_rank=3)
    x, _ = _solver(A, b, algorithm="amen", truncator=truncator, kick_rank=2,
                   max_iterations=4).solve()
    assert max(x.ranks) <= 3


@pytest.mark.parametrize("d", [1, 2, 4])
def test_amen_orders(d):
    np.random.seed(5)
    A = _laplacian(4, d, shift=1.0)
    b = tensap.TTTensor.randn([4] * d, [2] * (d - 1))
    x, _ = _solver(A, b, algorithm="amen", tolerance=1e-11, max_iterations=20).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-9)


def test_amen_matrix_free_non_symmetric():
    np.random.seed(6)
    n, d = 5, 3
    A = _laplacian(n, d, shift=5.0) + tensap.TTOperator(
        [(np.eye(n, k=1) * 3.0)[None, :, :, None]] + [np.eye(n)[None, :, :, None]] * 2)
    b = tensap.TTTensor.randn([n] * d, [2, 2])
    x, _ = _solver(A, b, algorithm="amen", dense_max_size=0, local_tolerance=1e-13,
                   tolerance=1e-9, max_iterations=20).solve()
    npt.assert_allclose(x.full().numpy().ravel(), _dense_solution(A, b), atol=1e-7)


def test_qtt_poisson_dirichlet_amen():
    grid = Grid([Axis(1.0, 4, Dirichlet(), Dirichlet()), Axis(1.0, 4, Dirichlet(), Dirichlet())],
                ordering="interleaved")
    A = -1.0 * grid.laplacian()
    xc, yc = np.meshgrid(grid.axes[0].cell_centers(), grid.axes[1].cell_centers(),
                         indexing="ij")
    f = np.exp(-20 * ((xc - 0.5) ** 2 + (yc - 0.4) ** 2))
    b = grid.quantize(f)
    truncator = tensap.Truncator(tolerance=1e-10, max_rank=30)
    x, output = _solver(A, b, algorithm="amen", truncator=truncator, tolerance=1e-8,
                        max_iterations=30).solve()
    reference = np.linalg.solve(-grid.laplacian_matrix().toarray(), f.ravel())
    npt.assert_allclose(grid.dequantize(x).ravel(), reference,
                        atol=1e-7 * np.abs(reference).max())
