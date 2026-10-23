"""
Unit tests of the loss functions, of the learning of linear models and of
model selection.

Least squares are checked through exact recovery of a target in the span of
the basis, and their leave-one-out error against a brute-force
leave-one-out; the density estimation in L2 through its closed form (mean of
the basis functions) and its leave-one-out risk against a brute force.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

C = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.2])  # sparse coefficients


@pytest.fixture
def basis():
    return tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(6))


@pytest.fixture
def x(rng):
    return rng.uniform(-1, 1, 30)


def _model(basis, x, y, **attributes):
    model = tensap.LinearModelLearningSquareLoss()
    model.basis = basis
    model.training_data = [x, np.array(y)]
    model.display = False
    for name, value in attributes.items():
        setattr(model, name, value)
    return model


# --- loss functions ---------------------------------------------------------------


def test_square_loss(rng):
    y = rng.standard_normal(10)
    y_pred = y + 0.1
    loss = tensap.SquareLossFunction()
    npt.assert_allclose(loss.eval(y_pred, [None, y]), 0.01)
    npt.assert_allclose(loss.risk_estimation(y_pred, [None, y]), 0.01)
    npt.assert_allclose(loss.relative_test_error(y_pred, [None, y]),
                        0.1 / np.sqrt(np.mean(y**2)))
    npt.assert_allclose(loss.test_error(y_pred, [None, y]), loss.relative_test_error(
        y_pred, [None, y]))
    loss.error_type = "absolute"
    npt.assert_allclose(loss.test_error(y_pred, [None, y]), 0.1)
    loss.error_type = "other"
    with pytest.raises(ValueError):
        loss.test_error(y_pred, [None, y])
    # a function, and vector-valued outputs
    x = rng.standard_normal((10, 1))
    f = tensap.UserDefinedFunction(lambda z: 2 * z[:, 0], 1)
    npt.assert_allclose(
        tensap.SquareLossFunction().eval(lambda z: 2 * z[:, 0], [x, 2 * x[:, 0]]), 0)
    npt.assert_allclose(tensap.SquareLossFunction().eval(f, [x, 2 * x[:, 0] + 1]), 1)
    Y = np.stack([y, y], 1)
    npt.assert_allclose(tensap.SquareLossFunction().eval(Y + 0.1, [None, Y]), 0.02)


def test_density_l2_loss(rng):
    # risk ||g||^2 - 2 E g(X) of a function g with known norm
    x = rng.uniform(-1, 1, 20)
    loss = tensap.DensityL2LossFunction()
    npt.assert_allclose(loss.eval(x, [x], 2.0), 4 - 2 * x)
    npt.assert_allclose(loss.test_error(x, [x], 2.0), 4 - 2 * np.mean(x))
    with pytest.raises(NotImplementedError):
        loss.relative_test_error(x, [x], 2.0)
    loss.error_type = "relative"
    with pytest.raises(AssertionError):
        loss.test_error(x, [x], 2.0)


def test_custom_loss(rng):
    y = rng.standard_normal(10)
    loss = tensap.CustomLossFunction(lambda y_true, y_pred: np.abs(y_true - y_pred))
    npt.assert_allclose(loss.risk_estimation(y + 0.5, [None, y]), 0.5)
    risk, risk_ref = loss.risk_estimation(y + 0.5, [None, y], nargout=2)
    npt.assert_allclose(risk_ref, np.mean(np.abs(y)))
    npt.assert_allclose(loss.test_error(y + 0.5, [None, y]), 0.5)
    loss.error_function = lambda f, s: 7.0
    assert loss.test_error(y, [None, y]) == 7.0
    with pytest.raises(AssertionError):
        loss.relative_test_error(y, [None, y])
    loss.relative_error_function = lambda f, s: 3.0
    assert loss.relative_test_error(y, [None, y]) == 3.0


def test_linear_model_factory():
    assert isinstance(tensap.Learning.linear_model(tensap.SquareLossFunction()),
                      tensap.LinearModelLearningSquareLoss)
    assert isinstance(tensap.Learning.linear_model(tensap.DensityL2LossFunction()),
                      tensap.LinearModelLearningDensityL2)
    with pytest.raises(AssertionError):
        tensap.Learning("not a loss")


# --- least squares --------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["qr", "solve"])
def test_least_squares_recover_a_target_in_the_span(basis, x, solver):
    y = basis.eval(x) @ C
    f, output = _model(basis, x, y, linear_solver=solver).solve()
    assert isinstance(f, tensap.FunctionalBasisArray)
    npt.assert_allclose(f.data.ravel(), C, atol=1e-12)


def test_least_squares_without_a_basis(basis, x):
    model = tensap.LinearModelLearningSquareLoss()
    model.basis_eval = basis.eval(x)
    model.training_data = [None, basis.eval(x) @ C]
    sol, _ = model.solve()
    npt.assert_allclose(sol, C, atol=1e-12)


def test_vector_valued_least_squares(basis, x):
    Y = np.stack([basis.eval(x) @ C, basis.eval(x) @ (2 * C)], 1)
    f, _ = _model(basis, x, Y).solve()
    npt.assert_allclose(f.data, np.stack([C, 2 * C], 1), atol=1e-12)


def test_weighted_least_squares(basis, x, rng):
    y = basis.eval(x) @ C + 0.1 * rng.standard_normal(x.size)
    w = rng.uniform(0.5, 2, x.size)
    f, _ = _model(basis, x, y, weights=w).solve()
    A = basis.eval(x)
    expected = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    npt.assert_allclose(f.data.ravel(), expected, atol=1e-12)


@pytest.mark.parametrize("solver", ["qr", "solve"])
def test_leave_one_out_error(basis, x, rng, solver):
    y = basis.eval(x) @ C + 0.1 * rng.standard_normal(x.size)
    model = _model(basis, x, y, error_estimation=True, linear_solver=solver)
    model.error_estimation_options["correction"] = False
    _, output = model.solve()
    A = basis.eval(x)
    delta = []
    for i in range(x.size):
        keep = np.arange(x.size) != i
        s = np.linalg.lstsq(A[keep], y[keep], None)[0]
        delta.append(y[i] - A[i] @ s)
    npt.assert_allclose(output["delta"], delta, atol=1e-12)
    npt.assert_allclose(output["error"], np.sqrt(np.mean(np.square(delta)) / np.mean(y**2)))
    # the correction increases the estimation
    model = _model(basis, x, y, error_estimation=True, linear_solver=solver)
    assert model.solve()[1]["error"] > output["error"]


@pytest.mark.parametrize("kind", ["k_fold", "residuals"])
def test_other_error_estimations(basis, x, rng, kind):
    y = basis.eval(x) @ C + 0.1 * rng.standard_normal(x.size)
    _, output = _model(basis, x, y, error_estimation=True, error_estimation_type=kind).solve()
    if kind == "residuals":
        A = basis.eval(x)
        r = y - A @ np.linalg.lstsq(A, y, None)[0]
        npt.assert_allclose(output["error"], np.sqrt(np.mean(r**2) / np.mean(y**2)))
    else:
        assert 0 < output["error"] < 1


def test_leave_one_out_error_with_too_few_samples(basis, x):
    y = basis.eval(x[:6]) @ C
    model = _model(basis, x[:6], y, error_estimation=True)
    assert model.solve()[1]["error"] == np.inf


# the orthogonal matching pursuit stops once the exact sparse solution is found
@pytest.mark.filterwarnings("ignore:Orthogonal matching pursuit ended prematurely")
@pytest.mark.parametrize("regularization", ["l0", "l1", "l2"])
def test_regularized_least_squares(basis, x, regularization):
    y = basis.eval(x) @ C
    model = _model(basis, x, y, regularization=True, regularization_type=regularization)
    f, output = model.solve()
    npt.assert_allclose(f.data.ravel(), C, atol=1e-6)
    if regularization != "l2":  # model selection along the path: the support
        npt.assert_array_equal(output["pattern"], C != 0)


def test_regularized_least_squares_errors(basis, x):
    model = _model(basis, x, basis.eval(x) @ C, regularization=True,
                   regularization_type="l3")
    with pytest.raises(ValueError):
        model.solve()


def test_basis_adaptation(basis, x):
    y = basis.eval(x) @ np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    f, output = _model(basis, x, y, basis_adaptation=True).solve()
    npt.assert_allclose(f.data.ravel(), [1.0, 0.5, 0, 0, 0, 0], atol=1e-12)
    assert output["ind"] == 1  # the second set of the adaptation path
    assert output["error_path"].size == 6


def test_test_error(basis, x, rng):
    y = basis.eval(x) @ C
    model = _model(basis, x, y, test_error=True)
    x_test = rng.uniform(-1, 1, 10)
    model.test_data = [x_test, basis.eval(x_test) @ C + 0.01]
    _, output = model.solve()
    npt.assert_allclose(output["test_error"],
                        0.01 / np.sqrt(np.mean(model.test_data[1] ** 2)), rtol=1e-8)


# --- density estimation in L2 -----------------------------------------------------------------


def _density_model(basis, x, **attributes):
    model = tensap.LinearModelLearningDensityL2()
    model.basis = basis
    model.training_data = x
    for name, value in attributes.items():
        setattr(model, name, value)
    return model


def test_density_estimation(basis, x):
    f, output = _density_model(basis, x, error_estimation=True).solve()
    A = basis.eval(x)
    npt.assert_allclose(f.data.ravel(), A.mean(0))
    # leave-one-out risk ||g_i||^2 - 2 g_i(x_i), g_i estimated without x_i
    risks = []
    for i in range(x.size):
        g = np.delete(A, i, 0).mean(0)
        risks.append(g @ g - 2 * A[i] @ g)
    npt.assert_allclose(output["error"], np.mean(risks))


def test_density_estimation_with_a_shift(basis, x):
    # training_data = [x, b]: estimation of the density minus b
    b = np.full(6, 0.1)
    f, _ = _density_model(basis, [x, b]).solve()
    npt.assert_allclose(f.data.ravel(), basis.eval(x).mean(0) - b)


@pytest.mark.parametrize("option", ["regularization", "basis_adaptation"])
def test_density_estimation_selects_coefficients(basis, x, option):
    f, output = _density_model(basis, x, **{option: True}).solve()
    coef = f.data.ravel()
    full = basis.eval(x).mean(0)
    kept = coef != 0
    npt.assert_allclose(coef[kept], full[kept])
    assert output["error"] == np.min(output["error_path"])


def test_density_estimation_test_error(basis, x):
    model = _density_model(basis, x, test_error=True)
    model.test_data = x[:10]
    f, output = model.solve()
    c = f.data.ravel()
    npt.assert_allclose(output["test_error"], c @ c - 2 * np.mean(basis.eval(x[:10]) @ c))


def test_custom_loss_learning_requires_tensorflow():
    pytest.importorskip("tensorflow")


# --- model selection --------------------------------------------------------------------------


@pytest.fixture
def selection():
    m = tensap.ModelSelection()
    m.data["complexity"] = [1, 2, 4, 8, 16]
    m.data["empirical_risk"] = [1.0, 0.5, 0.1, 0.09, 0.085]
    return m


def test_lambda_path(selection):
    path, m_path = selection.lambda_path()
    assert np.all(np.diff(path) > 0)
    # m_path[k] is the model selected for a penalty just above path[k]
    npt.assert_array_equal(selection.m_lambda(path * (1 + 1e-9)), m_path)
    npt.assert_array_equal(selection.m_lambda([0, 10]), [4, 0])


def test_slope_heuristic(selection):
    lambda_hat, m_hat, path, m_path = selection.slope_heuristic()
    npt.assert_array_equal(m_hat, selection.m_lambda(lambda_hat))
    # the largest jump of complexity along the path, from 8 to 4, at path[1]
    npt.assert_allclose(lambda_hat, selection.gap_factor * path[1])
    selection.data["complexity"] = [3, 3, 3]
    assert selection.slope_heuristic()[:2] == (0, 0)


def test_complexities(basis, x):
    f = tensap.FunctionalBasisArray(C, basis)
    assert tensap.ModelSelection.complexity(f) == 6
    assert tensap.ModelSelection.complexity([f, f]) == [6, 6]
    assert tensap.ModelSelection.complexity_functional_basis_array(f) == 6
    t = tensap.TreeBasedTensor.randn(tensap.DimensionTree.balanced(3), [1, 2, 2, 2, 2], [3, 3, 3])
    assert tensap.ModelSelection.complexity(t) == t.storage()
    MS = tensap.ModelSelection
    assert MS.complexity_tree_based_tensor(t, c_type="grassman") == t.storage() - np.sum(
        t.ranks**2)
    assert MS.complexity_tree_based_tensor(t, c_type="stiefel") == t.storage() - np.sum(
        t.ranks * (t.ranks + 1) / 2)
    assert MS.complexity_tree_based_tensor([t], "sparse_storage") == [t.sparse_storage()]
    with pytest.raises(ValueError):
        MS.complexity_tree_based_tensor(t, c_type="other")
    with pytest.raises(ValueError):
        MS.complexity("not a model")


# --- magic indices ------------------------------------------------------------------------------


@pytest.mark.parametrize("option", ["left_right", "left", "right"])
def test_magic_indices_give_invertible_submatrices(rng, option):
    F = rng.standard_normal((8, 5))
    rows, cols = tensap.magic_indices(F, 4, option)
    assert len(set(rows)) == len(rows) == 4 and len(set(cols)) == 4
    assert np.linalg.cond(F[np.ix_(rows, cols)]) < 1e6
    if option == "left":
        assert cols == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        tensap.magic_indices(F, 2, "other")
