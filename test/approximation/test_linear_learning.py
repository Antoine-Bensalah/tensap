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


# --- least squares --------------------------------------------------------------------


def test_weighted_least_squares(basis, x, rng):
    y = basis.eval(x) @ C + 0.1 * rng.standard_normal(x.size)
    w = rng.uniform(0.5, 2, x.size)
    f, _ = _model(basis, x, y, weights=w).solve()
    A = basis.eval(x)
    expected = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    npt.assert_allclose(f.data.ravel(), expected, atol=1e-12)


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


# --- density estimation in L2 -----------------------------------------------------------------


def _density_model(basis, x, **attributes):
    model = tensap.LinearModelLearningDensityL2()
    model.basis = basis
    model.training_data = x
    for name, value in attributes.items():
        setattr(model, name, value)
    return model


def test_density_estimation_test_error(basis, x):
    model = _density_model(basis, x, test_error=True)
    model.test_data = x[:10]
    f, output = model.solve()
    c = f.data.ravel()
    npt.assert_allclose(output["test_error"], c @ c - 2 * np.mean(basis.eval(x[:10]) @ c))


# --- model selection --------------------------------------------------------------------------


@pytest.fixture
def selection():
    m = tensap.ModelSelection()
    m.data["complexity"] = [1, 2, 4, 8, 16]
    m.data["empirical_risk"] = [1.0, 0.5, 0.1, 0.09, 0.085]
    return m


# --- magic indices ------------------------------------------------------------------------------
