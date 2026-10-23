"""
Unit tests of Poincaré learning: identities between the loss, its gradient,
its Hessian and the matrices of the quasi-Newton and surrogate methods,
minimizers on a target with an exact feature, benchmarks and the
scikit-learn estimator.

The target is u = f(g(x)) with one feature g(x) = G0^T phi(x) on a polynomial
basis phi of total degree at most 2 in dimension 3: the Poincaré loss
vanishes at G0, which the minimizers must recover.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap
from tensap.poincare_learning.utils import _loss, _loss_vector_space as lvs
from tensap.poincare_learning.benchmarks.poincare_benchmarks import build_benchmark

D, N, M = 3, 40, 1
X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)


@pytest.fixture
def basis():
    return lvs._build_ortho_poly_basis(X, 1, 2)


@pytest.fixture
def case(basis, rng):
    x = rng.uniform(-1, 1, (N, D))
    jac_basis = basis.eval_jacobian(x)  # (N, K, d)
    K = jac_basis.shape[1]
    G0 = rng.standard_normal((K, M))
    jac_g = lvs._eval_jac_g(G0, jac_basis)  # (N, M, d)
    z = basis.eval(x) @ G0
    jac_u = (np.cos(z)[:, :, None] * jac_g)  # u = sin(g)
    return x, jac_u, jac_basis, G0


@pytest.fixture
def loss(case, basis):
    _, jac_u, jac_basis, _ = case
    return tensap.PoincareLossVectorSpace(jac_u, jac_basis, basis)


def _random_G(case, rng, m=M):
    return rng.standard_normal((case[2].shape[1], m))


# --- the loss --------------------------------------------------------------------------


def test_poincare_loss_is_the_mean_squared_distance_to_the_row_space(rng):
    jac_u = rng.standard_normal((5, 1, 4))
    jac_g = rng.standard_normal((5, 2, 4))
    expected = 0
    for ju, jg in zip(jac_u, jac_g):
        q = np.linalg.qr(jg.T)[0]  # orthonormal basis of the row space of jg
        expected += np.sum((ju.T - q @ q.T @ ju.T) ** 2) / 5
    npt.assert_allclose(_loss.poincare_loss(jac_u, jac_g), expected)
    npt.assert_allclose(_loss.poincare_loss(jac_u, jac_u), 0, atol=1e-28)


def test_augmented_loss_tends_to_the_loss(rng):
    jac_u = rng.standard_normal((5, 1, 4))
    jac_g = rng.standard_normal((5, 2, 4))
    exact = _loss.poincare_loss(jac_u, jac_g)
    npt.assert_allclose(_loss.poincare_loss_augmented(jac_u, jac_g, 1e-6), exact, rtol=1e-6)
    assert _loss.poincare_loss_augmented(jac_u, jac_g, 1.0) > exact


def test_surrogate_loss(rng):
    jac_u = rng.standard_normal((5, 4))  # scalar function: one row per sample
    jac_g = rng.standard_normal((5, 2, 4))
    assert _loss.poincare_loss_surrogate(jac_u, jac_g) > 0
    # zero when the feature contains the gradient
    npt.assert_allclose(_loss.poincare_loss_surrogate(jac_u, jac_u[:, None, :]), 0, atol=1e-28)


def test_loss_vanishes_at_the_feature_of_the_target(loss, case, rng):
    G0 = case[3]
    npt.assert_allclose(loss.eval(G0), 0, atol=1e-25)
    assert loss.eval(_random_G(case, rng)) > 1e-3
    # invariance by a change of basis of the features
    npt.assert_allclose(loss.eval(3 * G0), 0, atol=1e-25)


def test_jacobian_of_the_features(case, basis):
    x, _, jac_basis, G0 = case
    jac_g = lvs._eval_jac_g(G0, jac_basis)
    h = 1e-6
    for j in range(D):
        e = np.zeros(D)
        e[j] = h
        fd = (basis.eval(x + e) @ G0 - basis.eval(x - e) @ G0) / (2 * h)
        npt.assert_allclose(jac_g[:, :, j], fd, atol=1e-8)
    npt.assert_allclose(lvs._eval_jac_g(np.ravel(G0, order="F"), jac_basis), jac_g)


# --- derivatives and matrices ---------------------------------------------------------------


def test_gradient_by_finite_differences(loss, case, rng):
    G = _random_G(case, rng, 2)
    grad = loss.eval_gradient(G)
    V = rng.standard_normal(G.shape)
    h = 1e-6
    fd = (loss.eval(G + h * V) - loss.eval(G - h * V)) / (2 * h)
    npt.assert_allclose(np.sum(grad * V), fd, rtol=1e-5)
    # the gradient is 2 (S_G - H_G) G
    npt.assert_allclose(grad, 2 * (loss.eval_SG_X(G, G) - loss.eval_HG_X(G, G)), atol=1e-12)


def test_hessian_by_finite_differences(loss, case, rng):
    G = _random_G(case, rng, 2)
    X_ = rng.standard_normal(G.shape)
    h = 1e-6
    fd = (loss.eval_gradient(G + h * X_) - loss.eval_gradient(G - h * X_)) / (2 * h)
    npt.assert_allclose(loss.eval_HessG_X(G, X_), fd, rtol=1e-4, atol=1e-7)
    H = loss.eval_HessG_full(G)
    npt.assert_allclose(H @ np.ravel(X_, order="F"), np.ravel(loss.eval_HessG_X(G, X_), "F"),
                        atol=1e-10)
    npt.assert_allclose(loss.eval_HessG_diag(G), np.diag(H), atol=1e-10)
    npt.assert_allclose(H, H.T, atol=1e-10)


def test_sigma_and_h_matrices(loss, case, rng):
    G = _random_G(case, rng, 2)
    X_ = rng.standard_normal(G.shape)
    x = np.ravel(X_, order="F")
    S, H = loss.eval_SG_HG_full(G)
    npt.assert_allclose(loss.eval_SG_full(G), S)
    npt.assert_allclose(S @ x, np.ravel(loss.eval_SG_X(G, X_), "F"), atol=1e-12)
    npt.assert_allclose(H @ x, np.ravel(loss.eval_HG_X(G, X_), "F"), atol=1e-12)
    npt.assert_allclose(loss.eval_SG_diag(G), np.diag(S), atol=1e-12)
    # S_G is symmetric positive definite, and SGinv solves S_G Y = X
    assert np.all(np.linalg.eigvalsh((S + S.T) / 2) > 0)
    Y = loss.eval_SGinv_X(G, X_, cg_kwargs={"rtol": 1e-12})
    npt.assert_allclose(loss.eval_SG_X(G, Y), X_, atol=1e-8)


def test_surrogate_matrices(loss, case, rng):
    A, B, C = loss.eval_surrogate_matrices()
    npt.assert_allclose(C, 0)
    for mat in (A, B):
        npt.assert_allclose(mat, mat.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(mat) > -1e-12)
    # B - A is positive semi-definite, singular in the direction of the feature
    G0 = case[3]
    assert np.all(np.linalg.eigvalsh(B - A) > -1e-10)
    npt.assert_allclose(G0.T @ (B - A) @ G0, 0, atol=1e-10)
    npt.assert_allclose(loss.eval_surrogate(G0), 0, atol=1e-25)


# --- minimizers --------------------------------------------------------------------------------


def _subspace_distance(G, G0):
    """1 - the smallest cosine of the principal angles between the spans."""
    q, q0 = np.linalg.qr(G)[0], np.linalg.qr(G0)[0]
    return 1 - np.linalg.svd(q.T @ q0, compute_uv=False).min()


def test_surrogate_minimization_recovers_the_feature(loss, case):
    G, value, surrogate = loss.minimize_surrogate(m=1)
    npt.assert_allclose(value, 0, atol=1e-20)
    assert _subspace_distance(G, case[3]) < 1e-10
    npt.assert_allclose(surrogate, 0, atol=1e-10)
    G, losses_opt, losses, surrogates = loss.minimize_surrogate_greedy(2)
    assert losses.size == 1 and losses[0] < 1e-7  # stops when the loss vanishes


@pytest.mark.parametrize("precond", ["sigma", "hessian", "cg_sigma"])
def test_quasi_newton_recovers_the_feature(loss, case, precond):
    G, value = loss.minimize_qn(m=1, init_method="surrogate", precond_method=precond,
                                precond_kwargs={"rtol": 1e-12} if precond == "cg_sigma" else {})
    assert value < 1e-10
    assert _subspace_distance(G, case[3]) < 1e-6


@pytest.mark.parametrize("init", ["random", "random_linear", "surrogate_greedy",
                                  "active_subspace"])
def test_initializations(loss, case, init):
    G0 = tensap.poincare_learning.poincare_loss_vector_space_learning._initialization(
        loss.jac_u, loss.jac_basis, 1, init, n_try=2, seed=0)
    assert G0.shape[-2:] == (case[2].shape[1], 1)
    G, values = loss.minimize_qn(G0, maxiter=5)
    assert np.all(np.isfinite(values))


def test_quasi_newton_without_preconditioner(loss):
    # fixed point iterations G <- orth(G + H_G G): no guarantee, only finite
    G, value = loss.minimize_qn(m=1, precond_method=None, maxiter=5)
    assert np.isfinite(value)


def test_wrong_options(loss):
    with pytest.raises(ValueError):
        loss.minimize_qn(m=1, init_method="other")
    with pytest.raises(ValueError):
        loss.minimize_qn(m=1, precond_method="other")
    with pytest.raises(ValueError):
        loss.minimize_qn(m=1, precond_method="cg_other")


def test_active_subspace():
    # u(x) = sin(w^T x): the active subspace is spanned by w
    w = np.array([1.0, 2.0, -1.0])
    x = np.random.uniform(-1, 1, (30, 3))
    jac_u = (np.cos(x @ w)[:, None] * w)[:, None, :]
    G = tensap.poincare_learning.poincare_loss_vector_space_learning._minimize_active_subspace(
        jac_u, None, 1)
    assert _subspace_distance(G, w[:, None]) < 1e-12


def test_pymanopt_minimization(loss, case):
    pytest.importorskip("pymanopt")
    G, value, _ = loss.minimize_pymanopt(m=1, init_method="surrogate")
    assert value < 1e-10


# --- projected and truncated losses ---------------------------------------------------------------


def test_project_basis(loss, case, rng):
    K = case[2].shape[1]
    sub = np.linalg.qr(rng.standard_normal((K, 4)))[0]
    projected = loss.project_basis(sub)
    G = rng.standard_normal((4, 1))
    npt.assert_allclose(projected.eval(G), loss.eval(sub @ G))
    assert isinstance(projected.basis, tensap.SubFunctionalBasis)
    loss.R = np.eye(K)
    npt.assert_allclose(loss.project_basis(sub).R, sub.T @ sub, atol=1e-14)


def test_truncated_loss(case, rng):
    _, _, jac_basis, _ = case
    jac_u = rng.standard_normal((N, 3, D))  # vector-valued function
    t = tensap.PoincareLossVectorSpaceTruncated(jac_u, jac_basis, m=1)
    for ju in t.jac_u:
        assert np.linalg.matrix_rank(ju) == 1  # rank one truncation of each Jacobian
    t.truncate(3)
    npt.assert_allclose(t.jac_u, jac_u)
    full = tensap.PoincareLossVectorSpaceTruncated(jac_u, jac_basis)
    npt.assert_allclose(full.jac_u, jac_u)


# --- benchmarks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_name, kwargs",
    [("sin_squared_norm", {"d": 4}),
     ("sin_squared_norm",
      {"d": 3, "R": np.array([[1.0, 2.0, 0], [0, 1.0, 0], [1.0, 0, 3.0]])}),
     ("cos_squared_norm", {"d": 4, "c": 0.5}),
     ("sum_cos_sin_squared_norm", {"d": 3, "R1": np.diag([1.0, 2.0, 3.0]),
                                   "R2": np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])}),
     ("exp_mean_sin_exp_cos", {"d": 5}),
     ("quartic_sin_collective", {"d": 4}),
     ("quartic_sin_collective", {"d": 4, "mat_lst": [np.eye(3), np.diag([1.0, 2.0, 3.0])]})],
)
def test_benchmark_jacobians_by_finite_differences(case_name, kwargs, rng):
    fun, fun_jac, X_ = build_benchmark(case_name, **kwargs)
    x = X_.random(6)
    jac = np.reshape(fun_jac(x), (6, -1, x.shape[1]))
    h = 1e-6
    for j in range(x.shape[1]):
        e = np.zeros(x.shape[1])
        e[j] = h
        fd = (np.reshape(fun(x + e), (6, -1)) - np.reshape(fun(x - e), (6, -1))) / (2 * h)
        npt.assert_allclose(jac[:, :, j], fd, rtol=1e-6, atol=1e-7)


def test_unknown_benchmark():
    with pytest.raises(NotImplementedError):
        build_benchmark("other")


# --- scikit-learn estimator ---------------------------------------------------------------


@pytest.mark.parametrize("innerp", ["l2", "h1_0", "other"])
def test_polynomial_feature_estimator(case, innerp):
    from tensap.poincare_learning.utils._sklearn_wrapper import PolynomialFeatureEstimator

    x, jac_u, _, G0 = case
    est = PolynomialFeatureEstimator(X, 1, 2, innerp, fit_method="surrogate",
                                     fit_parameters={"m": 1})
    est.fit(x, jac_u)
    assert est.G.shape == (G0.shape[0], 1)
    npt.assert_allclose(est.G.T @ est.G, np.eye(1), atol=1e-12)
    if innerp == "l2":
        assert _subspace_distance(est.G, G0) < 1e-8
        npt.assert_allclose(est.score(x, jac_u), 0, atol=1e-18)
    est.fit_method = "surrogate_greedy"
    est.fit_parameters = {"m_max": 2}
    est.fit(x, jac_u)
    est.fit_method = "other"
    with pytest.raises(NotImplementedError):
        est.fit(x, jac_u)
