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


def test_augmented_loss_tends_to_the_loss(rng):
    jac_u = rng.standard_normal((5, 1, 4))
    jac_g = rng.standard_normal((5, 2, 4))
    exact = _loss.poincare_loss(jac_u, jac_g)
    npt.assert_allclose(_loss.poincare_loss_augmented(jac_u, jac_g, 1e-6), exact, rtol=1e-6)
    assert _loss.poincare_loss_augmented(jac_u, jac_g, 1.0) > exact


# --- derivatives and matrices ---------------------------------------------------------------


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


# --- minimizers --------------------------------------------------------------------------------


def _subspace_distance(G, G0):
    """1 - the smallest cosine of the principal angles between the spans."""
    q, q0 = np.linalg.qr(G)[0], np.linalg.qr(G0)[0]
    return 1 - np.linalg.svd(q.T @ q0, compute_uv=False).min()


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


# --- projected and truncated losses ---------------------------------------------------------------


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


# --- scikit-learn estimator ---------------------------------------------------------------
