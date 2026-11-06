r"""
Poincaré learning with a surrogate
==================================

Learn a feature map by minimisation of a surrogate of the Poincaré loss.
Requires scikit-learn.

We look for one feature :math:`g : \mathbb{R}^8 \to \mathbb{R}` such that
:math:`u(X) \approx f(g(X))`. Instead of the Poincaré loss
:math:`\mathbb{E}[\|\Pi^\perp_{\nabla g(X)} \nabla u(X)\|^2]`, which is not
convex, we minimize the surrogate of Nouy and Pasco (2025),
:math:`\mathbb{E}[\|\nabla u(X)\|^2 \|\Pi^\perp_{\nabla u(X)} \nabla
g(X)\|^2]`, quadratic in :math:`g`, by solving an eigenvalue problem. We
then use its minimizer as an initial guess for the Poincaré loss. See
:ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/poincare_learning/tutorial_poincare_learning_surrogate.py``.
"""

# %%
# Imports
# ^^^^^^^
import numpy as np
import matplotlib.pyplot as plt
from tensap.poincare_learning.benchmarks.poincare_benchmarks import (
    build_benchmark,
)
from tensap.poincare_learning.utils._loss_vector_space import (
    _build_ortho_poly_basis,
)
from tensap.poincare_learning.poincare_loss_vector_space import (
    PoincareLossVectorSpace,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging

# Fixed seed, so that the example is reproducible.
np.random.seed(0)
LHS_RNG = np.random.default_rng(0)
logging.basicConfig(level=logging.INFO)

# %%
# Function to generate samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For a sample :math:`x_1, \dots, x_N` of :math:`X`, drawn by Latin
# hypercube sampling, we evaluate :math:`u`, its gradient, the basis
# :math:`\varphi = (\varphi_1, \dots, \varphi_K)` and its Jacobian. The
# feature map is :math:`g(x) = G^{\mathsf T} \varphi(x)`, with a matrix of
# coefficients :math:`G \in \mathbb{R}^{K \times m}`, so that
# :math:`\nabla g(x) = G^{\mathsf T} \nabla \varphi(x)`, and
# ``PoincareLossVectorSpace`` evaluates the empirical Poincaré loss
#
# .. math::
#
#    \mathcal{L}_N(G) = \frac{1}{N} \sum_{k=1}^{N}
#    \big\| \Pi^\perp_{\nabla g(x_k)} \nabla u(x_k) \big\|^2,
#
# where :math:`\Pi^\perp_{\nabla g(x)}` is the orthogonal projection onto
# the orthogonal complement of
# :math:`\mathrm{span}\{\nabla g_1(x), \dots, \nabla g_m(x)\}` in
# :math:`\mathbb{R}^d`: the part of :math:`\nabla u(x)` that the features
# do not see.


def generate_samples(N, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N, seed=LHS_RNG)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x_set)
    jac_basis_set = basis.eval_jacobian(x_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set = PoincareLossVectorSpace(
        jac_fun_set, jac_basis_set, basis, R
    )

    return (
        x_set,
        fun_set,
        jac_fun_set,
        basis_set,
        jac_basis_set,
        loss_set,
    )


# %%
# Functions to fit regressors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Once :math:`g` is learned, the profile
# :math:`f : \mathbb{R}^m \to \mathbb{R}` is fitted on the pairs
# :math:`(g(x_k), u(x_k))`, by kernel ridge regression or by polynomial
# regression, with hyperparameters chosen by cross-validation
# (scikit-learn). Since :math:`m` is small, this is a regression problem
# in low dimension.


def fit_krr_regressor(z_set, u_set):

    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        param_grid={
            "alpha": np.logspace(-11, -5, 10),
            "gamma": np.logspace(-6, 2, 10),
        },
        scoring="neg_mean_squared_error",
    )

    kr.fit(z_set, u_set)
    print(
        f"Best KRR with params: {kr.best_params_} and R2 score: "
        f"{kr.best_score_:.3f}"
    )

    return kr


def fit_poly_regressor(z_set, u_set):
    model = Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )

    param_grid = {"poly__degree": [1, 2, 3, 4, 5, 6]}

    cv = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=10,
    )

    cv.fit(z_set, u_set)
    print(
        f"Best Poly with params: {cv.best_params_} and MSE score: "
        f"{-cv.best_score_:.3f}"
    )

    return cv


# %%
# Definition of the benchmark
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With PyTorch, :math:`u` is the borehole function of 8 random
# parameters, whose gradient is computed by automatic differentiation.
# Otherwise, with :math:`X` uniform on :math:`(-1, 1)^8`,
#
# .. math::
#
#    u(x) = \exp\Big(\frac{1}{8} \sum_{\nu=1}^{8} \sin(x_\nu)\,
#    e^{\cos(x_\nu)}\Big),
#
# a function of a single feature, the sum inside the exponential, which
# is not a polynomial.

# if pytorch is installed
try:
    from tensap.poincare_learning.benchmarks import (
        poincare_benchmarks_torch as benchmarks_torch,
    )

    u, jac_u, X = benchmarks_torch.build_benchmark_torch("borehole")

except ImportError:
    u, jac_u, X = build_benchmark("exp_mean_sin_exp_cos", d=8)


# %%
# Build a polynomial basis
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The features are sought in the span of the polynomials
# :math:`\varphi_\alpha(x) = \prod_{\nu} p^{(\nu)}_{\alpha_\nu}(x_\nu)`,
# with :math:`p^{(\nu)}_k` the orthonormal polynomials of degree
# :math:`k` for the law of :math:`X_\nu`, and the multi-indices
# :math:`1 \le |\alpha|_1 \le 3`, numbered :math:`\varphi_1, \dots,
# \varphi_K`. The constant is left out, since its gradient is zero.
# :math:`R` is the Gram matrix of the gradients,
# :math:`R(i, j) = \mathbb{E}[\nabla \varphi_i(X) \cdot \nabla
# \varphi_j(X)]`, used to normalize the features,
# :math:`G^{\mathsf T} R G = I_m`.

p_norm = 1  # p-norm of the multi-indices
max_deg = 3  # bound on the p_norm
basis = _build_ortho_poly_basis(X, p=p_norm, m=max_deg)
K = basis.cardinal()
R = basis.gram_matrix_h1_0()

# %%
# Sampling
# ^^^^^^^^
# A training sample of size 300, on which the loss is minimized, and an
# independent test sample of size 500, to check that the learned features
# generalize.

N_train = 300
(
    x_train,
    u_train,
    jac_u_train,
    basis_train,
    jac_basis_train,
    loss_train,
) = generate_samples(N_train, X, u, jac_u, basis, R)


N_test = 500
x_test, u_test, jac_u_test, basis_test, jac_basis_test, loss_test = (
    generate_samples(N_test, X, u, jac_u, basis, R)
)


# %%
# Minimize the surrogate
# ^^^^^^^^^^^^^^^^^^^^^^
# For a single feature, the two losses are related pointwise by
#
# .. math::
#
#    \|\nabla g(x)\|^2\, \|\Pi^\perp_{\nabla g(x)} \nabla u(x)\|^2
#    = \|\nabla u(x)\|^2\, \|\Pi^\perp_{\nabla u(x)} \nabla g(x)\|^2,
#
# both sides being :math:`\|\nabla u(x)\|^2 \|\nabla g(x)\|^2
# \sin^2\theta(x)`, with :math:`\theta(x)` the angle between the two
# gradients. The surrogate drops the normalization by
# :math:`\|\nabla g(x)\|^2`, which makes the Poincaré loss non-convex.
# With :math:`g = G^{\mathsf T} \varphi`, the empirical surrogate is then
# a quadratic form :math:`G^{\mathsf T} H G`, with
#
# .. math::
#
#    H = \frac{1}{N} \sum_{k=1}^{N} \|\nabla u(x_k)\|^2\,
#    \nabla \varphi(x_k)\, \Pi^\perp_{\nabla u(x_k)}\,
#    \nabla \varphi(x_k)^{\mathsf T},
#
# and its minimizer under :math:`G^{\mathsf T} R G = 1` is the
# eigenvector of the smallest eigenvalue of :math:`H G = \lambda R G`
# (``minimize_surrogate``). The coefficients are then normalized, which
# only rescales the feature.
m = 1
G_surr, _, _ = loss_train.minimize_surrogate(m=m)
G_surr = np.linalg.svd(G_surr, full_matrices=False)[0]

# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The surrogate and the Poincaré loss of this feature, on the training and
# test samples.

print(f"\nPoincare loss and Surrogate on {G_surr.shape[1]} features")
print(
    "Surrogate on train set:      "
    f"{loss_train.eval_surrogate(G_surr):.3e}"
)
print(f"Poincare loss on train set:  {loss_train.eval(G_surr):.3e}")
print(
    "Surrogate on test set:       "
    f"{loss_test.eval_surrogate(G_surr):.3e}"
)
print(f"Poincare loss on test set:   {loss_test.eval(G_surr):.3e}")

# %%
# Plot for eyeball regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Scatter plots of :math:`u(x_k)` against each feature
# :math:`g_j(x_k)`, on the two samples: when :math:`u` is close to a
# function of a single feature, the points lie close to a curve.

z_surr_train = basis.eval(x_train) @ G_surr
z_surr_test = basis.eval(x_test) @ G_surr

fig, ax = plt.subplots(1, z_surr_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel("u(X)")

for i in range(z_surr_train.shape[1]):
    ax[i].scatter(z_surr_train[:, i], u_train, label="train")
    ax[i].scatter(z_surr_test[:, i], u_test, label="test")
    ax[i].set_xlabel(f"g_{i}(X)")

fig.suptitle(
    f"""
    Surrogate only
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples
    """,
    y=0.0,
)
plt.show()


# %%
# Fit regressor with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# A polynomial regression of :math:`u(x_k)` on the features (the kernel
# ridge regression is an alternative, commented out).

# regressor_surr = fit_krr_regressor(z_surr_train, u_train)
regressor_surr = fit_poly_regressor(z_surr_train, u_train)


def f_surr(z):
    return regressor_surr.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors of :math:`f \circ g` in the empirical :math:`L^2` norm,
# :math:`\big(\frac{1}{N} \sum_{k} (f(g(x_k)) - u(x_k))^2\big)^{1/2}`,
# absolute and relative to the norm of :math:`u`, on the training and
# test samples.

y_train = f_surr(z_surr_train)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_surr(z_surr_test)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(
    f"\nSurrogate only | Regression based on {G_surr.shape[1]} features"
)
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %%
# Plot final regression
# ^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k)` against the prediction :math:`f(g(x_k))`: the closer
# the points are to the diagonal, the better the approximation.

plt.scatter(y_train, u_train, label="train")
plt.scatter(y_test, u_test, label="test")
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate only
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""")
plt.show()


# %%
# (Optional) Minimization of the Poincaré loss on the Grassmann manifold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The minimizer of the surrogate is a good initial guess for the
# Poincaré loss itself, which we now minimize from it as in
# :ref:`sphx_glr_examples_poincare_poincare_learning.py`, then fit and
# evaluate the regression in the same way.

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        "beta_rule": "PolakRibiere",
        "max_iterations": 50,
        "verbosity": 2,
    }
    G_opt, _, _ = loss_train.minimize_pymanopt(
        G_surr, use_precond=True, optimizer_kwargs=optimizer_kwargs
    )

# quasi newton
except ImportError:
    G_opt, _ = loss_train.minimize_qn(
        G0=G_surr, m=m, maxiter=50, tol=1e-10
    )


# %%
# Plot for eyeball regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same scatter plots, for the features obtained after the
# minimization of the Poincaré loss.

z_opt_train = basis.eval(x_train) @ G_opt
z_opt_test = basis.eval(x_test) @ G_opt

fig, ax = plt.subplots(1, z_opt_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel("u(X)")

for i in range(z_opt_train.shape[1]):
    ax[i].scatter(z_opt_train[:, i], u_train, label="train")
    ax[i].scatter(z_opt_test[:, i], u_test, label="test")
    ax[i].set_xlabel(f"g_{i}(X)")

fig.suptitle(
    f"""
    Surrogate as init
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples
    """,
    y=0.0,
)
plt.show()


# %%
# Fit regression with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A polynomial regression of :math:`u(x_k)` on the features (the kernel
# ridge regression is an alternative, commented out).

# regressor_opt = fit_krr_regressor(z_opt_train, u_train)
regressor_opt = fit_poly_regressor(z_opt_train, u_train)


def f_opt(z):
    return regressor_opt.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors of :math:`f \circ g` in the empirical :math:`L^2` norm,
# :math:`\big(\frac{1}{N} \sum_{k} (f(g(x_k)) - u(x_k))^2\big)^{1/2}`,
# absolute and relative to the norm of :math:`u`, on the training and
# test samples.

y_train = f_opt(z_opt_train)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_opt(z_opt_test)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(
    f"\nSurrogate as init | Regression based on {G_opt.shape[1]} "
    "features"
)
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %%
# Plot final regression
# ^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k)` against the prediction :math:`f(g(x_k))`: the closer
# the points are to the diagonal, the better the approximation.

plt.scatter(y_train, u_train, label="train")
plt.scatter(y_test, u_test, label="test")
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate as init
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""")
plt.show()
