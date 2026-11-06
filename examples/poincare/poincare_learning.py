r"""
Poincaré learning
=================

Learn a feature map by minimisation of the Poincaré loss, then fit a
regressor on the features. Requires scikit-learn.

For a function :math:`u` of a random vector :math:`X = (X_1, \dots, X_d)`,
we look for a feature map :math:`g : \mathbb{R}^d \to \mathbb{R}^m`,
:math:`m \ll d`, and a profile :math:`f` such that
:math:`u(X) \approx f(g(X))`. The feature map minimizes the Poincaré-based
bound :math:`\mathbb{E}[\|\Pi^\perp_{\nabla g(X)} \nabla u(X)\|^2]` on the
error of the best profile, over polynomial features; here :math:`d = 8`
and :math:`m = 2`. Then :math:`f` is fitted by regression, and we look at
the error on a test sample. See :ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/poincare_learning/tutorial_poincare_learning.py``.
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
    # u, jac_u, X = build_benchmark("sin_squared_norm", d=8)

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
# Choosing hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^
# We learn :math:`m = 2` features. The loss is not convex in :math:`G`,
# and the result depends on the initial guess: the active subspace, that
# is the best linear features :math:`g(x) = V^{\mathsf T} x`, with
# :math:`V` the :math:`m` dominant eigenvectors of
# :math:`\mathbb{E}[\nabla u(X) \nabla u(X)^{\mathsf T}]`.

m = 2  # number of features to learn
init_method = "active_subspace"  # initialization method

# %%
# Minimize the Poincaré loss
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The loss only depends on the subspace spanned by the gradients of the
# features: replacing :math:`G` by :math:`G M`, with :math:`M` invertible,
# does not change it. It is thus minimized on the Grassmann manifold, by
# a conjugate gradient (``minimize_pymanopt``) when pymanopt is installed,
# or otherwise by the quasi-Newton method of Bigoni et al. (2022)
# (``minimize_qn``).

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        "beta_rule": "PolakRibiere",
        "max_iterations": 50,
        "verbosity": 2,
    }
    G, _, _ = loss_train.minimize_pymanopt(
        G0=None,
        m=m,
        init_method=init_method,
        optimizer_kwargs=optimizer_kwargs,
    )

# quasi newton
except ImportError:
    G, _ = loss_train.minimize_qn(
        G0=None, m=m, init_method=init_method, maxiter=50, tol=1e-10
    )


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The loss on the test sample estimates
# :math:`\mathbb{E}[\|\Pi^\perp_{\nabla g(X)} \nabla u(X)\|^2]`: when
# it stays close to the loss on the training sample, the features
# generalize.

print(f"\nPoincare loss on {G.shape[1]} features")
print(f"PLOSS on train set    : {loss_train.eval(G):.3e}")
print(f"PLOSS on test set     : {loss_test.eval(G):.3e}")

# %%
# Plot for eyeball regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Scatter plots of :math:`u(x_k)` against each feature
# :math:`g_j(x_k)`, on the two samples: when :math:`u` is close to a
# function of a single feature, the points lie close to a curve.

z_train = basis.eval(x_train) @ G
z_test = basis.eval(x_test) @ G

fig, ax = plt.subplots(1, z_train.shape[1])
if z_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel("u(X)")

for i in range(z_train.shape[1]):
    ax[i].scatter(z_train[:, i], u_train, label="train")
    ax[i].scatter(z_test[:, i], u_test, label="test")
    ax[i].set_xlabel(f"g_{i}(X)")

fig.suptitle(
    f"""
    Poly features m={z_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""",
    y=0.0,
)
plt.show()


# %%
# Fit regressor with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# A polynomial regression of :math:`u(x_k)` on the features (the kernel
# ridge regression is an alternative, commented out).

#  regressor = fit_krr_regressor(z_train, u_train)
regressor = fit_poly_regressor(z_train, u_train)


def f(z):
    return regressor.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors of :math:`f \circ g` in the empirical :math:`L^2` norm,
# :math:`\big(\frac{1}{N} \sum_{k} (f(g(x_k)) - u(x_k))^2\big)^{1/2}`,
# absolute and relative to the norm of :math:`u`, on the training and
# test samples.

y_train = f(z_train)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f(z_test)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\nRegression based on {G.shape[1]} features")
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
    Poly features m={z_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""")
plt.show()


# %%
