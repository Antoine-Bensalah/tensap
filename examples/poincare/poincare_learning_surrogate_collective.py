r"""
Collective Poincaré learning with a surrogate
=============================================

Poincaré learning on the collective benchmark, with tensorized samples and a
surrogate of the Poincaré loss. Requires scikit-learn.

The function :math:`u(x, y)` depends on :math:`x \in \mathbb{R}^8` and on
a parameter :math:`y \in \mathbb{R}`. We look for features
:math:`g : \mathbb{R}^8 \to \mathbb{R}^3` of :math:`x` only, shared by the
whole collection of functions :math:`u(\cdot, y)`:
:math:`u(x, y) \approx f(g(x), y)`, with a small Poincaré loss
:math:`\mathbb{E}[\|\Pi^\perp_{\nabla g(X)} \nabla_x u(X, Y)\|^2]`. We
compare a surrogate on a tensorized sample with the plain minimization of
the loss. See :ref:`theory-learning`.

This example comes from the script
``test/non_regression_scripts/poincare_learning/tutorial_poincare_learning_surrogate_collective.py``.
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
    PoincareLossVectorSpaceTruncated,
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
# Functions to generate samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The points are written :math:`(x, y)`, with :math:`x` the first 8
# components of :math:`X` (indices ``ind1``) and :math:`y` the last one.
# ``generate_samples`` draws :math:`N` independent points
# :math:`(x_k, y_k)` and builds the Poincaré loss of the gradients
# :math:`\nabla_x u(x_k, y_k)`, one per point, as in
# :ref:`sphx_glr_examples_poincare_poincare_learning.py`.
#
# ``generate_samples_tensorized`` draws instead :math:`N_1` points
# :math:`x_i` and :math:`N_2` points :math:`y_j`, and evaluates :math:`u`
# on the grid of all the pairs :math:`(x_i, y_j)`. At each :math:`x_i`,
# the :math:`N_2` gradients :math:`\nabla_x u(x_i, y_j)` form the Jacobian
# of the vector-valued function :math:`x \mapsto (u(x, y_1), \dots,
# u(x, y_{N_2}))`, and the loss becomes
#
# .. math::
#
#    \mathcal{L}(G) = \frac{1}{N_1} \sum_{i=1}^{N_1} \sum_{j=1}^{N_2}
#    \big\| \Pi^\perp_{\nabla g(x_i)} \nabla_x u(x_i, y_j) \big\|^2.
#
# ``PoincareLossVectorSpaceTruncated`` can moreover replace these
# gradients by their projection on their dominant directions in
# :math:`\mathbb{R}^8`.


def generate_samples(N, ind1, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N, seed=LHS_RNG)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)[:, ind1]
    basis_set = basis.eval(x_set[:, ind1])
    jac_basis_set = basis.eval_jacobian(x_set[:, ind1])

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


def generate_samples_tensorized(
    N1, N2, ind1, X, fun, jac_fun, basis, R=None
):

    ind2 = np.delete(np.arange(X.ndim()), ind1)

    X1 = X.marginal(ind1)
    X2 = X.marginal(ind2)

    x1_set = X1.lhs_random(N1, seed=LHS_RNG)
    x2_set = X2.lhs_random(N2, seed=LHS_RNG)

    x_set = np.zeros((N1 * N2, X.ndim()))
    jac_basis_set = basis.eval_jacobian(x1_set)
    jac_fun_set_tensorized = np.zeros((N1, N2, X.ndim() - 1))

    for i in range(N1):
        # x2_set = X2.lhs_random(N2, seed=LHS_RNG)
        x_set_i = np.zeros((N2, X.ndim()))
        x_set_i[:, ind1] = x1_set[i]
        x_set_i[:, ind2] = x2_set
        x_set[i * N2 : (i + 1) * N2] = x_set_i  # noqa: E203
        jac_fun_set_tensorized[i, :, :] = jac_fun(x_set_i)[:, ind1]

    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x1_set)
    jac_basis_set = basis.eval_jacobian(x1_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set_tensorized = PoincareLossVectorSpaceTruncated(
        jac_fun_set_tensorized, jac_basis_set, basis, R
    )

    return (
        x_set,
        fun_set,
        jac_fun_set_tensorized,
        basis_set,
        jac_basis_set,
        loss_set_tensorized,
    )


# %%
# Functions to fit regressors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Once :math:`g` is learned, the profile
# :math:`f : \mathbb{R}^m \times \mathbb{R} \to \mathbb{R}` is fitted on
# the pairs
# :math:`((g(x_k), y_k), u(x_k, y_k))`, by kernel ridge regression or by
# polynomial regression, with hyperparameters chosen by cross-validation
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
        f"Best KRR with params: {kr.best_params_} and MSE score: "
        f"{-kr.best_score_:.3e}"
    )

    return kr


def fit_poly_regressor(z_set, u_set):
    model = Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )

    param_grid = {"poly__degree": np.arange(10)}

    cv = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=10,
    )

    cv.fit(z_set, u_set)
    print(
        f"Best Poly with params: {cv.best_params_} and  score: "
        f"{-cv.best_score_:.3e}"
    )

    return cv


# %%
# Functions to build lists of matrices
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Three families of symmetric matrices :math:`M_1, \dots, M_n` of size 8,
# which define the features of the benchmark below.


def build_mat_lst_1(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.eye(dim)
        for j in range(1, i + 1):
            mat += np.diag(np.ones(dim - j), k=j)
            mat += np.diag(np.ones(dim - j), k=-j)
        mat = mat / i
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst_2(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.diag(np.ones(dim - i), k=i)
        mat = mat + mat.T
        mat = mat / 2
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst_3(dim, n_mat):
    mat_lst = []
    for i in range(n_mat):
        mat = np.random.RandomState(i).normal(size=(dim, dim))
        mat = mat.T @ mat
        mat = mat / np.linalg.norm(mat, ord=2)
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst(dim, n_mat, which=1):
    if which == 1:
        out = build_mat_lst_1(dim, n_mat)
    elif which == 2:
        out = build_mat_lst_2(dim, n_mat)
    elif which == 3:
        out = build_mat_lst_3(dim, n_mat)
    else:
        raise NotImplementedError
    return out


# %%
# Definition of the benchmark
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :math:`X` is uniform on :math:`(-1, 1)^9`, and
#
# .. math::
#
#    u(x, y) = \sum_{l=1}^{3} \big(x^{\mathsf T} M_l\, x\big)^2
#    \sin(c_l\, y), \qquad c_l = \frac{\pi l}{6},
#
# with :math:`M_1 = I` and :math:`M_2`, :math:`M_3` the symmetric
# matrices with entries :math:`1/2` on the first, or second, sub- and
# super-diagonals (``which=2``). For every :math:`y`, :math:`u(\cdot, y)`
# is a function of the three quadratic features
# :math:`g_l(x) = x^{\mathsf T} M_l\, x`.


d = 8 + 1
n_mat = 3
ind1 = np.arange(d - 1)
ind2 = np.delete(np.arange(d), ind1)
mat_lst = build_mat_lst(d - 1, n_mat, which=2)

# if pytorch is installed
try:
    from tensap.poincare_learning.benchmarks import (
        poincare_benchmarks_torch as benchmarks_torch,
    )

    u, jac_u, X = benchmarks_torch.build_benchmark_torch(
        "quartic_sin_collective", d=d, mat_lst=mat_lst
    )


except ImportError:
    u, jac_u, X = build_benchmark(
        "quartic_sin_collective", d=d, mat_lst=mat_lst
    )

# %%
# Build a polynomial basis
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The features are sought in the span of the polynomials
# :math:`\varphi_\alpha(x) = \prod_{\nu} p^{(\nu)}_{\alpha_\nu}(x_\nu)`,
# with :math:`p^{(\nu)}_k` the orthonormal polynomials of degree
# :math:`k` for the law of :math:`X_\nu`, and the multi-indices
# :math:`1 \le |\alpha|_1 \le 2` on the 8 components of :math:`x`,
# numbered :math:`\varphi_1, \dots, \varphi_K`. The constant is left
# out, since its gradient is zero. :math:`R` is the Gram matrix of the
# gradients, :math:`R(i, j) = \mathbb{E}[\nabla \varphi_i(X) \cdot
# \nabla \varphi_j(X)]`, used to normalize the features,
# :math:`G^{\mathsf T} R G = I_m`. Since the features depend on
# :math:`x` only, the expectation is over the law of these 8 components.

p_norm = 1  # p-norm of the multi-indices
max_deg = 2  # bound on the p_norm
basis = _build_ortho_poly_basis(X.marginal(ind1), p=p_norm, m=max_deg)
K = basis.cardinal()
R = basis.gram_matrix_h1_0()

# %%
# Tensorized sample with surrogate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The quadratic features belong to the space of polynomials of degree at
# most 2, so that they can be recovered exactly. But the surrogate can
# only recover exactly as many features as there are gradients per point:
# with a single gradient, :math:`\nabla_x u` gives at each point a single
# direction of the span of :math:`\nabla g_1, \nabla g_2, \nabla g_3`.
# The tensorized sample gives four gradients per point, enough to see the
# three directions at once.

# %%
# Sampling
# ^^^^^^^^
# A tensorized training sample of :math:`25 \times 4 = 100` points, and a
# test sample of 1000 independent points.

N1_train = 25
N2_train = 4
x_train, u_train, _, _, _, loss_train_tensorized = (
    generate_samples_tensorized(
        N1_train, N2_train, ind1, X, u, jac_u, basis, R
    )
)

N_test = 1000
x_test, u_test, _, _, _, loss_test = generate_samples(
    N_test, ind1, X, u, jac_u, basis, R
)

# %%
# Minimize the surrogate
# ^^^^^^^^^^^^^^^^^^^^^^
# The gradients at each :math:`x_i` are first projected on their
# :math:`m = 3` dominant directions (``truncate``). The surrogate for a
# vector-valued function is
#
# .. math::
#
#    \frac{1}{N_1} \sum_{i=1}^{N_1} \sigma_{\max}(x_i)^2\,
#    \big\| \Pi^\perp_{\nabla_x u(x_i)} \nabla g(x_i)^{\mathsf T}
#    \big\|^2,
#
# with :math:`\sigma_{\max}(x_i)` the largest singular value of the
# Jacobian :math:`\nabla_x u(x_i) = (\nabla_x u(x_i, y_j))_{j}` and
# :math:`\Pi^\perp_{\nabla_x u(x_i)}` the projection onto the orthogonal
# complement of its rows. It is quadratic in :math:`G`, and its
# :math:`m` minimizers under :math:`G^{\mathsf T} R G = I_m` are the
# eigenvectors of the :math:`m` smallest eigenvalues of a generalized
# eigenvalue problem (``minimize_surrogate``).

m = len(mat_lst) - 0
loss_train_tensorized.truncate(m)
G_surr, _, _ = loss_train_tensorized.minimize_surrogate(m=m)
G_surr = np.linalg.svd(G_surr, full_matrices=False)[0]

# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The surrogate and the Poincaré loss on the tensorized training sample,
# and the Poincaré loss on the test sample, of the features obtained.

print(f"\nPoincare loss and Surrogate on {G_surr.shape[1]} features")
print(
    "Surrogate on tensorized train set:      "
    f"{loss_train_tensorized.eval_surrogate(G_surr):.3e}"
)
print(
    "Poincare loss on tensorized train set : "
    f"{loss_train_tensorized.eval(G_surr):.3e}"
)
print(
    "Poincare loss on test set:              "
    f"{loss_test.eval(G_surr):.3e}"
)

# %%
# Plot for eyeball regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k, y_k)` against each feature :math:`g_l(x_k)`. Since
# :math:`u` also depends on :math:`y`, the points do not lie on a curve.

z_surr_train = basis.eval(x_train[:, ind1]) @ G_surr
z_surr_test = basis.eval(x_test[:, ind1]) @ G_surr

fig, ax = plt.subplots(1, z_surr_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel("u(X)")

for i in range(z_surr_train.shape[1]):
    for j in range(N1_train):
        ax[i].scatter(
            z_surr_train[j::N2_train, i],
            u_train[j::N2_train],
            label="train",
            s=5.0,
        )
    ax[i].set_xlabel(f"g_{i}(X)")

fig.suptitle(
    f"""
    Surrogate only tensorized sample
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples
    """,
    y=0.0,
)
plt.show()


# %%
# Fit regressor with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The profile :math:`f` takes the features and the parameter :math:`y` as
# inputs: a polynomial regression on :math:`(g(x_k), y_k)`.

# add the last parameter as a feature
zy_surr_train = np.hstack([z_surr_train, x_train[:, ind2]])
zy_surr_test = np.hstack([z_surr_test, x_test[:, ind2]])

# shuffle = np.random.permutation(len(zy_surr_train))
shuffle = np.arange(len(zy_surr_train))

# regressor_surr = fit_krr_regressor(
#     zy_surr_train[shuffle], u_train[shuffle]
# )
regressor_surr = fit_poly_regressor(
    zy_surr_train[shuffle], u_train[shuffle]
)


def f_surr(z):
    return regressor_surr.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors of the approximation in the empirical :math:`L^2` norm,
# :math:`\big(\frac{1}{N} \sum_{k} (f(g(x_k), y_k) -
# u(x_k, y_k))^2\big)^{1/2}`,
# absolute and relative to the norm of :math:`u`, on the training and
# test samples.

y_train = f_surr(zy_surr_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_surr(zy_surr_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(
    "\nSurrogate only tensorized sample | Regression based on "
    f"{G_surr.shape[1]} features"
)
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %%
# Plot final regression
# ^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k, y_k)` against the prediction :math:`f(g(x_k), y_k)`: the
# closer the points are to the diagonal, the better the approximation.

plt.scatter(y_test, u_test, label="test", s=5.0)
plt.scatter(y_train, u_train, label="train", s=5.0)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate only tensorized sample
    Poly features m={zy_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()

# %%
# (Optional) Minimization of the Poincaré loss on the Grassmann manifold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The minimizer of the surrogate is a good initial guess for the
# Poincaré loss itself, on the same tensorized sample, which we now
# minimize from it as in
# :ref:`sphx_glr_examples_poincare_poincare_learning.py`, then fit and
# evaluate the regression in the same way.

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        "beta_rule": "PolakRibiere",
        "max_iterations": 50,
        "verbosity": 2,
    }
    G_opt, _, _ = loss_train_tensorized.minimize_pymanopt(
        G_surr, use_precond=True, optimizer_kwargs=optimizer_kwargs
    )

# quasi newton
except ImportError:
    G_opt, _ = loss_train_tensorized.minimize_qn(
        G0=G_surr, m=m, maxiter=50, tol=1e-10
    )


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The same quantities for the features after the minimization.

print(
    f"\nPoincare loss and Surrogate as init on {G_opt.shape[1]} "
    "features"
)
print(
    "Surrogate on tensorized train set:      "
    f"{loss_train_tensorized.eval_surrogate(G_opt):.3e}"
)
print(
    "Poincare loss on tensorized train set : "
    f"{loss_train_tensorized.eval(G_opt):.3e}"
)
print(
    "Poincare loss on test set:              "
    f"{loss_test.eval(G_opt):.3e}"
)

# %%
# Plot for eyeball regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same scatter plots, for the features obtained after the
# minimization of the Poincaré loss.

z_opt_train = basis.eval(x_train[:, ind1]) @ G_opt
z_opt_test = basis.eval(x_test[:, ind1]) @ G_opt

fig, ax = plt.subplots(1, z_opt_train.shape[1])
if z_opt_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel("u(X)")

for i in range(z_opt_train.shape[1]):
    for j in range(N1_train):
        ax[i].scatter(
            z_opt_train[j::N2_train, i],
            u_train[j::N2_train],
            label="train",
            s=5.0,
        )
    ax[i].set_xlabel(f"g_{i}(X)")

fig.suptitle(
    f"""
    Surrogate as init tensorized sample
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples
    """,
    y=0.0,
)
plt.show()


# %%
# Fit regressor with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# As before, a polynomial regression on :math:`(g(x_k), y_k)`.

# add the last parameter as a feature
zy_opt_train = np.hstack([z_opt_train, x_train[:, ind2]])
zy_opt_test = np.hstack([z_opt_test, x_test[:, ind2]])

# shuffle = np.random.permutation(len(zy_opt_train))
shuffle = np.arange(len(zy_opt_train))

# regressor_opt = fit_krr_regressor(
#     zy_opt_train[shuffle], u_train[shuffle]
# )
regressor_opt = fit_poly_regressor(
    zy_opt_train[shuffle], u_train[shuffle]
)


def f_opt(z):
    return regressor_opt.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors in the empirical :math:`L^2` norm, as before.

y_train = f_opt(zy_opt_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_opt(zy_opt_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(
    "\nSurrogate as init tensorized sample | Regression based on "
    f"{G_opt.shape[1]} features"
)
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %%
# Plot final regression
# ^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k, y_k)` against the prediction :math:`f(g(x_k), y_k)`.

plt.scatter(y_test, u_test, label="test", s=5.0)
plt.scatter(y_train, u_train, label="train", s=5.0)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate as init tensorized sample
    Poly features m={zy_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()

# %%
# Comparison with a plain sample and the classical approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For comparison, the same budget of 100 evaluations of :math:`u` and of
# its gradient, at independent points :math:`(x_k, y_k)`, and the
# minimization of the Poincaré loss with one gradient per point, from the
# active subspace, as in
# :ref:`sphx_glr_examples_poincare_poincare_learning.py`.

# %%
# Resample
# ^^^^^^^^
x_train, u_train, _, _, _, loss_train = generate_samples(
    N1_train * N2_train, ind1, X, u, jac_u, basis, R
)

# %%
# Minimize the Poincaré loss
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The three features are learned at once, from the active subspace.

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        "beta_rule": "PolakRibiere",
        "max_iterations": 100,
        "verbosity": 2,
    }
    G_pmo, _, _ = loss_train.minimize_pymanopt(
        m=m,
        init_method="active_subspace",
        optimizer_kwargs={"max_iterations": 100},
    )

# quasi newton
except ImportError:
    G_pmo, _ = loss_train.minimize_qn(
        G0=None,
        m=m,
        init_method="active_subspace",
        maxiter=50,
        tol=1e-10,
    )

# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The Poincaré loss of these features on the training and test samples.

print(f"\nPoincare loss on {G_pmo.shape[1]} features")
print(f"Poincare loss on train set:  {loss_train.eval(G_pmo):.3e}")
print(f"Poincare loss on test set:   {loss_test.eval(G_pmo):.3e}")


# %%
# Fit regressor with sklearn
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The same polynomial regression on :math:`(g(x_k), y_k)`.

# add the last parameter as a feature
zy_pmo_train = np.hstack(
    [basis.eval(x_train[:, ind1]) @ G_pmo, x_train[:, ind2]]
)
zy_pmo_test = np.hstack(
    [basis.eval(x_test[:, ind1]) @ G_pmo, x_test[:, ind2]]
)

regressor_pmo = fit_poly_regressor(zy_pmo_train, u_train)


def f_pmo(z):
    return regressor_pmo.predict(z)


# %%
# Evaluate performances
# ^^^^^^^^^^^^^^^^^^^^^
# The errors in the empirical :math:`L^2` norm, to compare with those
# obtained with the tensorized sample.


y_train = f_pmo(zy_pmo_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train) ** 2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_pmo(zy_pmo_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test) ** 2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(
    f"\npmoogate only | Regression based on {G_pmo.shape[1]} features"
)
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %%
# Plot final regression
# ^^^^^^^^^^^^^^^^^^^^^
# :math:`u(x_k, y_k)` against the prediction :math:`f(g(x_k), y_k)`.

plt.scatter(y_test, u_test, label="test", s=5.0)
plt.scatter(y_train, u_train, label="train", s=5.0)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    pymanopt optim
    Poly features m={zy_pmo_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()
# %%
