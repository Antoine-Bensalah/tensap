"""
Borehole, episode 2: the quantity that drives the flow
======================================================

In :ref:`the first episode <sphx_glr_use_cases_borehole.py>`, the engineer
studying the borehole built a surrogate model of the flow :math:`Q` from 200
runs of the model, and found with the Sobol indices that five of the eight
parameters matter. No single parameter explains the flow, though. Is there a
single *combination* of the parameters that does, a quantity
:math:`g(X)` such that :math:`Q \\approx f(g(X))`? Such a quantity would
reduce the study to one dimension, and would tell the engineer what the
flow really depends on.

The simulation code now also returns the gradient of the flow with respect
to the parameters, as codes with an adjoint solver do, for about the cost of
one more run. Gradients tell in which directions :math:`Q` varies at each
point, and this is what a feature :math:`g` must capture. We learn
:math:`g` by minimizing a loss derived from Poincaré inequalities
:cite:p:`Zahm2020,Bigoni2022`, first with linear features (the active
subspace :cite:p:`Constantine2014`), then with polynomial features, using
the convex surrogate of the loss of :cite:t:`NouyPasco2026`.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from time import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import tensap
from tensap.poincare_learning.benchmarks.poincare_benchmarks import _get_shift_scaling
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis

warnings.filterwarnings("ignore")
np.random.seed(0)

# %%
# The model and its gradient
# --------------------------
#
# The flow is the one of the first episode,
#
# .. math::
#
#    Q = \frac{2 \pi T_u (H_u - H_l)}
#             {\ln(r / r_w) \left( 1 + \dfrac{2 L T_u}{\ln(r / r_w)\, r_w^2 K_w}
#              + \dfrac{T_u}{T_l} \right)},
#
# with the same eight random parameters :math:`r_w, r, T_u, H_u, T_l, H_l,
# L, K_w`. The parameters have very different scales, so we work with
# standardized variables :math:`\xi`: each parameter is
# :math:`x_\nu = c_\nu + s_\nu \xi_\nu`, where :math:`\xi_\nu` is a standard
# normal variable for :math:`r_w` and :math:`\ln r`, and a uniform variable
# on :math:`[-1, 1]` for the others. The function below returns the flow and
# its gradient with respect to :math:`\xi`, written by hand from
# :math:`\ln Q` (here it plays the part of the adjoint solver).

_, X_phys = tensap.multivariate_functions_benchmark("borehole")
shift, scaling = _get_shift_scaling(X_phys)
X = X_phys.get_standard_random_vector()
d = X.size
NAMES = ["$r_w$", "$r$", "$T_u$", "$H_u$", "$T_l$", "$H_l$", "$L$", "$K_w$"]


def model(xi):
    """Flow Q and its gradient with respect to the standardized variables."""
    x = shift + scaling * xi
    r_w, z, T_u, H_u, T_l, H_l, L, K_w = x.T
    log_r = 7.71 + 1.0056 * z - np.log(r_w)  # ln(r / r_w)
    A = 2 * L * T_u / (log_r * r_w**2 * K_w)
    D = 1 + A + T_u / T_l
    Q = 2 * np.pi * T_u * (H_u - H_l) / (log_r * D)
    # Gradient of ln Q with respect to the physical parameters.
    dlog = np.empty_like(x)
    dlog[:, 0] = 1 / (r_w * log_r) - A * (1 / (r_w * log_r) - 2 / r_w) / D
    dlog[:, 1] = -1.0056 / log_r + 1.0056 * A / (log_r * D)
    dlog[:, 2] = 1 / T_u - (A / T_u + 1 / T_l) / D
    dlog[:, 3] = 1 / (H_u - H_l)
    dlog[:, 4] = T_u / (T_l**2 * D)
    dlog[:, 5] = -1 / (H_u - H_l)
    dlog[:, 6] = -A / (L * D)
    dlog[:, 7] = A / (K_w * D)
    return Q, Q[:, None] * dlog * scaling


# %%
# A check of the gradient by centred finite differences, and of the flow
# against the borehole function of tensap.

xi = X.random(5)
Q, dQ = model(xi)
h = 1e-6
dQ_fd = np.stack(
    [(model(xi + h * e)[0] - model(xi - h * e)[0]) / (2 * h) for e in np.eye(d)],
    axis=1,
)
borehole, _ = tensap.multivariate_functions_benchmark("borehole")
print(f"Gradient, relative difference: {np.abs(dQ - dQ_fd).max() / np.abs(dQ).max():.1e}")
print(f"Flow, difference: {np.abs(borehole(shift + scaling * xi) - Q).max():.1e}")

# %%
# The engineer runs the model at :math:`n = 200` random points, the budget of
# the first episode, now with the gradients. As in the first episode, a large
# test sample, affordable only because the model is cheap, measures the
# errors.

n = 200
x_train = X.random(n)
y_train, dy_train = model(x_train)
x_test = X.random(10000)
y_test, dy_test = model(x_test)

# %%
# Learning a feature from the gradients
# -------------------------------------
#
# Let :math:`g : \mathbb{R}^d \to \mathbb{R}^m` be a feature map, with
# :math:`m` small. The best approximation of :math:`Q` by a function of
# :math:`g(X)` is the conditional expectation :math:`\mathbb{E}(Q \mid g(X))`.
# Under assumptions on :math:`X` and :math:`g` (Poincaré inequalities on the
# level sets of :math:`g`), its error is bounded by
# :cite:p:`Zahm2020,Bigoni2022`
#
# .. math::
#
#    \mathbb{E}\left[ (Q - \mathbb{E}(Q \mid g(X)))^2 \right]
#    \le C\, \mathcal{J}(g),
#    \qquad
#    \mathcal{J}(g) = \mathbb{E}\left[ \left\| \Pi_{g}(X)^\perp
#    \nabla Q(X) \right\|^2 \right],
#
# where :math:`\Pi_g(x)^\perp` is the orthogonal projection onto the
# orthogonal complement of the span of the gradients of
# :math:`g_1, \dots, g_m` at :math:`x`. :math:`\mathcal{J}(g)` is the part of
# the gradient of :math:`Q` that :math:`g` misses: it vanishes when the level
# sets of :math:`g` are level sets of :math:`Q`. It depends only on the
# gradients, and is estimated by its empirical mean on the training points.
# The strategy is in two steps:
#
# 1. learn the feature :math:`g` by minimizing the empirical
#    :math:`\mathcal{J}(g)`;
# 2. learn the profile :math:`f`, a function of :math:`m` variables, by
#    least squares on the points :math:`(g(x^k), Q(x^k))`.
#
# The features are sought in the span of a basis
# :math:`\varphi = (\varphi_1, \dots, \varphi_K)` of polynomials, orthonormal
# for the distribution of :math:`\xi`, of total degree at most :math:`p`,
# without the constant: :math:`g(\xi) = G^T \varphi(\xi)` with
# :math:`G \in \mathbb{R}^{K \times m}`. For :math:`p = 1`, the features are
# linear.
#
# We report the relative loss
# :math:`\mathcal{J}(g) / \mathbb{E}(\|\nabla Q\|^2)`, estimated on 2000 test
# points, and the relative error
# :math:`\|Q - f \circ g\| / \|Q\|` on the test sample. The profile is a
# polynomial whose degree, from 1 to 6, is selected by 10-fold
# cross-validation.


def feature_space(p, x, dy):
    """Polynomial basis of degree p and the losses on the training and test points."""
    basis = _build_ortho_poly_basis(X, p=1, m=p)
    R = basis.gram_matrix_h1_0()
    loss = PoincareLossVectorSpace(dy[:, None, :], basis.eval_jacobian(x), basis, R)
    loss_test = PoincareLossVectorSpace(
        dy_test[:2000, None, :], basis.eval_jacobian(x_test[:2000]), basis, R
    )
    return basis, loss, loss_test


def fit_profile(z, y):
    """Polynomial profile f, of degree selected by cross-validation."""
    cv = GridSearchCV(
        Pipeline([("poly", PolynomialFeatures()), ("lin", LinearRegression())]),
        {"poly__degree": [1, 2, 3, 4, 5, 6]},
        scoring="neg_mean_squared_error",
        cv=10,
    )
    return cv.fit(z, y)


GRAD2 = np.mean(np.sum(dy_test[:2000] ** 2, axis=1))


def evaluate(basis, loss_test, G):
    """Relative loss on the test points, profile, and relative error of f o g."""
    # The sign of a feature is arbitrary: choose g increasing with Q.
    G *= np.sign(np.corrcoef((basis.eval(x_train) @ G)[:, 0], y_train)[0, 1])
    f = fit_profile(basis.eval(x_train) @ G, y_train)
    y_pred = f.predict(basis.eval(x_test) @ G)
    err = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)
    return loss_test.eval(G) / GRAD2, f, err


# %%
# Linear features: the active subspace
# ------------------------------------
#
# For linear features :math:`g(\xi) = G^T \xi`, the projection
# :math:`\Pi_g^\perp` does not depend on :math:`x`, and
# :math:`\mathcal{J}(g) = \mathbb{E}(\|\nabla Q\|^2) - \operatorname{tr}(P^T H P)`
# with :math:`H = \mathbb{E}(\nabla Q \nabla Q^T)` and :math:`P` an
# orthonormal basis of the span of :math:`G`. The minimizer is spanned by the
# :math:`m` leading eigenvectors of :math:`H`: it is the *active subspace*
# :cite:p:`Constantine2014,Zahm2020`.

basis_1, loss_1, loss_1_test = feature_space(1, x_train, dy_train)
results = {}
print(f"{'features':>22}{'rel. loss':>11}{'error':>10}")
for m in [1, 2, 3]:
    G = loss_1.minimize_active_subspace(m)
    G = G[0] if isinstance(G, tuple) else G
    J, f, err = evaluate(basis_1, loss_1_test, G)
    results[("linear", m)] = (J, err)
    print(f"{f'linear, m = {m}':>22}{J:11.1e}{err:10.1e}")
    if m == 1:
        G_lin, f_lin = G, f

# %%
# One linear feature leaves 1.4 % of the squared gradient unexplained, and
# gives an error of about 3 %. Three linear features only reach
# :math:`6 \times 10^{-3}`, less than the sparse polynomials of the first
# episode. The flow is not a function of a linear combination of the
# parameters.
#
# Nonlinear features
# ------------------
#
# For polynomial features of degree :math:`p > 1`, the gradients of
# :math:`g` vary with :math:`x`, and :math:`\mathcal{J}` is not a quadratic
# function of :math:`G` anymore. :cite:t:`Bigoni2022` minimize it with a
# quasi-Newton method on the Grassmann manifold
# (:meth:`~tensap.poincare_learning.PoincareLossVectorSpace.minimize_qn`),
# started from the active subspace. For one feature,
# :math:`\mathcal{J}(g) = \mathbb{E}(\|\nabla Q\|^2 \sin^2 \theta)`, where
# :math:`\theta(x)` is the angle between :math:`\nabla Q(x)` and
# :math:`\nabla g(x)`. :cite:t:`NouyPasco2026` replace it with the
# surrogate
#
# .. math::
#
#    \mathcal{S}(g) = \mathbb{E}\left( \|\nabla Q\|^2 \|\nabla g\|^2
#    \sin^2 \theta \right),
#
# a quadratic form in :math:`G`. Under the normalization
# :math:`\mathbb{E}(\|\nabla g\|^2) = 1`, its minimizer is the eigenvector of
# a generalized eigenvalue problem of size :math:`K`
# (:meth:`~tensap.poincare_learning.PoincareLossVectorSpace.minimize_surrogate`),
# and they show that a small surrogate controls the loss. We compare the
# two, with one feature, for degrees 2, 3 and 4.

print(f"{'features':>22}{'K':>5}{'rel. loss':>11}{'error':>10}{'time (s)':>10}")
for p in [2, 3, 4]:
    basis, loss, loss_test = feature_space(p, x_train, dy_train)
    for method in ["quasi-Newton", "surrogate"]:
        if method == "quasi-Newton" and p == 4:
            continue  # its cost grows quickly with K; the surrogate is enough
        t = time()
        if method == "quasi-Newton":
            G, _ = loss.minimize_qn(m=1, init_method="active_subspace", maxiter=100, tol=1e-10)
        else:
            G, _, _ = loss.minimize_surrogate(m=1)
        t = time() - t
        J, f, err = evaluate(basis, loss_test, G)
        results[(method, p)] = (J, err)
        print(f"{f'degree {p}, {method}':>22}{basis.cardinal():5d}{J:11.1e}{err:10.1e}{t:10.2f}")
basis_4, G_4, f_4 = basis, G, f

# %%
# With one polynomial feature of degree 4, :math:`f \circ g` reaches a
# relative error of about :math:`10^{-4}`, fifty times smaller than with three
# linear features. The surrogate is as good as the direct minimization of the
# loss, and much faster: one eigenvalue problem, without iterations. The
# degree 4 basis has :math:`K = 494` functions, more than the number of
# runs, but the gradients give :math:`n d = 1600` equations.
#
# The figure shows the flow against the feature, on the test points, for the
# best linear feature and for the polynomial feature of degree 4: the cloud
# collapses on the curve of the profile :math:`f`.

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
for ax, b, G, f, title in [
    (axes[0], basis_1, G_lin, f_lin, "linear feature"),
    (axes[1], basis_4, G_4, f_4, "polynomial feature, degree 4"),
]:
    z = (b.eval(x_test[:2000]) @ G).ravel()
    zz = np.linspace(z.min(), z.max(), 200)
    ax.plot(z, y_test[:2000], ".", ms=2, color="#9ab", label="test points")
    ax.plot(zz, f.predict(zz[:, None]), color="#e8862a", label="profile $f$")
    ax.set_xlabel("$g(X)$")
    ax.set_title(title)
axes[0].set_ylabel("flow $Q$ (m$^3$/year)")
axes[0].legend(fontsize="small")
fig.tight_layout()

# %%
# The error against the complexity of the features: the number of linear
# features, and the degree of one polynomial feature.

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
axes[0].semilogy([1, 2, 3], [results[("linear", m)][1] for m in [1, 2, 3]], "o-",
                 color="#127a8a")
axes[0].set_xticks([1, 2, 3])
axes[0].set_xlabel("number $m$ of linear features")
axes[0].set_ylabel("relative error of $f \\circ g$")
errors = [results[("linear", 1)][1]] + [results[("surrogate", p)][1] for p in [2, 3, 4]]
axes[1].semilogy([1, 2, 3, 4], errors, "o-", color="#e8862a")
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xlabel("degree $p$ of one feature")
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()

# %%
# Reading the feature
# -------------------
#
# Which parameters does the feature depend on? The mean squared derivatives
# :math:`\mathbb{E}((\partial_{\xi_\nu} g)^2)`, normalized to sum to 1, are
# compared with those of :math:`Q`.

jac_g = np.einsum("k,nkj->nj", G_4[:, 0], basis_4.eval_jacobian(x_test[:2000]))
share_g = np.mean(jac_g**2, axis=0)
share_Q = np.mean(dy_test[:2000] ** 2, axis=0)
share_g, share_Q = share_g / share_g.sum(), share_Q / share_Q.sum()

fig, ax = plt.subplots(figsize=(7, 3))
pos = np.arange(d)
ax.bar(pos - 0.2, share_Q, 0.4, color="#9ab", label="flow $Q$")
ax.bar(pos + 0.2, share_g, 0.4, color="#127a8a", label="feature $g$")
ax.set_xticks(pos)
ax.set_xticklabels(NAMES)
ax.set_ylabel("share of $\\mathbb{E}(\\|\\nabla \\cdot\\|^2)$")
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# The feature depends on :math:`r_w`, :math:`H_u`, :math:`H_l`, :math:`L`
# and :math:`K_w`, the parameters found by the Sobol indices in the first
# episode, in the same proportions as the flow. Can we name it? With the
# values of the parameters, :math:`T_u / T_l` is about :math:`10^3` and
# :math:`2 L T_u / (\ln(r / r_w)\, r_w^2 K_w)` about :math:`3 \times 10^5`:
# the second term dominates the denominator of :math:`Q`, and
#
# .. math::
#
#    Q \approx Q_D = \frac{\pi r_w^2 K_w (H_u - H_l)}{L}.
#
# This is Darcy's law in the borehole: the flow through a pipe of section
# :math:`\pi r_w^2`, of length :math:`L`, filled with a medium of
# conductivity :math:`K_w`, under the difference of heads
# :math:`H_u - H_l`. The resistance of the borehole limits the flow, and the
# aquifers hardly matter. The learned feature is an increasing function of
# :math:`Q_D`:

x_phys = shift + scaling * x_test[:2000]
r_w, _, _, H_u, _, H_l, L, K_w = x_phys.T
Q_D = np.pi * r_w**2 * K_w * (H_u - H_l) / L
z = (basis_4.eval(x_test[:2000]) @ G_4).ravel()
rank = np.corrcoef(np.argsort(np.argsort(z)), np.argsort(np.argsort(Q_D)))[0, 1]
print(f"Rank correlation between g(X) and Q_D: {rank:.5f}")

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(Q_D, z, ".", ms=2, color="#127a8a")
ax.set_xlabel("Darcy flow $Q_D$ (m$^3$/year)")
ax.set_ylabel("learned feature $g(X)$")
fig.tight_layout()

# %%
# The feature was learned from 200 runs and their gradients, without any
# knowledge of the formula. It is more than :math:`Q_D`, though: the best
# profile of :math:`Q_D` has an error of about :math:`1.5 \times 10^{-3}`,
# since :math:`Q_D` neglects the resistance of the aquifers, while the
# learned feature also captures this correction.

y = y_test[:2000]
f_D = fit_profile(Q_D[:, None], y)
err_D = np.linalg.norm(f_D.predict(Q_D[:, None]) - y) / np.linalg.norm(y)
print(f"Best profile of Q_D, fitted on the test points: {err_D:.1e}")
print(f"f o g, fitted on the 200 training points:       {results[('surrogate', 4)][1]:.1e}")

# %%
# Back to the first episode
# -------------------------
#
# The relative errors with 200 runs, for the surrogate models of the first
# episode, built from the values of :math:`Q` only, and for :math:`f \circ g`,
# built from the values and the gradients:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Method
#      - Data
#      - Error
#    * - Sparse polynomials
#      - :math:`Q`
#      - :math:`3.6 \times 10^{-3}`
#    * - Canonical format
#      - :math:`Q`
#      - :math:`6.5 \times 10^{-4}`
#    * - Tensor train
#      - :math:`Q`
#      - :math:`2.9 \times 10^{-5}`
#    * - Adaptive tree-based format
#      - :math:`Q`
#      - :math:`1.0 \times 10^{-6}`
#    * - One linear feature
#      - :math:`Q, \nabla Q`
#      - :math:`2.8 \times 10^{-2}`
#    * - One polynomial feature, degree 4
#      - :math:`Q, \nabla Q`
#      - :math:`1.3 \times 10^{-4}`
#
# The tree-based tensor formats remain the most accurate surrogate models of
# the flow. The feature gives something else: a model with one input,
# :math:`Q \approx f(g(X))`, that can be plotted, inverted
# (:math:`f` is increasing, so the flow exceeds a threshold exactly when
# :math:`g(X)` exceeds its preimage) and interpreted.
#
# Discussion
# ----------
#
# - When the gradients are available, the Poincaré loss turns them into
#   features, without any regression on the values of :math:`Q`. The values
#   are only used for the profile, a function of few variables.
# - Linear features (the active subspace) are cheap and robust but, here,
#   miss the nonlinear structure of the flow. One polynomial feature finds
#   it, and the convex surrogate finds it in one eigenvalue problem.
# - The feature tells the engineer what to measure: the flow is governed by
#   the Darcy flow of the borehole, and a better knowledge of :math:`r_w`,
#   which appears squared, matters most.
# - For several outputs (the flow at several times, for instance), the same
#   loss learns features shared by all the outputs; see the examples on
#   Poincaré learning in the :doc:`/examples/index`.
