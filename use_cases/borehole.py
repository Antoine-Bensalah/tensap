"""
Surrogate model of a borehole
=============================

An engineer studies the flow of water through a borehole that connects two
aquifers. Eight physical parameters of the site are uncertain. How large is
the flow, how likely is it to exceed a threshold, and which parameters
matter? The model is assumed to be expensive, so that only a few hundred
runs are affordable: we build a surrogate model from these runs, compare
several approximation methods at the same budget, and answer the questions
on the surrogate.

The borehole function comes from a study of the safety of a nuclear waste
repository :cite:p:`Harper1983`, and is a standard benchmark of uncertainty
quantification since :cite:t:`Morris1993` (see also :cite:t:`VLSE`).
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from contextlib import redirect_stdout
import io
from time import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensap

warnings.filterwarnings("ignore")
np.random.seed(0)

# %%
# The model
# ---------
#
# A borehole of radius :math:`r_w` and length :math:`L` crosses an upper
# aquifer and a lower aquifer, of transmissivities :math:`T_u` and
# :math:`T_l` and of potentiometric heads :math:`H_u` and :math:`H_l`. In a
# steady state, the flow of water from the upper to the lower aquifer is
#
# .. math::
#
#    Q = \frac{2 \pi T_u (H_u - H_l)}
#             {\ln(r / r_w) \left( 1 + \dfrac{2 L T_u}{\ln(r / r_w)\, r_w^2 K_w}
#              + \dfrac{T_u}{T_l} \right)}
#    \quad \text{(m}^3\text{/year)},
#
# where :math:`r` is the radius of influence of the borehole and :math:`K_w`
# the hydraulic conductivity of the borehole. The eight parameters are
# modelled as independent random variables :cite:p:`Harper1983`:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Parameter
#      - Meaning
#      - Unit
#      - Distribution
#    * - :math:`r_w`
#      - radius of the borehole
#      - m
#      - normal, mean 0.10, standard deviation 0.0161812
#    * - :math:`r`
#      - radius of influence
#      - m
#      - lognormal: :math:`\ln r` normal, mean 7.71, standard deviation 1.0056
#    * - :math:`T_u`
#      - transmissivity of the upper aquifer
#      - m²/year
#      - uniform on [63070, 115600]
#    * - :math:`H_u`
#      - potentiometric head of the upper aquifer
#      - m
#      - uniform on [990, 1110]
#    * - :math:`T_l`
#      - transmissivity of the lower aquifer
#      - m²/year
#      - uniform on [63.1, 116]
#    * - :math:`H_l`
#      - potentiometric head of the lower aquifer
#      - m
#      - uniform on [700, 820]
#    * - :math:`L`
#      - length of the borehole
#      - m
#      - uniform on [1120, 1680]
#    * - :math:`K_w`
#      - hydraulic conductivity of the borehole
#      - m/year
#      - uniform on [9855, 12045]
#
# The model is available in tensap. The second variable is the standard
# normal variable :math:`\xi` such that :math:`\ln r = 7.71 + 1.0056\, \xi`.

model, X = tensap.multivariate_functions_benchmark("borehole")
d = X.size
NAMES = ["$r_w$", "$r$", "$T_u$", "$H_u$", "$T_l$", "$H_l$", "$L$", "$K_w$"]

# %%
# Here a run of the model takes a microsecond: we can afford a large
# reference sample, used to measure the errors of the surrogate models and to
# compute reference values of the quantities of interest. In a real study,
# this sample would not exist.

x_ref = X.random(100000)
y_ref = model(x_ref)
x_test, y_test = x_ref[:10000], y_ref[:10000]

# %%
# Approximation methods
# ---------------------
#
# A surrogate model is an approximation :math:`f` of the model :math:`Q`,
# built from :math:`n` runs :math:`(x^k, Q(x^k))_{1 \le k \le n}` at random
# points :math:`x^k`, by minimizing the empirical least-squares error
#
# .. math::
#
#    \frac{1}{n} \sum_{k=1}^n \left( Q(x^k) - f(x^k) \right)^2
#
# over a set of functions. All the methods below expand :math:`f` on the
# tensor product of the bases of polynomials :math:`(\phi^\nu_i)_{i \ge 0}`
# orthonormal with respect to the distribution of each parameter
# :math:`X_\nu`,
#
# .. math::
#
#    f(x) = \sum_{i_1, \ldots, i_8} a_{i_1 \ldots i_8}\,
#           \phi^1_{i_1}(x_1) \cdots \phi^8_{i_8}(x_8),
#
# and differ in the structure imposed on the tensor of coefficients
# :math:`a`, which has :math:`9^8 \approx 4 \cdot 10^7` entries for
# polynomials of degree 8:
#
# - **sparse polynomials**: few nonzero coefficients :math:`a_i`, among the
#   multi-indices of total degree at most 5, selected along the
#   :math:`\ell_1`-regularization path, the final model being chosen by
#   cross-validation :cite:p:`Blatman2011`;
# - **canonical format**: :math:`a` is a sum of :math:`r` rank-one tensors;
# - **tensor train**: :math:`a` is in tree-based format for a linear
#   dimension tree;
# - **adaptive tree-based format**: :math:`a` is in tree-based format, and
#   both the ranks and the dimension tree are adapted to the data
#   :cite:p:`Grelier2018`. See :ref:`learning in tensor formats
#   <theory-tensor-learning>` and
#   :cite:t:`Nouy2017book`.
#
# In the three tensor formats, the ranks are increased greedily, the degree
# of the polynomials is selected for each parameter, and the final model is
# chosen by cross-validation, so that the test sample is never used to build
# a surrogate.

BASES = tensap.FunctionalBases(
    [
        tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(9))
        for x in X.random_variables
    ]
)


def sparse_polynomials(x, y):
    degree = 5
    indices = tensap.MultiIndices.with_bounded_norm(d, 1, degree)
    bases = tensap.FunctionalBases(
        [
            tensap.PolynomialFunctionalBasis(
                v.orthonormal_polynomials(), range(degree + 1)
            )
            for v in X.random_variables
        ]
    )
    solver = tensap.LinearModelLearningSquareLoss()
    solver.basis = tensap.SparseTensorProductFunctionalBasis(bases, indices)
    solver.training_data = [x, y]
    solver.regularization = True
    solver.regularization_type = "l1"
    solver.regularization_options = {"alpha": 0.0}
    solver.model_selection = True
    f, _ = solver.solve()
    return f, int(np.count_nonzero(f.data))


def canonical(x, y):
    solver = tensap.CanonicalTensorLearning(d, tensap.SquareLossFunction())
    solver.bases = BASES
    solver.training_data = [x, y]
    solver.initialization_type = "mean"
    solver.rank_adaptation = True
    solver.rank_adaptation_options["max_iterations"] = 15
    solver.tolerance["on_error"] = 1e-6
    solver.alternating_minimization_parameters["max_iterations"] = 50
    solver.alternating_minimization_parameters["stagnation"] = 1e-8
    solver.alternating_minimization_parameters["one_by_one_factor"] = False
    solver.alternating_minimization_parameters["inner_loops"] = 2
    solver.alternating_minimization_parameters["display"] = False
    solver.linear_model_learning.regularization = False
    solver.linear_model_learning.basis_adaptation = True
    solver.model_selection = True
    solver.model_selection_options["type"] = "cv_error"
    solver.display = False
    f, _ = solver.solve()
    return f, f.tensor.storage()


def tree_based(x, y, adapt_tree):
    if adapt_tree:
        solver = tensap.TreeBasedTensorLearning.tensor_train_tucker(
            d, tensap.SquareLossFunction()
        )
    else:
        solver = tensap.TreeBasedTensorLearning.tensor_train(
            d, tensap.SquareLossFunction()
        )
    solver.bases = BASES
    solver.bases_eval = BASES.eval(x)
    solver.training_data = [None, y]
    solver.initialization_type = "canonical"
    solver.tolerance["on_stagnation"] = 1e-6
    solver.tolerance["on_error"] = 1e-6
    solver.rank_adaptation = True
    solver.rank_adaptation_options["max_iterations"] = 15
    solver.rank_adaptation_options["theta"] = 0.8
    # Stop the rank adaptation when the error increases ten times: with few
    # runs, large ranks overfit, and the alternating minimization diverges.
    solver.rank_adaptation_options["early_stopping"] = True
    solver.rank_adaptation_options["early_stopping_factor"] = 10
    solver.tree_adaptation = adapt_tree
    solver.tree_adaptation_options["max_iterations"] = 100
    solver.alternating_minimization_parameters["max_iterations"] = 5
    solver.alternating_minimization_parameters["stagnation"] = 1e-10
    solver.alternating_minimization_parameters["display"] = False
    solver.linear_model_learning.regularization = False
    solver.linear_model_learning.basis_adaptation = True
    solver.linear_model_learning.error_estimation = True
    solver.model_selection = True
    solver.model_selection_options["type"] = "cv_error"
    solver.display = False
    f, _ = solver.solve()
    return f, f.tensor.storage()


METHODS = {
    "Sparse polynomials": sparse_polynomials,
    "Canonical format": canonical,
    "Tensor train": lambda x, y: tree_based(x, y, adapt_tree=False),
    "Adaptive tree-based format": lambda x, y: tree_based(x, y, adapt_tree=True),
}


def relative_error(f):
    return np.linalg.norm(f(x_test) - y_test) / np.linalg.norm(y_test)


# %%
# Convergence with the number of runs
# -----------------------------------
#
# Each method is trained on the same samples of :math:`n` runs, for
# increasing budgets :math:`n`, and evaluated by its relative error on the
# test sample,
# :math:`\| Q - f \|_{\text{test}} / \| Q \|_{\text{test}}`.

BUDGETS = [100, 200, 400, 800]
results = {name: {"error": [], "size": [], "time": []} for name in METHODS}
surrogates = {}
for n in BUDGETS:
    x_train = X.random(n)
    y_train = model(x_train)
    for name, method in METHODS.items():
        t0 = time()
        with redirect_stdout(io.StringIO()):  # messages of the solvers
            f, size = method(x_train, y_train)
        results[name]["time"].append(time() - t0)
        results[name]["error"].append(relative_error(f))
        results[name]["size"].append(size)
        surrogates[name, n] = f

SHORT = ["sparse", "canonical", "TT", "adaptive"]
for key, title, form in [
    ("error", "relative test error", "{:10.1e}"),
    ("size", "number of parameters", "{:10d}"),
    ("time", "time of construction (s)", "{:10.0f}"),
]:
    print(title)
    print("     n" + "".join(f"{name:>10}" for name in SHORT))
    for k, n in enumerate(BUDGETS):
        values = [form.format(results[name][key][k]) for name in METHODS]
        print(f"{n:>6}" + "".join(values))
    print()

MARKERS = ["s", "^", "v", "o"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for marker, name in zip(MARKERS, METHODS):
    axes[0].loglog(BUDGETS, results[name]["error"], marker=marker, label=name)
    axes[1].loglog(
        results[name]["size"], results[name]["error"], marker=marker, ls=""
    )
axes[0].set_xlabel("number of runs $n$")
axes[0].set_ylabel("relative test error")
axes[0].legend(fontsize="small")
axes[1].set_xlabel("number of parameters of the surrogate")
axes[1].set_ylabel("relative test error")
for ax in axes:
    ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

# %%
# The sparse polynomials need several hundred runs to reach an error of
# :math:`10^{-3}`. The tensor formats are more accurate by one to three orders
# of magnitude at the same budget. The adaptive tree-based format reaches an
# error of about :math:`10^{-6}` from 200 runs, with the fewest parameters:
# the dimension tree is adapted to the interactions between the parameters,
# which keeps the ranks small. With more runs, the tensor train catches up.
# The errors also depend on the sample: a careful study would repeat each
# point on several samples, at a higher cost.
#
# From now on, the engineer keeps the adaptive tree-based surrogate built
# from 200 runs.

N = 200
f = surrogates["Adaptive tree-based format", N]

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(y_test[:2000], f(x_test[:2000]), ".", ms=2)
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, "k-", lw=0.8)
ax.set_xlabel("$Q(x)$ (m³/year)")
ax.set_ylabel("surrogate $f(x)$ (m³/year)")
ax.set_title(f"Test sample, surrogate built from {N} runs")
fig.tight_layout()

# %%
# Uncertainty of the flow
# -----------------------
#
# The surrogate costs nothing to evaluate: the statistics of the flow are
# computed on :math:`10^5` evaluations of the surrogate, and compared with
# the same statistics on :math:`10^5` runs of the model. Mean and variance
# can also be computed exactly on the surrogate, from its coefficients: the
# bases being orthonormal, :math:`\mathbb{E}(f(X)) = a_{0 \ldots 0}` and
# :math:`\mathbb{E}(f(X)^2) = \|a\|^2` (Parseval), where the norm of a
# tensor in tree-based format is computed from its cores, without forming
# :math:`a`.


def mean_and_variance(g):
    mean = g.mean()
    return mean, g.tensor.norm() ** 2 - mean**2


THRESHOLD = 150.0
f_ref = f(x_ref)
mean, variance = mean_and_variance(f)
rows = [
    ("mean", np.mean(y_ref), mean),
    ("standard deviation", np.std(y_ref), np.sqrt(variance)),
    ("1% quantile", np.quantile(y_ref, 0.01), np.quantile(f_ref, 0.01)),
    ("99% quantile", np.quantile(y_ref, 0.99), np.quantile(f_ref, 0.99)),
    (
        f"P(Q > {THRESHOLD:.0f})",
        np.mean(y_ref > THRESHOLD),
        np.mean(f_ref > THRESHOLD),
    ),
]
print(f"{'':>20}  {'model (1e5 runs)':>17}  {f'surrogate ({N} runs)':>20}")
for label, ref, sur in rows:
    print(f"{label:>20}  {ref:17.4g}  {sur:20.4g}")

fig, ax = plt.subplots(figsize=(6, 3.5))
bins = np.linspace(0, 250, 101)
ax.hist(y_ref, bins, density=True, histtype="step", lw=2, label="model, $10^5$ runs")
ax.hist(
    f_ref, bins, density=True, histtype="step", ls="--", lw=1.5,
    label=f"surrogate built from {N} runs",
)
ax.axvline(THRESHOLD, color="0.4", lw=0.8)
ax.text(THRESHOLD + 3, ax.get_ylim()[1] * 0.85, "threshold", color="0.4")
ax.set_xlabel("flow $Q$ (m³/year)")
ax.set_ylabel("probability density")
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# The distribution of the flow, its quantiles and the probability of
# exceeding the threshold, a small probability, are obtained from 200 runs of
# the model, where a direct Monte Carlo estimation would need thousands.
#
# Which parameters matter?
# ------------------------
#
# The Sobol indices :cite:p:`Sobol2001` split the variance of the flow
# between the parameters. The first-order index of :math:`X_\nu` is the part
# of the variance explained by :math:`X_\nu` alone, and the total index the
# part explained by :math:`X_\nu` and its interactions with the other
# parameters:
#
# .. math::
#
#    S_\nu = \frac{\mathbb{V}(\mathbb{E}(Q \mid X_\nu))}{\mathbb{V}(Q)},
#    \qquad
#    S^T_\nu = 1 - \frac{\mathbb{V}(\mathbb{E}(Q \mid X_{-\nu}))}{\mathbb{V}(Q)},
#
# where :math:`X_{-\nu}` denotes all the parameters but :math:`X_\nu`. For
# the surrogate, a function in tensor format on orthonormal bases, a
# conditional expectation is again a function in tensor format, obtained by
# contracting the cores with the means of the basis functions
# (:meth:`~tensap.FunctionalTensor.conditional_expectation`): its variance,
# and the indices, are computed exactly, as above. For
# the model, they are estimated by Monte Carlo with the estimators of
# :cite:t:`Saltelli2010`, at the cost of :math:`(d + 2) \times 10^5` runs.

S, ST = np.zeros(d), np.zeros(d)
for nu in range(d):
    others = [mu for mu in range(d) if mu != nu]
    S[nu] = mean_and_variance(f.conditional_expectation([nu]))[1] / variance
    ST[nu] = 1 - mean_and_variance(f.conditional_expectation(others))[1] / variance

M = 100000
A, B = X.random(M), X.random(M)
yA, yB = model(A), model(B)
var_ref = np.var(np.concatenate([yA, yB]))
S_ref, ST_ref = np.zeros(d), np.zeros(d)
for nu in range(d):
    AB = A.copy()
    AB[:, nu] = B[:, nu]
    yAB = model(AB)
    S_ref[nu] = np.mean(yB * (yAB - yA)) / var_ref
    ST_ref[nu] = 0.5 * np.mean((yA - yAB) ** 2) / var_ref

# Monte Carlo estimates of indices close to 0 can be slightly negative.
S_ref, ST_ref = np.maximum(S_ref, 0), np.maximum(ST_ref, 0)
print(f"{'':>6}{'S, model':>11}{'S, surr.':>11}{'ST, model':>11}{'ST, surr.':>11}")
for nu in range(d):
    print(
        f"{NAMES[nu].strip('$'):>6}{S_ref[nu]:11.3f}{S[nu]:11.3f}"
        f"{ST_ref[nu]:11.3f}{ST[nu]:11.3f}"
    )

fig, ax = plt.subplots(figsize=(7, 3.5))
pos = np.arange(d)
w = 0.2
ax.bar(pos - 1.5 * w, S_ref, w, color="#9ab", label="first order, model (MC)")
ax.bar(pos - 0.5 * w, S, w, color="#127a8a", label="first order, surrogate")
ax.bar(pos + 0.5 * w, ST_ref, w, color="#f3c89c", label="total, model (MC)")
ax.bar(pos + 1.5 * w, ST, w, color="#e8862a", label="total, surrogate")
ax.set_xticks(pos)
ax.set_xticklabels(NAMES)
ax.set_ylabel("Sobol index")
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# The radius of the borehole :math:`r_w` explains most of the variance; then
# come the heads :math:`H_u` and :math:`H_l`, the length :math:`L` and the
# conductivity :math:`K_w`. The transmissivities :math:`T_u`, :math:`T_l` and
# the radius of influence :math:`r` hardly matter: the engineer can fix them
# at nominal values in further studies, and should measure :math:`r_w`
# first. The total indices are close to the first-order ones: the
# interactions between parameters are weak.
#
# Reading the surrogate
# ---------------------
#
# The dimension tree found by the algorithm, with the parameters at the
# leaves and the ranks at the internal nodes. A node of rank 1 separates the
# variables below it from the others: the surrogate is then a product of a
# function of these variables and of a function of the others. Here the
# algorithm finds, from the data only, that the flow is close to such a
# product, a factor depending on the heads :math:`H_u`, :math:`H_l` and on
# :math:`T_u` (the formula contains :math:`T_u (H_u - H_l)`), times a factor
# depending on the geometry of the borehole.

tree = f.tensor.tree
labels = [
    NAMES[tree.dims[node - 1][0]] if tree.is_leaf[node - 1]
    else str(f.tensor.ranks[node - 1])
    for node in range(1, tree.nb_nodes + 1)
]
tree.plot_with_labels_at_nodes(labels, title="Dimension tree and ranks")

# %%
# Discussion
# ----------
#
# - With a budget of a few hundred runs, a surrogate in tensor format gives
#   the distribution of the flow, small probabilities and the hierarchy of
#   the parameters, with an accuracy that a direct Monte Carlo study would
#   reach with thousands of runs.
# - The adaptive tree-based format is the most accurate and the most compact
#   at small budgets; its cost of construction, a few tens of seconds, is
#   negligible before the cost of a real simulation of the borehole.
# - Sparse polynomials are a good first choice when the budget is very small
#   or the function very smooth. Tensor formats take over when the degree
#   needed in each variable grows, since their number of parameters grows
#   linearly with the dimension, and not exponentially.
# - The test sample, and the Monte Carlo references, are only possible here
#   because the model is cheap. In practice, the cross-validation error,
#   computed by the methods from the training runs only, is the estimate of
#   the error to rely on.
