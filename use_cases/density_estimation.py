"""
Density estimation: five methods on a hidden structure
======================================================

We observe :math:`n` independent samples of a random vector of dimension 6,
and want its probability density. The density is not Gaussian: it has a
curved ridge, two modes, and correlations. It also has a hidden structure,
which the data do not show at first sight. We compare five estimators, from
the classical ones to tensors in tree-based format, on their accuracy, their
cost, and what they tell about the distribution.

The example is synthetic, so that the exact density is known and the errors
can be measured. It extends the example
:doc:`/examples/learning/tree_based_tensor_density_learning` of the gallery,
which estimates a Gaussian density.
"""

# Copyright (c) 2020-2026, Anthony Nouy, Erwan Grelier and contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from contextlib import redirect_stdout
import io
from time import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
import tensap

warnings.filterwarnings("ignore")
np.random.seed(0)

# %%
# The distribution
# ----------------
#
# The random vector :math:`X = (X_1, \dots, X_6)` is made of three
# independent pairs of variables, built from independent standard normal
# variables :math:`Z_1, \dots, Z_6` and a random sign :math:`S = \pm 1`:
#
# - a *banana*: :math:`(Z_1,\ \tfrac12 (Z_1^2 - 1) + \tfrac12 Z_2)`;
# - two *modes*: :math:`(\tfrac32 S + \tfrac12 Z_3,\ \tfrac32 S + \tfrac12 Z_4)`;
# - a *correlated* pair: :math:`(Z_5,\ 0.8 Z_5 + 0.6 Z_6)`, of correlation
#   0.8.
#
# The six variables are then shuffled: the pairs are
# :math:`(X_5, X_3)`, :math:`(X_2, X_4)` and :math:`(X_6, X_1)`. The
# estimators do not know it.

PERM = [5, 2, 1, 3, 0, 4]  # X_{j+1} is the variable PERM[j] of the pairs
d = 6
NAMES = [f"$X_{j + 1}$" for j in range(d)]


def normal_pdf(t, s=1.0):
    return np.exp(-(t**2) / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)


def sample(n):
    z = np.random.randn(n, 6)
    sign = np.where(np.random.rand(n) < 0.5, -1, 1)
    y = np.column_stack([
        z[:, 0], 0.5 * (z[:, 0] ** 2 - 1) + 0.5 * z[:, 1],
        1.5 * sign + 0.5 * z[:, 2], 1.5 * sign + 0.5 * z[:, 3],
        z[:, 4], 0.8 * z[:, 4] + 0.6 * z[:, 5],
    ])
    return y[:, PERM]


def banana(a, b):
    return normal_pdf(a) * normal_pdf(b - 0.5 * (a**2 - 1), 0.5)


def modes(a, b):
    return 0.5 * (normal_pdf(a + 1.5, 0.5) * normal_pdf(b + 1.5, 0.5)
                  + normal_pdf(a - 1.5, 0.5) * normal_pdf(b - 1.5, 0.5))


def correlated(a, b):
    return normal_pdf(a) * normal_pdf(b - 0.8 * a, 0.6)


def pdf(x):
    y = np.empty_like(x)
    y[:, PERM] = x
    return banana(y[:, 0], y[:, 1]) * modes(y[:, 2], y[:, 3]) * correlated(y[:, 4], y[:, 5])


# %%
# The tensor methods approximate the density on a bounded box,
# :math:`[-4, 4]^6`, which contains all but 0.3 % of the samples (the others
# are discarded). The errors are relative errors in :math:`L^2([-4, 4]^6)`,
# estimated on :math:`2 \times 10^4` uniform points of the box.

BOX = 4.0
VOLUME = (2 * BOX) ** d
REF = tensap.RandomVector([tensap.UniformRandomVariable(-BOX, BOX) for _ in range(d)])
x_err = REF.random(20000)
p_err = pdf(x_err)


def error(p):
    return np.linalg.norm(p - p_err) / np.linalg.norm(p_err)


def data(n):
    x = sample(n)
    return x[np.all(np.abs(x) < BOX, axis=1)]


# %%
# The five estimators
# -------------------
#
# **Gaussian.** The normal density with the empirical mean and covariance.
#
# **Kernel density estimator (KDE).**
# :math:`\hat p(x) = \frac1n \sum_k K_h(x - x^k)`, a Gaussian kernel with the
# bandwidth of Scott's rule :cite:p:`Scott1992`
# (:class:`scipy.stats.gaussian_kde`).
#
# The three other estimators are tensors on a basis of polynomials of degree
# at most 20 in each variable, orthonormal on :math:`[-4, 4]`. Let
# :math:`p = q / 8^6`, where :math:`q` is the density of :math:`X` with
# respect to the uniform measure :math:`\mu` on the box. The function
# :math:`q` is approximated by minimizing the empirical :math:`L^2` risk
#
# .. math::
#
#    \mathcal{R}(f) = \|f\|_{L^2_\mu}^2 - \frac{2}{n} \sum_{k=1}^n f(x^k),
#
# an unbiased estimate, up to a constant, of :math:`\|q - f\|_{L^2_\mu}^2`
# (:class:`~tensap.DensityL2LossFunction`).
#
# **Independent marginals.** The product of the six one-dimensional
# estimates, a tensor of rank one: each one is the empirical projection
# :math:`\hat q_\nu = \sum_i \big( \frac1n \sum_k \varphi_i(x^k_\nu) \big)
# \varphi_i`.
#
# **Tensor train.** A tensor in tree-based format with the linear tree
# :math:`\{1\}, \{1,2\}, \{1,2,3\}, \dots` (with Tucker leaves), and ranks
# adapted to the data :cite:p:`Grelier2019`.
#
# **Adaptive tree.** The same, but the dimension tree is also adapted to the
# data :cite:p:`Grelier2018,Grelier2019`.
#
# For the tensor formats, the ranks are selected with the risk on a test
# sample of size :math:`n / 5`.

bases = tensap.FunctionalBases([
    tensap.PolynomialFunctionalBasis(v.orthonormal_polynomials(), range(21))
    for v in REF.random_variables
])


def gaussian(x):
    g = multivariate_normal(x.mean(axis=0), np.cov(x.T))
    return g.pdf, g


def kde(x):
    k = gaussian_kde(x.T)
    return (lambda y: k(y.T)), k


def independent(x):
    coefs = [np.mean(b.eval(x[:, nu]), axis=0) for nu, b in enumerate(bases.bases)]

    def p(y):
        values = [b.eval(y[:, nu]) @ c for nu, (b, c) in enumerate(zip(bases.bases, coefs))]
        return np.prod(values, axis=0) / VOLUME
    return p, coefs


def tree_based(x, adapt_tree):
    x_test = data(len(x) // 5)
    s = tensap.TreeBasedTensorLearning.tensor_train_tucker(d, tensap.DensityL2LossFunction())
    s.bases = bases
    s.bases_eval = bases.eval(x)
    s.training_data = x
    s.test_error = True
    s.test_data = x_test
    s.initialization_type = "random"
    s.tolerance["on_stagnation"] = 1e-6
    s.tolerance["on_error"] = -np.inf  # the risk is negative
    s.linear_model_learning.regularization = False
    s.linear_model_learning.basis_adaptation = True
    s.linear_model_learning.error_estimation = True
    s.rank_adaptation = True
    s.rank_adaptation_options["max_iterations"] = 10
    s.rank_adaptation_options["theta"] = 0.8
    s.rank_adaptation_options["early_stopping"] = True
    s.rank_adaptation_options["early_stopping_factor"] = 0.1
    s.tree_adaptation = adapt_tree
    if adapt_tree:
        s.tree_adaptation_options["tolerance"] = 1e-6
    s.alternating_minimization_parameters["stagnation"] = 1e-6
    s.alternating_minimization_parameters["max_iterations"] = 50
    s.alternating_minimization_parameters["display"] = False
    s.display = False
    s.model_selection = True
    s.model_selection_options["type"] = "test_error"
    with redirect_stdout(io.StringIO()):
        f, _ = s.solve()
    return (lambda y: f(y) / VOLUME), f


METHODS = {
    "Gaussian": gaussian,
    "KDE": kde,
    "independent marginals": independent,
    "tensor train": lambda x: tree_based(x, False),
    "adaptive tree": lambda x: tree_based(x, True),
}

# %%
# Convergence
# -----------
#
# Each estimator is built from :math:`n = 1000`, 5000 and 20000 samples.

SIZES = [1000, 5000, 20000]
errors = {name: [] for name in METHODS}
times = {name: [] for name in METHODS}
models = {}
for n in SIZES:
    x = data(n)
    for name, method in METHODS.items():
        t = time()
        p, model = method(x)
        times[name].append(time() - t)
        errors[name].append(error(p(x_err)))
        models[name] = (p, model)

print(f"{'':>22}" + "".join(f"{f'n = {n}':>12}" for n in SIZES) + f"{'time (s)':>12}")
for name in METHODS:
    print(f"{name:>22}" + "".join(f"{e:12.3f}" for e in errors[name])
          + f"{times[name][-1]:12.1f}")

fig, ax = plt.subplots(figsize=(6, 4))
for name, color in zip(METHODS, ["#9ab", "#8a6", "#bbb", "#f3a35c", "#127a8a"]):
    ax.loglog(SIZES, errors[name], "o-", color=color, label=name)
ax.set_xlabel("number of samples $n$")
ax.set_ylabel("relative $L^2$ error")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# The Gaussian and the independent marginals cannot represent the density,
# whatever the number of samples: the first misses the shapes, the second
# the dependence. The KDE converges, but slowly, as expected in dimension 6.
# The tensor formats are much more accurate, and the adaptive tree is the
# best at every sample size, and faster than the tensor train. The times are
# the times of construction for the largest sample; the KDE is built at once,
# but each evaluation sums :math:`n` kernels.
#
# The densities by tensors are not constrained to be nonnegative: an
# :math:`L^2` approximation oscillates slightly around zero where the density
# is close to zero. Here the negative values stay small, about 2 % of the
# maximum of the density, and the integral of the estimate, computed
# exactly, is close to 1.

p_tree, f_tree = models["adaptive tree"]
values = p_tree(x_err)
print(f"Adaptive tree: integral {f_tree.mean():.4f} (exact),"
      f" min p = {values.min():.1e}, max p = {values.max():.1e}")

# %%
# Marginals, exactly
# ------------------
#
# A marginal density of a tensor is again a tensor, obtained by integrating
# the other variables, that is by contracting the cores with the means of
# the basis functions
# (:meth:`~tensap.FunctionalTensor.conditional_expectation`): it is computed
# exactly, without sampling. For the Gaussian, the marginals are known; for
# the KDE, the marginal of :math:`(X_\nu, X_\mu)` is the KDE of the samples
# of :math:`(X_\nu, X_\mu)` with the same bandwidth. The figure shows the
# marginal densities of the three pairs.

# The pairs (X_5, X_3), (X_2, X_4), (X_6, X_1), and their exact densities.
PAIRS = [(4, 2, banana), (1, 3, modes), (5, 0, correlated)]
grid = np.linspace(-BOX, BOX, 121)
G1, G2 = np.meshgrid(grid, grid)
points = np.column_stack([G1.ravel(), G2.ravel()])

_, gauss = models["Gaussian"]
_, kde_model = models["KDE"]


fig, axes = plt.subplots(3, 4, figsize=(9, 7), sharex=True, sharey=True)
for row, (nu, mu, exact) in enumerate(PAIRS):
    pair = [nu, mu]
    gauss_pair = multivariate_normal(gauss.mean[pair], gauss.cov[np.ix_(pair, pair)])
    kde_pair = gaussian_kde(kde_model.dataset[pair], bw_method=kde_model.factor)
    # The marginal is a function of the variables in increasing order.
    tree_pair = f_tree.conditional_expectation(pair)
    order = np.argsort(pair)
    estimates = [
        ("exact", exact(points[:, 0], points[:, 1])),
        ("Gaussian", gauss_pair.pdf(points)),
        ("KDE", kde_pair(points.T)),
        ("adaptive tree", tree_pair(points[:, order]) / (2 * BOX) ** 2),
    ]
    for col, (title, z) in enumerate(estimates):
        ax = axes[row, col]
        ax.contourf(G1, G2, np.maximum(z, 0).reshape(G1.shape), levels=12, cmap="Blues")
        if row == 0:
            ax.set_title(title, fontsize="medium")
        if col == 0:
            ax.set_ylabel(NAMES[mu])
        if row == 2:
            ax.set_xlabel(NAMES[nu])
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
fig.tight_layout()

# %%
# The adaptive tree reproduces the banana, the two modes and the
# correlation. The Gaussian only sees ellipses, and the KDE, with a
# bandwidth chosen for six dimensions, blurs the shapes.
#
# The structure found by the tree
# -------------------------------
#
# The dimension tree learned by the algorithm, with the variables at the
# leaves and the ranks at the internal nodes.

tree = f_tree.tensor.tree
labels = [
    NAMES[tree.dims[node - 1][0]] if tree.is_leaf[node - 1]
    else str(f_tree.tensor.ranks[node - 1])
    for node in range(1, tree.nb_nodes + 1)
]
tree.plot_with_labels_at_nodes(labels, title="Dimension tree and ranks")

groups = [
    sorted(int(nu) + 1 for nu in tree.dims[node])
    for node in range(tree.nb_nodes)
    if not tree.is_leaf[node] and len(tree.dims[node]) == 2
]
print("Pairs of variables grouped by the tree:", groups)

# %%
# From the samples only, the algorithm groups the variables by pairs, the
# three hidden pairs: the tree recovers the structure of the distribution,
# that the shuffling of the variables had hidden. The tensor train, whose
# tree follows the order of the variables, separates the variables of each
# pair, and needs larger ranks for a worse accuracy.
#
# Discussion
# ----------
#
# - The Gaussian and the product of marginals are fast and robust, but
#   biased: they fit only a family of shapes, and do not improve with more
#   samples.
# - The KDE makes no assumption, but its accuracy degrades quickly with the
#   dimension, and it gives a sum of :math:`n` kernels: evaluating it, or
#   computing its marginals, costs :math:`O(n)` per point.
# - The tensor formats exploit the low-rank structure of the density. The
#   cost of construction is larger, but the result is a compact function (a
#   few hundred parameters), whose marginals and conditional densities are
#   computed exactly.
# - The adaptive tree finds by itself a good grouping of the variables. With
#   a fixed tree, the order of the variables matters; adapting the tree
#   removes the need to know it.
# - Unlike the KDE, an :math:`L^2` estimate in tensor format can be slightly
#   negative where the density is close to zero.
