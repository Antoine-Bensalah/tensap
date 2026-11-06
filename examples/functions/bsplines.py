r"""
B-splines
=========

Cardinal B-splines, B-splines with extra knots, dilated B-splines, and
interpolation of a function.

A B-spline basis :math:`\{\varphi_i\}_{i \in I}` of degree :math:`s` spans
the piecewise polynomials of degree :math:`s` on a set of simple knots, with
:math:`s - 1` continuous derivatives. Such univariate bases are building
blocks of the tensor product bases of tensap. Here we plot them with their
derivatives, and use them to interpolate a function of one variable.
See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_BSplines.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Cardinal BSplines
# ^^^^^^^^^^^^^^^^^
# The cardinal B-spline :math:`B_m` of degree :math:`m` is built on the
# integer knots :math:`0, 1, \dots, m + 1`: :math:`B_0` is the indicator of
# :math:`[0, 1)`, and :math:`B_m` is obtained from :math:`B_{m-1}` by a
# convolution with :math:`B_0`,
#
# .. math::
#
#    B_m(x) = \int_0^1 B_{m-1}(x - t) \, dt .
#
# It is supported on :math:`[0, m + 1]`. The top row shows
# :math:`B_0, \dots, B_4` and the bottom row their first derivatives,
# computed with ``eval_derivative``.
plt.figure(1)
mmax = 5
fig, axes = plt.subplots(2, mmax, figsize=(15, 5))
xplot = np.linspace(0, mmax + 1, 400)
for m in range(mmax):
    h = tensap.BSplinesFunctionalBasis.cardinal_bspline(m)
    axes[0, m].plot(xplot, h.eval(xplot))
    axes[0, m].set_title(f"B_{m}")
    axes[0, m].set_xlim([0, m + 1])
    axes[1, m].plot(xplot, h.eval_derivative(1, xplot))
    axes[1, m].set_title(f"dB_{m}/dx")
    axes[1, m].set_xlim([0, m + 1])
plt.show()

# %%
# BSplines with extra knots
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# To get a basis of the splines of degree :math:`s` on an interval
# :math:`[t_1, t_n]`, ``with_extra_knots`` adds :math:`s` knots on each side
# of the given knots :math:`t_1 < \dots < t_n`, with the spacing of the first
# and last intervals. The B-splines built on these knots, restricted to
# :math:`[t_1, t_n]`, form a basis of the splines of degree :math:`s` on this
# interval. Here :math:`s = 3` with three knots on :math:`[-1, 1]`; we plot
# the basis functions and their first two derivatives.
knots = np.linspace(-1, 1, 3)
s = 3  # degree
h = tensap.BSplinesFunctionalBasis.with_extra_knots(knots, s)

xplot = np.linspace(-1, 1, 400)

# Evaluate B-splines and their derivatives at the points in xplot
B_spline_values = h.eval(xplot)
B_spline_deriv_1 = h.eval_derivative(1, xplot)
B_spline_deriv_2 = h.eval_derivative(2, xplot)

# Create figure and subplots
plt.figure(1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# First subplot: B-Splines
axes[0].plot(xplot, B_spline_values)
axes[0].set_title("BSplines")
axes[0].set_xlim([-1, 1])

# Second subplot: First derivative of B-Splines
axes[1].plot(xplot, B_spline_deriv_1)
axes[1].set_title("BSplines first derivative")
axes[1].set_xlim([-1, 1])

# Third subplot: Second derivative of B-Splines
axes[2].plot(xplot, B_spline_deriv_2)
axes[2].set_title("BSplines second derivative")
axes[2].set_xlim([-1, 1])

# Show the plot
plt.tight_layout()
plt.show()

# %%
# Interpolation of a function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We now interpolate :math:`f(x) = \cos(4 \pi x)` on :math:`[-1, 1]` with
# splines of degree 4 on 30 knots. The interpolant
# :math:`I f = \sum_{i \in I} a(i) \varphi_i` satisfies
# :math:`I f(x_k) = f(x_k)` at :math:`\# I` points :math:`x_k`. These are the
# *magic points* of the basis, selected greedily among 1000 candidate points
# so that the interpolation is well conditioned (``magic_points``). The
# relative error is then estimated on a random sample of the uniform measure
# on :math:`[-1, 1]` with ``test_error``.
knots = np.linspace(-1, 1, 30)
s = 4
H = tensap.BSplinesFunctionalBasis.with_extra_knots(knots, s)
f = tensap.UserDefinedFunction("np.cos(4*np.pi*x0)", 1)

X = tensap.UniformRandomVariable(-1, 1)
a = H.magic_points(np.linspace(-1, 1, 1000))[0]
If = H.interpolate(f, a)

plt.plot(xplot, If.eval(xplot), xplot, f.eval(xplot), a, f.eval(a), ".")
plt.legend(["If", "f"])

ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (magic points)  = %2.5e" % ERR_L2)

# %%
# The interpolant and the function cannot be told apart on the plot, with a
# relative error of the order of :math:`10^{-4}`.


# %%
# Dilated BSplines
# ^^^^^^^^^^^^^^^^
# Dilated B-splines give a multilevel basis on :math:`(0, 1)`. A function is
# indexed by a level :math:`l` and a local index :math:`j`,
#
# .. math::
#
#    \varphi_{l, j}(x) = b^{l/2} \, B_s(b^l x - j),
#
# with :math:`b` the base of dilation, and ``with_level_bounded_by`` keeps
# the levels bounded by :math:`r`. With :math:`s = 0`, these are the scaled
# indicators of the :math:`b`-adic intervals of each level.
s = 0  # degree
r = 2  # resolution
b = 2  # base of dilation
h = tensap.DilatedBSplinesFunctionalBasis.with_level_bounded_by(s, b, r)

plt.figure(2)
plt.clf()
x = np.linspace(0, 1, 100)
plt.plot(x, h.eval(x))
plt.legend(range(h.cardinal()))


# %%
# Dilated BSplines
# ^^^^^^^^^^^^^^^^
# The same construction with quadratic B-splines (:math:`s = 2`), and their
# first and second derivatives. The derivative of order :math:`k` of
# :math:`\varphi_{l, j}` carries a factor :math:`b^{l k}`, so the functions of
# the finer levels have larger derivatives.
s = 2  # degree
r = 2  # resolution
b = 2  # base of dilation
h = tensap.DilatedBSplinesFunctionalBasis.with_level_bounded_by(s, b, r)

plt.figure(3)
x = np.linspace(0, 1, 100)
y = h.eval(x)
plt.plot(x, y)
plt.title("BSplines")

plt.figure(4)
y_derivative_1 = h.eval_derivative(1, x)
plt.plot(x, y_derivative_1)
plt.title("BSplines first derivative")

plt.figure(5)
# Placeholder for second derivative evaluation
y_derivative_2 = h.eval_derivative(2, x)  # Example second derivative
plt.plot(x, y_derivative_2)
plt.title("BSplines second derivative")
