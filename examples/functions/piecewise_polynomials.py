r"""
Piecewise polynomials
=====================

Piecewise polynomial bases with given points and degrees, adapted to a
singularity, and bivariate piecewise polynomials.

For points :math:`a = t_0 < t_1 < \dots < t_n = b`, a
:class:`~tensap.PiecewisePolynomialFunctionalBasis` spans the functions whose
restriction to each element :math:`(t_{e-1}, t_e)` is a polynomial of degree
:math:`p_e`, with no continuity imposed across the points. The basis is
orthonormal for the uniform measure on :math:`(a, b)`. We use it to
interpolate functions of one and two variables, smooth or with a
singularity. See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_piecewise_polynomials.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Piecewise polynomial basis with given points and degrees
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Two elements :math:`(0, 0.2)` and :math:`(0.2, 1)`, with degrees 1 and 2.
# In each element, the basis functions are orthonormal polynomials of the
# local variable, extended by zero outside; the basis has
# :math:`\sum_e (p_e + 1)` functions. We plot them and their first
# derivatives.
points = [0, 0.2, 1]
p = [1, 2]
H = tensap.PiecewisePolynomialFunctionalBasis(points, p)
xplot = np.linspace(0, 1, 1000)
plt.figure(1)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))

plt.figure(2)
plt.plot(xplot, H.eval_derivative(1, xplot))
plt.legend(np.arange(H.cardinal()))


# %%
# Piecewise polynomial basis with constant degree and mesh size
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``hp(a, b, h, p)`` builds a uniform mesh of :math:`(a, b)` with mesh size
# :math:`h`, here :math:`h = 1/4`, and the degree :math:`p = 1` in every
# element.
h = 2 ** (-2)
p = 1
H = tensap.PiecewisePolynomialFunctionalBasis.hp(0, 1, h, p)
xplot = np.linspace(0, 1, 1000)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))

# %%
# Piecewise polynomial basis with constant degree and given number of elements
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``np(a, b, n, p)`` gives the same kind of basis with :math:`n` elements
# instead, here 2 elements and the degree 4.
n = 2
p = 4
H = tensap.PiecewisePolynomialFunctionalBasis.np(0, 1, n, p)
xplot = np.linspace(0, 1, 1000)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))


# %%
# Interpolation of a function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We interpolate :math:`f(x) = \cos(4 \pi x)` with 4 elements of degree 3.
# The interpolant :math:`I f = \sum_i a(i) \varphi_i` matches :math:`f` at
# :math:`p + 1` points in each element: the Chebyshev points
# (``interpolation_points``), or the magic points of the polynomial basis of
# the element (``magic_points``). The relative error is estimated on a
# random sample of the uniform measure on :math:`(0, 1)`.

H = tensap.PiecewisePolynomialFunctionalBasis.np(0, 1, 4, 3)
f = tensap.UserDefinedFunction("np.cos(4*np.pi*x0)", 1)

# using chebychev points in each interval
g = H.interpolation_points()
If = H.interpolate(f, g)
plt.plot(xplot, If.eval(xplot), xplot, f.eval(xplot))
plt.legend(["If", "f"])

X = tensap.UniformRandomVariable(0, 1)
ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (chebychev points)  = %2.5e" % ERR_L2)

# using magic points in each interval
g = H.magic_points()
If = H.interpolate(f, g)
ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (magic points) = %2.5e" % ERR_L2)

# %%
# The two sets of points give errors of the same order, a few
# :math:`10^{-3}`, with only 16 degrees of freedom for two periods of the
# cosine.


# %%
# Singularity adapted Piecewise polynomial basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The function :math:`f(x) = \sqrt{x}` is not smooth at :math:`x = 0`, and a
# uniform mesh would converge slowly. ``singularityhp_adapted`` builds an
# *hp* mesh: the elements are refined geometrically towards the singularity,
# with points :math:`0, 2^{-12}, 2^{-11}, \dots, 1/2, 1` for
# :math:`h = 2^{-12}`, and the degree grows away from it, from 0 in the
# smallest element. The function is interpolated at the Chebyshev points of
# the elements.
f = tensap.UserDefinedFunction("np.sqrt(x0)", 1)
h = 2 ** (-12)
H = tensap.PiecewisePolynomialFunctionalBasis.singularityhp_adapted(
    0, 1, [0], h
)

xI = H.interpolation_points()
If = H.interpolate(f, xI)
xplot = np.linspace(0, 1, 1000)
plt.plot(
    xplot, If.eval(xplot), xplot, f.eval(xplot), xI, f.eval(xI), "."
)
plt.legend(["log If", "log f"])
X = tensap.UniformRandomVariable(0, 1)
ERR_L2, ERR_L_INF = f.test_error(If, 100, X)
print("Mean squared error (random sample) = %2.5e" % ERR_L2)

# %%
# A random sample of 100 points rarely falls in the small elements near 0,
# where the error concentrates. The relative :math:`L^1` and :math:`L^2`
# errors are therefore also computed with a composite Gauss-Legendre rule
# with 20 points in each element of the mesh, for instance
#
# .. math::
#
#    \frac{\|f - I f\|_{L^2}}{\|f\|_{L^2}} \approx
#    \Big(\frac{\sum_k w_k |f(x_k) - I f(x_k)|^2}
#    {\sum_k w_k |f(x_k)|^2}\Big)^{1/2}.
#
# These errors are larger than the one on the random sample, the
# :math:`L^2` error being of the order of :math:`10^{-4}`.

g = tensap.IntegrationRule.gauss_legendre_composite(H.points, 20)
L1error = np.dot(
    g.weights, abs(f.eval(g.points) - If.eval(g.points))
) / np.dot(g.weights, abs(f.eval(g.points)))
print("L1 error = %2.5e" % L1error[0])
L2error = np.sqrt(
    np.dot(g.weights, abs(f.eval(g.points) - If.eval(g.points)) ** 2)
    / np.dot(g.weights, abs(f.eval(g.points)) ** 2)
)
print("L2 error = %2.5e" % L2error[0])

# %%
# Bivariate piecewise polynomials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In dimension 2, with the same basis :math:`\{\varphi_i\}_{i \in I}` of
# piecewise cubic polynomials on a uniform mesh of size :math:`h = 2^{-4}`
# for each variable, the
# :class:`~tensap.FullTensorProductFunctionalBasis` of the products
# :math:`\varphi_{i_1}(x_1) \varphi_{i_2}(x_2)` gives the interpolant
#
# .. math::
#
#    I f(x_1, x_2) = \sum_{i_1 \in I} \sum_{i_2 \in I} a(i_1, i_2) \,
#    \varphi_{i_1}(x_1) \, \varphi_{i_2}(x_2)
#
# of :math:`f(x) = \cos(4 \pi x_1) \cos(2 \pi x_2)` on the product grid of the
# univariate interpolation points (``tensor_product_interpolation``). The
# surfaces of :math:`f` and :math:`I f` are plotted.
d = 2
f = tensap.UserDefinedFunction(
    "np.cos(4*np.pi*x0)*np.cos(2*np.pi*x1)", d
)
f.evaluation_at_multiple_points = True
h = 2**-4
p = 3
bases = tensap.PiecewisePolynomialFunctionalBasis.hp(0, 1, h, p)
bases = tensap.FunctionalBases.duplicate(bases, d)
H = tensap.FullTensorProductFunctionalBasis(bases)
If, OUTPUT = H.tensor_product_interpolation(f)
ERR_L2, ERR_L_INF = f.test_error(If, 100, If.measure)

print("Mean squared error = %2.5e" % ERR_L2)

f.measure = If.measure
axf = f.surf([100, 100])
axf.set_title("f")
axIf = If.surf([100, 100])
axIf.set_title("If")
