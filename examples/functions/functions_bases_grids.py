r"""
Functions, tensor grids and functional bases
============================================

Define scalar and vector-valued functions, evaluate them on full and sparse
tensor grids, build functional bases and project a function by quadrature.

The functions of :math:`d` variables considered in tensap are expanded on
tensor product bases, :math:`f(x) = \sum_{i} a(i) \varphi_i(x)` with
:math:`\varphi_i(x) = \varphi^{(1)}_{i_1}(x_1) \cdots
\varphi^{(d)}_{i_d}(x_d)`, where :math:`i` runs over a product set or a
subset :math:`\Lambda` of multi-indices. This example walks through the
pieces: functions, grids, full and sparse bases, and the computation of the
coefficients :math:`a` by quadrature in dimension 3.
See :ref:`theory-functions` and :ref:`theory-multi-indices`.

This example comes from the script
``test/non_regression_scripts/functions/tutorial_functions_bases_grids.py``.
"""

# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Scalar-valued UserDefinedFunction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.UserDefinedFunction` wraps an expression of the variables
# ``x0, ..., x{d-1}``, here :math:`f(x) = x_1 + x_2 + x_3^4` on
# :math:`\mathbb{R}^3`. With ``evaluation_at_multiple_points = True`` the
# expression is evaluated at all the points at once. It is evaluated at 4
# random points, given as the rows of an array of shape ``(4, d)``; ``eval``
# returns a column and the call ``fun(x)`` a flat array.
d = 3
fun = tensap.UserDefinedFunction("x0+x1+x2**4", d)
fun.evaluation_at_multiple_points = True
x = np.random.rand(4, d)
print("fun.eval(x) = \n%s\n" % fun.eval(x))
print("fun(x) =      \n%s\n" % fun(x))

# %%
# Vector-valued UserDefinedFunction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A function can also take its values in :math:`\mathbb{R}^2`, or in arrays
# of shape ``[1, 2]``. With ``evaluation_at_multiple_points = False`` the
# expression is evaluated point by point; both modes give the same values.
f = tensap.UserDefinedFunction("[x0, 100*x1]", d, 2)
f.evaluation_at_multiple_points = False
print(
    "With f.evaluation_at_multiple_points == False, f(x) = \n%s\n"
    % f(x)
)
f.evaluation_at_multiple_points = True
print(
    "With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n"
    % f(x)
)

f = tensap.UserDefinedFunction("[x1,100*x2]", d, [1, 2])
f.evaluation_at_multiple_points = False
print(
    "With f.evaluation_at_multiple_points == False, f(x) = \n%s\n"
    % f(x)
)
f.evaluation_at_multiple_points = True
print(
    "With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n"
    % f(x)
)

# %%
# Evaluation on a FullTensorGrid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.FullTensorGrid` is a product
# :math:`G_1 \times G_2 \times G_3` of grids of one variable, here 50 points
# of :math:`[-1, 1]` in each direction, so :math:`50^3` points in total.
# ``eval_on_tensor_grid`` returns the evaluations of the function on it.
g = tensap.FullTensorGrid(np.linspace(-1, 1, 50), d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker="x")

# %%
# Evaluation on a SparseTensorGrid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A :class:`~tensap.SparseTensorGrid` only keeps the points
# :math:`(x_{i_1}, x_{i_2}, x_{i_3})` for the multi-indices :math:`i` of a
# set :math:`\Lambda`. Here the univariate grid has 6 points on
# :math:`[0, 1]` and :math:`\Lambda = \{i \in \mathbb{N}_0^3 :
# i_1 + i_2 + i_3 \le 5\}`, which is much smaller than the full grid of
# :math:`6^3` points, as the plot shows.
I0 = tensap.MultiIndices.with_bounded_norm(d, 1, 5)
g = tensap.SparseTensorGrid(np.linspace(0, 1, 6), I0, d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker="x")

# %%
# Functional Bases
# ^^^^^^^^^^^^^^^^
# A :class:`~tensap.PolynomialFunctionalBasis` gives the univariate basis of
# the monomials :math:`1, x, \dots, x^4`, and ``FunctionalBases.duplicate``
# uses it for each of the :math:`d` variables. ``eval`` returns, for each
# variable :math:`\nu`, the matrix of the evaluations of the
# :math:`\varphi^{(\nu)}_{i}` at the points.
h = tensap.PolynomialFunctionalBasis(
    tensap.CanonicalPolynomials(), range(5)
)
H = tensap.FunctionalBases.duplicate(h, d)
grid = tensap.FullTensorGrid(np.linspace(-1, 1, 10), d)
x = grid.array()
Hx = H.eval(x)

# %%
# Sparse tensor functional basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In dimension :math:`d = 2`, a
# :class:`~tensap.SparseTensorProductFunctionalBasis` keeps the products
# :math:`\varphi^{(1)}_{i_1}(x_1) \varphi^{(2)}_{i_2}(x_2)` for
# :math:`i \in \Lambda`, with
# :math:`\Lambda = \{i : \|i\|_{\ell^1} \le 4\}`: the polynomials of total
# degree at most 4. The multi-indices are printed (they are the degrees,
# hence start from 0), and each basis function is plotted on a fine grid.
d = 2
p = 1
m = 4
h = tensap.PolynomialFunctionalBasis(
    tensap.CanonicalPolynomials(), range(5)
)
H = tensap.FunctionalBases.duplicate(h, d)
I0 = tensap.MultiIndices.with_bounded_norm(d, p, m)
Psi = tensap.SparseTensorProductFunctionalBasis(H, I0)

print("Multi-indices: \n%s\n" % I0.array)

finegrid = tensap.FullTensorGrid(np.arange(-1, 1.1, 0.1), d)
x = finegrid.array()
Psix = Psi.eval(x)

for i in range(I0.cardinal()):
    finegrid.plot(Psix[:, i])
    plt.title(str(I0.array[i, :]))

# %%
# Projection on polynomial space through quadrature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Back to :math:`d = 3`, with a standard Gaussian vector
# :math:`X = (X_1, X_2, X_3)` of law :math:`\mu`, and the function
# :math:`f(x) = x_1 + x_2^2 + x_3^3`. With the orthonormal Hermite
# polynomials of degree at most 3 in each variable, the orthogonal
# projection of :math:`f` in :math:`L^2_\mu` has the coefficients
#
# .. math::
#
#    a(i) = \mathbb{E}\big(f(X) \varphi_i(X)\big)
#    \approx \sum_{k} w_k \, f(x_k) \, \varphi_i(x_k),
#
# computed with the tensor product of Gauss-Hermite rules with 5 points per
# variable (``tensor_product_projection``). This rule is exact for the
# products :math:`f \varphi_i`, and :math:`f` is in the space, so the test
# error is at the level of machine precision.
d = 3
p = 3
fun = tensap.UserDefinedFunction("x0+x1**2+x2**3", d)
fun.evaluation_at_multiple_points = True
v = tensap.NormalRandomVariable()
X = tensap.RandomVector(v, d)
I0 = v.gauss_integration_rule(5).tensorize(d)
u = fun.eval_on_tensor_grid(I0.points)

h = tensap.PolynomialFunctionalBasis(
    tensap.HermitePolynomials(), range(p + 1)
)
H = tensap.FunctionalBases.duplicate(h, d)
H = tensap.FullTensorProductFunctionalBasis(H)

f, _ = H.tensor_product_projection(fun, I0)

N_test = 100
x_test = X.random(N_test)
err_test = np.linalg.norm(f(x_test) - fun(x_test)) / np.linalg.norm(
    fun(x_test)
)
print("Test error = %2.5e\n" % err_test)
