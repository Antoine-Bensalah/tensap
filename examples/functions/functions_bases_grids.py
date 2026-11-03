"""
Functions, tensor grids and functional bases
============================================

Define scalar and vector-valued functions, evaluate them on full and sparse
tensor grids, build functional bases and project a function by quadrature.

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
d = 3
fun = tensap.UserDefinedFunction("x0+x1+x2**4", d)
fun.evaluation_at_multiple_points = True
x = np.random.rand(4, d)
print("fun.eval(x) = \n%s\n" % fun.eval(x))
print("fun(x) =      \n%s\n" % fun(x))

# %%
# Vector-valued UserDefinedFunction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
f = tensap.UserDefinedFunction("[x0, 100*x1]", d, 2)
f.evaluation_at_multiple_points = False
print("With f.evaluation_at_multiple_points == False, f(x) = \n%s\n" % f(x))
f.evaluation_at_multiple_points = True
print("With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n" % f(x))

f = tensap.UserDefinedFunction("[x1,100*x2]", d, [1, 2])
f.evaluation_at_multiple_points = False
print("With f.evaluation_at_multiple_points == False, f(x) = \n%s\n" % f(x))
f.evaluation_at_multiple_points = True
print("With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n" % f(x))

# %%
# Evaluation on a FullTensorGrid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
g = tensap.FullTensorGrid(np.linspace(-1, 1, 50), d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker="x")

# %%
# Evaluation on a SparseTensorGrid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I0 = tensap.MultiIndices.with_bounded_norm(d, 1, 5)
g = tensap.SparseTensorGrid(np.linspace(0, 1, 6), I0, d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker="x")

# %%
# Functional Bases
# ^^^^^^^^^^^^^^^^
h = tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(), range(5))
H = tensap.FunctionalBases.duplicate(h, d)
grid = tensap.FullTensorGrid(np.linspace(-1, 1, 10), d)
x = grid.array()
Hx = H.eval(x)

# %%
# Sparse tensor functional basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
d = 2
p = 1
m = 4
h = tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(), range(5))
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
d = 3
p = 3
fun = tensap.UserDefinedFunction("x0+x1**2+x2**3", d)
fun.evaluation_at_multiple_points = True
v = tensap.NormalRandomVariable()
X = tensap.RandomVector(v, d)
I0 = v.gauss_integration_rule(5).tensorize(d)
u = fun.eval_on_tensor_grid(I0.points)

h = tensap.PolynomialFunctionalBasis(tensap.HermitePolynomials(), range(p + 1))
H = tensap.FunctionalBases.duplicate(h, d)
H = tensap.FullTensorProductFunctionalBasis(H)

f, _ = H.tensor_product_projection(fun, I0)

N_test = 100
x_test = X.random(N_test)
err_test = np.linalg.norm(f(x_test) - fun(x_test)) / np.linalg.norm(fun(x_test))
print("Test error = %2.5e\n" % err_test)
