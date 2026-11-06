r"""
Integration rules
=================

Gauss, piecewise Gauss-Legendre and tensor product integration rules.

An integration rule approximates an integral with respect to a measure
:math:`\mu` by a weighted sum of evaluations,
:math:`\int f \, d\mu \approx \sum_{k=1}^{n} w_k f(x_k)`. We test Gauss rules
on an interval, a composite rule, and a tensor product rule for a product
measure :math:`\mu_1 \otimes \mu_2`, on functions whose integral is known.
See :ref:`theory-functions`.

This example comes from the script
``test/non_regression_scripts/approximation/tutorial_integration_rules.py``.
"""

import numpy as np
import tensap

# Fixed seed, so that the example is reproducible.
np.random.seed(0)

# %%
# Gauss Integration Rule
# ^^^^^^^^^^^^^^^^^^^^^^
# The Gauss rule with :math:`n` points for a measure :math:`\mu` is exact for
# the polynomials of degree at most :math:`2n - 1`; its points are the roots
# of the orthonormal polynomial of degree :math:`n`. For the Lebesgue measure
# on :math:`[0, 2]` and 10 points, we integrate :math:`f(x) = e^x`, whose
# integral is :math:`e^2 - 1`, and print the relative error.
mu = tensap.LebesgueMeasure(0, 2)
G = mu.gauss_integration_rule(10)
# Function f(x) = exp(x)
f = tensap.UserDefinedFunction("np.exp(x0)", 1)
Iapp = G.integrate(f)
Iex = np.exp(2) - 1
print("Error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))

# %%
# Piecewise Gauss-Legendre Integration Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A composite rule splits the interval at given points, here
# :math:`[0, 1] = [0, 0.5] \cup [0.5, 1]`, and uses a Gauss-Legendre rule
# with 6 points on each piece
# (``IntegrationRule.gauss_legendre_composite``). This is the rule to use
# for piecewise smooth functions, with breakpoints at the singularities.
g = tensap.IntegrationRule.gauss_legendre_composite([0, 0.5, 1], 6)
f = tensap.UserDefinedFunction("np.exp(x0)", 1)
Iapp = g.integrate(f)
Iex = np.exp(1) - 1

print("Integration error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))


# %%
# Tensor product integration rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For a :class:`~tensap.ProductMeasure` :math:`\mu = \mu_1 \otimes \mu_2`,
# here the Lebesgue measure on :math:`[0, 5] \times [0, 1]`, the tensor
# product of Gauss rules with :math:`n_1` and :math:`n_2` points has the
# points :math:`(x^{(1)}_{k_1}, x^{(2)}_{k_2})` and the weights
# :math:`w^{(1)}_{k_1} w^{(2)}_{k_2}`. With 10 and 2 points, it integrates
# :math:`f(x) = e^{x_1} x_2`, whose integral is :math:`(e^5 - 1) / 2`; the
# second rule is exact since :math:`f` is linear in :math:`x_2`.
mu = tensap.ProductMeasure(
    [tensap.LebesgueMeasure(0, 5), tensap.LebesgueMeasure(0, 1)]
)
G = mu.gauss_integration_rule([10, 2])
G = G.integration_rule()
# Function f(x0,x1) = exp(x0)*x1
f = tensap.UserDefinedFunction("np.exp(x0)*x1", 2)
f.evaluation_at_multiple_points = True
Iapp = G.integrate(f)
Iex = (np.exp(5) - 1) / 2
print("Error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))

# %%
# In the three cases, the relative error is at the level of machine
# precision: these rules are very accurate for smooth integrands.
