(theory-functions)=
# Measures, bases and functions

To approximate a function of $d$ variables in a tensor format, the function
is expanded on a tensor product of bases of functions of one variable: its
coefficients form a tensor, which is then stored in a low-rank format. This
page introduces the measures that define the inner products, the bases of
functions (in particular orthonormal polynomials), and the functions given by
a tensor of coefficients.

## Measures and random variables

The functions are defined on a set $\mathcal{X} = \mathcal{X}_1 \times
\cdots \times \mathcal{X}_d$ equipped with a product measure
$\mu = \mu_1 \otimes \cdots \otimes \mu_d$, often a probability measure: the
law of a random vector $X = (X_1, \dots, X_d)$ with independent components.
The space $L^2_\mu(\mathcal{X})$ is then the tensor product of the spaces
$L^2_{\mu_\nu}(\mathcal{X}_\nu)$.

tensap provides discrete, uniform, normal and empirical random variables (the
last one estimated from a sample by a kernel density, with Scott's rule for
the bandwidth), random vectors with independent components, and the Lebesgue
measure on an interval. They can be sampled, and their density, cumulative
distribution function and its inverse can be evaluated.

## Orthonormal polynomials

```{prf:definition} Orthonormal polynomials
:label: def-orthonormal-polynomials

For a measure $\mu$ on $\mathbb{R}$, the orthonormal polynomials
$(p_n)_{n \ge 0}$, with $p_n$ of degree $n$, satisfy
$\int p_m\, p_n\, d\mu = \delta_{mn}$.
```

```{prf:proposition} Three-term recurrence
:label: prop-three-term

The monic orthogonal polynomials $\tilde p_n$ satisfy $\tilde p_{-1} = 0$,
$\tilde p_0 = 1$ and

$$
\begin{aligned}
\tilde p_{n+1}(x) &= (x - a_n)\, \tilde p_n(x) - b_n\, \tilde p_{n-1}(x), \\
a_n &= \frac{\int x\, \tilde p_n^2\, d\mu}{\int \tilde p_n^2\, d\mu},
\qquad
b_n = \frac{\int \tilde p_n^2\, d\mu}{\int \tilde p_{n-1}^2\, d\mu},
\end{aligned}
$$

and $p_n = \tilde p_n / \|\tilde p_n\|_{L^2_\mu}$.
```

A family of orthonormal polynomials is thus defined by its recurrence
coefficients and norms. tensap provides the Legendre polynomials (uniform
measure on $[-1, 1]$), the Hermite polynomials (standard normal measure),
discrete polynomials (for a discrete measure) and empirical polynomials (for
an empirical measure). For a measure obtained from one of these by an affine
change of variable, such as a uniform measure on $[a, b]$ or a normal measure
with any mean and standard deviation, the polynomials are shifted and
scaled accordingly. Each random variable gives its orthonormal polynomials.

## Bases of functions

A *basis* $\{\varphi_i\}_{i \in I}$ of functions of one variable can be made
of polynomials, piecewise polynomials, B-splines, or any user-defined
functions. For $d$ variables, with a basis
$\{\varphi^{(\nu)}_{i_\nu}\}_{i_\nu \in I_\nu}$ for each variable, the
products

$$
\varphi_i(x) = \varphi^{(1)}_{i_1}(x_1) \cdots \varphi^{(d)}_{i_d}(x_d),
\qquad i = (i_1, \dots, i_d),
$$

form a basis indexed by $I_1 \times \cdots \times I_d$ (a *full tensor
product basis*), or by a subset $\Lambda$ of it (a *sparse tensor product
basis*, see {ref}`theory-multi-indices`). With orthonormal univariate bases,
the products are orthonormal in $L^2_\mu$.

## Functions in a basis, functional tensors

A function expanded on a basis, $f(x) = \sum_{i \in I} a(i)\, \varphi_i(x)$,
is given by its coefficients. With a full tensor product basis, the
coefficients form a tensor of order $d$:

$$
\begin{aligned}
f(x_1, \dots, x_d) &= \sum_{i_1 \in I_1} \cdots \sum_{i_d \in I_d}
  a(i_1, \dots, i_d)\, \varphi^{(1)}_{i_1}(x_1) \cdots
  \varphi^{(d)}_{i_d}(x_d) \\
&= \big(a, \Phi(x)\big),
\end{aligned}
$$

with $\Phi(x) = \varphi^{(1)}(x_1) \otimes \cdots \otimes
\varphi^{(d)}(x_d)$ the tensor of the evaluations of the bases. This is a
*functional tensor*: when the tensor $a$ is in a low-rank format (canonical,
Tucker, tree-based or tensor train), so is $f$, as a function of the
variables. For instance, a tensor $a$ of canonical rank $r$ gives a sum of
$r$ products of functions of one variable,
$f(x) = \sum_{k=1}^{r} f^{(1)}_k(x_1) \cdots f^{(d)}_k(x_d)$. An evaluation
of $f$ at a point is a contraction of $a$ with the vectors
$\varphi^{(\nu)}(x_\nu)$, at a cost proportional to the storage of $a$.

With an orthonormal basis, the $L^2_\mu$ norm of $f$ is the canonical norm of
its tensor of coefficients: the approximation of $f$ is the approximation of
$a$, and the truncation of $a$ ({ref}`theory-truncation`) controls the error
on $f$.

**Tensorized functions.** A function of one variable on $[0, 1)$ can also be
identified with a function of $L + 1$ variables, by the digits of its argument
in base $b$, and approximated in a tensor format: see
{ref}`theory-qtt`.

## In tensap

- Measures and random variables: {py:class}`~tensap.UniformRandomVariable`,
  {py:class}`~tensap.NormalRandomVariable`,
  {py:class}`~tensap.DiscreteRandomVariable`,
  {py:class}`~tensap.EmpiricalRandomVariable`,
  {py:class}`~tensap.RandomVector`, {py:class}`~tensap.LebesgueMeasure`.
- Polynomials: {py:class}`~tensap.LegendrePolynomials`,
  {py:class}`~tensap.HermitePolynomials`,
  {py:class}`~tensap.DiscretePolynomials`,
  {py:class}`~tensap.EmpiricalPolynomials`,
  {py:class}`~tensap.ShiftedOrthonormalPolynomials`; the method
  `orthonormal_polynomials` of a random variable or a measure.
- Bases: {py:class}`~tensap.PolynomialFunctionalBasis`,
  {py:class}`~tensap.UserDefinedFunctionalBasis`,
  {py:class}`~tensap.FunctionalBases`,
  {py:class}`~tensap.FullTensorProductFunctionalBasis`,
  {py:class}`~tensap.SparseTensorProductFunctionalBasis`.
- Functions: {py:class}`~tensap.FunctionalBasisArray` (a function on a
  basis), {py:class}`~tensap.FunctionalTensor` (coefficients in a tensor
  format), {py:class}`~tensap.TensorizedFunction`.
- Examples: {ref}`sphx_glr_examples_functions_measures.py`,
  {ref}`sphx_glr_examples_functions_orthonormal_polynomials.py`,
  {ref}`sphx_glr_examples_functions_functions_bases_grids.py`.
