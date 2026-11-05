# Measures, bases and functions

## `RandomVariable`

A random variable $X$ can be created by calling its name: for
instance, `X = tensap.UniformRandomVariable(a, b)` creates a random
variable with a uniform distribution on the interval $[a, b]$. The
random variables currently implemented in tensap are:

- `tensap.DiscreteRandomVariable(v, p)`: a random variable with
  discrete values $v$ and associated probabilities $p$,
- `tensap.UniformRandomVariable(a, b)`: a uniform random variable on
  $[a, b]$,
- `tensap.NormalRandomVariable(m, s)`: a normal random variable with
  mean $m$ and standard deviation $s$,
- `tensap.EmpiricalRandomVariable(S)`: a random variable created from
  a sample $S$ using kernel density estimation with Scott's rule
  of thumb to determine the bandwidth.

A new random variable can easily be implemented in tensap by making its
class inheriting from {py:class}`~tensap.RandomVariable` and implementing the few
methods necessary for its creation.

Once a random variable $X$ is created, one can for instance
generate $n$ random numbers according to its distribution with
`X.random(n)`, create the orthonormal polynomials associated with its
measure with `X.orthonormal_polynomials()` (as presented in Section
{ref}`theory-polynomials`), or evaluate its probability density function
(`X.pdf(x)`), cumulative distribution function (`X.cdf(x)`) or
inverse cumulative distribution function (`X.icdf(x)`).

## `RandomVector`

A random vector $X$ if defined in tensap by a list of
{py:class}`~tensap.RandomVariable` objects and a {py:class}`~tensap.Copula`, characterizing the
dependencies between the random variables. Currently, only the
independent copula {py:class}`~tensap.IndependentCopula` is implemented.

Given a list of {py:class}`~tensap.RandomVariable` `random_variables` and a `Copula
C`, a random vector can be created with `X =
tensap.RandomVector(random_variables, copula=C)`.

Once a random vector $X$ is created, one can for instance generate
$n$ random numbers according to its distribution with
`X.random(n)`, create the orthonormal polynomials associated with its
measure with `X.orthonormal_polynomials()` (as presented in Section
{ref}`theory-polynomials`), or evaluate its probability density function
(`X.pdf(x)`) or cumulative distribution function (`X.cdf(x)`).

(theory-polynomials)=
## `Polynomials`

Families of univariate polynomials $(p_i)_{i\ge 0}$ are
represented in tensap with classes inheriting from
{py:class}`~tensap.UnivariatePolynomials`. The $i$-th polynomial $p_i$
represented by a {py:class}`~tensap.UnivariatePolynomials` object `P` can be evaluated
with `P.polyval(x, i)`, as well as its first order derivative
(`P.d_polyval(x, i)`) and its $n$-th order derivative
(`P.dn_polyval(x, n, i)`).

Given a measure $\mu$, the moments
$\int  p_{i_1}(x)...p_{i_k}(x) d\mu(x)$ for
$(i_1,...,i_k) \in {\mathbb{N}}^{k}$ can be obtained with
`P.moment(I, mu)`, with $\texttt{I}$ a
$n$-by-$k$ array representing $n$ tuples
$(i_1,...,i_k)$. `P.moment(I, X)` with
$\texttt{X}$ a random variable considers for
$\mu$ the probability distribution of $X$.

### `CanonicalPolynomials`
The family of canonical polynomials is implemented in the class
{py:class}`~tensap.CanonicalPolynomials`. It is such that its $i$-th polynomial is
$p_i(x) = x^i$.

### `OrthonormalPolynomials`
Orthonormal polynomials are families of polynomials
$(p_i)_{i \geq 0}$ that satisfy

$$\int p_i(x) p_j(x) d\mu(x)= \delta_{ij}$$

with $\delta_{ij}$ the Kronecker delta, and with $\mu$ some
measure.

In tensap, the orthonormal polynomials $p_i$, $i \geq 0$, are defined using the three-term recurrence relation

$$\begin{aligned}
&\tilde p_{-1}(x) = 0, \quad \tilde p_0(x) = 1, \\
&\tilde p_{i+1}(x) = (x - a_i)\tilde p_{i}(x) - b_i \tilde p_{i-1}(x), \quad i \geq 0,\\
&p_i(x) = \frac{\tilde p_i(x)}{n_i}, \quad i \geq 0
\end{aligned}$$

with $a_i$ and $b_i$ the recurrence coefficients, and
$n_i$ the norm of $\tilde p_i$, defined by

$$\begin{aligned}
a_i &= \frac{\int x \, \tilde p_i(x)^2 \, d\mu(x)}{\int \tilde p_i(x)^2 \, d\mu(x)}, \\
b_i &= \frac{\int \tilde p_i(x)^2 \, d\mu(x)}{\int \tilde p_{i-1}(x)^2 \, d\mu(x)}, \\
n_i &= \sqrt{\int \tilde p_i(x)^2 \, d\mu(x)}.
\end{aligned}$$

Implementing a new family of orthonormal polynomials in tensap is
easy: one only needs to create a class with a method providing the
recurrence coefficients $a_i$, $b_i$ and the norms
$n_i$, $\forall i \geq 0$.

Are currently implemented in tensap:

- {py:class}`~tensap.DiscretePolynomials`: discrete polynomials orthonormal with
  respect to the measure of a {py:class}`~tensap.DiscreteRandomVariable`;
- {py:class}`~tensap.LegendrePolynomials`: polynomials defined on $[-1,1]$ and
  orthonormal with respect to the uniform measure on $[-1,1]$
  with density $\frac{1}{2}\mathbf{1}_{[-1,1]}(x)$;
- {py:class}`~tensap.HermitePolynomials`: polynomials defined on ${\mathbb{R}}$
  and orthonormal with respect to the standard gaussian measure with
  density $\exp(-x^2/2)/\sqrt{2\pi}$;
- {py:class}`~tensap.EmpiricalPolynomials`: polynomials orthonormal with respect to the
  measure of an {py:class}`~tensap.EmpiricalRandomVariable`.

If `mu` is a {py:class}`~tensap.LebesgueMeasure` on $[-1,1]$, `mu.orthonormal_polynomials()` returns a {py:class}`~tensap.LegendrePolynomials` with suitably normalized coefficients. If `mu` is a {py:class}`~tensap.LebesgueMeasure` on $[a,b]$ different from $[-1,1]$, `mu.orthonormal_polynomials()` returns a {py:class}`~tensap.ShiftedOrthonormalPolynomials`.  
If $\texttt{X}$ is a {py:class}`~tensap.DiscreteRandomVariable`, a {py:class}`~tensap.UniformRandomVariable`, a {py:class}`~tensap.NormalRandomVariable`, or a {py:class}`~tensap.EmpiricalRandomVariable`, the corresponding family of orthonormal polynomials can be created with the command `X.orthonormal_polynomials()`. If $\texttt{X}$ does not correspond to a default measure but can be obtained as the push-forward measure of a default measure by an affine transformation (e.g. a uniform measure on $[a,b] \neq [-1,1]$, or a gaussian measure with mean $a$ and standard deviation $\sigma$ with $(a,\sigma)\neq (0,1)$.), the returned object is a {py:class}`~tensap.ShiftedOrthonormalPolynomials`.

## `FunctionalBasis`

Bases of functions can be implemented in tensap by inheriting from {py:class}`~tensap.FunctionalBasis`. The basis functions of a {py:class}`~tensap.FunctionalBasis` object `H` can be evaluated with `H.eval(x)`, as well as their $i$-th order derivative with `H.eval_derivative(i, x)`.  
We present below some specific bases implemented in tensap. New bases can easily be implemented by making their class inherit from {py:class}`~tensap.FunctionalBasis`.

### `PolynomialFunctionalBasis`
The command {py:class}`~tensap.PolynomialFunctionalBasis` `(basis, indices)`,
with `basis` a {py:class}`~tensap.UnivariatePolynomials` and `indices` a list,
returns the basis of polynomials $(p_i)_{i \in I}$ with $I$
given by `indices`.

### `UserDefinedFunctionalBasis`
Given a list of functions `fun`, taking each as inputs $d$
variables, and a `Measure mu`, the command
`tensap.UserDefinedFunctionalBases(fun, mu, d)` returns a basis whose
functions are the ones given in `fun`, with a domain equipped with the
measure $mu$.

### `FullTensorProductFunctionalBasis`
A {py:class}`~tensap.FullTensorProductFunctionalBasis` object represents a basis of
multivariate functions
$\{\phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d)\}_{i_1 \in I^1, \ldots, i_d \in I^d}$.
It is obtained with the command
`tensap.FullTensorProductFunctionalBasis(bases)`, where `bases` is a
list of {py:class}`~tensap.FunctionalBasis` or a {py:class}`~tensap.FunctionalBases`, containing the
different bases $\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}$,
$\nu = 1, \ldots, d$.

### `SparseTensorProductFunctionalBasis`
A {py:class}`~tensap.SparseTensorProductFunctionalBasis` object represents a basis of
multivariate functions
$\{\phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d)\}_{(i_1, \ldots, i_d) \in \Lambda}$,
with $\Lambda \subset I^1 \times \cdots \times I^d$ a set of
multi-indices. It is obtained with the command
`tensap.SparseTensorProductFunctionalBasis(bases, indices)`, where
`bases` is a list of {py:class}`~tensap.FunctionalBasis` or a {py:class}`~tensap.FunctionalBases`,
containing the different bases
$\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}$,
$\nu = 1, \ldots, d$, and `indices` is a {py:class}`~tensap.MultiIndices`
representing the set of multi-indices $\Lambda$.

## `FunctionalBases`

The command `tensap.FunctionalBases(bases)`, with `bases` a list of
{py:class}`~tensap.FunctionalBasis`, returns an object representing a collections of
bases. To obtain a collection of $d$ identical bases, one can use
`tensap.FunctionalBases.duplicate(basis, d)`.

Similarly to {py:class}`~tensap.FunctionalBasis`, the basis functions of a
{py:class}`~tensap.FunctionalBases` object `H` can be evaluated with `H.eval(x)`, as
well as their $i$-th order derivative with `H.eval_derivative(i,
x)`.

## `FunctionalBasisArray`

Given a basis of functions $\{\phi_{i}\}_{i \in I}$, a
{py:class}`~tensap.FunctionalBasisArray` object represents a function $f$ that
writes

$$f(x) = \sum_{i \in I} a_i \phi_i(x),$$

with some coefficients $a_i$, $i \in I$, and can be created
with the command `f = tensap.FunctionalBasisArray(a, basis, shape)`,
with `shape` the output shape of $f$.

A {py:class}`~tensap.FunctionalBasisArray` is a {py:class}`~tensap.Function`. It can be evaluated with
the command `f.eval(x)`, and one can obtain its derivatives with
`f.eval_derivative(n, x)`.

## `FunctionalTensor`

Given $d$ bases of functions
$\{\phi^\nu_{i_\nu}\}_{i_\nu \in I^\nu}$,
$\nu = 1, \ldots, d$, and a tensor
$a \in {\mathbb{R}}^{I^1 \times \cdots \times I^d}$, a
{py:class}`~tensap.FunctionalTensor` object represents a function $f$ that writes

$$f(x) = \sum_{i_1 \in I^1} \cdots \sum_{i_d \in I^d} a_{i_1, \ldots, i_d} \phi^1_{i_1}(x_1) \cdots \phi^d_{i_d}(x_d).$$

The tensor $a$ can be in different tensor formats
({py:class}`~tensap.FullTensor`, {py:class}`~tensap.TreeBasedTensor`, \...).

A {py:class}`~tensap.FunctionalTensor` is a {py:class}`~tensap.Function`. It can be evaluated with the
command `f.eval(x)`, and one can obtain its derivatives with
`f.eval_derivative(n, x)`.

## `Tensorizer` and `TensorizedFunction`

For an introduction to tensorization of functions, see {cite:p}`Ali2020I`, {cite:p}`Ali2020II`, {cite:p}`Ali2021III`.  
We consider functions defined on the interval $I = [0,1)$. For a given $b \in \{2,3,\ldots, \}$ and $d\in {\mathbb{N}}$, an element $x \in I$ can be identified with the tuple $(i_1,\ldots,i_d,y)$, such that

$$
            x = t_{b,d}(i_1,\ldots,i_d,y) = \sum_{k=1}^d i_kb^{-k} + b^{-d}y$$

with $i_k \in I_b = \{0,\ldots,b-1\}$, $k = 1,\ldots,d$,
and $y = b^d x - \lfloor b^d x \rfloor \in [0,1)$. The tuple
$(i_1,\ldots,i_d)$ is the representation in base $b$ of
$\lfloor b^d x \rfloor$. This defines a bijective map
$t_{b,d}$ from $\{0,\ldots,b-1\}^{d} \times [0,1)$ to
$[0,1)$.

Such a mapping is represented in tensap by the object {py:class}`~tensap.Tensorizer`:
`t = tensap.Tensorizer(b, d)`. For a given $x$ in $[0,1)$,
on obtains the corresponding tuple $(i_1, ..., i_d,y)$ with the
command `= t.map(x)`. For a given tuple $(i_1, ..., i_d,y)$, on
obtains the corresponding $x$ with `t.inverse_map([i_1, ...,
i_d,y])`.

This identification is generalized to functions of $D$ variables
with `t = tensap.Tensorizer(b, d, D)`.

The map $t_{b,d}$ allows to define a tensorization map $T_{b,d}$, which associates to a univariate function $F$ defined on $[0,1)$ the multivariate function $f = F \circ t_{b,d}$ defined on $I_b^d \times I$, such that

$$f(i_1,\ldots,i_d,y) = F(t_{b,d}(i_1,\ldots,i_d,y)).$$

Such a function is represented in tensap by a {py:class}`~tensap.TensorizedFunction`,
and can be created with `f = tensap.TensorizedFunction(fun, t)`,
with `fun` a `function` or {py:class}`~tensap.Function` and `t` a
{py:class}`~tensap.Tensorizer`. The {py:class}`~tensap.TensorizedFunction` `f` is a function of
$d+1$ variables that can be evaluated with `f.eval(x)`, with
`x` a list or `numpy.ndarray` with $d+1$ columns.
