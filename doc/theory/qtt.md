(theory-qtt)=
# Tensorization and quantized tensor trains

A vector of $N = b^L$ values can be seen as a tensor of order $L$ with modes
of size $b$, by writing its index in base $b$. For the samples of many
smooth functions, and for the matrices of many discretized operators, this
tensor has small tensor-train ranks, independent of $L$: the storage grows
like $\log N$ instead of $N$ {cite:p}`Khoromskij2011,Oseledets2010`. This is
the *quantized tensor train* (QTT) format.

## Tensorization

**Vectors.** A vector $v \in \mathbb{R}^N$ with $N = b^L$ is identified with
a tensor $\boldsymbol{v} \in (\mathbb{R}^b)^{\otimes L}$ by writing its index
in base $b$, most significant digit first:

$$
v(i) = \boldsymbol{v}(i_1, \dots, i_L), \qquad
i = \sum_{k=1}^{L} i_k\, b^{L-k}, \quad i_k \in \{0, \dots, b - 1\}.
$$

This is the reshape of $v$ in row-major order, a linear isometry. $b$ is the
*base* and $L$ the *resolution*. The QTT representation of $v$ is the TT
representation of $\boldsymbol{v}$ ({ref}`theory-tensor-trains`), with a
storage $L b r^2$ instead of $b^L$. A matrix acting on such vectors is
tensorized with the same digits for its rows and its columns: a TT operator
whose cores, of size $b \times b$, are indexed by the pairs of digits
$(i_k, j_k)$.

**Tensors and orderings.** A tensor of
$\mathbb{R}^{N_1} \otimes \cdots \otimes \mathbb{R}^{N_d}$ with
$N_\nu = b^{L_\nu}$ is identified with a tensor of order $\sum_\nu L_\nu$,
with the digits $(i_{\nu,1}, \dots, i_{\nu,L_\nu})$ of each coordinate. Two
orderings of the digits are used:

- *coordinate-wise*: all the digits of the first coordinate, then those of
  the second one, and so on;
- *resolution-wise*: the first digit of every coordinate, then the second
  one, and so on. With different resolutions, the digits are aligned on the
  most significant one.

The ordering has an impact on the ranks.

**Functions.** For a function $f$ on $[0, 1)$, every $x \in [0, 1)$ is
written $x = b^{-L}(i + y)$ with $i = [i_1 \dots i_L]_b$ and
$y \in [0, 1)$, so that $f$ is identified with a function of $L + 1$
variables,

$$
f(x) = f_L(i_1, \dots, i_L, y), \qquad
f_L \in (\mathbb{R}^b)^{\otimes L} \otimes \mathbb{R}^{[0, 1)},
$$

and in dimension $d$ with a function of $(L + 1) d$ variables. The map
$f \mapsto f_L$ is a linear isometry of $L^p$ spaces. Sampling $f_L$ at a
fixed $y = y_0$ gives the tensorized vector of the samples of $f$ on the grid
$x_i = b^{-L}(i + y_0)$: the nodes for $y_0 = 0$, the centres of the cells
for $y_0 = 1/2$. In tensap, this map is a {py:class}`~tensap.Tensorizer`,
which also carries the geometry of the grid and the ordering of the digits.

## Ranks of tensorized functions

```{prf:proposition} QTT ranks of elementary functions {cite:p}`Khoromskij2011`
:label: prop-qtt-ranks

For every set $\alpha$ of digits, the $\alpha$-ranks of the tensorization of

1. an exponential $e^{a x}$ are equal to 1;
2. $\cos(\omega x + \varphi)$ are equal to 2, and those of a trigonometric
   polynomial of degree $p$ are at most $2 p + 1$;
3. a polynomial of degree $p$ are at most $p + 1$.

These ranks do not depend on the resolution $L$.
```

```{prf:proof}
1. With $x = b^{-L}(\sum_k i_k b^{L-k} + y)$,
   $e^{a x} = \prod_k e^{a b^{-k} i_k}\, e^{a b^{-L} y}$ is an elementary
   tensor.
2. $e^{\mathrm{i} \omega x}$ has rank 1 for the same reason, and the cosine
   is a sum of two such terms.
3. Write $x = g(i_\alpha) + \tilde g(i_{\alpha^c}, y)$, the contributions of
   the two groups of digits. By the binomial formula,
   $q(g + \tilde g) = \sum_{j \le p} g^j h_j(\tilde g)$, a sum of $p + 1$
   products of a function of $i_\alpha$ and a function of the other
   variables.
```

Splines with a few $b$-adic breakpoints also have small ranks, and so do
some functions without classical regularity: the indicator of the Cantor
set ($b = 3$) is an elementary tensor. A function that is smooth except on
a small set, such as a front or a corner, has ranks that grow slowly with
the resolution. Approximation classes of tensor networks with tensorization
are studied in {cite:p}`AliNouy2023`.

## QTT matrices

**The shift.** Let $(S v)(j) = v(j + 1)$, the matrix with ones on the
superdiagonal. Adding 1 to an index written in base $b$ increments one digit
$k$, from $a < b - 1$ to $a + 1$, and resets the digits after it, all equal
to $b - 1$, to 0 (a carry); the digits before $k$ are unchanged.

```{prf:proposition} QTT representation of the shift {cite:p}`Kazeev2012`
:label: prop-qtt-shift

With $J = \sum_{a < b - 1} e_a e_{a+1}^{\mathsf T}$ and
$J' = e_{b-1} e_0^{\mathsf T}$,

$$
S = \sum_{k=1}^{L} I^{\otimes (k-1)} \otimes J \otimes J'^{\otimes (L-k)}
  = \begin{pmatrix} I & J \end{pmatrix} \bowtie
    \begin{pmatrix} I & J \\ 0 & J' \end{pmatrix}^{\bowtie (L - 2)} \bowtie
    \begin{pmatrix} J \\ J' \end{pmatrix} ,
$$

a TT operator of ranks 2, with $\bowtie$ the strong Kronecker product
({prf:ref}`ex-laplacian-tt`).
```

```{prf:proof}
The first equality is the digit-by-digit description above: the pair (row
digit, column digit) is $(i, i)$ before $k$, $(a, a + 1)$ at $k$, and
$(b - 1, 0)$ after $k$. The product of the block matrices sums over the
paths of the rank index: the index is 0 as long as the cores emit $I$,
switches to 1 when a core emits $J$, and stays 1 while the cores emit $J'$.
These paths are exactly the $L$ terms of the sum; the two values of the rank
index are the two states of the carry.
```

The periodic shift adds the corner $e_{N-1} e_0^{\mathsf T} =
J'^{\otimes L}$, the path that starts in the state 1, at no cost in rank.
Every power $S^c$ has rank at most 2, by the same argument applied to the
addition of a constant. Other primitives follow: the identity and the
elementary matrices $e_i e_j^{\mathsf T} = \bigotimes_k e_{i_k}
e_{j_k}^{\mathsf T}$ have rank 1; a banded Toeplitz matrix
$\sum_c t_c S^c$ is a short sum of powers of the shift, rounded, and the
tridiagonal matrix $\mathrm{tridiag}(-1, 2, -1)$ has QTT ranks 3
{cite:p}`Kazeev2012`; a dense block in a corner, a sum of elementary
matrices, has a rank independent of $N$. A banded matrix whose first and
last rows are modified, as by a boundary condition, is a banded matrix plus
two corner blocks: a low-rank correction. All these ranks are independent of
$N$.

**Assembly in dimension $d$.** With the coordinate-wise ordering, the
operator $A_1 \otimes A_2$ on tensorized arrays is the concatenation of the
chains of cores of $A_1$ and $A_2$, with a rank 1 at the junction. In a
general ordering, the cores of each factor are placed at the positions of
the digits of its coordinate, and the bonds carry the tuple of the current
ranks of all the factors. Sums such as $I \otimes T + T \otimes I$ are then
rounded.

## Cost

The quantization of a dense array is a TT-SVD of its $N$ entries, only
possible for moderate sizes. The primitives are built directly, in
$O(L b^2)$ operations; a banded matrix with $K$ diagonals costs $K$
products by the shift, followed by roundings.

## In tensap

- {py:class}`~tensap.Tensorizer`, with one resolution per dimension and the
  two orderings; {py:func}`~tensap.quantize`, {py:func}`~tensap.dequantize`,
  {py:func}`~tensap.quantize_function`, {py:func}`~tensap.quantize_operator`.
- The primitives: {py:func}`~tensap.qtt_shift`, {py:func}`~tensap.qtt_banded`,
  {py:func}`~tensap.qtt_corner`, {py:func}`~tensap.qtt_identity`,
  {py:func}`~tensap.qtt_elementary_matrix`, and the assembly in dimension $d$
  {py:func}`~tensap.qtt_layout_product`, {py:func}`~tensap.qtt_embed`.
- Example: {ref}`sphx_glr_examples_tensors_qtt.py`.
