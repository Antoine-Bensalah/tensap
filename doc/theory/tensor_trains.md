(theory-tensor-trains)=
# Tensor trains

The core of a Tucker tensor has $\prod_\nu R_\nu$ entries, exponential in
the dimension ({ref}`theory-tucker-canonical`). The tensor-train (TT) format
{cite:p}`Oseledets2011` replaces it by a chain of tensors of order three,
with a storage linear in $d$ and a truncation by singular value
decompositions, which the canonical format lacks. In quantum physics, it is
the matrix product state.

## The format

```{prf:definition} Tensor-train format
:label: def-tt

A tensor in TT format is given by cores
$C^\nu \in \mathbb{R}^{r_{\nu-1} \times N_\nu \times r_\nu}$, with
$r_0 = r_d = 1$:

$$
u(i_1, \dots, i_d)
= \sum_{k_1 = 1}^{r_1} \cdots \sum_{k_{d-1} = 1}^{r_{d-1}}
  C^1(i_1, k_1)\, C^2(k_1, i_2, k_2) \cdots C^d(k_{d-1}, i_d)
= C^1(i_1)\, C^2(i_2) \cdots C^d(i_d),
$$

with $C^\nu(i_\nu) \in \mathbb{R}^{r_{\nu-1} \times r_\nu}$ a matrix (the
first one a row, the last one a column). The $r_\nu$ are its TT ranks, and
its storage is $\sum_\nu r_{\nu-1} N_\nu r_\nu = O(d N r^2)$.

An operator in TT format has cores
$A^{(\nu)} \in \mathbb{R}^{r_{\nu-1} \times N_\nu \times N'_\nu \times
r_\nu}$, that is matrices $A^{(\nu)}_{k k'} \in \mathbb{R}^{N_\nu \times
N'_\nu}$, and
$A = \sum_k A^{(1)}_{1, k_1} \otimes A^{(2)}_{k_1, k_2} \otimes \cdots
\otimes A^{(d)}_{k_{d-1}, 1}$.
```

Seen with the paired index $(i_\nu, j_\nu)$, of size $N_\nu N'_\nu$, an
operator is a TT tensor: every vector-space operation is the same for
vectors and for operators. The TT format is the tree-based format of the
linear tree $\{1\}, \{1, 2\}, \dots, \{1, \dots, d\}$; it is stored as a list
of cores because every TT algorithm is a sweep along this chain.

```{prf:example} The Laplacian in TT format
:label: ex-laplacian-tt

With block matrices whose entries are matrices, multiplied with the
Kronecker product of the entries (the strong Kronecker product $\bowtie$),
the Laplacian
$A = \sum_\mu I \otimes \cdots \otimes K \otimes \cdots \otimes I$ is

$$
A = \begin{pmatrix} K & I \end{pmatrix} \bowtie
    \begin{pmatrix} I & 0 \\ K & I \end{pmatrix}^{\bowtie (d-2)} \bowtie
    \begin{pmatrix} I \\ K \end{pmatrix} ,
$$

a TT operator of ranks 2. The same formula with the canonical vectors $e_1$,
$e_2$ in place of $K$ and $I$ is a TT core of the Tucker representation of
{prf:ref}`ex-laplacian`.
```

## Ranks and the TT-SVD

Grouping the first $\nu$ cores and the last $d - \nu$,

$$
u(i_1, \dots, i_d) = \sum_{k = 1}^{r_\nu}
\underbrace{C^1(i_1) \cdots C^\nu(i_\nu)}_{C^{\le \nu}(i_1 \dots i_\nu,\, k)}\;
\underbrace{C^{\nu+1}(i_{\nu+1}) \cdots C^d(i_d)}_{C^{> \nu}(k,\, i_{\nu+1} \dots i_d)}:
$$

the matricization $M_{\le \nu}(u)$, with the first $\nu$ indices as rows, is
the product $C^{\le \nu} C^{> \nu}$, of rank at most $r_\nu$. Conversely, the
TT-SVD builds a representation with $r_\nu = \rank M_{\le \nu}(u)$: the
minimal TT ranks are the ranks of these sequential matricizations.

```{prf:algorithm} TT-SVD
:label: alg-tt-svd

**Inputs** a full tensor $u$, a relative precision $\varepsilon$

1. Truncated SVD of $M_{\le 1}(u) = U_1 \Sigma_1 W_1^{\mathsf T}$ at the
   precision $\varepsilon / \sqrt{d - 1}$; $U_1$ is the first core.
2. Reshape $\Sigma_1 W_1^{\mathsf T}$, of size $r_1 \times (N_2 \cdots N_d)$,
   into $(r_1 N_2) \times (N_3 \cdots N_d)$, and repeat on it.
```

As for the higher-order SVD ({ref}`theory-truncation`), the $d - 1$ errors
are orthogonal, so that $\|u - u_r\| \le \varepsilon \|u\|$, and the result
is quasi-optimal with the factor $\sqrt{d - 1}$ {cite:p}`Oseledets2011`.

## Orthogonality

A core is *left-orthogonal* if its matricization of size
$r_{\nu-1} N_\nu \times r_\nu$ has orthonormal columns, and
*right-orthogonal* if its matricization of size
$r_{\nu-1} \times N_\nu r_\nu$ has orthonormal rows. If $C^1, \dots, C^\nu$
are left-orthogonal, the interface $C^{\le \nu}$ has orthonormal columns (by
induction, it is a product of matrices with orthonormal columns), and
symmetrically on the right. With an *orthogonality center* $\mu$, the cores
before $\mu$ being left-orthogonal and those after right-orthogonal,

$$
\operatorname{vec}(u) = \big(C^{\le \mu - 1} \otimes I_{N_\mu} \otimes
C^{> \mu\, \mathsf T}\big) \operatorname{vec}(C^\mu)
\quad \text{so that} \quad \|u\| = \|C^\mu\| ,
$$

and a change of $C^\mu$ changes $u$ by the same amount in norm. The
orthogonalization is a sweep of QR factorizations: from the left,
$C^\nu = Q R$, $C^\nu \leftarrow Q$, $C^{\nu+1} \leftarrow R\, C^{\nu+1}$,
which does not change the product $C^\nu(i_\nu) C^{\nu+1}(i_{\nu+1})$.

## Operations on the cores

**Inner product.** For $u$ and $w$ in TT format, $(u, w)$ is a product of
$d$ matrices, contracted from the left without forming the tensors:

$$
Z_0 = 1, \qquad
Z_\nu = \sum_{i_\nu} C^\nu_u(i_\nu)^{\mathsf T} Z_{\nu-1} C^\nu_w(i_\nu),
\qquad (u, w) = Z_d ,
$$

in $O(d N r^3)$ operations.

**Sums and products.** Products of block matrices give the cores directly:

- $u + w$ has the block-diagonal cores
  $\begin{pmatrix} C^\nu_u(i) & 0 \\ 0 & C^\nu_w(i) \end{pmatrix}$ (the
  first one a row, the last one a column): ranks $r_\nu + r'_\nu$;
- the entrywise (Hadamard) product $u \odot w$ has the cores
  $C^\nu_u(i) \otimes C^\nu_w(i)$, since the product of Kronecker products
  is the Kronecker product of the products: ranks $r_\nu r'_\nu$;
- $A u$ has the cores $\sum_j A^{(\nu)}(i, j) \otimes C^\nu_u(j)$: ranks
  $r^A_\nu r_\nu$. The same formula composes two operators.

## Rounding

```{prf:algorithm} TT rounding
:label: alg-tt-rounding

**Inputs** the cores of $u$, a relative precision $\varepsilon$

1. Orthogonalize $u$ with the center on the first core.
2. For $\nu = 1, \dots, d - 1$: truncated SVD
   $C^\nu \approx U \Sigma W^{\mathsf T}$ of the matricization of size
   $r_{\nu-1} N_\nu \times r_\nu$, at the precision
   $\varepsilon / \sqrt{d - 1}$; $C^\nu \leftarrow U$, and
   $C^{\nu+1} \leftarrow \Sigma W^{\mathsf T} C^{\nu+1}$, which moves the
   center to $\nu + 1$.
```

At the step $\nu$, the cores on the left are left-orthogonal and those on
the right right-orthogonal: the matricization of $C^\nu$ has the singular
values of $M_{\le \nu}(u)$, the step is a step of the TT-SVD, and the same
bound holds. A redundant representation, such as $u + u$, is reduced to its
minimal ranks.

## Entrywise product at a given precision

The exact product $z = u \odot w$ has ranks $r_u r_w$, and its rounding
orthogonalizes cores of that size: $O(d N r^6)$ for $r_u = r_w = r$. It can
be computed directly at the precision $\varepsilon$ by a two-site fit of the
projection, in the manner of the two-site DMRG algorithm for linear
systems. With the cores of $z$ left-orthogonal on the
left of the bond $(\nu, \nu + 1)$ and right-orthogonal on its right, the best
pair of cores is given by the supercore

$$
S(s, i, j, s') = \sum L_\nu(s, a, b)\, U^\nu(a, i, a')\, W^\nu(b, i, b')\,
U^{\nu+1}(a', j, a'')\, W^{\nu+1}(b', j, b'')\, R_{\nu+1}(s', a'', b''),
$$

where $L_\nu$ and $R_{\nu+1}$ contract the left and right parts of $z$ with
those of $u$ and $w$. Its truncated SVD at the precision
$\varepsilon / \sqrt{d - 1}$ gives the two cores and the rank of the bond.
Every contraction involves one core of each of $z$, $u$ and $w$, for
$O(d N^2 r^4)$ operations per sweep. The sweeps alternate, from an initial
guess (the rounding of $u$), until $z$ stagnates. If the product has no
component on the bases of the initial guess, a supercore is zero and the fit
cannot proceed: the exact product is then rounded instead.

In the numerical experiments of tensap, on smooth two-dimensional fields in
QTT format, this fit was about 4 to 16 times faster than the rounding of the exact
product for the same achieved precision; this is an observation, which
depends on the ranks.

## Tucker tensors with a TT core

A Tucker tensor whose core $G$ is a TT tensor over the Tucker ranks is a TT
tensor in the physical variables, with the cores
$C^\nu(a, i, b) = \sum_l G^\nu(a, l, b)\, U^{(\nu)}(i, l)$. Its truncation
orthonormalizes the frames, rounds $G$, then, for each dimension $\nu$ with
the center of $G$ on $\nu$, truncates the SVD of the matricization of
$G^\nu$ along its middle index, $W \Sigma Z^{\mathsf T}$: the frame is
rotated by $W$, and $G^\nu$ replaced by $\Sigma Z^{\mathsf T}$. These
$2d - 1$ truncations are orthogonal projections of the same kind, hence the
precision $\varepsilon / \sqrt{2d - 1}$ for each of them.

**The canonical format embeds into TT.** A diagonal tensor
$\sum_k \sigma_k e_k \otimes \cdots \otimes e_k$ has TT ranks $R$, with the
cores $C^\nu(a, i, b) = \delta_{a i b}$ (the first one multiplied by
$\operatorname{diag} \sigma$): a canonical tensor is a Tucker tensor with a
TT core of ranks $R$.

## Cost

With $N$ and $r$ bounds on the sizes and the ranks: storage $d N r^2$;
inner product, orthogonalization and rounding $O(d N r^3)$; $A u$ multiplies
the ranks and costs $O(d N N' r_A^2 r^2)$.

## In tensap

- {py:class}`~tensap.TTTensor`, {py:class}`~tensap.TTOperator`; rounding and
  the product at a given precision by a {py:class}`~tensap.Truncator`
  (`truncate`, `hadamard_product`).
- The Laplacian in TT format:
  {py:class}`~tensap.LaplacianLikeOperatorFactory` (`make_tt_operator`, and
  `make_tt_tucker_tensor` for the Tucker format with a TT core).
- Example: {ref}`sphx_glr_examples_tensors_tt_tensor.py`.
