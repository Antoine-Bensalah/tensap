(theory-full-tensors)=
# Tensors and ranks

This page introduces algebraic tensors, the operations on them, and the
notions of matricization and rank on which every low-rank format is built.
For an introduction to tensor calculus, see {cite:p}`Hackbusch2019`.

## Tensors

Given $d$ index sets $I_\nu = \{1, \dots, N_\nu\}$, $1 \le \nu \le d$, and
$I = I_1 \times \cdots \times I_d$, a *tensor of order $d$* is an element of
the space $\mathbb{R}^I$, of dimension $\prod_\nu N_\nu$. It is represented
by the multidimensional array of its entries

$$
u(i) = u(i_1, \dots, i_d), \qquad i = (i_1, \dots, i_d) \in I .
$$

The *tensor product* of vectors $v^{(\nu)} \in \mathbb{R}^{I_\nu}$ is the
*elementary tensor* $v = v^{(1)} \otimes \cdots \otimes v^{(d)}$, with the
entries

$$
v(i_1, \dots, i_d) = v^{(1)}(i_1) \cdots v^{(d)}(i_d) ,
$$

and $\mathbb{R}^I = \mathbb{R}^{I_1} \otimes \cdots \otimes \mathbb{R}^{I_d}$
is the span of the elementary tensors. For $d = 2$, $v \otimes w$ is
identified with the matrix $v w^{\mathsf T}$.

The *canonical inner product* $(u, v) = \sum_{i \in I} u(i)\, v(i)$ and the
associated norm $\|u\|$, the Frobenius norm, make $\mathbb{R}^I$ a Hilbert
space. On elementary tensors, the inner product factorizes,

$$
\big(v^{(1)} \otimes \cdots \otimes v^{(d)},\;
     w^{(1)} \otimes \cdots \otimes w^{(d)}\big)
= \prod_{\nu=1}^{d} \big(v^{(\nu)}, w^{(\nu)}\big),
$$

and in particular $\|v^{(1)} \otimes \cdots \otimes v^{(d)}\| =
\prod_\nu \|v^{(\nu)}\|$ (crossnorm property). This identity is behind all
the inner products of the low-rank formats.

**Storage.** A full tensor stores its $\prod_\nu N_\nu$ entries: this is
{py:class}`~tensap.FullTensor`. A *sparse* tensor stores only its non-zero
entries and their indices ({py:class}`~tensap.SparseTensor`), and a
*diagonal* tensor, with $N_1 = \dots = N_d = N$ and $u(i_1, \dots, i_d) = 0$
unless $i_1 = \dots = i_d$, stores its $N$ diagonal entries
({py:class}`~tensap.DiagonalTensor`). A diagonal tensor is the core of a
tensor in canonical format ({ref}`theory-tucker-canonical`).

## Operations

Let $u \in \mathbb{R}^{I_1 \times \cdots \times I_d}$ and
$v \in \mathbb{R}^{J_1 \times \cdots \times J_e}$.

- **Outer product.** $u \otimes v$ is the tensor of order $d + e$ with the
  entries $u(i_1, \dots, i_d)\, v(j_1, \dots, j_e)$.
- **Kronecker product.** For $e = d$, the Kronecker product is the tensor of
  order $d$ and size $N_1 M_1 \times \cdots \times N_d M_d$ with the entries
  $u(i_1, \dots, i_d)\, v(j_1, \dots, j_d)$, the pair $(i_\nu, j_\nu)$ being
  merged into one index for each $\nu$.
- **Hadamard product.** For two tensors of the same size, $u \odot v$ is the
  tensor of the entrywise products, $(u \odot v)(i) = u(i)\, v(i)$.
- **Contracted product.** For sets of dimensions $\alpha$ of $u$ and $\beta$
  of $v$ with the same sizes, the contraction of $u$ and $v$ along $\alpha$
  and $\beta$ sums over the indices of these dimensions, paired in order. For
  instance, for two tensors of order 4,

  $$
  w(i_3, i_4, j_1, j_4) = \sum_{i_1, i_2}
    u(i_1, i_2, i_3, i_4)\, v(j_1, i_1, i_2, j_4)
  $$

  is the contraction along $\alpha = \{1, 2\}$ and $\beta = \{2, 3\}$. The
  inner product $(u, v)$ is the contraction along all the dimensions.
- **Products with matrices.** For a matrix $M$ with as many columns as
  $N_\nu$, the *mode product* $u \times_\nu M$ applies $M$ to the dimension
  $\nu$:

  $$
  (u \times_\nu M)(i_1, \dots, i'_\nu, \dots, i_d)
  = \sum_{i_\nu} M(i'_\nu, i_\nu)\, u(i_1, \dots, i_\nu, \dots, i_d).
  $$

  Products in different dimensions commute. The products with one matrix
  per dimension, $u \times_1 M^{(1)} \cdots \times_d M^{(d)}$, apply the
  Kronecker product $M^{(1)} \otimes \cdots \otimes M^{(d)}$ to $u$, and the
  products with vectors contract the dimensions.

Some computations only need the *diagonal* of a result (the entries whose
indices are equal in some pairs of dimensions), which can be obtained at a
much lower cost than the full result: the diagonal of a tensor, or the
diagonal of a contracted product or of products with matrices.

## Matricizations and ranks

For a non-empty set of dimensions $\alpha \subset D = \{1, \dots, d\}$ and
its complement $\alpha^c = D \setminus \alpha$, the *$\alpha$-matricization*
$M_\alpha(u)$ is the matrix of the entries of $u$, with the indices
$i_\alpha = (i_\nu)_{\nu \in \alpha}$ as rows and $i_{\alpha^c}$ as columns:

$$
\begin{aligned}
M_\alpha(u)\big(i_\alpha, i_{\alpha^c}\big) &= u(i_1, \dots, i_d), \\
M_\alpha(u) &\in \mathbb{R}^{N_\alpha \times N_{\alpha^c}},
\qquad N_\alpha = \prod_{\nu \in \alpha} N_\nu .
\end{aligned}
$$

It identifies $u$ with an element of
$\mathbb{R}^{I_\alpha} \otimes \mathbb{R}^{I_{\alpha^c}}$, a tensor of order
two. The multi-indices $i_\alpha$ are numbered in a fixed order; the
vectorization of $u$ is the case of a single group of all the dimensions.

```{prf:definition} $\alpha$-rank, minimal subspace
:label: def-alpha-rank

The *$\alpha$-rank* of $u$ is $\rank_\alpha(u) = \rank M_\alpha(u)$, the
smallest $r$ such that $u = \sum_{k = 1}^{r} v_k \otimes w_k$ with
$v_k \in \mathbb{R}^{I_\alpha}$ and $w_k \in \mathbb{R}^{I_{\alpha^c}}$. The
*minimal subspace* $U^{\min}_\alpha(u)$ is the column space of
$M_\alpha(u)$, the smallest subspace $U_\alpha$ such that
$u \in U_\alpha \otimes \mathbb{R}^{I_{\alpha^c}}$, and
$\rank_\alpha(u) = \dim U^{\min}_\alpha(u)$. Moreover
$\rank_\alpha(u) = \rank_{\alpha^c}(u)$.
```

For a collection $T$ of subsets of $D$, the tuple
$\rank_T(u) = (\rank_\alpha(u))_{\alpha \in T}$ is the *$T$-rank* of $u$.
The Tucker rank is the case of the singletons, the tensor-train ranks that of
the sets $\{1, \dots, \nu\}$, and the tree-based ranks that of the nodes of a
dimension tree ({ref}`theory-tree-based-tensors`).

**$\alpha$-singular values.** The singular value decomposition of the
matricization,

$$
M_\alpha(u) = \sum_{k \ge 1} \sigma^\alpha_k\, v^\alpha_k \otimes w^\alpha_k ,
$$

with orthonormal families $(v^\alpha_k)$ and $(w^\alpha_k)$ and
$\sigma^\alpha_1 \ge \sigma^\alpha_2 \ge \dots \ge 0$, defines the
*$\alpha$-singular values* $\sigma^\alpha_k$ and the *$\alpha$-principal
components* $v^\alpha_k$ of $u$. The $\alpha$-rank is the number of non-zero
$\alpha$-singular values, and the truncation of this decomposition gives the
best approximation with a bounded $\alpha$-rank ({ref}`theory-truncation`).

**The canonical rank.** The smallest number of elementary tensors whose sum
is $u$ is its *canonical rank* $\rank(u)$, and
$\rank_\alpha(u) \le \rank(u)$ for every $\alpha$
({ref}`theory-tucker-canonical`). For $d = 2$, all these ranks coincide
with the rank of the matrix.

## In tensap

- {py:class}`~tensap.FullTensor`, {py:class}`~tensap.SparseTensor`,
  {py:class}`~tensap.DiagonalTensor`.
- The operations above are the methods `tensordot` (outer and contracted
  products), `kron`, the product `*` (Hadamard), `dot`,
  `tensor_matrix_product` and `tensor_vector_product`; the diagonals of
  results are given by `eval_diag`, `tensordot_eval_diag` and
  `tensor_matrix_product_eval_diag`.
- Matricizations and singular values: `matricize(alpha)`,
  `singular_values()`, `alpha_principal_components(alpha)`, and
  `orth(alpha)` for a representation whose $\alpha$-matricization has
  orthonormal columns. `FullTensor.reshape` and the matricizations number
  the multi-indices in column-major order, the first index varying fastest,
  as in Matlab.
- Example: {ref}`sphx_glr_examples_tensors_full_tensor.py`.
