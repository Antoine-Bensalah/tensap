(theory-tucker-canonical)=
# Tucker and canonical formats

A full tensor $u \in \mathbb{R}^{N_1 \times \cdots \times N_d}$ has
$\prod_\nu N_\nu$ entries: $10^{12}$ numbers for $N_\nu = 100$ and $d = 6$.
Low-rank formats store $u$ through small objects whose size grows linearly
with $d$, by exploiting the fact that $u$ often lives in a much smaller tensor
space $U_1 \otimes \cdots \otimes U_d$, with $U_\nu \subset \mathbb{R}^{N_\nu}$.
This page defines the Tucker format and its special case, the canonical
format, for tensors and for operators, and the arithmetic on them.

## Notation

$V = \mathbb{R}^{N_1} \otimes \cdots \otimes \mathbb{R}^{N_d}$ is the tensor
space, with the canonical inner product $(u, v) = \sum_i u(i)\, v(i)$ and the
norm $\|u\| = (u, u)^{1/2}$. On elementary tensors the inner product
factorizes,

$$
\big(v^{(1)} \otimes \cdots \otimes v^{(d)},\;
     w^{(1)} \otimes \cdots \otimes w^{(d)}\big)
= \prod_{\nu=1}^{d} \big(v^{(\nu)}, w^{(\nu)}\big),
$$

by expanding the sum over $(i_1, \dots, i_d)$ as a product of sums. For a
dimension $\nu$, the matricization $M_\nu(u) \in \mathbb{R}^{N_\nu \times
\prod_{\mu \ne \nu} N_\mu}$ arranges the entries of $u$ with $i_\nu$ as row
index, and $\rank_\nu(u) = \rank M_\nu(u)$. For a tensor $C$ and a matrix $Q$
with as many columns as the size of the dimension $\nu$ of $C$, the mode
product $C \times_\nu Q$ applies $Q$ to the dimension $\nu$:
$M_\nu(C \times_\nu Q) = Q\, M_\nu(C)$. Products in different dimensions
commute, and $\operatorname{vec}(C \times_1 Q_1 \cdots \times_d Q_d) =
(Q_1 \otimes \cdots \otimes Q_d) \operatorname{vec}(C)$, the Kronecker product
corresponding to the tensor product through the row-major (C) order.

## The Tucker format

```{prf:definition} Tucker format
:label: def-tucker

Given, for each dimension $\nu$, a family of $R_\nu$ vectors
$v^{(\nu)}_1, \dots, v^{(\nu)}_{R_\nu} \in \mathbb{R}^{N_\nu}$, called a
*frame* and gathered as the columns of a matrix
$U^{(\nu)} \in \mathbb{R}^{N_\nu \times R_\nu}$, and a *core*
$C \in \mathbb{R}^{R_1 \times \cdots \times R_d}$, the Tucker tensor
$(C, \{U^{(\nu)}\})$ is

$$
u = \sum_{k_1 = 1}^{R_1} \cdots \sum_{k_d = 1}^{R_d}
    C(k_1, \dots, k_d)\; v^{(1)}_{k_1} \otimes \cdots \otimes v^{(d)}_{k_d}
  = C \times_1 U^{(1)} \times_2 \cdots \times_d U^{(d)} .
$$

The tuple $(R_1, \dots, R_d)$ is its *representation rank*. Its storage is
$\sum_\nu N_\nu R_\nu + \prod_\nu R_\nu$.
```

The core may itself be stored in a structured format: full, diagonal (the
canonical format below), sparse, or tree-based. In tensap, a Tucker tensor
is a {py:class}`~tensap.TuckerTensorVector`, its frames a
{py:class}`~tensap.TFrameVector` made of one {py:class}`~tensap.FrameVector`
per dimension.

### Which ranks are needed

```{prf:proposition} Tucker rank
:label: prop-tucker-rank

A tensor $u$ has a Tucker representation with ranks $(R_\nu)$ if and only if
$R_\nu \ge r_\nu := \rank_\nu(u)$ for all $\nu$. The tuple
$(r_1, \dots, r_d)$ is the *Tucker rank* of $u$.
```

```{prf:proof}
Matricizing the definition in the dimension $\nu$ gives

$$
M_\nu(u) = U^{(\nu)}\, M_\nu(C)
\Big(\bigotimes_{\mu \ne \nu} U^{(\mu)}\Big)^{\mathsf T},
$$

so that the columns of $M_\nu(u)$ lie in the range of $U^{(\nu)}$, and
$R_\nu \ge r_\nu$. Conversely, let $U^{(\nu)}$ be an orthonormal basis of the
column space of $M_\nu(u)$ (its left singular vectors), and
$C = u \times_1 U^{(1)\mathsf T} \cdots \times_d U^{(d)\mathsf T}$. Then
$C \times_1 U^{(1)} \cdots \times_d U^{(d)} = u \times_1 P_1 \cdots \times_d
P_d$ with the orthogonal projections $P_\nu = U^{(\nu)} U^{(\nu)\mathsf T}$,
and each $P_\nu$ leaves $u$ unchanged, since it leaves the columns of
$M_\nu(u)$ unchanged.
```

The Tucker rank is computable by $d$ singular value decompositions; this is
the starting point of the truncation by higher-order singular value
decomposition (HOSVD).

### Non-uniqueness and orthonormal frames

For invertible matrices $Q^{(\nu)} \in \mathbb{R}^{R_\nu \times R_\nu}$,

$$
C \times_1 U^{(1)} \cdots \times_d U^{(d)}
= \big(C \times_1 Q^{(1)} \cdots \times_d Q^{(d)}\big)
  \times_1 U^{(1)} Q^{(1)-1} \cdots \times_d U^{(d)} Q^{(d)-1},
$$

since $(C \times_\nu Q) \times_\nu (U Q^{-1}) = C \times_\nu U$: the
representation is defined up to this change of bases. It is used to make the
frames orthonormal. With the QR factorizations $U^{(\nu)} = Q^{(\nu)}
T^{(\nu)}$, $u = (C \times_1 T^{(1)} \cdots \times_d T^{(d)}) \times_1 Q^{(1)}
\cdots \times_d Q^{(d)}$, and since a Kronecker product of matrices with
orthonormal columns has orthonormal columns,

$$
\|u\| = \Big\|\Big(\bigotimes_\nu Q^{(\nu)}\Big)
        \operatorname{vec}(C')\Big\| = \|C'\|,
\qquad C' = C \times_1 T^{(1)} \cdots \times_d T^{(d)} .
$$

With orthonormal frames, the norms, inner products and truncations of $u$ are
those of its small core.

## The canonical format

```{prf:definition} Canonical format
:label: def-canonical

The canonical format is the Tucker format with a *diagonal* core:
$R_\nu = R$ for all $\nu$, and $C(k, \dots, k) = \sigma_k$, the other entries
being zero,

$$
u = \sum_{k = 1}^{R} \sigma_k\; v^{(1)}_k \otimes \cdots \otimes v^{(d)}_k ,
$$

a sum of $R$ elementary tensors, with storage $R \sum_\nu N_\nu + R$. The
smallest such $R$ is the *canonical rank* $\rank(u)$, and $\mathcal{R}_r$
denotes the set of tensors of canonical rank at most $r$.
```

In tensap, the canonical format is a {py:class}`~tensap.TuckerTensorVector`
whose core is a {py:class}`~tensap.DiagonalTensor`.

```{prf:proposition}
:label: prop-canonical-ranks

For every subset $\alpha$ of dimensions, $\rank_\alpha(u) \le \rank(u)$; the
converse inequality fails in general.
```

```{prf:proof}
Grouping the factors of each term, $M_\alpha(u) = \sum_k \sigma_k
(\bigotimes_{\nu \in \alpha} v^{(\nu)}_k) \otimes (\bigotimes_{\nu \notin
\alpha} v^{(\nu)}_k)$ is a sum of at most $R$ rank-one matrices. A full core
of Tucker rank $(r, \dots, r)$, on the other hand, needs in general many
more than $r$ elementary tensors.
```

### The set of canonical tensors is not closed

For $d \ge 3$, the set $\mathcal{R}_r$ is not closed. The classical example,
with independent vectors $a$ and $b$ and $\varepsilon > 0$, is

$$
u_\varepsilon = \frac{1}{\varepsilon}
  \big((a + \varepsilon b)^{\otimes 3} - a^{\otimes 3}\big)
= a \otimes a \otimes b + a \otimes b \otimes a + b \otimes a \otimes a
  + O(\varepsilon).
$$

Each $u_\varepsilon$ has rank 2, while the limit has rank 3: the limit is
approximated arbitrarily well by tensors of rank 2, whose terms blow up like
$1/\varepsilon$, and it has no best approximation of rank 2. This is why the
canonical format has no truncation by singular value decompositions: it is
replaced by greedy algorithms. The sets of tensors with bounded Tucker rank,
and more generally the tree-based formats, are closed, and have one
{cite:p}`Hackbusch2019`.

## Operators

The same formats apply to operators, with matrices
$A^{(\nu)}_k \in \mathbb{R}^{N_\nu \times N'_\nu}$ in place of vectors, and
Kronecker products:

$$
A = \sum_{k} C_A(k)\;
    A^{(1)}_{k_1} \otimes \cdots \otimes A^{(d)}_{k_d},
\qquad
\Big(\bigotimes_\nu A^{(\nu)}\Big) \Big(\bigotimes_\nu v^{(\nu)}\Big)
= \bigotimes_\nu A^{(\nu)} v^{(\nu)} .
$$

In tensap, an operator in Tucker format is a
{py:class}`~tensap.TuckerTensorOperator`, with frames of matrices
({py:class}`~tensap.FrameOperator`).

### The Laplacian

```{prf:example} Laplacian-like operators
:label: ex-laplacian

The discrete Laplacian on a uniform tensor grid of $(0, 1)^d$ is

$$
A = \sum_{\mu = 1}^{d}
    M \otimes \cdots \otimes M \otimes \underbrace{K}_{\mu}
    \otimes M \otimes \cdots \otimes M ,
$$

with $K = h^{-2}\,\mathrm{tridiag}(-1, 2, -1)$ and $M = I$ for finite
differences, and the 1D stiffness and mass matrices for finite elements. It
has three exact low-rank representations:

- canonical, with $d$ terms: in the dimension $\mu$, the term $k$ is $K$ if
  $k = \mu$ and $M$ otherwise, and $\sigma = (1, \dots, 1)$;
- Tucker, of rank 2: the frames $\{K, M\}$ in every dimension, and the core
  $C \in \mathbb{R}^{2 \times \cdots \times 2}$ equal to 1 where exactly one
  index selects $K$, and to 0 elsewhere ($d$ non-zero entries among $2^d$);
- tree-based, with the same frames and a tree-based core of ranks 2.

{py:class}`~tensap.LaplacianLikeOperatorFactory` builds them from $K$ and
$M$.
```

## Arithmetic and rank growth

```{prf:proposition} Operations in Tucker format
:label: prop-tucker-arithmetic

Let $u = (C_u, \{v^{(\nu)}_k\})$ and $w = (C_w, \{w^{(\nu)}_l\})$ be Tucker
tensors of representation ranks $(R_\nu)$ and $(R'_\nu)$, and
$A = (C_A, \{A^{(\nu)}_k\})$ an operator of representation ranks
$(R^A_\nu)$.

1. $u + w$ has the concatenated frames
   $\{v^{(\nu)}_1, \dots, v^{(\nu)}_{R_\nu}, w^{(\nu)}_1, \dots,
   w^{(\nu)}_{R'_\nu}\}$ and the block-diagonal core of $C_u$ and $C_w$: its
   ranks are $R_\nu + R'_\nu$.
2. $A u$ has the frames $\{A^{(\nu)}_k v^{(\nu)}_l\}_{k, l}$ and the core
   $C_A \otimes C_u$: its ranks are $R^A_\nu R_\nu$.
3. $(u, w) = \big(C_u,\; C_w \times_1 G^{(1)} \cdots \times_d G^{(d)}\big)$,
   with the Gram matrices $G^{(\nu)}(k, l) = (v^{(\nu)}_k, w^{(\nu)}_l)$.
```

```{prf:proof}
(1) The sum of the two expansions is an expansion on the concatenated frames,
whose coefficients are $C_u$ in the block of the first $R_\nu$ indices,
$C_w$ in the block of the last $R'_\nu$, and zero in the mixed blocks.
(2) By multilinearity, $A u = \sum_{k, l} C_A(k)\, C_u(l)
\bigotimes_\nu A^{(\nu)}_{k_\nu} v^{(\nu)}_{l_\nu}$: the pair
$(k_\nu, l_\nu)$ indexes the new frame, and $C_A(k) C_u(l)$ is the
Kronecker product of the cores. (3) By bilinearity and the factorization of
the inner product on elementary tensors,

$$
(u, w) = \sum_{k, l} C_u(k)\, C_w(l)
         \prod_\nu \big(v^{(\nu)}_{k_\nu}, w^{(\nu)}_{l_\nu}\big)
       = \sum_k C_u(k) \sum_l \prod_\nu G^{(\nu)}(k_\nu, l_\nu)\, C_w(l),
$$

and the inner sum is $(C_w \times_1 G^{(1)} \cdots \times_d G^{(d)})(k)$.
```

Sums and products are exact and cheap, but the ranks grow: an iterative
method must either truncate its iterates, or build the solution with a
controlled rank.

**Diagonal cores stay diagonal.** The Kronecker product and the
block-diagonal sum of diagonal tensors are diagonal: in canonical format,
$A v$ and $b - A v$ remain canonical, of ranks $R^A R$ and $R^b + R^A R$.
The inner product needs no full core: with $C_u = \operatorname{diag}(\sigma)$
and $C_w = \operatorname{diag}(\tau)$, only the terms with all their indices
equal remain,

$$
(u, w) = \sigma^{\mathsf T}
         \big(G^{(1)} \odot \cdots \odot G^{(d)}\big)\, \tau ,
$$

with $\odot$ the entrywise (Hadamard) product. With a full core, on the
contrary, the core of $A u$ has $\prod_\nu R^A_\nu R_\nu$ entries: $(2m)^d$
for the Tucker Laplacian applied to a tensor of rank $(m, \dots, m)$.

## Cost

With $N$ a bound on the $N_\nu$ and $R$ on the representation ranks:

| Format | Storage | Inner product $(u, w)$ |
| --- | --- | --- |
| full | $N^d$ | $N^d$ |
| Tucker, full core | $d N R + R^d$ | $d N R^2 + d R^{d+1}$ |
| canonical | $d N R + R$ | $d N R^2$ |

The canonical format is linear in $d$ everywhere. The Tucker format keeps a
core whose size is exponential in $d$; for large $d$, the core itself must be
stored in a low-rank format, tree-based or tensor train
{cite:p}`Hackbusch2019,Kolda2009`.

## In tensap

- Classes: {py:class}`~tensap.TuckerTensorVector`,
  {py:class}`~tensap.TuckerTensorOperator`, {py:class}`~tensap.FrameVector`,
  {py:class}`~tensap.FrameOperator`, {py:class}`~tensap.TFrameVector`,
  {py:class}`~tensap.TFrameOperator`,
  {py:class}`~tensap.LaplacianLikeOperatorFactory`.
- Example: {ref}`sphx_glr_examples_tensors_tucker_tensor.py`.
