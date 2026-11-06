(theory-truncation)=
# Truncation

Sums and operator applications make the ranks of a tensor in low-rank
format grow ({ref}`theory-tucker-canonical`). Given a tensor $u$ and a
relative precision $\varepsilon$, the truncation computes a tensor $u_r$ of
smaller ranks such that $\|u - u_r\| \le \varepsilon \|u\|$, in the
canonical (Frobenius) norm. The tool is the truncated singular value
decomposition (SVD), applied to the matricizations of $u$: the higher-order
SVD for the Tucker format, the hierarchical SVD for the tree-based formats.
For a tensor in Tucker format, only its core is decomposed.

## Truncated SVD of an order-two tensor

Every $u \in \mathbb{R}^{N_1} \otimes \mathbb{R}^{N_2}$, a matrix, has a
singular value decomposition

$$
u = \sum_{k \ge 1} \sigma_k\, u^1_k \otimes u^2_k ,
$$

with $(u^1_k)$ and $(u^2_k)$ orthonormal and the singular values
$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Its truncation
$u_r = \sum_{k \le r} \sigma_k\, u^1_k \otimes u^2_k$ satisfies
$\|u - u_r\|^2 = \sum_{k > r} \sigma_k^2$, since the elementary tensors
$u^1_k \otimes u^2_k$ are orthonormal.

```{prf:theorem} Eckart-Young
:label: thm-eckart-young

$u_r$ is a best approximation of $u$ among the tensors of rank at most $r$:
$\|u - u_r\| = \min_{\rank(v) \le r} \|u - v\|$.
```

```{prf:proof}
For $v$ of rank at most $r$ with a fixed column space $U$ of dimension $r$,
the best choice is $(P_U \otimes \mathrm{id})\, u$, the orthogonal projection
of the columns of $u$, with the error $\|u\|^2 - \|(P_U \otimes \mathrm{id})
u\|^2$. Maximizing $\|(P_U \otimes \mathrm{id}) u\|^2 = \operatorname{tr}(P_U
u u^{\mathsf T})$ over the subspaces of dimension $r$ selects the leading
eigenvectors of $u u^{\mathsf T}$ (Ky Fan's maximum principle), which are the
$u^1_k$, $k \le r$.
```

**Relative precision.** Given $\varepsilon$, the truncation keeps the
smallest $r$ such that

$$
\sum_{k > r} \sigma_k^2 \le \varepsilon^2 \sum_{k \ge 1} \sigma_k^2 ,
\qquad \text{hence} \qquad \|u - u_r\| \le \varepsilon \|u\| ,
$$

and $r$ can further be capped by a maximal rank.

## Higher-order SVD

For a set $\alpha$ of dimensions, the truncated SVD of the matricization
$M_\alpha(u)$ at rank $r_\alpha$ gives the *dominant $\alpha$-principal
subspace* $U^\alpha_{r_\alpha}$, spanned by its first $r_\alpha$ left
singular vectors. The orthogonal projection $\mathcal P_\alpha$ onto
$U^\alpha_{r_\alpha} \otimes \mathbb{R}^{N_{\alpha^c}}$ gives, by
{prf:ref}`thm-eckart-young`,

$$
\|u - \mathcal P_\alpha u\|^2
= \min_{\rank_\alpha(v) \le r_\alpha} \|u - v\|^2
= \sum_{k > r_\alpha} (\sigma^\alpha_k)^2 ,
$$

with $\sigma^\alpha_k$ the singular values of $M_\alpha(u)$, the
*$\alpha$-singular values* of $u$.

```{prf:algorithm} Higher-order SVD (HOSVD)
:label: alg-hosvd

**Inputs** a tensor $u$, a relative precision $\varepsilon$

**Output** a Tucker tensor $u_r$ with $\|u - u_r\| \le \varepsilon \|u\|$

1. For $\nu = 1, \dots, d$: compute $W^{(\nu)}$, the first $r_\nu$ left
   singular vectors of $M_\nu(u)$, with $r_\nu$ chosen for the relative
   precision $\varepsilon / \sqrt{d}$.
2. Return the Tucker tensor with the frames $W^{(\nu)}$ and the core
   $u \times_1 W^{(1)\mathsf T} \cdots \times_d W^{(d)\mathsf T}$, that is
   $u_r = (P_1 \otimes \cdots \otimes P_d)\, u$ with
   $P_\nu = W^{(\nu)} W^{(\nu)\mathsf T}$.
```

```{prf:proposition} Error of the HOSVD {cite:p}`DeLathauwer2000,Hackbusch2019`
:label: prop-hosvd

$$
\|u - u_r\|^2 \le \sum_{\nu = 1}^{d} \|u - \mathcal P_\nu u\|^2 ,
$$

so that the precision $\varepsilon / \sqrt{d}$ in each dimension gives
$\|u - u_r\| \le \varepsilon \|u\|$. Moreover $u_r$ is quasi-optimal:
$\|u - u_r\| \le \sqrt{d}\, \min \|u - v\|$ over the tensors $v$ of Tucker
rank at most $(r_1, \dots, r_d)$.
```

```{prf:proof}
The projections $\mathcal P_\nu$ in different dimensions commute. By
telescoping,

$$
u - \mathcal P_1 \cdots \mathcal P_d u
= (I - \mathcal P_1) u + \mathcal P_1 (I - \mathcal P_2) u + \dots
  + \mathcal P_1 \cdots \mathcal P_{d-1} (I - \mathcal P_d) u .
$$

The terms are mutually orthogonal: the $\nu$-th one is in the range of
$I - \mathcal P_\nu$, the following ones in the range of $\mathcal P_\nu$.
Since projections do not increase norms,
$\|u - u_r\|^2 \le \sum_\nu \|(I - \mathcal P_\nu) u\|^2$. For the
quasi-optimality, if $u^\star$ is a best approximation, each
$\|(I - \mathcal P_\nu) u\|$ is at most $\|u - u^\star\|$ by
{prf:ref}`thm-eckart-young`, since $\rank_\nu(u^\star) \le r_\nu$.
```

The *sequentially truncated* HOSVD {cite:p}`VanNieuwenhoven2012` projects
after each dimension: $W^{(1)}$ from $M_1(u)$, then $u_1 = u \times_1
W^{(1)\mathsf T}$, $W^{(2)}$ from $M_2(u_1)$, and so on. It satisfies the
same bound and is cheaper, since every SVD is computed on an already reduced
tensor.

## Hierarchical SVD

For a dimension tree $T$, the tensors of the tree-based format of ranks
$(r_\alpha)_{\alpha \in T}$ are those with $\rank_\alpha(v) \le r_\alpha$ for
every node $\alpha$. The same principle, applied from the leaves to the root,
gives the hierarchical SVD:

1. for each leaf $\alpha$, $U^\alpha$ is the dominant $\alpha$-principal
   subspace of $u$;
2. for each interior node $\alpha$ other than the root, with
   $W_\alpha = \bigotimes_{\beta} U^\beta$ the already truncated subspaces of
   its children, $U^\alpha \subset W_\alpha$ is the dominant
   $\alpha$-principal subspace of the projection of $u$ onto $W_\alpha$;
3. $u_r$ is the orthogonal projection of $u$ onto the tensor product of the
   subspaces of the children of the root.

```{prf:proposition} Error of the hierarchical SVD {cite:p}`Grasedyck2010,Hackbusch2019`
:label: prop-hsvd

$$
\|u - u_r\|^2 \le \sum_{\alpha \in T \setminus \{D\}}
                  \sum_{k > r_\alpha} (\sigma^\alpha_k)^2 ,
$$

hence $\|u - u_r\| \le C(T) \min \|u - v\|$ over the tree-based tensors of
ranks $(r_\alpha)$, with $C(T) = \sqrt{\# T - 1} \le \sqrt{2d - 2}$, and the
relative precision $\varepsilon$ is reached with the precision
$\varepsilon / C(T)$ at each node.
```

As for the HOSVD, each step is an orthogonal projection, and the errors add
up in squares. The HOSVD is the case of the Tucker tree, with
$C(T) = \sqrt{d}$. A variant from the root to the leaves
{cite:p}`Grasedyck2010` has the same guarantees; a recursive implementation
is more stable with respect to round-off errors {cite:p}`Baboulin2025`.

## Thresholding of the singular values

Instead of a rank or a precision, a threshold $\tau$ can be given. For an
order-two tensor $u = \sum_k \sigma_k u^1_k \otimes u^2_k$:

- the *hard thresholding*
  $\mathrm{HT}_\tau(u) = \sum_k \sigma_k \mathbf{1}_{\sigma_k > \tau}\,
  u^1_k \otimes u^2_k$ is a solution of
  $\min_v \|u - v\|^2 + \tau^2 \rank(v)$;
- the *soft thresholding*
  $\mathrm{ST}_\tau(u) = \sum_k (\sigma_k - \tau)_+\, u^1_k \otimes u^2_k$
  is the solution of $\min_v \frac12 \|u - v\|^2 + \tau \|\sigma(v)\|_1$,
  with the nuclear norm, a convex relaxation of the rank. As a proximal
  operator, it is *non-expansive*:
  $\|\mathrm{ST}_\tau(u) - \mathrm{ST}_\tau(v)\| \le \|u - v\|$.

For tensors of higher order, the thresholdings act on the $\alpha$-singular
values, node by node. Composed over the nodes of a tree from the root to the
leaves, the soft thresholding is still non-expansive {cite:p}`Bachmayr2016`,
which is what the analysis of iterative methods with truncation requires.
In tensap, the thresholding is applied in every truncated SVD, combined with
the relative precision (the smaller rank wins), and the nodes are processed
from the leaves to the root.

## Tucker tensors: only the core is decomposed

If the frames of a Tucker tensor $u = (C, \{U^{(\nu)}\})$ are orthonormal,

$$
M_\nu(u) = U^{(\nu)} M_\nu(C)
\Big(\bigotimes_{\mu \ne \nu} U^{(\mu)}\Big)^{\mathsf T}
$$

has the same singular values as $M_\nu(C)$, and its left singular vectors
are the $U^{(\nu)} w$, for the left singular vectors $w$ of $M_\nu(C)$:
multiplying by matrices with orthonormal columns does not change the
singular values. The truncation of $u$ reduces to that of its core.

```{prf:algorithm} Truncation of a Tucker tensor with a full core
:label: alg-truncate-tucker

**Inputs** $u = (C, \{U^{(\nu)}\})$, a relative precision $\varepsilon$

1. Orthonormalize the frames: $U^{(\nu)} = Q^{(\nu)} T^{(\nu)}$ (QR), and
   $C \leftarrow C \times_1 T^{(1)} \cdots \times_d T^{(d)}$.
2. For $\nu = 1, \dots, d$: compute $W^{(\nu)}$, the left singular vectors
   of $M_\nu(C)$ at the relative precision $\varepsilon / \sqrt{d}$.
3. Return the core $C \times_1 W^{(1)\mathsf T} \cdots \times_d
   W^{(d)\mathsf T}$ and the frames $Q^{(\nu)} W^{(\nu)}$.
```

The cost is that of the QR factorizations of the frames,
$\sum_\nu N_\nu R_\nu^2$, and of the SVDs of the matricizations of the core,
$\sum_\nu R_\nu \prod_\mu R_\mu$: nothing depends on $\prod_\nu N_\nu$, but
the core still has $\prod_\nu R_\nu$ entries.

**Order two.** For $d = 2$, $u = U^{(1)} C\, U^{(2)\mathsf T}$ is a matrix,
and the SVD of its small core $C = W \Sigma Z^{\mathsf T}$, after
orthonormalization of the frames, gives the best approximation
$u_r = \sum_{k \le r} \sigma_k\, (U^{(1)} w_k) \otimes (U^{(2)} z_k)$,
whatever the type of the core. The result has a diagonal core: the canonical
format is preserved.

**Diagonal core.** The canonical format has no SVD, and the set of tensors
of canonical rank at most $r$ is not closed for $d \ge 3$
({ref}`theory-tucker-canonical`). A tensor $u$ with a diagonal core is
recompressed by the greedy algorithm ({ref}`theory-greedy`) applied to the
identity operator, $I v = u$: the energy is then
$\frac12 \|v - u\|^2 - \frac12 \|u\|^2$, so that each correction is a
best rank-one approximation of the remainder $u - u^{n-1}$, computed by
alternating least squares, and is followed by an update of all the rank-one
factors ({ref}`theory-galerkin-updates`). The algorithm stops when
$\|u - u^n\| \le \varepsilon \|u\|$, or after as many terms as $u$ has;
there is no guarantee that the precision is reached with fewer terms.

**Tree-based core.** If the core is a tree-based tensor, absorbing the frames
into its leaves gives a tree-based tensor in the physical variables, on the
same tree: the leaf $C^\nu$ becomes $U^{(\nu)} C^\nu$. This tensor is
truncated by the hierarchical SVD, which never forms the full core, and its
leaves are moved back to the frames.

**Operators** are truncated as their vectorization, the matrices of their
frames being seen as vectors: the norm is then the Frobenius norm.

## In tensap

- {py:class}`~tensap.Truncator`: `truncate` dispatches on the format of the
  tensor and of its core; `hosvd`, `hsvd` and `trunc_svd` are the
  algorithms above.
- Examples: {ref}`sphx_glr_examples_tensors_tensor_truncation.py`,
  {ref}`sphx_glr_examples_tensors_tucker_tensor.py`.
