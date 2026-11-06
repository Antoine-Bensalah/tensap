(theory-tree-contractions)=
# Tree contractions and local problems on tree-based cores

A tree-based tensor $C$ is a network of node tensors $\{C^\alpha\}$ on a
dimension tree. Two operations on such tensors need contractions of the
whole network: the inner product with a *rank-one metric*,

$$
(C, M D), \qquad M = M_1 \otimes \cdots \otimes M_d ,
$$

and the local problem of a node in an alternating minimization, for instance
the update of the tree-based core of a Tucker tensor
({ref}`theory-galerkin-updates`). Forming the full tensors costs
$\prod_\nu N_\nu$, exponential in $d$; contracting along the tree costs a
polynomial in the ranks. This is the tree version of the interfaces of the
solvers in tensor-train format ({ref}`theory-tt-solvers`), where the tree is
a chain. The leaves are *inactive* here: the index of the parent of a leaf
$\nu$ is directly the index $i_\nu$ of the dimension $\nu$.

## Contractions along the tree

**Bottom contractions.** For a node $\beta$ other than the root, with the
children $\gamma_1, \dots, \gamma_p$, the contraction of its subtree is the
matrix $B_\beta$ of size $r_\beta(C) \times r_\beta(D)$:

$$
B_\nu = M_\nu \ \text{(leaf of the dimension } \nu\text{)}, \qquad
B_\beta(k, l) = \sum C^\beta(k_1, \dots, k_p, k)
  \prod_{t = 1}^{p} B_{\gamma_t}(k_t, l_t)\, D^\beta(l_1, \dots, l_p, l),
$$

computed from the leaves to the root, one node tensor at a time.

**Top contractions.** For a node $\gamma$ other than the root, child number
$s$ of its parent $\beta$, the contraction of everything outside its
subtree is

$$
T_\gamma(k_s, l_s) = \sum C^\beta(k_1, \dots, k_p, k)
  \prod_{t \ne s} B_{\gamma_t}(k_t, l_t)\, T_\beta(k, l)\,
  D^\beta(l_1, \dots, l_p, l),
$$

without the factor $T_\beta$ when $\beta$ is the root, computed from the
root to the leaves.

```{prf:proposition} Reduction at a node
:label: prop-tree-reduction

For any node $\alpha$ with the children $\gamma_1, \dots, \gamma_p$,

$$
(C, M D) = \sum C^\alpha(k_1, \dots, k_p, k)
  \prod_{t} B_{\gamma_t}(k_t, l_t)\, T_\alpha(k, l)\,
  D^\alpha(l_1, \dots, l_p, l).
$$
```

The matrices $B_{\gamma_1}, \dots, B_{\gamma_p}, T_\alpha$ are the
*reduction* of the inner product at the node $\alpha$. At a leaf $\nu$,
$T_\nu$ is the contraction of $C$ and $D$ through all the matrices but
$M_\nu$.

## Local problem of a node

For a Tucker tensor with orthonormal frames and a tree-based core $C$, the
Galerkin problem on the core is $A_c C = b_c$, with
$A_c = \sum_k C_A(k) \bigotimes_\nu P^{(\nu)}_{k_\nu}$ and $C_A$ the
tree-based core of the operator ({ref}`theory-galerkin-updates`). With the
representation orthogonal at the node $\alpha$, $C = \Psi_\alpha c$ is
linear in the node tensor $c$, with $\Psi_\alpha$ a matrix with orthonormal
columns, and the Galerkin problem on the node is

$$
\Psi_\alpha^{\mathsf T} A_c \Psi_\alpha\, c = \Psi_\alpha^{\mathsf T} b_c .
$$

The local operator is SPD when $A$ is. Since $A_c$ is a tree network of
operators, the reduction applies to the three networks $C$ (test), $C_A$ and
$C$ (trial): the interfaces $L_t(i, k, j)$ of the children and of the parent
($i$ the test index, $k$ the index of $C_A$, $j$ the trial index) give

$$
\big(\Psi_\alpha^{\mathsf T} A_c \Psi_\alpha\, c\big)(i_1, \dots, i_p, i_0)
= \sum C_A^\alpha(k_1, \dots, k_p, k_0) \prod_{t = 0}^{p} L_t(i_t, k_t, j_t)\,
  c(j_1, \dots, j_p, j_0),
$$

with $t = 0$ the parent, absent at the root, and
$\Psi_\alpha^{\mathsf T} b_c$ likewise with the core of $b$. The explicit map
$\Psi_\alpha$ would have $\prod_\nu R_\nu$ rows, out of reach for large $d$;
the interfaces have a few thousand entries.

## Solving the local systems

A node of a binary tree with ranks $r$ has $r^3$ unknowns: about $27\,000$
for $r = 30$, a dense matrix of several gigabytes. Small local systems are
assembled and solved directly; larger ones are solved by the conjugate
gradient, which only needs the product by the local operator

$$
M = \sum_k C_A^\alpha(k)\, L_1^{k_1} \otimes \cdots \otimes L_p^{k_p}
\otimes L_0^{k_0}, \qquad L_t^{k_t} = L_t(\cdot, k_t, \cdot),
$$

a sum of Kronecker products, applied one interface after the other, then
contracted with the node tensor of $C_A$: $O(r_A^3 r^4)$ for a binary tree.

**A Kronecker-sum preconditioner.** The Kronecker sums
$S_1 \otimes I \otimes \cdots + \dots + I \otimes \cdots \otimes S_n$ form a
linear space. The orthogonal projection of $M$ onto it, for the Frobenius
inner product, is

$$
\sum_t I \otimes \cdots \otimes T_t \otimes \cdots \otimes I
- (n - 1)\, \tau\, I, \qquad
T_t = \frac{\operatorname{tr}_{s \ne t}(M)}{\prod_{s \ne t} n_s}, \qquad
\tau = \frac{\operatorname{tr} M}{\prod_s n_s},
$$

with the partial traces
$\operatorname{tr}_{s \ne t}(M) = \sum_k C_A^\alpha(k)\, L_t^{k_t}
\prod_{s \ne t} \operatorname{tr}(L_s^{k_s})$, cheap to compute. Its inverse
is applied by *fast diagonalization*: with the eigendecompositions
$T_t = Q_t \operatorname{diag}(\lambda_t) Q_t^{\mathsf T}$, it is
$(\bigotimes_t Q_t) \operatorname{diag}\big(1 / (\sum_t \lambda_{t, i_t} -
(n - 1) \tau)\big) (\bigotimes_t Q_t)^{\mathsf T}$, in $O(n r^4)$ operations
after $n$ eigendecompositions of size $r$.

For the Laplacian, $M$ *is* a Kronecker sum, and the preconditioner is the
exact inverse: with the representation orthogonal at $\alpha$, the interface
of a subtree that does not contain the operator $K$ is the identity. For an
operator far from a Kronecker sum, the preconditioner is only approximate,
and may not even be positive definite: the projection onto the Kronecker
sums is not a projection onto the positive definite ones.

```{prf:example} A projection that is not positive
:label: ex-kronecker-sum-not-spd

For $M = \operatorname{diag}(1, 1, 1, 100)$ on $\mathbb{R}^2 \otimes
\mathbb{R}^2$, SPD, both partial traces are $\operatorname{diag}(1, 50.5)$,
$\tau = 25.75$, and the projected Kronecker sum is
$\operatorname{diag}(-23.75, 25.75, 25.75, 75.25)$.
```

The conjugate gradient needs an SPD preconditioner: the denominators
$\sum_t \lambda_{t, i_t} - (n - 1) \tau$ are checked, and the local system is
solved without preconditioner if one of them is not positive.

## Cost

With $r$ and $r_A$ the ranks of $C$ and $C_A$ and binary trees, a bottom or
top contraction costs $O(r^4)$ between two networks and $O(r_A^3 r^4)$ with
the operator; the local system of an interior node has $r^3$ unknowns,
assembled in $O(r_A^3 r^6)$ or applied in $O(r_A^3 r^4)$.

## In tensap

- {py:meth}`~tensap.TreeBasedTensor.reduce_dot_with_rank_one_metric_at_node`
  computes the reduction at a node; the local problems are those of the
  update of a tree-based core by
  {py:class}`~tensap.TuckerTensorALSLinearSolver`, with the Kronecker-sum
  preconditioner by default.
- Use case: {ref}`sphx_glr_use_cases_high_dimensional_poisson.py`.
