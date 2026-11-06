(theory-galerkin-updates)=
# Galerkin updates

The iterate of the pure greedy algorithm, $u^n = \sum_{i \le n} c_i\, w_i$
with rank-one terms $w_i = w^{(1)}_i \otimes \cdots \otimes w^{(d)}_i$, is
not the best approximation of the solution among the tensors of rank $n$
({ref}`theory-greedy`). At fixed rank, it can be improved by minimizing the
energy $J(v) = \frac12 (A v, v) - (b, v)$ over a linear space attached to
the iterate: the parameters of the format, the core or the factors of one
dimension, are updated one group at a time, the others being fixed. Each
update is a Galerkin projection, so that $J$ decreases and the rank does not
change. $A$ is symmetric positive definite (SPD) throughout.

## Update of the core

With the frames fixed, $J(\sum_i c_i w_i) = \frac12 c^{\mathsf T} G c -
g^{\mathsf T} c$ is a quadratic function of the coefficients, minimized
where its gradient vanishes: the Galerkin problem on
$\operatorname{span}\{w_1, \dots, w_n\}$,

$$
G\, c = g, \qquad
G(j, i) = (A w_i, w_j)
= \sum_k a_k \prod_{\nu = 1}^{d} \big(A^{(\nu)}_k w^{(\nu)}_i,
  w^{(\nu)}_j\big), \qquad
g(j) = (b, w_j),
$$

for $A = \sum_k a_k \bigotimes_\nu A^{(\nu)}_k$. The entries of $G$
factorize over the dimensions, as every inner product of elementary tensors.
With this update after each correction, the greedy algorithm becomes the
*orthogonal* greedy algorithm {cite:p}`Temlyakov2008`: $u^n$ is the energy
projection of $u$ onto the span of the rank-one terms. The cost is
$O(d R^A n^2 N)$ for the inner products and $O(n^3)$ for the solve.

**Full core.** For a Tucker tensor with orthonormal frames,
$v = (U^{(1)} \otimes \cdots \otimes U^{(d)}) \operatorname{vec}(C)$, and
the Galerkin problem on the core is $A_c\, C = b_c$, with the projected
operator and right-hand side

$$
A_c = \sum_k C_A(k)\, P^{(1)}_{k_1} \otimes \cdots \otimes P^{(d)}_{k_d},
\quad P^{(\nu)}_k = U^{(\nu)\mathsf T} A^{(\nu)}_k U^{(\nu)}, \qquad
b_c = \sum_l C_b(l) \bigotimes_\nu U^{(\nu)\mathsf T} b^{(\nu)}_l ,
$$

of size $\prod_\nu R_\nu$. It is assembled and solved directly for small
ranks; beyond, it is solved by the conjugate gradient, $A_c$ being applied by
mode products over the non-zero entries of $C_A$, without being assembled.

## Update of the frames

With the coefficients and the frames of the other dimensions fixed, the
unknowns are the factors of one dimension $\mu$. Writing
$v = \sum_i c_i\, w^{(\mu)}_i \otimes z_i$ with
$z_i = \bigotimes_{\nu \ne \mu} w^{(\nu)}_i$ fixed, the energy is quadratic
in the matrix $W = [c_1 w^{(\mu)}_1, \dots, c_n w^{(\mu)}_n]$ of the
terms of dimension $\mu$.

```{prf:proposition} Local system of the frames update
:label: prop-frames-update

The minimizer of $J$ over $W$ solves

$$
\Big(\sum_k \Gamma_k \otimes A^{(\mu)}_k\Big) \operatorname{vec}(W)
= \operatorname{vec}(F), \qquad
\Gamma_k(j, i) = a_k \prod_{\nu \ne \mu}
\big(A^{(\nu)}_k w^{(\nu)}_i, w^{(\nu)}_j\big),
$$

with $F = \sum_l b^{(\mu)}_l f_l^{\mathsf T}$,
$f_l(j) = \tau_l \prod_{\nu \ne \mu} (b^{(\nu)}_l, w^{(\nu)}_j)$, for
$b = \sum_l \tau_l \bigotimes_\nu b^{(\nu)}_l$. The system has $N_\mu n$
unknowns, in $n \times n$ blocks of size $N_\mu \times N_\mu$, sparse if the
$A^{(\mu)}_k$ are.
```

```{prf:proof}
With $W_i$ the column $i$ of $W$, $v = \sum_i W_i \otimes z_i$ and, by the
factorization of the inner products,
$(A v, v) = \sum_k \sum_{i, j} \Gamma_k(j, i)\,
(A^{(\mu)}_k W_i, W_j) = \sum_k (A^{(\mu)}_k W \Gamma_k^{\mathsf T}, W)$ and
$(b, v) = (F, W)$. The gradient $\sum_k A^{(\mu)}_k W \Gamma_k^{\mathsf T} -
F$ vanishes at the minimizer, and
$\operatorname{vec}(A W \Gamma^{\mathsf T}) = (\Gamma \otimes A)
\operatorname{vec}(W)$.
```

This is alternating least squares for tensors of rank $n$, one dimension at
a time. The columns of $W$ are then normalized, and their norms become the
new coefficients $c_i$.

**Why the terms, and not the factors.** Taking the factors $w^{(\mu)}_i$ as
unknowns, with the coefficients $c_i$ in the matrix, multiplies its blocks by
$c_i c_j$: the coefficients enter the matrix squared. Once the greedy
algorithm has converged, the new terms have coefficients many orders of
magnitude smaller than the first ones, and the condition number of the
system explodes, while the same Galerkin problem written on the terms
$c_i w^{(\mu)}_i$ does not depend on the coefficients.

**Limitation of the canonical format.** The block $\sum_k \Gamma_k$
involves the Gram matrix of the terms $z_i$, the Hadamard product of the Gram
matrices of the frames of the other dimensions. When the terms are
redundant, which happens when a greedy algorithm adds corrections beyond the
rank needed, the $z_i$ become numerically linearly dependent, and the system
is singular. In the canonical format the terms cannot be orthonormalized, so
that this cannot be avoided; it can only be detected, at the cost of an
$n \times n$ eigenvalue problem, and the update is then skipped. A Tucker
tensor with a full or structured core does not have this limitation: its
frames can be orthonormalized, and a redundancy shows in its core.

## Alternating minimization

```{prf:algorithm} Alternating least squares on a low-rank iterate
:label: alg-als-updates

**Inputs** $A$, $b$, an initial tensor $v$ of rank $n$

1. Repeat until the relative residual is small, or $v$ stagnates:
   1. update the core of $v$;
   2. repeat until stagnation, or for a maximal number of sweeps: for
      $\mu = 1, \dots, d$, update the frame of the dimension $\mu$.
```

**Properties.** Every update minimizes $J$ on a linear space containing the
current iterate: $J$ decreases and the rank does not change. The update of
the frames minimizes over a larger space than the update of the core (it
includes rescalings of the terms), so that it is more effective per step,
and more expensive. With updates after each correction, the greedy algorithm
keeps its convergence {cite:p}`FalcoNouy2012,Cances2011`, and usually
converges much faster.

## In tensap

- {py:class}`~tensap.TuckerTensorALSLinearSolver`: `update_core`,
  `update_frames_by_als`, and `solve` for the alternation. The two updates
  can be given as the `update` of a {py:class}`~tensap.GreedyLinearSolver`.
- Example: {ref}`sphx_glr_examples_solvers_greedy_with_updates.py`.
- Use case: {ref}`sphx_glr_use_cases_parametric_diffusion_reaction.py`.
