(theory-inverse-approximation)=
# Low-rank approximation of the inverse

The iterative methods with truncation converge in a number of iterations
that grows with the condition number of the operator
({ref}`theory-truncated-iterations`); a preconditioner $P \approx A^{-1}$ is
needed, and it must be in low-rank format too. This page shows how to
approximate the inverse of an operator $A = \sum_i C_A(i)
\bigotimes_\nu A^{(\nu)}_{i_\nu}$ in canonical format, by the same
variational and greedy approach as for the solution of $A u = b$
({ref}`theory-greedy`), with operators as unknowns. $A$ is symmetric
positive definite (SPD) throughout.

## Variational formulation

Operators form a Hilbert space with the Frobenius inner product
$(P, Q) = \operatorname{tr}(P Q^{\mathsf T})$. More generally than
$P A = I$, consider $P A = B$ for a given operator $B$.

```{prf:proposition}
:label: prop-inverse-functional

The functional

$$
\mathcal{J}(P) = \tfrac12 (P A, P) - (B, P)
= \tfrac12 \operatorname{tr}(P A P^{\mathsf T})
  - \operatorname{tr}(B P^{\mathsf T})
$$

is strictly convex, with the gradient $P A - B$ and the minimizer
$B A^{-1}$. Up to a constant,
$\mathcal{J}(P) = \frac12 \|(P - B A^{-1}) A^{1/2}\|_F^2$.
```

```{prf:proof}
With $A$ symmetric, $\mathcal{J}(P + H) = \mathcal{J}(P) + (P A - B, H) +
\frac12 (H A, H)$, and $(H A, H) = \|H A^{1/2}\|_F^2 > 0$ for $H \ne 0$.
Expanding $\frac12 \|(P - B A^{-1}) A^{1/2}\|_F^2$ gives $\mathcal{J}(P)$
plus $\frac12 \operatorname{tr}(B A^{-1} B^{\mathsf T})$.
```

Minimizing $\mathcal{J}$ on a set of low-rank operators gives the best
approximation of $B A^{-1}$ in a norm weighted by $A$.

## Rank-one correction by alternating least squares

For a rank-one operator $W = a\, v^{(1)} \otimes \cdots \otimes v^{(d)}$,
with matrices $v^{(\nu)}$, the traces factorize:

$$
\begin{aligned}
\operatorname{tr}(W A W^{\mathsf T})
&= a^2 \sum_i C_A(i) \prod_\nu
  \big(A^{(\nu)}_{i_\nu}, v^{(\nu)\mathsf T} v^{(\nu)}\big), \\
\operatorname{tr}(B W^{\mathsf T})
&= a \sum_l C_B(l) \prod_\nu \big(B^{(\nu)}_{l_\nu}, v^{(\nu)}\big).
\end{aligned}
$$

With all the factors but $v = v^{(\mu)}$ fixed, and $a$ absorbed into $v$,
$\mathcal{J}$ is a quadratic function of $v$, and its minimizer solves the
matrix equation

$$
v\, Q_\mu = Z_\mu, \qquad
Q_\mu = \sum_i \gamma_i A^{(\mu)}_{i_\mu}, \qquad
Z_\mu = \sum_l \delta_l B^{(\mu)}_{l_\mu},
$$

where $\gamma$ and $\delta$ gather the core and the products over the other
dimensions. $Q_\mu$ is SPD, as the restriction of the SPD quadratic form
$P \mapsto \operatorname{tr}(P A P^{\mathsf T})$, so that
$v = Z_\mu Q_\mu^{-1}$. Constraints can be imposed on each factor:

- **symmetric**: the optimality condition becomes the Lyapunov equation
  $v Q + Q v = Z + Z^{\mathsf T}$. With $Q = U \operatorname{diag}(q)
  U^{\mathsf T}$ and $v = U Y U^{\mathsf T}$, its solution is
  $Y_{ij} = (U^{\mathsf T} (Z + Z^{\mathsf T}) U)_{ij} / (q_i + q_j)$;
- **sparse**: each row $v_k$ of $v$ solves the least-squares problem
  $v_k Q|_{J_k} = Z_k$ on a sparsity pattern $J_k$, as for a sparse
  approximate inverse; the pattern can be built adaptively, by adding to
  each row the entry that decreases its residual the most.

## Greedy approximation

```{prf:algorithm} Greedy approximation of the inverse
:label: alg-greedy-inverse

**Inputs** $A$, a number of terms

1. $P^0 = 0$, $B^0 = I$.
2. For $l = 1, 2, \dots$:
   1. $W_l$ = rank-one approximation of the solution of $W A = B^{l-1}$ by
      alternating least squares;
   2. $P^l = P^{l-1} + W_l$;
   3. stop when $P$ stagnates, or when $\|I - P^l A\|_F / \|I\|_F$ is
      small;
   4. $B^l = I - P^l A$.
```

As a greedy algorithm on the space of operators, it converges to $A^{-1}$ in
the norm of $\mathcal{J}$ {cite:p}`FalcoNouy2012`. A minimal residual
variant replaces $(A, B)$ by $(A A^{\mathsf T}, A^{\mathsf T})$, so that the
minimizer of $\mathcal{J}$ minimizes $\|I - P A\|_F$.

**What the approximation is, and is not.** The minimization over a low-rank
set gives an approximate inverse $P$, with no guarantee that $P$ is SPD: the
factors are unconstrained, and even symmetric factors give a signed sum of
Kronecker products. The preconditioned conjugate gradient assumes an SPD
preconditioner, and its rate depends on the spectrum of $P^{1/2} A P^{1/2}$,
which a small $\|I - P A\|_F$ does not control independently of the
dimension. What matters for preconditioning is the spectrum of $P A$: on the
Poisson problems tested in tensap, a few terms make $\kappa(P A)$ close to
1, but this is an observation, to be checked on each problem.

**Why the inverse of the Laplacian has low rank.** $A^{-1}$ is not of low
rank, but it is well approximated by short sums of Kronecker products
{cite:p}`Hackbusch2019`. An exponential sum $t^{-1} \approx \sum_{k=1}^{R}
\omega_k e^{-\alpha_k t}$ on the spectrum of $A$ has an error decreasing
exponentially with $R$; for the Kronecker sum
$A = \sum_\nu I \otimes \cdots \otimes K \otimes \cdots \otimes I$, whose
terms commute, $e^{-\alpha A} = \bigotimes_\nu e^{-\alpha K}$, so that
$A^{-1} \approx \sum_k \omega_k \bigotimes_\nu e^{-\alpha_k K}$ has
canonical rank $R$. If the relative error of the exponential sum is at most
$\delta$ on the spectrum, the approximation $P$ is spectrally equivalent to
$A^{-1}$, with $\kappa(P A) \le (1 + \delta) / (1 - \delta)$.

## Cost

Each local solve needs the reduced quantities
$(A^{(\nu)}_i, v^{(\nu)\mathsf T} v^{(\nu)})$, dense products of size
$N_\nu \times N_\nu$, and a dense solve of size $N_\mu$ with $N_\mu$
right-hand sides, $O(N_\mu^3)$. The residual $I - P^l A$ has rank
$1 + l R_A$.

## In tensap

- {py:class}`~tensap.RankOneInverseApproximatorByALS`,
  {py:class}`~tensap.GreedyInverseApproximator`; the factors are computed with
  {py:func}`~tensap.lyapsym` (symmetric) and {py:func}`~tensap.gspai`
  (sparse).
- Example: {ref}`sphx_glr_examples_solvers_poisson_tucker.py`, where the
  approximate inverse preconditions the truncated iterations.
