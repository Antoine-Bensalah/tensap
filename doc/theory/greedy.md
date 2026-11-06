(theory-greedy)=
# Greedy algorithms and alternating least squares

Instead of iterating on a full-rank approximation and truncating it
({ref}`theory-truncated-iterations`), the solution of $A u = b$ can be built
one rank-one term at a time, without fixing the rank in advance. Each term
is computed by alternating least squares, which turns a nonlinear problem on
rank-one tensors into a sequence of small linear systems. This is the
*proper generalized decomposition* (PGD) in its progressive form
{cite:p}`Ammar2006,Nouy2017book`.

Throughout, $A$ is symmetric positive definite (SPD), and the solution $u$
minimizes the energy $J(v) = \frac12 (A v, v) - (b, v)$, with
$J(v) - J(u) = \frac12 \|v - u\|_A^2$: minimizing $J$ on a set of tensors is
approximating $u$ in the energy norm.

## Rank-one approximation by alternating least squares

The problem is to find a rank-one tensor

$$
v = \lambda\, w^{(1)} \otimes \cdots \otimes w^{(d)}, \qquad \|w^{(\nu)}\| = 1,
$$

minimizing $J$ on the set $\mathcal{R}_1$ of rank-one tensors. This set is
not a linear space, but the map $(w^{(1)}, \dots, w^{(d)}) \mapsto
w^{(1)} \otimes \cdots \otimes w^{(d)}$ is multilinear: the problem is
linear in each factor separately.

```{prf:proposition} Local problem
:label: prop-als-local

Let $A = \sum_k \sigma_k \bigotimes_\nu A^{(\nu)}_k$ and
$b = \sum_l \tau_l \bigotimes_\nu b^{(\nu)}_l$. With all the factors but
$w^{(\mu)}$ fixed, and $w = \lambda w^{(\mu)}$ carrying the scale, the
minimizer of $J$ over $w$ solves

$$
A_\mu\, w = b_\mu, \qquad
A_\mu = \sum_k \sigma_k\, \gamma_k\, A^{(\mu)}_k, \qquad
b_\mu = \sum_l \tau_l\, \delta_l\, b^{(\mu)}_l,
$$

with $\gamma_k = \prod_{\nu \ne \mu} (A^{(\nu)}_k w^{(\nu)}, w^{(\nu)})$ and
$\delta_l = \prod_{\nu \ne \mu} (b^{(\nu)}_l, w^{(\nu)})$. The matrix
$A_\mu$, of size $N_\mu$, is SPD.
```

```{prf:proof}
By the factorization of the inner product on elementary tensors,
$(A v, v) = \sum_k \sigma_k (A^{(\mu)}_k w, w) \gamma_k$ and
$(b, v) = \sum_l \tau_l (b^{(\mu)}_l, w) \delta_l$. Hence
$J = \frac12 w^{\mathsf T} A_\mu w - b_\mu^{\mathsf T} w$, a quadratic
function of $w$ alone, whose gradient vanishes at the minimizer. For
$w \ne 0$, $w^{\mathsf T} A_\mu w = (A v, v) > 0$, since $A$ is SPD.
```

For the Laplacian, $A_\mu$ is a shifted 1D Laplacian. The alternating least
squares (ALS) algorithm solves these local problems for
$\mu = 1, \dots, d$ in turn, normalizes $w^{(\mu)} = w / \|w\|$ with
$\lambda = \|w\|$, and repeats the sweeps until $\lambda$ stagnates.

**Properties.** Each local solve minimizes $J$ exactly in one block of
variables, so that $J$ decreases at each step: ALS is a block coordinate
descent method. It converges in general only to a stationary point, which
depends on the initialization; there is no guarantee of a global minimizer,
even for a convex $J$. Local convergence results are known for the canonical
and tensor-train formats {cite:p}`Uschmajew2012,Holtz2012`.

**Non-symmetric operators.** Without symmetry there is no energy, and the
Galerkin condition (the same local system, now non-symmetric) can be
unstable. A Petrov-Galerkin variant tests with $A_\mu^{-\mathsf T}
w^{(\mu)}$ instead of $w^{(\mu)}$ in some dimensions, which amounts to a
local minimal residual condition.

**Cost.** Each local solve needs the reduced quantities of the other
dimensions, $(A^{(\nu)}_k w^{(\nu)}, w^{(\nu)})$, which are stored and only
updated for the dimension just solved, the assembly of $A_\mu$, and a solve
of size $N_\mu$: the cost of a sweep is linear in $d$.

## Greedy rank-one corrections

```{prf:algorithm} Greedy algorithm with rank-one corrections
:label: alg-greedy

**Inputs** $A$, $b$, an update $\mathcal{U}$ (default: the identity)

1. $u^0 = 0$.
2. For $n = 1, 2, \dots$:
   1. $s^{n-1} = b - A u^{n-1}$;
   2. $w_n$ = rank-one ALS for $A w = s^{n-1}$, that is
      $w_n \in \arg\min_{w \in \mathcal{R}_1} J(u^{n-1} + w)$;
   3. $u^n = \mathcal{U}(u^{n-1} + w_n)$;
   4. stop when the relative residual $\|s^n\| / \|b\|$ or the relative
      change $\|u^n - u^{n-1}\| / \|u^n\|$ is small.
```

Computing the rank-one approximation of $A w = s^{n-1}$ is indeed minimizing
$J(u^{n-1} + w)$ over $w$, since

$$
J(u^{n-1} + w) = J(u^{n-1}) + \tfrac12 (A w, w) - (s^{n-1}, w):
$$

each correction is the best rank-one improvement of the current iterate.
With $\mathcal{U}$ the identity (pure greedy algorithm),
$u^n = \sum_{k \le n} w_k$ is a tensor of canonical rank $n$. The update can
recompute the coefficients of the terms, or all their factors, by Galerkin
projections ({ref}`theory-galerkin-updates`).

```{prf:theorem} Convergence of the greedy algorithm {cite:p}`LeBris2009,Cances2011,FalcoNouy2012`
:label: thm-greedy

If $A$ is SPD and each correction is an exact minimizer of
$J(u^{n-1} + w)$ over $\mathcal{R}_1$, then $\|u^n - u\|_A \to 0$. The
result extends to strongly convex functionals with a Lipschitz differential,
to any closed dictionary containing $\mathcal{R}_1$ and spanning the space,
with or without updates, and to corrections that achieve at least a fixed
fraction of the best improvement (weak greedy algorithm).
```

The argument: $J(u^n)$ decreases and is bounded below, so the gains
$J(u^{n-1}) - J(u^n) = \frac12 \max_{w \in \mathcal{R}_1}
(s^{n-1}, w)^2 / \|w\|_A^2$ tend to 0; a residual whose inner products with
all the rank-one tensors tend to 0 tends to 0 weakly, and convexity upgrades
this to the energy norm.

**What the theorem does not cover.** The correction computed by ALS, with a
fixed number of sweeps or stopped on stagnation, may be a stationary point
that does not achieve a fixed fraction of the best improvement. The theorem
describes the ideal algorithm; for the implemented one, the decrease of the
residual is measured at each iteration.

**Rate.** The pure greedy algorithm can converge slowly compared with the
best approximation of rank $n$: each term is optimal given the previous
ones, which are never revised. Greedy approximation theory relates the rate
to the approximability of $u$ by sums of rank-one terms
{cite:p}`Temlyakov2008`. The updates revise the previous terms.

**Error estimate.** The algorithm monitors the residual. For $A$ SPD,
$\|u^n - u\| / \|u\| \le \kappa(A)\, \|s^n\| / \|b\|$: a small residual
means a small error only up to the condition number.

**Minimal residual variant.** The same algorithm applied to
$A^{\mathsf T} A u = A^{\mathsf T} b$ minimizes the residual norm at each
correction, for non-symmetric operators, at the price of squaring the
condition number.

**Cost.** Each iteration runs one rank-one ALS on the residual, of canonical
rank $R^b + R^A (n - 1)$, and evaluates the residual: the cost is linear in
$d$, and no truncation is needed in canonical format.

## Greedy construction of tensor subspaces

The set of tensors of Tucker rank at most $(r_1, \dots, r_d)$ is also the
union of the tensor subspaces $U_1 \otimes \cdots \otimes U_d$ with
$\dim U_\nu = r_\nu$: minimizing $J$ on it is finding optimal subspaces, a
multilinear version of projection-based model order reduction. Greedy
algorithms can build increasing tensor subspaces
$U^n_1 \otimes \cdots \otimes U^n_d$ {cite:p}`Nouy2017book`. At the step
$n$, with a set $D_n$ of dimensions to enrich:

1. compute a rank-one correction $w_n = w^{(1)}_n \otimes \cdots \otimes
   w^{(d)}_n$ minimizing $J(u^{n-1} + w)$;
2. enrich $U^n_\nu = U^{n-1}_\nu + \operatorname{span}(w^{(\nu)}_n)$ for
   $\nu \in D_n$, and keep $U^n_\nu = U^{n-1}_\nu$ otherwise;
3. compute $u^n$ minimizing $J$ on $U^n_1 \otimes \cdots \otimes U^n_d$: a
   Galerkin problem on the core of a Tucker tensor
   ({ref}`theory-galerkin-updates`), the core being itself in a structured
   format for large $d$. A diagonal core gives back the greedy algorithm in
   canonical format.

With all the dimensions enriched at each step, this is the greedy algorithm
with an update of the core. Enriching only the dimensions with the largest
estimated errors gives anisotropic subspaces, smaller for the same accuracy.

## In tensap

- {py:class}`~tensap.RankOneALSLinearSolver`,
  {py:class}`~tensap.GreedyLinearSolver`; the updates are methods of
  {py:class}`~tensap.TuckerTensorALSLinearSolver`.
- Examples: {ref}`sphx_glr_examples_solvers_poisson_tucker.py`,
  {ref}`sphx_glr_examples_solvers_greedy_with_updates.py`.
- Use case: {ref}`sphx_glr_use_cases_parametric_diffusion_reaction.py`.
