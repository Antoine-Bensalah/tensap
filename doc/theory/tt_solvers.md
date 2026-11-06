(theory-tt-solvers)=
# Alternating solvers in tensor-train format: ALS, DMRG and AMEn

To solve $A u = b$ with $A$ a TT operator and $b$ a TT tensor, the solution
is sought in the tensor-train format ({ref}`theory-tensor-trains`). The
parametrization of a tensor by its cores is multilinear: minimizing the
energy $J(v) = \frac12 (A v, v) - (b, v)$ over one core (ALS), or over two
neighbouring cores merged into a supercore (DMRG), is a sequence of linear
problems {cite:p}`Holtz2012`. AMEn keeps the one-core steps of ALS and adapts
the ranks by enriching each core with directions of the residual
{cite:p}`Dolgov2014`. $A$ is symmetric positive definite (SPD) unless
stated otherwise.

## The local problem

With the orthogonality center on the core $\nu$,
$\operatorname{vec}(v) = U_\nu \operatorname{vec}(C^\nu)$, with
$U_\nu = C^{\le \nu - 1} \otimes I_{N_\nu} \otimes C^{> \nu\, \mathsf T}$ a
matrix with orthonormal columns ({ref}`theory-tensor-trains`). Minimizing
$J(U_\nu c)$ over $c$ is the Galerkin projection on the range of $U_\nu$:

$$
A_\nu\, c = b_\nu, \qquad
A_\nu = U_\nu^{\mathsf T} A U_\nu, \qquad b_\nu = U_\nu^{\mathsf T} b .
$$

Since $U_\nu$ has orthonormal columns, $A_\nu$ is SPD with
$\kappa(A_\nu) \le \kappa(A)$: its Rayleigh quotients are those of $A$ on a
subspace. In TT form, $A_\nu$ is the contraction of three small tensors: the
left interface $L_\nu$, contraction of the cores before $\nu$ of $v$, $A$
and $v$; the core $A^{(\nu)}$; and the right interface $R_\nu$, for the
cores after $\nu$:

$$
(A_\nu c)(k, i, k') = \sum L_\nu(k, a, l)\, A^{(\nu)}(a, i, j, a')\,
R_\nu(k', a', l')\, c(l, j, l') ,
$$

and $b_\nu$ likewise with the interfaces of $b$. $L_{\nu+1}$ is obtained from
$L_\nu$ by contracting one more core, and $R_{\nu-1}$ from $R_\nu$: the
interfaces are stored and updated when the center moves, so that a sweep
costs $O(d)$ contractions instead of $O(d^2)$.

## The three algorithms

```{prf:algorithm} ALS in TT format
:label: alg-tt-als

**Inputs** $A$, $b$, an initial guess $v$ in TT format, of the desired ranks

1. Orthogonalize $v$ with the center on the first core, and compute the
   right interfaces.
2. Repeat sweeps, from left to right then from right to left: for each core
   $\nu$,
   1. solve the local problem $A_\nu c = b_\nu$, and set $C^\nu = c$;
   2. move the center to the next core by a QR factorization of $C^\nu$,
      which does not change $v$, and update the interface.
3. Stop when $v$ stagnates, or when the relative residual is small.
```

**DMRG** takes as local unknown the supercore $W$ of two neighbouring cores
$\nu$ and $\nu + 1$, of shape
$r_{\nu-1} \times N_\nu \times N_{\nu+1} \times r_{\nu+1}$, with the local
operator built from $L_\nu$, $A^{(\nu)}$, $A^{(\nu+1)}$ and $R_{\nu+1}$.
After the local solve, the truncated SVD of $W$ at the relative precision
$\varepsilon$ gives the two cores and the rank $r_\nu$: the ranks are set by
the singular values of the supercore. Since the other cores are orthogonal,
$\|W\| = \|v\|$, and the truncation is relative to the norm of the iterate.

```{prf:algorithm} AMEn
:label: alg-amen

**Inputs** $A$, $b$, an initial guess $v$, a small rank $\rho$ (the *kick
rank*)

Sweep from left to right; with the center on the core $\nu$:

1. solve the local problem of ALS for $C^\nu$;
2. truncate $C^\nu \approx U V$ by an SVD at the relative precision
   $\varepsilon$, which may decrease the rank;
3. enrich: with a TT tensor $z$ of rank $\rho$ approximating the residual
   $b - A v$, compute the projection $S^\nu$ of the residual on the left
   interface of $v$ and the right interface of $z$, replace $U$ by the
   orthonormal basis $Q$ of $[U, S^\nu]$, and multiply the next core by the
   corresponding coefficients: the tensor is unchanged, but the next core
   sees $\rho$ more directions, those in which the residual is the largest;
4. update $z$, by one step of ALS for the residual itself, and the
   interfaces.
```

The ranks increase by at most $\rho$ per sweep and decrease by the
truncation: they are adapted at the cost of one-core steps. Since the
enrichment directions enter the basis of the next core, the next
minimization is over a space that contains approximate residual directions.

## Properties

- **ALS.** Each step minimizes $J$ exactly over one core, so that $J$
  decreases monotonically. With full ranks, the format is the whole space
  and one sweep gives the exact solution. The ranks are those of the initial
  guess: ALS stagnates at the accuracy they allow.
- **DMRG.** Each step minimizes $J$ over two cores, then truncates: $J$ may
  increase by the truncation error, small at a small $\varepsilon$. It adapts
  the ranks, but the cost of a step grows like $N^2$, since the supercore
  has two physical indices: cheap in QTT format ($N = 2$), costly for large
  $N_\nu$.
- **AMEn.** It has the cost of ALS, one physical index per step, and adapts
  the ranks. The energy does not decrease monotonically either, but a sweep
  decreases the error at least as much as a step of steepest descent with a
  low-rank approximation of the residual {cite:p}`Dolgov2014`, which is what
  makes AMEn converge where ALS stagnates at a too small rank.
- **Convergence.** There is no guarantee of a global minimizer; local
  convergence results are known for ALS in TT format {cite:p}`Holtz2012`.
  In practice, the stopping test is the stagnation of the iterate: the
  relative residual of a badly conditioned problem stays at about
  $\kappa(A)$ times the truncation precision, even when the solution is
  accurate.

## Local solvers

Up to a given size, the local matrix is assembled and the system solved
directly. Beyond, it is solved without assembling the matrix: the conjugate
gradient (or GMRES for a non-symmetric operator) only needs the product
$c \mapsto A_\nu c$, computed by the three contractions above, from the
current core as initial guess. The local systems inherit the conditioning
of $A$; the conjugate gradient is preconditioned by the *block Jacobi*
preconditioner of the physical indices: for each pair $(k, k')$ of rank
indices, the block $\sum_{a, b} L_\nu(k, a, k)\, A^{(\nu)}(a, \cdot, \cdot,
b)\, R_\nu(k', b, k')$ of size $N_\nu$, a principal submatrix of $A_\nu$,
hence SPD.

**Non-symmetric operators.** Without symmetry there is no energy; the
Galerkin conditions still define the local problems, solved by GMRES. The
normal equations $A^{\mathsf T} A u = A^{\mathsf T} b$ restore the
minimization (of the residual norm) at the price of squaring the condition
number.

## Cost

With $r$, $r^A$ and $N$ bounds on the ranks and the sizes, a local step costs
$O(r^3 r^A N + r^2 (r^A)^2 N^2)$ for the update of an interface and for a
matrix-free product (twice $N$ for DMRG), $O(s^3)$ for a dense local solve
of size $s = r^2 N$, and $O(r^3 N^3)$ for the SVD of a supercore. A sweep
is $d$ such steps.

## In tensap

- {py:class}`~tensap.TTTensorALSLinearSolver`, with `algorithm` set to
  `'als'`, `'dmrg'` or `'amen'`; the truncation of DMRG and AMEn by a
  {py:class}`~tensap.Truncator`.
- Example: {ref}`sphx_glr_examples_solvers_tt_solvers.py`.
