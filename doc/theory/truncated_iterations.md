(theory-truncated-iterations)=
# Iterative methods with truncation

Classical iterative methods for $A u = b$ only need products by $A$, sums
and inner products, which are available in every low-rank format
({ref}`theory-tucker-canonical`). But the ranks grow at each operation: one
step of the Richardson iteration multiplies them by about the rank of $A$.
Iterative methods with truncation keep every iterate in low-rank format by
truncating it after each step ({ref}`theory-truncation`). This page presents
the truncated Richardson iteration and the truncated preconditioned
conjugate gradient, and what the truncation does to their convergence.

Throughout, $A$ is symmetric positive definite (SPD). The solution $u$ then
minimizes the energy $J(v) = \frac12 (A v, v) - (b, v)$, with
$J(v) - J(u) = \frac12 \|v - u\|_A^2$ and $\|v\|_A = (A v, v)^{1/2}$. The
residual of $v$ is $s = b - A v$, and $\Pi$ denotes a truncation.

## Truncated Richardson iterations

```{prf:algorithm} Truncated Richardson iterations
:label: alg-truncated-richardson

**Inputs** $A$, $b$, a step $\omega$, a preconditioner $P$ (default $I$), a
truncation $\Pi$

1. $u^0 = 0$.
2. For $n = 0, 1, \dots$: $u^{n+1} = \Pi\big(u^n - \omega\, P (A u^n - b)\big)$;
   stop when the relative change $\|u^{n+1} - u^n\| / \|u^{n+1}\|$ or the
   relative residual $\|b - A u^{n+1}\| / \|b\|$ is small.
```

**Without truncation** {cite:p}`Saad2003`, the iteration is the fixed point
iteration of $F(v) = v - \omega P (A v - b)$, and
$F(v) - F(w) = (I - \omega P A)(v - w)$. For $P = I$, each component in the
eigenbasis of $A$ is multiplied by $1 - \omega \lambda_j$: the iteration
converges if and only if $0 < \omega < 2 / \lambda_{\max}(A)$, with the
contraction constant $\rho = \max_j |1 - \omega \lambda_j|$. The best step
$\omega^\star = 2 / (\lambda_{\min} + \lambda_{\max})$ gives

$$
\rho^\star = \frac{\kappa - 1}{\kappa + 1},
\qquad \kappa = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)} ,
$$

and about $\kappa$ iterations are needed per decade of error. For the
discrete Laplacian with $N$ points per direction, $\kappa \approx 4 / (\pi
h)^2 \sim N^2$, independently of the dimension $d$. If $P$ is SPD, $P A$ is
self-adjoint for the energy inner product, and the same analysis holds in the
energy norm with $\kappa(P A)$: a good approximation $P \approx A^{-1}$
({ref}`theory-inverse-approximation`) makes $\kappa(P A)$ close to 1 and
$\omega = 1$ a good step.

### What the truncation does

With truncation, $u^{n+1} = \Pi(v^{n+1})$ with $v^{n+1} = F(u^n)$, and

$$
u^{n+1} - u = \big(F(u^n) - F(u)\big) + \big(\Pi(v^{n+1}) - v^{n+1}\big):
$$

the iteration is a perturbed fixed point iteration. Its behaviour depends on
the kind of truncation {cite:p}`Hackbusch2008,Bachmayr2016`.

```{prf:proposition} Truncation at a relative precision
:label: prop-richardson-relative

If $\|\Pi(v) - v\| \le \varepsilon \|v\|$ for all $v$ and
$\gamma = \rho (1 + \varepsilon) < 1$, then

$$
\|u - u^n\| \le \gamma^n \|u - u^0\| + \frac{\varepsilon}{1 - \gamma}\,
\|u\| .
$$
```

```{prf:proof}
$\|u^{n+1} - u\| \le \rho \|u^n - u\| + \varepsilon \|v^{n+1}\|$, and
$\|v^{n+1}\| \le \|u\| + \rho \|u^n - u\|$. Hence
$\|u^{n+1} - u\| \le \gamma \|u^n - u\| + \varepsilon \|u\|$, and the bound
follows by induction.
```

The iterates enter a neighbourhood of $u$ of radius
$\varepsilon \|u\| / (1 - \gamma)$: a slow contraction ($\rho$ close to 1)
amplifies the truncation error, and to reach an error $\eta$ the truncation
must be at the precision of order $(1 - \rho)\, \eta$. The ranks adapt to the
precision but are not controlled, and may become large during the
iterations. Two remarks: both norms must be the same (the truncation is
controlled in the Frobenius norm, the contraction of the preconditioned
iteration in the energy norm, and mixing them costs a factor
$\sqrt{\kappa(A)}$); and a maximal rank may prevent the truncation from
reaching the requested precision, in which case the bound does not apply.

**Truncation at a fixed rank.** If $\Pi(v)$ is a quasi-optimal approximation
in a set $\mathcal M_r$ of tensors of rank at most $r$,
$\|\Pi(v) - v\| \le C \min_{w \in \mathcal M_r} \|v - w\|$ (with
$C = \sqrt{d}$ for the HOSVD), then with $u_r$ a best approximation of $u$
in $\mathcal M_r$ and $\gamma = \rho (1 + C)$,

$$
\|u - u^n\| \le \gamma^n \|u - u^0\| + \frac{C}{1 - \gamma}\, \|u - u_r\| .
$$

The ranks are controlled, but $\gamma < 1$ requires $\rho < 1 / (1 + C)$, a
very good preconditioner.

**Non-expansive truncation.** If $\|\Pi(v) - \Pi(w)\| \le \|v - w\|$, as for
the soft thresholding of the singular values ({ref}`theory-truncation`),
$\Pi \circ F$ is contractive with the same constant $\rho$: the iterates
converge to its unique fixed point $u^\star$, with
$(1 + \rho)^{-1} \|u - \Pi(u)\| \le \|u - u^\star\| \le (1 - \rho)^{-1}
\|u - \Pi(u)\|$. A sequence of thresholds decreasing to 0 gives convergence
to $u$, with guarantees on the ranks for a well-chosen sequence
{cite:p}`Bachmayr2016`.

**Evolution equations.** A time-stepping scheme $u^{n+1} = F_n(u^n)$ is
truncated in the same way, $u^{n+1} = \Pi \circ F_n(u^n)$; controlling the
errors and the ranks over many steps then requires a careful adaptation of
the truncation parameters {cite:p}`Bachmayr2025`.

## Truncated preconditioned conjugate gradient

```{prf:algorithm} Truncated preconditioned conjugate gradient
:label: alg-truncated-pcg

**Inputs** $A$, $b$, a preconditioner $P$ (default $I$), a truncation $\Pi$

1. $v = 0$, $s = b - A v$, $z = P s$, $p = z$, $\pi = (p, A p)$.
2. For $k = 1, 2, \dots$:
   1. $\omega = (s, p) / \pi$, and $v \leftarrow \Pi(v + \omega\, p)$;
   2. $s \leftarrow \Pi(b - A v)$, recomputed from $v$;
   3. stop on the stagnation of $v$, or when $\|b - A v\| / \|b\|$ is small;
   4. $z = P s$, $\beta = -(z, A p) / \pi$, $p \leftarrow \Pi(z + \beta\, p)$,
      $\pi = (p, A p)$.
```

The conjugate gradient minimizes the energy along successive directions $p$:
the exact line search $\min_\omega J(v + \omega p)$ gives
$\omega = (s, p) / (A p, p)$, and the new direction is the preconditioned
residual made $A$-orthogonal to the previous one, which gives $\beta$. In
exact arithmetic, the directions are then $A$-orthogonal to all the previous
ones, and $\beta$ equals the textbook value $(s_{k+1}, z_{k+1}) / (s_k,
z_k)$. Two details differ from the textbook algorithm {cite:p}`Saad2003`,
to cope with the truncations: the residual is recomputed from $v$ instead of
updated recursively, so that the truncation errors do not accumulate in it;
and $\beta$ uses the local $A$-orthogonality, which remains meaningful when
the global conjugacy is lost.

**Without truncation**, the energy error decreases at least like
$2 \big((\sqrt{\kappa} - 1) / (\sqrt{\kappa} + 1)\big)^k$ with
$\kappa = \kappa(P A)$ {cite:p}`Saad2003`: $\sqrt{\kappa}$ instead of
$\kappa$ for the Richardson iteration.

**With truncation**, the directions lose their global conjugacy; the method
behaves like a nonlinear iteration, and stagnates at a level set by the
truncation precision, as the Richardson iteration does
{cite:p}`Hackbusch2008`. In the numerical experiments of tensap (QTT
Poisson problems), the truncation had to be about 100 times finer than the
requested relative residual: with equal tolerances, the residual stayed above
the tolerance. This is an observation on the tested problems, not a general
prescription.

**Breakdown.** A non-positive curvature $(p, A p) \le 0$ with a non-zero
residual, which can happen if $A$ or $P$ is not SPD, is reported as a
breakdown.

## Cost

Both methods only use the algebra of the formats (products by $A$ and $P$,
sums, inner products) and a truncation: they apply unchanged to the Tucker,
tensor-train and quantized formats. Per iteration, the Richardson iteration
applies $A$ once and truncates once, on ranks $(2 + R^A) R_n + R^b$ for an
iterate of rank $R_n$; the truncated PCG applies $A$ twice and $P$ once, and
truncates three times. The cost of the truncations dominates.

## In tensap

- {py:class}`~tensap.TruncatedRichardsonIterations`,
  {py:class}`~tensap.TruncatedPCG`, with a {py:class}`~tensap.Truncator`.
- Example: {ref}`sphx_glr_examples_solvers_poisson_tucker.py`.
