(theory-staggered)=
# Finite volumes on a staggered grid in QTT format

For the incompressible Navier-Stokes equations, the staggered (MAC) grid
puts the pressure at the centres of the cells and each velocity component at
the faces normal to its direction. Its discrete divergence and gradient
satisfy exact identities, which make the projection onto the divergence-free
fields exact. This page gives the one-dimensional blocks, their assembly in
QTT format ({ref}`theory-qtt`), the Helmholtz-Hodge projection and the
convective term. The cell-centred blocks are those of
{ref}`theory-finite-volumes`.

## Locations and storage

An axis $(0, L)$ has $N = 2^l$ cells of size $h = L / N$, with the centres
$(i + \frac12) h$ and the faces $i h$, $i = 0, \dots, N$. There are $N + 1$
faces, not a power of two: the $N$ *left* faces $i = 0, \dots, N - 1$ are
stored. On a periodic axis, the face $N$ is the face 0. On a bounded axis,
the face 0 is a boundary face whose value is known (a wall, an inflow): it
stays in the vector, and every operator has a zero row there, so that a time
step leaves it unchanged; the face $N$ is not stored, and its value comes
from the condition of the field at $x = L$: $f_N = g$ (Dirichlet) or
$f_N = f_{N-1} + h g$ (Neumann, an outflow). In dimension $d$, every field
(the pressure, each velocity component, the corners) has $2^{l_\nu}$ values
per axis, and is stored as a QTT tensor in the usual layout.

## The one-dimensional blocks

| Block | Formula and closure |
| --- | --- |
| gradient $G$, cells to faces | $(c_i - c_{i-1}) / h$; face 0: zero row (bounded) or $(c_0 - c_{N-1}) / h$ (periodic) |
| divergence $D$, faces to cells | $(f_{i+1} - f_i) / h$, with $f_N$ from the condition |
| face Laplacian | $(f_{i-1} - 2 f_i + f_{i+1}) / h^2$; face 0: zero row; $f_N$ from the condition |
| cell Laplacian | the cell-centred closures of {ref}`theory-finite-volumes` |
| average, faces to cells | $(f_i + f_{i+1}) / 2$ |
| average, cells to faces | $(c_{i-1} + c_i) / 2$ |

The data of the conditions enter a source on the last cell or face, as for
the cell-centred finite volumes. Each block is a banded Toeplitz matrix plus
one or two corner corrections: its QTT ranks are at most 5, independently of
$N$.

```{prf:proposition} Exact identities of the staggered grid
:label: prop-staggered-identities

With the velocity known at both ends of the bounded axes (walls), or on
periodic axes,

1. $D\, G = L_p$, the cell-centred Laplacian with homogeneous Neumann
   closures (or the periodic one), in any dimension:
   $\sum_k D_k G_k = L_p$;
2. $G = -D^{\mathsf T}$ on the interior faces.
```

```{prf:proof}
Inside the axis, $(D G c)_i = ((G c)_{i+1} - (G c)_i) / h =
(c_{i+1} - 2 c_i + c_{i-1}) / h^2$. At the ends, $(G c)_0 = 0$ and
$(G c)_N = 0$ give $(c_1 - c_0)/h^2$ and $-(c_{N-1} - c_{N-2})/h^2$, the
cell-centred Laplacian with homogeneous Neumann closures. On a periodic axis
both matrices are circulant, and the identity holds without boundary terms.
In dimension $d$, each $D_k G_k$ is the one-dimensional identity embedded in
the layout. The second identity is read on the formulas of $G$ and $D$.
```

## The Helmholtz-Hodge projection

Every face velocity $w$ splits into $w = u + G \varphi$ with $D u = 0$: by
the first identity, $L_p \varphi = D w$, and $u = w - G \varphi$ is
divergence-free up to the precision of the solver. The system is solved in
the form $-L_p \varphi = -D w$, symmetric positive semi-definite. With walls
or periodic axes, its kernel is spanned by the constants, and it is
regularized by a rank-one term as in {prf:ref}`prop-fv-regularization`. The
regularization is added to $-L_p$: $L_p + \sigma q q^{\mathsf T}$ would be
invertible but indefinite, outside the assumptions of the conjugate
gradient and of the energy interpretation of the alternating solvers.

**Compatibility.** The singular problem has a solution only if the mean of
$D w$ vanishes, the discrete balance of the fluxes through the boundary. The
regularization does not repair an incompatible field: the defect is measured,
rejected above a tolerance, and removed below it, where it comes from
round-off and truncation errors.

**Orthogonality.** By the second identity, for $u$ divergence-free and with
homogeneous boundary data, $(u, G \varphi) = -(D u, \varphi) = 0$ and
$\|w\|^2 = \|u\|^2 + \|G \varphi\|^2$: $u$ is the orthogonal projection of
$w$ onto the divergence-free fields, which is the projection step of the
Chorin method.

**Outlet.** At an outlet $x = L$ with $p = 0$, the gradient on the face $N$
uses the ghost value $-p_{N-1}$, and the pressure problem becomes the
cell-centred Laplacian with a Dirichlet closure at the outlet, which is not
singular. The corrected velocity is divergence-free only with its corrected
outlet face $N$, which is not stored: the projection returns it.

**Discrete curl.** With a stream function $\psi$ at the corners, the
differences $u_x = D_y \psi$ and $u_y = -D_x \psi$ give a field with
$D u = D_x D_y \psi - D_y D_x \psi = 0$ exactly, since operators on different
axes commute: this builds exactly divergence-free fields.

## The convective term

In conservative form, $N_j = \sum_k \partial_k (u_j u_k)$ at the location of
$u_j$:

$$
N_j = G_j \big( (A^{f \to c}_j u_j)^2 \big)
  + \sum_{k \ne j} D_k \big( (A^{c \to f}_k u_j) \odot
  (A^{c \to f}_j u_k) \big),
$$

with $A^{f \to c}$ and $A^{c \to f}$ the averages of the table above, and
$\odot$ the entrywise product. For $k = j$, the product is formed at the
cells along $j$; for $k \ne j$, at the corners of the plane $(j, k)$. In QTT
format, each entrywise product is computed at a given precision
({ref}`theory-tensor-trains`).

```{prf:proposition} Conservation by the convective term
:label: prop-convection-conservation

On a periodic grid, $\sum_i (N_j)_i = 0$: the momentum is conserved. For a
field with $D u = 0$, $(u, N(u)) = 0$, with periodic axes or walls: the
kinetic energy is conserved by the convection.
```

The first property holds because $N_j$ is a sum of differences; the second
is the property of the conservative staggered schemes
{cite:p}`Morinishi1998`. On smooth fields the error against $(u \cdot \nabla) u$ is
$O(h^2)$.

## Time integration

**Heat and unsteady Stokes equations.** The viscous operators are the
Laplacians of the velocity components, with their rows of the stored
boundary faces masked. The sine modes are eigenvectors of the discrete
Laplacian with homogeneous Dirichlet conditions, on the faces as on the
cells, which gives exact references for explicit and implicit Euler. For the
unsteady Stokes equations, the incremental pressure correction

$$
\begin{aligned}
(I - \Delta t\, \nu L)\, u^* &= u^n - \Delta t\, G p^n, &
L_p \varphi &= D u^*, \\
u^{n+1} &= u^* - G \varphi, &
p^{n+1} &= p^n + \varphi / \Delta t,
\end{aligned}
$$

has the discrete steady Stokes solutions as fixed points, for every time
step.

**Navier-Stokes equations.** On a periodic domain, the method of lines on
the projected equations $\partial_t u = P(-N(u) + \nu L u)$, with $P$ the
Helmholtz-Hodge projection, and Heun's method (an explicit Runge-Kutta
method of order 2), each stage being projected. Since $P$ is linear and
idempotent, this is the projection of Heun's method for the semi-discrete
system on the divergence-free fields: second order in time.

## In tensap

- {py:class}`~tensap.tensorized_pde_discretization.finite_volumes.StaggeredAxis`
  (the 1D blocks) and
  {py:class}`~tensap.tensorized_pde_discretization.finite_volumes.StaggeredGrid`
  (assembled operators, `project`, `convection`, `curl`, boundary sources).
- Use case: {ref}`sphx_glr_use_cases_taylor_green_vortex.py`.
