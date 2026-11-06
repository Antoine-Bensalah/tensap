(theory-finite-volumes)=
# Cell-centred finite volumes in QTT format

The Poisson problem $-\Delta u = f$ on a box $\Omega = \prod_\nu (0, a_\nu)$,
with boundary conditions on its faces, is discretized by cell-centred finite
volumes on a dyadic grid. The grid functions and the operators are tensor
trains in the quantized format ({ref}`theory-qtt`), so that the cost depends
on their ranks, and only logarithmically on the number of cells.

## The scheme

Each axis is divided into $N_\nu = 2^{L_\nu}$ cells of size
$h_\nu = a_\nu / N_\nu$, and the unknowns $u_i$ approximate $u$ at the cell
centres $x_i = (i + \frac12) h_\nu$. These are the cells of the dyadic grid
of a {py:class}`~tensap.Tensorizer` with $y_0 = 1/2$: the $L_\nu$ digits of
$i$ are the levels of the dyadic refinement of the axis.

**From the fluxes.** Integrating $-u'' = f$ on the cell
$K_i = (x_{i - 1/2}, x_{i + 1/2})$ gives the exact balance of the fluxes
through its two faces,
$-\big(u'(x_{i+1/2}) - u'(x_{i-1/2})\big) = \int_{K_i} f$. Approximating the
flux at a face by the centred difference of the two neighbouring cells, and
dividing by $h$,

$$
-\frac{u_{i-1} - 2 u_i + u_{i+1}}{h^2} = f_i ,
\qquad f_i \approx \frac{1}{h} \int_{K_i} f ,
$$

with a local truncation error $O(h^2)$ in the interior cells.

**Closures.** At a face, the missing neighbour is a ghost value $u_g$ set by
the boundary condition:

- Dirichlet $u = g$: the face value is the mean of the ghost and of the first
  cell, $(u_g + u_0)/2 = g$, so that $u_g = 2 g - u_0$; the first row becomes
  $(-3 u_0 + u_1)/h^2$, and the source $2 g / h^2$ goes to the right-hand
  side;
- Neumann $\partial_n u = g$, with the outward normal: the centred flux at
  the face gives $u_g = u_0 + h g$; the first row becomes $(-u_0 + u_1)/h^2$,
  with the source $g / h$;
- periodic axis: the neighbour across the face is the cell at the other end
  (a circulant matrix).

In QTT format, the 1D operator is a banded matrix (rank 3) plus two corner
corrections of rank 1, rounded ({ref}`theory-qtt`): its QTT ranks are at most
5, independently of $N$. The operator only depends on the *types* of the
conditions; their data enter a source $s$, so that $-\Delta u = f$ becomes
$-\Delta_h u = f + s$. On a face normal to the axis $\nu$, the source is
$w\, e_0 \otimes g$, with the weight $w$ of the closure, $e_0$ the first
canonical vector on the digits of the axis $\nu$ (rank 1), and $g$ the data
on the face as a QTT tensor on the other axes: no array of the size of the
grid is formed.

**Assembly.** $\Delta_h = \sum_\nu I \otimes \cdots \otimes D_2^{(\nu)}
\otimes \cdots \otimes I$, assembled in the layout of the
{py:class}`~tensap.Tensorizer` ({ref}`theory-qtt`): with the coordinate-wise
ordering, the chains of cores are concatenated; with the resolution-wise
ordering, the ranks of the axes multiply where their digits alternate. The
ordering has a much larger impact on the ranks of the solutions than on those
of the operator: separable functions are better compressed coordinate-wise.

## Spectra and accuracy

```{prf:proposition} Spectra of the 1D operators
:label: prop-fv-spectra

The matrix $-D_2$ is symmetric, its eigenvalues are
$\frac{4}{h^2} \sin^2 \frac{\theta}{2}$, and its eigenvectors are sampled
sines or cosines $v_i$, $i = 0, \dots, N - 1$:

| Closures | Eigenvectors | Frequencies $\theta$ |
| --- | --- | --- |
| Dirichlet / Dirichlet | $\sin(\theta (i + \frac12))$ | $k \pi / N$, $k = 1, \dots, N$ |
| Dirichlet / Neumann | $\sin(\theta (i + \frac12))$ | $(2k + 1) \pi / (2N)$, $k = 0, \dots, N - 1$ |
| Neumann / Neumann | $\cos(\theta (i + \frac12))$ | $k \pi / N$, $k = 0, \dots, N - 1$ |
| periodic | $e^{\mathrm{i} \theta i}$ | $2 k \pi / N$, $k = 0, \dots, N - 1$ |
```

```{prf:proof}
In the interior, $-(v_{i-1} - 2 v_i + v_{i+1}) = (2 - 2 \cos\theta)\, v_i =
4 \sin^2(\theta / 2)\, v_i$ for $v_i$ a sine, a cosine or an exponential of
$\theta i + \varphi$. The closures select $\theta$: the Dirichlet closure
$v_{-1} = -v_0$ holds for $\sin(\theta(i + \frac12))$, odd about
$i = -\frac12$; the Neumann closure $v_{-1} = v_0$ for
$\cos(\theta(i + \frac12))$, even about it; the same symmetry about
$i = N - \frac12$ quantizes $\theta$.
```

With at least one Dirichlet face, $-D_2$ is positive definite, with a
condition number of order $N^2$. In dimension $d$, the eigenvalues of
$-\Delta_h$ are the sums of the eigenvalues of the axes.

**Order of convergence.** Near a Dirichlet face, the ghost value is a linear
extrapolation, which gives a local truncation error $O(1)$ in the first cell
only. The error is nevertheless $O(h^2)$ everywhere (*supraconvergence*): in
the error equation, the discrete Green function next to a Dirichlet face is
$O(h)$, so that the truncation error of one cell of width $h$ contributes
$O(h^2)$. The Neumann closure is consistent of order 1 in its cell, and also
gives a global $O(h^2)$.

## Singular problems

With every axis periodic, or with Neumann conditions on all the faces, the
constants span the kernel of $-\Delta_h$. The problem is regularized by a
term of rank one,

$$
A_\sigma = -\Delta_h + \frac{\sigma}{N}\, \mathbf{1} \mathbf{1}^{\mathsf T},
\qquad N = \prod_\nu N_\nu .
$$

```{prf:proposition} Rank-one regularization
:label: prop-fv-regularization

$A_\sigma$ is symmetric positive definite, with the eigenvalues of
$-\Delta_h$ in which 0 is replaced by $\sigma$. The solution of
$A_\sigma u = b$ is $u = u^\perp + (\bar b / \sigma) \mathbf{1}$, with
$\bar b$ the mean of $b$ and $u^\perp$ the solution of zero mean of
$-\Delta_h u^\perp = b - \bar b\, \mathbf{1}$. For compatible data,
$\bar b = 0$, it is the solution of zero mean.
```

```{prf:proof}
Write $u = u^\perp + c\, \mathbf{1}$ with $u^\perp$ of zero mean. Since
$-\Delta_h \mathbf{1} = 0$ and $\mathbf{1}^{\mathsf T} u^\perp = 0$,
$A_\sigma u = -\Delta_h u^\perp + \sigma c\, \mathbf{1}$. Projecting
$A_\sigma u = b$ on $\mathbf{1}$ and on its orthogonal complement gives
$c = \bar b / \sigma$ and the equation of $u^\perp$.
```

The compatibility condition $\bar b = 0$ is the discrete form of
$\int_\Omega f + \int_{\partial\Omega} g = 0$: summing the equations over
the cells, the interior fluxes cancel two by two. The conditioning of
$A_\sigma$ is that of $-\Delta_h$ on the functions of zero mean if $\sigma$
is at least the smallest non-zero eigenvalue $\lambda_1$, the default. In
QTT format, $\mathbf{1} \mathbf{1}^{\mathsf T}$ has rank 1. The alternatives
(fixing one cell, a Lagrange multiplier, a shift $-\Delta_h + \varepsilon
I$) break the symmetry or the tensor structure, or change the solution.

## In tensap

- {py:class}`~tensap.tensorized_pde_discretization.Dirichlet`,
  {py:class}`~tensap.tensorized_pde_discretization.Neumann`: the conditions,
  independent of the grid.
- {py:class}`~tensap.tensorized_pde_discretization.finite_volumes.Axis` and
  {py:class}`~tensap.tensorized_pde_discretization.finite_volumes.Grid`: the
  1D and assembled operators, the boundary sources and the regularization.
- Example: {ref}`sphx_glr_examples_solvers_qtt_poisson.py`.
