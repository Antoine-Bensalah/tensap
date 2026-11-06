(theory-multi-indices)=
# Multi-indices and grids

Sparse tensor product bases are indexed by sets of multi-indices, and
interpolation or quadrature uses tensor product grids or sparse grids. This
page defines the sets used in tensap.

## Sets of multi-indices

A *multi-index* is a tuple $i = (i_1, \dots, i_d) \in \mathbb{N}_0^d$, and a
set of multi-indices $\Lambda \subset \mathbb{N}_0^d$ indexes the products
$\varphi^{(1)}_{i_1}(x_1) \cdots \varphi^{(d)}_{i_d}(x_d)$ of a sparse tensor
product basis ({ref}`theory-functions`). Usual sets are:

- the *product set* $\Lambda = I_1 \times \cdots \times I_d$, which gives a
  full tensor product basis;
- the *box* $\Lambda = \{i : i_\nu \le m_\nu,\ 1 \le \nu \le d\}$;
- the *$\ell^p$ ball* $\Lambda = \{i : \|i\|_{\ell^p} \le m\}$, which gives
  for $p = 1$ the polynomials of total degree at most $m$, and, for $p < 1$,
  hyperbolic cross sets.

```{prf:definition} Downward closed set, margins
:label: def-downward-closed

A set $\Lambda$ is *downward closed* if $i \in \Lambda$ and $j \le i$
(componentwise) imply $j \in \Lambda$. The *margin* of $\Lambda$ is the set
of the multi-indices $i \notin \Lambda$ such that $i - e_\nu \in \Lambda$ for
some $\nu$ with $i_\nu > 0$, and its *reduced margin* is the set of the
$i \notin \Lambda$ such that $i - e_\nu \in \Lambda$ for every $\nu$ with
$i_\nu > 0$.
```

Adding an element of the reduced margin to a downward closed set gives a
downward closed set: the adaptive algorithms that enrich a sparse polynomial
basis select the new multi-indices in the reduced margin.

## Grids

A *tensor product grid* is a product $G_1 \times \cdots \times G_d$ of grids
of one variable, of size $\prod_\nu \#G_\nu$. A *sparse grid* is the set of
the points $(x_{1, i_1}, \dots, x_{d, i_d})$ for the multi-indices $i$ of a
set $\Lambda$, which avoids the exponential growth of the full grid with the
dimension.

## In tensap

- {py:class}`~tensap.MultiIndices`: product sets (`product_set`), sets with
  a bounded norm (`with_bounded_norm`), boxes (`bounded_by`), margins
  (`get_margin`, `get_reduced_margin`).
- {py:class}`~tensap.FullTensorGrid`, {py:class}`~tensap.SparseTensorGrid`.
- Examples: {ref}`sphx_glr_examples_functions_multi_indices.py`,
  {ref}`sphx_glr_examples_functions_functions_bases_grids.py`.
