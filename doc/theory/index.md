# Theory & Methods

This section presents the mathematical objects of tensap and the methods that
work on them, with the notations used throughout the documentation. Each page
links the classes it describes to their entry in the {doc}`/api/index`, and
to the examples of the {doc}`/examples/index` that use them. References are
listed in the {ref}`bibliography`.

```{note}
These pages come from the tensap manual (`manual/tensap_manual.tex`), which
they will replace. They are still organised by class, as the manual is; they
will be reorganised by concept and by method.
```

## Conventions

- A tensor of order $d$ is an array $X \in \mathbb{R}^{N_1 \times \cdots \times N_d}$.
  Its indices start at 0, as in Python: $i_\nu \in \{0, \ldots, N_\nu - 1\}$.
- The modes, or dimensions, are numbered $\nu = 1, \ldots, d$, and
  $D = \{1, \ldots, d\}$. In the code they are numbered from 0.
- For a subset $\alpha \subset D$, $\alpha^c = D \setminus \alpha$ is its
  complement, and $\#\alpha$ its cardinality.
- $T$ denotes a dimension tree over $D$, $S(\alpha)$ the children of a node
  $\alpha \in T$ and $\mathcal{L}(T)$ its leaves (see {doc}`tensor_formats`).
- The macros of the manual (`doc/math_notations.sty`) are available in these
  pages: `\Rbb` gives $\Rbb$, `\Ebb` gives $\Ebb$, `\Lc` gives $\Lc$.

## Contents

| Page | What it covers |
| --- | --- |
| {doc}`full_tensors` | Full tensors: creation, entries, reshaping, norms, products and contractions |
| {doc}`tensor_formats` | Canonical, diagonal, sparse and tree-based tensors, dimension trees and truncation |
| {doc}`functions` | Random variables and vectors, polynomials, functional bases and functional tensors |
| {doc}`multi_indices_grids` | Multi-indices and tensor grids |
| {doc}`learning` | Principal component analysis, loss functions and learning algorithms |

```{toctree}
:hidden:
:maxdepth: 2
:numbered: 3

full_tensors
tensor_formats
functions
multi_indices_grids
learning
```
