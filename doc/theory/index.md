# Theory & Methods

This section presents the mathematical objects of tensap and the methods that
work on them, with the notations used throughout the documentation. Each page
links the classes it describes to their entry in the {doc}`/api/index`, and
to the examples of the {doc}`/examples/index` that use them. References are
listed in the {ref}`bibliography`.

The pages follow the plan of A. Nouy's course on tensor networks for
scientific computing, restricted to what tensap implements: tensors and
tensor networks, tensorization, truncation, learning, equations in tensor
formats, and the approximation of functions, followed by the discretization
of PDEs in QTT format. Each concept page gives definitions, properties and
their proofs or references; each method page states the problem, the
assumptions, the algorithm, its properties and its cost, and distinguishes
established results from numerical observations. The pages replace the tensap
manual (`manual/tensap_manual.tex`).

## Conventions

The notation is that of A. Nouy's courses and survey {cite:p}`Nouy2017book`.

- **Tensors.** $d$ is the order and $D = \{1, \dots, d\}$ the set of the
  dimensions, indexed by $\nu$ (or $\mu$). The dimension $\nu$ has the index
  set $I_\nu = \{1, \dots, N_\nu\}$, and the tensor space is
  $V = \mathbb{R}^{I_1} \otimes \cdots \otimes \mathbb{R}^{I_d}$. The entries
  of a tensor $u \in V$ are written $u(i_1, \dots, i_d)$, or $u(i)$ with
  $i = (i_1, \dots, i_d)$, with a single level of indices. In the formulas
  the indices start at 1; in the code they start at 0.
- **Elementary tensors.** $v^{(1)} \otimes \cdots \otimes v^{(d)}$ has the
  entries $v^{(1)}(i_1) \cdots v^{(d)}(i_d)$; the superscript $(\nu)$ marks an
  object of the dimension $\nu$.
- **Inner product and norm.** $(u, v) = \sum_i u(i)\, v(i)$ and
  $\|u\| = (u, u)^{1/2}$, the canonical (Frobenius) norm.
- **Groups of dimensions.** For $\alpha \subset D$, $\alpha^c = D \setminus
  \alpha$ is its complement and $\#\alpha$ its cardinality. The
  matricization $M_\alpha(u)$ is the matrix of the entries of $u$ with the
  indices of $\alpha$ as rows, and $\rank_\alpha(u)$ its rank.
- **Ranks.** $r$ denotes a rank (or a bound on a rank), and $R$ a
  representation rank, the size of a representation, which may be larger.
- **Trees.** $T$ is a dimension tree over $D$, $S(\alpha)$ the children of a
  node $\alpha \in T$ and $\mathcal{L}(T)$ its leaves.
- **Linear systems.** $A u = b$, with the residual $s = b - A v$ of an
  approximation $v$, and, for a symmetric positive definite $A$, the energy
  $J(v) = \frac12 (A v, v) - (b, v)$. A truncation is written $\Pi$.
- The macros of the manual (`doc/math_notations.sty`) are available in these
  pages: `\Rbb` gives $\Rbb$, `\Ebb` gives $\Ebb$, `\Lc` gives $\Lc$.

## Contents

| Part | What it covers |
| --- | --- |
| {doc}`part_tensors` | Tensors and ranks; canonical, Tucker, tree-based and tensor-train formats; low-rank formats for functions |
| {doc}`part_tensorization` | Tensorization of vectors and functions, quantized tensor trains |
| {doc}`part_truncation` | Truncation by singular value decompositions in every format |
| {doc}`part_learning` | Learning with linear models and tensor formats, principal component analysis |
| {doc}`part_equations` | Linear systems in tensor formats: truncated iterations, greedy and alternating solvers, preconditioning |
| {doc}`part_pde` | Discretization of PDEs in QTT format: finite volumes, staggered grid |
| {doc}`part_approximation` | Multi-indices and grids for sparse approximation |

The {doc}`glossary` gives short definitions of the terms used in these
pages, with links to the pages that develop them.

```{toctree}
:hidden:
:maxdepth: 2
:numbered: 3

part_tensors
part_tensorization
part_truncation
part_learning
part_equations
part_pde
part_approximation
glossary
```
