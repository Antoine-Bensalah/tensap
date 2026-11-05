# Multi-indices and grids

## `MultiIndices`

A multi-index is a tuple $(i_1,\ldots,i_d) \in {\mathbb{N}}_0^d$.
A set $I \subset {\mathbb{N}}_0^d$ of multi-indices is represented
with an object {py:class}`~tensap.MultiIndices`.

To create a multi-index set $I$, we use the command
`tensap.MultiIndices(I)` with `I` a numpy array of size
$\#I \times d$.

A product set $I = I_1 \times \ldots \times I_d$ can be obtained
with `tensap.MultiIndices.product_set([I1,...,Id])`.

The set of multi-indices

$$I = \{i  \in {\mathbb{N}}_0^d : \Vert i \Vert_{\ell^p} \le m\}$$

can be obtained with `tensap.MultiIndices.with_bounded_norm(d, p,
m)`

The set of multi-indices

$$I = \{i  \in {\mathbb{N}}_0^d : i_\nu \le m_\nu , 1\le \nu \le d\}$$

can be obtained with `tensap.MultiIndices.bounded_by(d, p, m)`. If
$m$ is of length $1$, it uses $m_\nu = m$ for all
$\nu$.

For obtaining the margin or reduced margin of an multi-index set
$I$, we can use the methods `I.get_margin()` and `I.get_reduced_margin()`.

## `TensorGrid`, `FullTensorGrid` and `SparseTensorGrid`

Tensor product grids or sparse grids are represented with classes {py:class}`~tensap.FullTensorGrid` and {py:class}`~tensap.SparseTensorGrid`, that inherit from {py:class}`~tensap.TensorGrid`.
