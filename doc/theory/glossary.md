(theory-glossary)=
# Glossary

Short definitions of the terms used in Theory & Methods, with the page that
develops each of them. The terms are also listed in the {ref}`genindex`.

```{glossary}
:sorted:

alpha-rank
  The rank of the matricization $M_\alpha(u)$ of a tensor $u$ for a set
  $\alpha$ of dimensions. See {doc}`tucker_canonical`.

alpha-singular values
  The singular values of the matricization $M_\alpha(u)$; they give the
  error of the best approximation with a bounded $\alpha$-rank. See
  {doc}`truncation`.

canonical format
  A sum of $R$ elementary tensors; the Tucker format with a diagonal core.
  See {doc}`tucker_canonical`.

canonical rank
  The smallest number of elementary tensors whose sum is a given tensor.
  See {doc}`tucker_canonical`.

core
  The tensor of coefficients of a Tucker tensor on the products of the
  vectors of its frames; it can be full, diagonal, tree-based or a tensor
  train. See {doc}`tucker_canonical`.

elementary tensor
  A tensor product $v^{(1)} \otimes \cdots \otimes v^{(d)}$ of vectors, also
  called a rank-one tensor.

frame
  The family of vectors (or matrices, for an operator) of one dimension of a
  Tucker tensor. See {doc}`tucker_canonical`.

hierarchical SVD
  The truncation in a tree-based format by truncated SVDs from the leaves
  to the root of the dimension tree. See {doc}`truncation`.

HOSVD
  Higher-order singular value decomposition: the truncation in Tucker
  format by a truncated SVD of each matricization. See {doc}`truncation`.

matricization
  The matrix $M_\alpha(u)$ of the entries of a tensor $u$, with the indices
  of the dimensions $\alpha$ as rows and the others as columns. See
  {doc}`tucker_canonical`.

representation rank
  The sizes of a representation in a low-rank format, which may exceed
  the ranks of the tensor. See {doc}`tucker_canonical`.

truncation
  The approximation of a tensor by a tensor of smaller ranks, at a
  relative precision or with a maximal rank. See {doc}`truncation`.

Tucker format
  A core contracted with one frame per dimension. See
  {doc}`tucker_canonical`.

Tucker rank
  The tuple of the ranks of the matricizations $M_\nu(u)$, the smallest
  representation ranks in Tucker format. See {doc}`tucker_canonical`.

alternating least squares
  ALS: the minimization of a functional over a low-rank format, one group
  of parameters at a time, each step being a linear problem. See
  {doc}`greedy`.

energy
  The functional $J(v) = \frac12 (A v, v) - (b, v)$, minimized by the
  solution of $A u = b$ for a symmetric positive definite $A$; its
  minimization on a set of tensors is the best approximation in the energy
  norm.

Galerkin update
  The minimization of the energy over the core or over the frames of a
  low-rank iterate, the other parameters being fixed. See
  {doc}`galerkin_updates`.

greedy algorithm
  The construction of an approximation by successive rank-one
  corrections, also called proper generalized decomposition (PGD). See
  {doc}`greedy`.

truncated iteration
  A classical iterative method (Richardson, conjugate gradient) whose
  iterates are truncated after each step. See {doc}`truncated_iterations`.

orthogonality center
  The core of a tensor train whose left cores are left-orthogonal and
  right cores right-orthogonal; the norm of the tensor is that of this core.
  See {doc}`tensor_trains`.

rounding
  The truncation of a tensor train by a sweep of truncated SVDs of its
  cores. See {doc}`tensor_trains`.

tensor train
  TT: a tensor written as a product of matrices $C^1(i_1) \cdots
  C^d(i_d)$, one core per dimension. See {doc}`tensor_trains`.

TT rank
  The ranks $r_\nu$ of a tensor train, those of its sequential
  matricizations. See {doc}`tensor_trains`.

AMEn
  Alternating minimal energy: ALS in TT format with the cores enriched by
  directions of the residual, which adapts the ranks. See {doc}`tt_solvers`.

DMRG
  Density matrix renormalization group: ALS in TT format over two
  neighbouring cores, whose truncated SVD adapts the rank. See
  {doc}`tt_solvers`.

interface
  The contraction of the cores on one side of a core of a tensor network
  with those of the operator and of the iterate, which defines the local
  problems of the alternating solvers. See {doc}`tt_solvers`.
```
