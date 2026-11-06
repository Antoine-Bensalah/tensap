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
```
