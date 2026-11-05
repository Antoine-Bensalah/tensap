# Tensor formats

Here we present tensor formats available in tensap, which are structured
formats of tensors in $\mathbb{R}^{N_1\times \ldots \times N_d}.$
For a detailed description of methods, see the description of the
corresponding methods for {py:class}`~tensap.FullTensor` in . For an introduction to
tensor formats, we refer to the monograph {cite:p}`Hackbusch2019` and the survey
{cite:p}`Nouy2017book`.

## `CanonicalTensor`

The entries of an order-$d$ tensor
$X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$ in canonical
format can be written

$$
            X_{i_1, \ldots, i_d} = \sum_{k=1}^r C_k U^1_{i_1, k} \cdots U^d_{i_d, k},$$

with $r$ the canonical rank, and where the
$U_\nu = (U^\nu_{i_\nu, k})_{1\le i_\nu \le N_\nu , 1\le k \le r}$
are order-two tensors.

### Creating a `CanonicalTensor`
To create a canonical tensor in tensap, one can use the command
`tensap.CanonicalTensor(C, U)`, where `C` contains the
$(C_k)_{k=1}^d$, and `U` is a list containing the $U^\nu$,
$1\le \nu\le d$.

The storage complexity of such a tensor, obtained with `X.storage()`, is equal to $r(1 + N_1 + \cdots + N_d)$.  
It is also possible to generate a {py:class}`~tensap.CanonicalTensor` with entries

- equal to 0 with `tensap.CanonicalTensor.zeros(r, [N_1, ...,
  N_d])`,
- equal to 1 with `tensap.CanonicalTensor.ones(r, [N_1, ...,
  N_d])`,
- drawn randomly according to the uniform distribution on
  $[0, 1]$ with `tensap.CanonicalTensor.rand(r, [N_1, ...,
  N_d])`,
- drawn randomly according to the standard gaussian distribution with
  `tensap.CanonicalTensor.randn(r, [N_1, ..., N_d])`,
- generated using a provided `generator` with
  `tensap.CanonicalTensor.create` `(generator, r, [N_1, ...,
  N_d])`.

### Converting a `CanonicalTensor` to a `FullTensor`
A {py:class}`~tensap.CanonicalTensor` `X` can be converted to a {py:class}`~tensap.FullTensor`
(introduced in {ref}`theory-full-tensors`) with the command `X.full()`.

### Converting a `CanonicalTensor` to a `TreeBasedTensor`
A {py:class}`~tensap.CanonicalTensor` `X` can be converted to a {py:class}`~tensap.TreeBasedTensor`
(introduced in {ref}`theory-tree-based-tensor`) with the command
`X.tree_based_tensor(tree, is_active_node)`, with `tree` a
{py:class}`~tensap.DimensionTree` object, and `is_active_node` a list or array of
booleans indicating if each node of the tree is active.

### Accessing the diagonal of a `CanonicalTensor`
For a canonical tensor
$X \in {\mathbb{R}}^{N \times \ldots \times N}$, the command
`X.eval_diag()` returns the diagonal $X_{i, \ldots, i}$,
$i = 1, \ldots, N$, of the tensor. The method `eval_diag` can
also be used to evaluate the diagonal in a subset of dimensions `dims`
of the tensor with `X.eval_diag(dims)`, which returns a
{py:class}`~tensap.CanonicalTensor`.

### Computing the Frobenius norm of a `CanonicalTensor`
The command `X.norm()` returns the Frobenius norm of $X$. The
Frobenius norm of $X$ is equal to the Frobenius norm of its core
$C$ if `X.is_orth` is `True`.

### Computing the derivative of `CanonicalTensor` with respect to one of its parameters
Given an order-$d$ canonical tensor $X$ in
${\mathbb{R}}^{N \times \cdots \times N}$, the command
`X.parameter_gradient_eval_diag(k)`, for $1 \leq k \leq d$,
returns the derivative

$$\left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial U^k}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N.$$

The derivative of $X$ with respect to its core $C$, that
writes

$$\left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial C}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N,$$

is obtained with `X.parameter_gradient_eval_diag(d+1)`.

The method `parameter_gradient_eval_diag` is used in the
statistical learning algorithms presented in Section
{ref}`theory-tensor-learning`.

### Performing operations with `CanonicalTensor`
Some operations between tensors are implemented for {py:class}`~tensap.DiagonalTensor`
(see for a detailed description of the operations): the Kronecker
product with `kron`, the contraction with matrices with
`tensor_matrix_product`, the evaluation of the diagonal of a
contraction with matrices with `tensor_matrix_product_eval_diag`,
the dot product with `dot`.

Given a tensor $X$ and a list of matrices
$M = [M^1, ..., M^d]$, the command `Z =
X.tensor_matrix_product(M)` returns an order-d tensor $Z$ whose
entries are

$$\begin{aligned}
Z_{i_1, \ldots, i_d} = \sum_{k=1}^r \sum_{\substack{k_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} C_k U^1_{k_1, k} \cdots U^d_{k_d, k} \prod_{\nu = 1}^d M^\nu_{i_\nu, k_\nu}
\end{aligned}$$

The method `tensor_matrix_product_eval_diag` evaluates the
diagonal of the result of `tensor_matrix_product`.

The dot product of two canonical tensors $X$ and $Y$ with
same shape $[N_1, \ldots, N_d]$ can be obtained with `X.dot(Y)`.

## `DiagonalTensor`

A diagonal tensor
$X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$ is a tensor
whose entries $X_{i_1, \ldots, i_d}$ are non-zero only if
$i_1 = \cdots = i_d$.

### Creating a `DiagonalTensor`
To create a diagonal tensor in tensap, one can use the command
`tensap.DiagonalTensor(D, d)`, where `D` (of length $r$)
contains the diagonal of the tensor, and `d` is the order of the
tensor. The result if an order $d$ tensor in
$\mathbb{R}^{r\times \ldots \times r} = \mathbb{R}^{ r^d}$.

The sparse storage complexity of such a tensor, obtained with
`X.sparse_storage()`, is equal to `r = len(D)`. Its storage
complexity, not taking into account the fact that only the diagonal is
non-zero, is equal to $r^d$ and obtained with `X.storage()`.

It is also possible to generate a {py:class}`~tensap.DiagonalTensor` with entries

- equal to 0 with `tensap.DiagonalTensor.zeros(r, d)`,
- equal to 1 with `tensap.DiagonalTensor.ones(r, d)`,
- drawn randomly according to the uniform distribution on
  $[0, 1]$ with `tensap.DiagonalTensor.rand(r, d)`,
- drawn randomly according to the standard gaussian distribution with
  `tensap.DiagonalTensor.randn(r, d)`,
- generated using a provided `generator` with
  `tensap.DiagonalTensor.create` `(generator, r, d)`.

### Converting a `DiagonalTensor` to a `FullTensor`
A {py:class}`~tensap.DiagonalTensor` `X` can be converted to a {py:class}`~tensap.FullTensor`
(introduced in {ref}`theory-full-tensors`) with the command `X.full()`.

### Converting a `DiagonalTensor` to a `SparseTensor`
A {py:class}`~tensap.DiagonalTensor` `X` can be converted to a {py:class}`~tensap.SparseTensor`
(introduced in {ref}`theory-sparse-tensor`) with the command
`X.sparse()`.

### Converting a `DiagonalTensor` to a `TreeBasedTensor`
A {py:class}`~tensap.DiagonalTensor` `X` can be converted to a {py:class}`~tensap.TreeBasedTensor`
(introduced in {ref}`theory-tree-based-tensor`) with the command
`X.tree_based_tensor(tree, is_active_node)`, with `tree` a
{py:class}`~tensap.DimensionTree` object, and `is_active_node` a list or array of
booleans indicating if each node of the tree is active.

### Accessing the entries of a `DiagonalTensor`
The entries of the tensor `X` can be accessed with the method `eval_at_indices`: `X.eval_at_indices(ind)` returns the entries of $X$ indexed by the list `ind` containing the indices to access in each dimension.  
A sub-tensor can be extracted from `X` with the method `sub_tensor`.

For a tensor $X \in {\mathbb{R}}^{N \times \ldots \times N}$, the
command `X.eval_diag()` returns the diagonal
$X_{i, \ldots, i}$, $i = 1, \ldots, N$, of the tensor. The
method `eval_diag` can also be used to evaluate the diagonal in some
dimensions `dims` of the tensor with `X.eval_diag(dims)`.

### Computing the Frobenius norm of a `DiagonalTensor`
The command `X.norm()` returns the Frobenius norm of $X$.

### Performing operations with `DiagonalTensor`
Some operations between tensors are implemented for {py:class}`~tensap.DiagonalTensor`
(see for a detailed description of the operations): the outer product
with `tensordot`, the evaluation of the diagonal (or subtensors) of an
outer product with `tensordot_eval_diag`, the Kronecker product with
`kron`, the contraction with matrices or vectors with
`tensor_matrix_product` or `tensor_vector_product` respectively,
the evaluation of the diagonal of a contraction with matrices with
`tensor_matrix_product_eval_diag`, the dot product with `dot`.

(theory-sparse-tensor)=
## `SparseTensor`

A sparse tensor
$X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$ is a tensor
whose entries $X_{i_1, \ldots, i_d}$ are non-zero only for
$(i_1, \ldots, i_d) \in I$, with $I$ a set of multi-indices.

### Creating a `SparseTensor`
To create a sparse tensor $\texttt{X}$ in tensap,
one can use the command `tensap.SparseTensor(D, I, [N_1, ...,
N_d])`, where `D` contains the non-zero entries of $X$, `I`
is a {py:class}`~tensap.MultiIndices` containing the indices of its non-zero
enties, and where $N_1, \ldots, N_d$ is its shape.

The sparse storage complexity of such a tensor, obtained with
`X.sparse_storage()`, is equal to $\text{card}(I)$. Its storage
complexity, not taking into account the sparsity, is equal to
$N_1 \cdots N_d$ and can be accessed with `X.storage()`.

### Converting a `SparseTensor` to a `FullTensor`
A {py:class}`~tensap.SparseTensor` `X` can be converted to a {py:class}`~tensap.FullTensor`
(introduced in {ref}`theory-full-tensors`) with the command `X.full()`.

### Converting a `FullTensor` to a `SparseTensor`
A {py:class}`~tensap.FullTensor` `X` can be converted to a {py:class}`~tensap.SparseTensor`
(introduced in {ref}`theory-sparse-tensor`) with the command
`X.sparse()`.

### Accessing the entries of a `SparseTensor`
The entries of the tensor `X` can be accessed with the method
`eval_at_indices`: `X.eval_at_indices(ind)` returns the entries
of $X$ indexed by the list `ind` containing the indices to
access in each dimension.

A sub-tensor can be extracted from `X` with the method
`sub_tensor`.

For a tensor $X \in {\mathbb{R}}^{N, \ldots, N}$, the command
`X.eval_diag()` returns the diagonal $X_{i, \ldots, i}$,
$i = 1, \ldots, N$, of the tensor. The method `eval_diag` can
also be used to evaluate the diagonal in some dimensions `dims` of the
tensor with `X.eval_diag(dims)`.

### Reshaping a `SparseTensor`
The method `reshape` reshapes a {py:class}`~tensap.SparseTensor` using the
Fortran-like index order of numpy's reshape function.

The methods `transpose` and `itranspose` permute the dimensions of a
tensor `X`, given a permutation `dims` of $\{1, \ldots, d\}$.
They are such that `X = X.transpose(dims).itranspose(dims)`.

### Computing the Frobenius norm of a `SparseTensor`
The command `X.norm()` returns the Frobenius norm of $X$.

### Performing operations with `SparseTensor`
Some operations between tensors are implemented for {py:class}`~tensap.SparseTensor`
(see for a detailed description of the operations): the Kronecker
product with `kron`, the contraction with matrices or vectors with
`tensor_matrix_product` or `tensor_vector_product` respectively,
the evaluation of the diagonal of a contraction with matrices with
`tensor_matrix_product_eval_diag`, the dot product with `dot`.

## `TreeBasedTensor` and `DimensionTree`

We present in this section the {py:class}`~tensap.DimensionTree` and {py:class}`~tensap.TreeBasedTensor`
objects.

### `DimensionTree`

A dimension tree $T$ is a collection of non-empty subsets of $D = \{1, \ldots, d\}$ which is such that (i) all nodes $\alpha \in T$ are non-empty subsets of $D$, (ii) $D$ is the root of $T$, (iii) every node $\alpha \in T$ with $\#\alpha \ge 2$ has at least two children and the set of children of $\alpha$, denoted by $S(\alpha)$, is a non-trivial partition of $\alpha$, and (iv) every node $\alpha$ with $\#\alpha = 1$ has no child and is called a leaf (see for example {numref}`fig-dimension-trees`).  
We let $\operatorname{depth}(T) = \max_{\alpha \in T} \operatorname{level}(\alpha)$ be the depth of $T$, and ${\mathcal{L}}(T)$ be the set of leaves of $T$, which are such that $S(\alpha) = \emptyset$ for all $\alpha \in {\mathcal{L}}(T)$.


```{figure} images/dimension_trees.svg
:name: fig-dimension-trees
:alt: A trivial tree, a linear binary tree and a balanced binary tree, for d = 4.

Examples of dimension trees over $D = \{1, 2, 3, 4\}$: a trivial tree (left),
a linear binary tree (middle) and a balanced binary tree (right).
```

#### Creating a `DimensionTree`
A {py:class}`~tensap.DimensionTree` is characterized by its adjacency matrix and the
dimension associated with each leaf node: `T =
tensap.DimensionTree(dims, adjacency_matrix)`. The adjacency matrix of
a dimension tree $T$ can be accessed with `T.adjacency_matrix`.
The dimension associated with each leaf node can be accessed with
`T.dim2ind`.

Denoting by `order` the number of leaf nodes, it is possible to create

- a trivial tree with `tensap.DimensionTree.trivial(order)` ({numref}`fig-dimension-trees`, left),
- a linear tree with `tensap.DimensionTree.linear(order)` ({numref}`fig-dimension-trees`, middle),
- a balanced tree with
  `tensap.DimensionTree.balanced(order)` ({numref}`fig-dimension-trees`, right),
- a random tree with `tensap.DimensionTree.random(order, arity)`,
  with `arity` the arity of the tree, equal to the maximum number of
  children per node (randomly selected in an interval if provided).

Finally, a dimension tree can be created by extracting a sub-tree from
an existing tree $T$ with `T.sub_dimension_tree(root)` where
`root` is the node in $T$ that will become the root node of the
extracted tree.

#### Displaying a `DimensionTree`
A {py:class}`~tensap.DimensionTree` can be displayed with the command `T.plot()`. The
dimension associated with each leaf node can be plotted on the tree with
`T.plot_dims()`. Finally, the tree can be plotted with some quantity
displayed at each node with `T.plot_with_labels_at_nodes(labels)`.

#### Accessing properties of the tree
The number of nodes of a dimension tree $T$ is given by
`T.nb_nodes`.

The parent of $\alpha$, denoted by $P(\alpha)$, can be
obtained with `T.parent(alpha)`, and its ascendants $A(\alpha)$
and descendants $D(\alpha)$ by `T.ascendants(alpha)` and
`T.descendants(alpha)`, respectively. The children of $\alpha$
are given by `T.children(alpha)`. The command
`T.child_number(alpha)` returns $i^\gamma_\alpha$, for
$\alpha \in T \setminus \{D\}$ and $\gamma = P(\alpha)$,
which is such that $\alpha$ is the $i^\gamma_\alpha$-th
child of $\gamma$. For instance, in the linear tree of
{numref}`fig-dimension-trees`, the node $\alpha = \{3,4\}$ is the second child
of $\gamma = \{2,3,4\}$.

The level of a node $\alpha$ is denoted by
$\operatorname{level}(\alpha)$. The levels are defined such that
$\operatorname{level}(D) = 0$ and
$\operatorname{level}(\beta) = \operatorname{level}(\alpha) + 1$
for $\beta \in S(\alpha)$. The nodes of $T$ with level
$l$ are returned by `T.nodes_with_level(l)`.

The leaf nodes $\alpha \in {\mathcal{L}}(T)$ are such that
`T.is_leaf[alpha-1]` is `True`.

(theory-tree-based-tensor)=
### `TreeBasedTensor`

Given a dimension tree $T$, a {py:class}`~tensap.TreeBasedTensor` `X` is a tensor in *tree-based format* (see {cite:p}`Falco2018`, {cite:p}`Hackbusch2019`). It represents an order $d$ tensor $X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$ in the set of tensors with $\alpha$-ranks bounded by some integer $r_\alpha$, $\alpha \in T$. Such a tensor admits a representation

$$X_{i_1, \ldots, i_d} = \sum_{\substack{1 \leq k_\beta \leq r_\beta \beta \in T\setminus \{D\}}} \prod_{\alpha \in T\setminus {\mathcal{L}}(T)} C^\alpha_{(k_\beta)_{\beta \in S(\alpha)},k_\alpha} \prod_{\alpha \in {\mathcal{L}}(T)} C^{\alpha}_{i_\alpha,k_\alpha},$$

with $C^\alpha$, $\alpha \in T$, some tensors that
parameterize the representation of $X$. When $T$ is a
binary tree, the corresponding format is the so-called hierarchical
Tucker (HT) format. The particular case of a linear binary tree is the
tensor train Tucker format.

The *Tucker format* corresponds to a trivial tree $T=\{\{1\},\ldots,\{d\},\{1,\ldots,d\}\}$ and admits the representation

$$X_{i_1, \ldots, i_d} = \sum_{k_1=1}^{r_1} \ldots \sum_{k_d=1}^{r_d} C^{1,\ldots,d}_{k_1,\ldots,k_d} C^{1}_{i_1,k_1} \ldots C^{d}_{i_d,k_d}.$$

A *degenerate tree-based format* is defined as the set of tensors with
$\alpha$-ranks bounded by some integer $r_\alpha$, for all
$\alpha$ in a subset $A$ of $T$. The set $A$
corresponds to active nodes, which should contain all interior nodes
$T\setminus{\mathcal{L}}(T)$. A {py:class}`~tensap.TreeBasedTensor` `X` with
active nodes $A$ admits a representation.

$$X_{i_1, \ldots, i_d} = \sum_{\substack{1 \leq k_\beta \leq r_\beta \beta \in A \setminus \{D\}}} \prod_{\alpha \in A\setminus {\mathcal{L}}(T)} C^\alpha_{(k_\beta)_{\beta \in S(\alpha)},k_\alpha} \prod_{\alpha \in {\mathcal{L}}(T) \cap A} C^{\alpha}_{i_\alpha,k_\alpha},$$

with $C^\alpha$, $\alpha \in A$, some tensors that
parameterize the representation of $X$.

The *tensor train format* is a degenerate tree-based format with a linear tree $T$ and all leaf nodes inactive except the first one, that means $A = \{\{1\},\{1,2\}, \ldots, \{1,\ldots,d\}\}$. A tensor $X$ in tensor train format admits a representation

$$X_{i_1,\ldots,i_d} = \sum_{k_1=1}^{r_1} \ldots \sum_{k_{d-1}=1}^{r_{d-1}} C^1_{1,i_1,k_1} C^2_{k_1,i_2,k_2} \ldots C^{d-1}_{k_{d-2},i_{d-1},k_{d-1}} C^d_{k_{d-1},i_d,1}$$

with tensor $C^\nu$ and rank $r_\nu$ associated with the
node $\alpha = \{1,\ldots,\nu\}$.

For a more detailed presentation of tree-based formats (possibly degenerate) and more examples, see {cite:p}`Nouy2017book`.  
If the rank $r_D$ associated with the root node is different from $1$, a {py:class}`~tensap.TreeBasedTensor` `X` represents a tensor of order $d+1$ with entries $X_{i_1,\ldots,i_d,k_D}$, $1\le k_D \le r_D$. I can be used to defined vector-valued functional tensors (see ).

#### Creating a `TreeBasedTensor`
A {py:class}`~tensap.TreeBasedTensor` is created with the command `X = tensap.TreeBasedTensor(C, T)`, with `C` the list of {py:class}`~tensap.FullTensor` objects representing the $C^\alpha$, $\alpha \in T$, and `T` a {py:class}`~tensap.DimensionTree`. If some entries of the list `C` corresponding to leaf nodes are empty, it creates a degenerate tensor format, with $T\setminus A$ corresponding to the empty entries of `C`.  
It is possible to create a {py:class}`~tensap.TreeBasedTensor` in tensor-train format with the command `tensap.TreeBasedTensor.tensor_train(C)`, with `C` a list containing the tensors $C^1,\ldots,C^d$.  
Given a {py:class}`~tensap.DimensionTree` `T`, it is also possible to generate a {py:class}`~tensap.TreeBasedTensor` with entries

- equal to 0 with `tensap.TreeBasedTensor.zeros(T, r, s, I)`,
- equal to 1 with `tensap.TreeBasedTensor.ones(T, r, s, I)`,
- drawn randomly according to the uniform distribution on
  $[0, 1]$ with `tensap.TreeBasedTensor.rand(T, r, s, I)`,
- drawn randomly according to the standard gaussian distribution with
  `tensap.TreeBasedTensor.randn(T, r, s, I)`,
- generated using a provided `generator` with
  `tensap.TreeBasedTensor.create` `(generator, T, r, s, I)`,

where `r` is a list containing the $\alpha$-ranks,
$\alpha \in T$, or `'random'`, `s` is a list containing the
sizes $N_1, \ldots, N_d$, or `'random'`, and `I` is a list of
booleans indicating if the node $\alpha$ is active,
$\alpha \in T$, or `'random'`.

#### Storage complexity
The storage complexity of `X` is given by $\texttt{X.size = X.storage()}$ and returns the number of entries in tensors $C^\alpha$, $\alpha\in A$.  
The storage complexity of `X` taking into account the sparsity in the $C^\alpha$, $\alpha \in T$, is given by `X.sparse_storage()`. It returns the number of non-zero entries in tensors $C^\alpha$, $\alpha\in A$.  
The storage complexity of `X` taking into account the sparsity only in the leaf nodes is given by `X.sparse_leaves_storage()`.

#### Displaying a `TreeBasedTensor`
A graphical representation of a {py:class}`~tensap.TreeBasedTensor` `X` can be
obtained with the command `X.plot()`. Labels can be added to the nodes
of the tree, as well as a title, with `X.plot(labels, title)`.

#### Converting a `TreeBasedTensor` to a `FullTensor`
A {py:class}`~tensap.TreeBasedTensor` `X` can be converted to a {py:class}`~tensap.FullTensor`
(introduced in {ref}`theory-full-tensors`) with the command `X.full()`.

#### Converting a `FullTensor` to a `TreeBasedTensor`
A {py:class}`~tensap.FullTensor` `X` can be converted to a {py:class}`~tensap.TreeBasedTensor` with
the command `X.tree_based_tensor()`. The associated dimension tree
is a trivial tree with active nodes.

#### Accessing the entries of a `TreeBasedTensor`
The entries of the tensor `X` can be accessed with the method
`eval_at_indices`: `X.eval_at_indices(ind)` returns the entries
of $X$ indexed by the list `ind` containing the indices to
access in each dimension.

A sub-tensor can be extracted from `X` with the method `sub_tensor`
(see )

For a tensor $X \in {\mathbb{R}}^{N, \ldots, N}$, the command
`X.eval_diag()` returns the diagonal $X_{i, \ldots, i}$,
$i = 1, \ldots, N$, of the tensor. The method `eval_diag` can
also be used to evaluate the diagonal in some dimensions `dims` of the
tensor with `X.eval_diag(dims)`.

#### Obtaining an orthonormal representation of a `TreeBasedTensor`
The command `X.orth()` returns a representation of `X` where all the
core tensors except the root core represent orthonormal bases of
principal subspaces.

The command `X.orth_at_node(alpha)` returns a representation of
`X` where all the core tensors except the one of node $\alpha$
represent orthonormal bases of principal subspaces. The core tensor
$C^\alpha$ of the node $\alpha$ is such that the tensor
writes

$$X_{i_\alpha, i_{\alpha^c}} = \sum_{k} \sum_{l} C^\alpha_{k,l} u_l(i_\alpha) w_k(i_{\alpha^c}),$$

where the $u_l$ are orthonormal tensors and the $w_k$ are
orthonormal tensors. This orthonormality of the representation can be
checked by computing the Gram matrices of the bases of minimal subspaces
associated with the nodes of the tree with `X.gramians()`.

#### Modifying the tree structure of a `TreeBasedTensor`
It is possible to modify the tree of a {py:class}`~tensap.TreeBasedTensor` `X` by
permuting two of its nodes $\alpha$ and $\beta$ given a
relative tolerance `tol` with `X.permute_nodes([alpha, beta],
tol)`.

The leaves of the tree can also be permuted with the command
`X.permute_leaves(perm, tol)`, where `perm` is a permutation of
$(1, \ldots, d)$.

The method `optimize_dimension_tree` tries random permutations of
nodes to minimize the storage complexity of a tree-based tensor
$X$: `X.optimize_dimension_tree(tol, n)` tries $n$
random permutations and returns a {py:class}`~tensap.TreeBasedTensor` `Y` which is
such that `Y.storage()` is less or equal than `X.storage()`. The
nodes to permute are drawn according to probability measures favoring
high decreases of the ranks while maintaining a permutation cost as low
as possible (see {cite:p}`Grelier2019`).

The similar method `optimize_leaves_permutations` focuses on the
permutation of the leaf nodes to try to reduce the storage complexity of
a {py:class}`~tensap.TreeBasedTensor`.

#### Computing the Frobenius norm of a `TreeBasedTensor`
The command `X.norm()` returns the Frobenius norm of $X$.

#### Computing the $\alpha$-singular values of a `TreeBasedTensor`
For all $\alpha \in T$, the $\alpha$-singular values of
$X$ can be obtained with `X.singular_values()`.

The method `rank` uses the method `singular_values` to compute the
$\alpha$-ranks, $\alpha \in T$, of a {py:class}`~tensap.TreeBasedTensor`.

#### Computing the derivative of `TreeBasedTensor` with respect to one of its parameters
For an order-$d$ tree-based tensor `X` in
${\mathbb{R}}^{N \times \cdots \times N}$,
`X.parameter_gradient_eval_diag(alpha)`, for $\alpha \in T$,
returns the derivative

$$\left.\frac{\partial X_{i_1, \ldots, i_d}}{\partial C^\alpha}\right|_{i_1=\cdots=i_d=i}, \; i = 1, \ldots, N.$$

The method `parameter_gradient_eval_diag` is used in the
statistical learning algorithms presented in Section
{ref}`theory-tensor-learning`.

#### Performing operations with `TreeBasedTensor`
Some operations between tensors are implemented for {py:class}`~tensap.TreeBasedTensor`
(see for a detailed description of the operations): the Kronecker
product with `kron`, the contraction with matrices or vectors with
`tensor_matrix_product` or `tensor_vector_product` respectively,
the evaluation of the diagonal of a contraction with matrices with
`tensor_matrix_product_eval_diag`, the dot product with `dot`.

`Z = X.tensor_matrix_product(M)` `tensor_vector_product`
`tensor_matrix_product_eval_diag` `X.kron(Y)` `X.dot(Y)`

## Tensor truncation with `Truncator`

The object {py:class}`~tensap.Truncator` embeds several methods of truncation of tensors
in different formats. Given a tolerance `tol` and a maximum rank or
tuple of ranks `r`, a {py:class}`~tensap.Truncator` object can be created with `t =
tensap.Truncator(tol, r)`. The thresholding type (`'hard'` or
`'soft'`) can also be specified as a third argument.

### Truncation
The generic method `truncate` calls one of the methods presented
below, based on the type and order of its input, to obtain a truncation
of the provided tensor satisfying the relative prevision and maximal
rank requirements.

For an order $2$ tensor, the method `svd` is called. For a
tensor of order greater than $2$, the method `hosvd` is called
for a {py:class}`~tensap.FullTensor`, and `hsvd` for a {py:class}`~tensap.TreeBasedTensor`.

### Truncated singular value decomposition
The method `svd` computes the truncated singular value decomposition
of an order $2$ tensor. The input tensor can be a
`numpy.ndarray`, a `tensorflow.Tensor`, a {py:class}`~tensap.FullTensor` or a
{py:class}`~tensap.CanonicalTensor`, in which case the method `trunc_svd` is called,
or a {py:class}`~tensap.TreeBasedTensor`, in which case the method `hsvd` is called.

The method `trunc_svd` computes the truncated singular value
decomposition of a matrix, with a given relative precision in Schatten
$p$-norm (with a specified value for $p$) and given maximal
rank. The returned truncation is a {py:class}`~tensap.CanonicalTensor`.

### Truncated higher-order singular value decomposition
A truncated higher-order singular value decomposition of a
`numpy.ndarray`, a {py:class}`~tensap.FullTensor` or a {py:class}`~tensap.TreeBasedTensor` can be
computed with the method `hosvd`. The output is either a
{py:class}`~tensap.CanonicalTensor` for an order $2$ tensor, or a
{py:class}`~tensap.TreeBasedTensor` with a trivial tree for a tensor of order greater
than $2$.

### Truncation in tree-based tensor format
The method `hsvd` computes, given a {py:class}`~tensap.TreeBasedTensor` or a
{py:class}`~tensap.FullTensor` with a tree and a set of active nodes, a truncation in
tree-based tensor format.

### Truncation in tensor train format
The method `ttsvd`, given a {py:class}`~tensap.FullTensor`, calls the method `hsvd`
with a linear tree and all the leaf nodes inactive except the first one,
resulting in a truncation in tensor-train format.
