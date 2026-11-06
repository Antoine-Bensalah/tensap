(theory-tree-based-tensors)=
# Tree-based tensors

(theory-tree-based-tensor)=

Tree-based tensor formats, or tree tensor networks, are defined by bounds on
the $\alpha$-ranks of a tensor ({ref}`theory-full-tensors`) for the nodes
$\alpha$ of a dimension tree. They include the Tucker format, the
hierarchical Tucker format and the tensor-train format, and their storage
grows linearly with the dimension for bounded ranks
{cite:p}`Falco2018,Hackbusch2019,Nouy2017book`.

## Dimension trees

```{prf:definition} Dimension tree
:label: def-dimension-tree

A *dimension tree* $T$ over $D = \{1, \dots, d\}$ is a collection of
non-empty subsets of $D$ such that

1. $D \in T$ is the root;
2. every node $\alpha \in T$ with $\#\alpha \ge 2$ has a set of children
   $S(\alpha) \subset T$ that is a partition of $\alpha$ into at least two
   sets;
3. every node $\alpha$ with $\#\alpha = 1$ has no child: it is a *leaf*.
```

$\mathcal{L}(T)$ denotes the set of the leaves, $P(\alpha)$ the parent of a
node $\alpha \ne D$, and the *level* of a node is its distance to the root,
$\operatorname{level}(D) = 0$ and $\operatorname{level}(\beta) =
\operatorname{level}(\alpha) + 1$ for $\beta \in S(\alpha)$; the depth of
$T$ is the largest level.

```{figure} images/dimension_trees.svg
:name: fig-dimension-trees
:alt: A trivial tree, a linear binary tree and a balanced binary tree, for d = 4.

Dimension trees over $D = \{1, 2, 3, 4\}$: a trivial tree (left), a linear
binary tree (middle) and a balanced binary tree (right).
```

## The tree-based format

```{prf:definition} Tree-based format
:label: def-tree-based

Given a dimension tree $T$ and ranks $r = (r_\alpha)_{\alpha \in T}$ with
$r_D = 1$, the *tree-based format* is the set of tensors

$$
\mathcal{T}^T_r = \{ u : \rank_\alpha(u) \le r_\alpha,\ \alpha \in T \}.
$$

A tensor of $\mathcal{T}^T_r$ admits the representation

$$
u(i_1, \dots, i_d) = \sum_{\substack{k_\beta = 1, \dots, r_\beta \\
  \beta \in T \setminus \{D\}}}
  \prod_{\alpha \in T \setminus \mathcal{L}(T)}
    C^\alpha\big((k_\beta)_{\beta \in S(\alpha)}, k_\alpha\big)
  \prod_{\alpha \in \mathcal{L}(T)} C^\alpha(i_\alpha, k_\alpha),
$$

by a *tree tensor network*: one tensor $C^\alpha$ per node, with one index
per child and one index for the node itself (absent at the root), and, at a
leaf $\alpha = \{\nu\}$, the physical index $i_\nu$.
```

Its storage is

$$
\sum_{\alpha \in \mathcal{L}(T)} N_\alpha r_\alpha
+ \sum_{\alpha \in T \setminus \mathcal{L}(T)} r_\alpha
  \prod_{\beta \in S(\alpha)} r_\beta ,
$$

linear in $d$ for bounded ranks and a tree whose nodes have a bounded number
of children. Particular cases:

- the **Tucker format**, for the trivial tree
  $T = \{\{1\}, \dots, \{d\}, D\}$: the core $C^D$ and the leaves
  $C^\nu$, $u(i_1, \dots, i_d) = \sum_{k} C^D(k_1, \dots, k_d)\,
  C^1(i_1, k_1) \cdots C^d(i_d, k_d)$ ({ref}`theory-tucker-canonical`);
- the **hierarchical Tucker format**, for a binary tree;
- the **tensor train Tucker format**, for a linear binary tree.

**Active nodes.** A *degenerate* tree-based format bounds the $\alpha$-ranks
only for the nodes of a subset $A \subset T$, the *active* nodes, which
contains all the interior nodes. An inactive leaf has no tensor: the index of
its parent is directly the physical index. The **tensor-train format** is the
degenerate format of a linear tree in which only the first leaf is active,
$A = \{\{1\}, \{1, 2\}, \dots, D\}$ ({ref}`theory-tensor-trains`).

**Vector-valued tensors.** If the rank $r_D$ of the root is larger than 1,
the network represents a tensor of order $d + 1$, with the additional index
$k_D$; this is used for vector-valued functions.

## Orthogonality and singular values

The representation is not unique: for each node $\alpha \ne D$, an
invertible matrix can be applied to the index $k_\alpha$ of $C^\alpha$ and
its inverse to the same index of the parent. Using these changes of bases,
from the leaves to the root, by QR factorizations, the tensors of all the
nodes but the root can be made to represent *orthonormal* bases of their
subspaces. Then $\|u\| = \|C^D\|$.

More generally, the representation can be made orthonormal at any node
$\alpha$: all the other tensors represent orthonormal bases, and

$$
u(i_\alpha, i_{\alpha^c}) = \sum_{k, l} C^\alpha(k, l)\,
  v_k(i_\alpha)\, w_l(i_{\alpha^c})
$$

with orthonormal families $(v_k)$ and $(w_l)$. The $\alpha$-singular values
of $u$ are then those of the small matrix $C^\alpha$, matricized with its
parent index on one side: they are computed node by node without forming the
tensor, and give the $\alpha$-ranks and the truncation of $u$
({ref}`theory-truncation`).

## Changing the tree

The ranks of a tensor depend on the tree. Two nodes of a tree can be
exchanged, or the leaves permuted, the tensor being represented in the new
tree up to a given precision. Since the storage depends on the ranks, the
tree can be optimized for a given tensor: random permutations of nodes are
tried, drawn with probabilities that favour large decreases of the ranks and
small costs of permutation, and those that reduce the storage are kept
{cite:p}`Grelier2019`.

## Derivatives with respect to the parameters

The map $(C^\alpha)_{\alpha \in T} \mapsto u$ is multilinear: $u$ is linear
in each tensor $C^\alpha$, the others being fixed. Its derivative with
respect to $C^\alpha$ is therefore a linear map, given by the contraction of
all the other tensors of the network. Evaluated at the entries of interest,
it gives the linear problems solved at each step of the alternating
algorithms of learning ({ref}`theory-tensor-learning`) and of the solution of
equations ({ref}`theory-galerkin-updates`).

## In tensap

- {py:class}`~tensap.DimensionTree`: the trivial, linear, balanced and
  random trees, the parents, children, levels and leaves of the nodes, and
  sub-trees.
- {py:class}`~tensap.TreeBasedTensor`: construction from the tensors of the
  nodes and the tree (empty tensors for inactive leaves), conversion from and
  to full tensors, `orth`, `orth_at_node` and `gramians` for orthonormal
  representations, `singular_values` and `rank`, `permute_nodes`,
  `permute_leaves`, `optimize_dimension_tree` and
  `optimize_leaves_permutations` for the tree, and
  `parameter_gradient_eval_diag` for the derivatives.
- {py:class}`~tensap.CanonicalTensor` represents a canonical tensor
  $\sum_k C(k)\, U^1(i_1, k) \cdots U^d(i_d, k)$ by its coefficients and
  its factors, with the same operations.
- Examples: {ref}`sphx_glr_examples_tensors_dimension_tree.py`,
  {ref}`sphx_glr_examples_tensors_tree_based_tensor.py`.
