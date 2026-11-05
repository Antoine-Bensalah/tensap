(theory-full-tensors)=
# Full tensors

A {py:class}`~tensap.FullTensor` `X` represents an order $d$ tensor
$X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$, or
multidimensional array of size $N_1\times \ldots \times N_d$.
The entries of $X$ are $X_{i_1, \ldots, i_d}$, with
$(i_1,\ldots,i_d)$ a tuple of indices, where
$i_\nu \in \{0,\ldots,N_\nu-1\}$ is related to the
$\nu$-th mode of the tensor.

We present in this section how to create a {py:class}`~tensap.FullTensor` using tensap,
and several possible operations with such an object. For an introduction
to tensor calculus, we refer to the monograph {cite:p}`Hackbusch2019`.

For examples of use, see the example
{ref}`sphx_glr_examples_tensors_full_tensor.py`.

## Creating a `FullTensor`

Provided with an array `data` of shape `[N_1, ..., N_d]`, the command `X = tensap.FullTensor(data)` returns a tensor $X \in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$, with order `X.order = d` and shape `X.shape = (N_1, ..., N_d)`. The number of entries of `X` is given by $\texttt{X.size = X.storage()} = \prod_{i=1}^d N_i$. The number of nonzero entries of `X` is given by `X.sparse_storage()`.

It is also possible to generate a {py:class}`~tensap.FullTensor` with entries:

- equal to 0 with `tensap.FullTensor.zeros([N_1, ..., N_d])`,
- equal to 1 with `tensap.FullTensor.ones([N_1, ..., N_d])`,
- drawn randomly according to the uniform distribution on
  $[0, 1]$ with `tensap.FullTensor.rand([N_1, ..., N_d])`,
- drawn randomly according to the standard gaussian distribution with
  `tensap.FullTensor.randn([N_1, ..., N_d])`,
- different from 0 only on the diagonal, provided in `diag_data`,
  with `tensap.FullTensor.diag(diag_data, d)` (generating a tensor
  of order d and shape $[N, \ldots, N]$, with $N = len(diag_data)$,
- generated using a provided `generator` with
  `tensap.FullTensor.create(generator, [N_1, ..., N_d])`.

## Accessing the entries of a `FullTensor`

The entries of a tensor `X` can be accessed with the method
`eval_at_indices`: `X.eval_at_indices(ind)` returns the entries
of $X$ indexed by the list `ind` containing the indices to
access in each dimension.

### Extracting diagonal entries
For a tensor $X \in {\mathbb{R}}^{N, \ldots, N}$, the command
`X.eval_diag()` returns the diagonal entries
$X_{i, \ldots, i}$, $i = 1, \ldots, N$, of the tensor. The
command `X.eval_diag(dims)` returns the entries
$X_{i, \ldots, i}$, with $i$ in `ind`.

### Extracting a sub-tensor
A sub-tensor can be extracted from `X` with the method
`sub_tensor`: for an order-3 {py:class}`~tensap.FullTensor` `X` of size
$N_1\times N_2 \times N_3$, `X.sub_tensor([0, 1], ':', 2)`
returns a sub-tensor of size $2\times N_2 \times 1$ containing the
entries $X_{i_1,i_2,i_3}$ with $i_1\in \{0,1\}$,
$0\le i_2 \le N_2-1$ and $i_3=2$.

## Permuting the modes of a `FullTensor`

The methods `transpose` and `itranspose` permute the dimensions of a
tensor `X`, given a permutation `dims` of $\{1, \ldots, d\}$.
They are such that `X = X.transpose(dims).itranspose(dims)`.

## Reshaping a `FullTensor`
The command `X.reshape(shape)` reshapes a {py:class}`~tensap.FullTensor` using a
column-major order (e.g. used in Fortran, Matlab, R). It relies on the
numpy's reshape function with Fortran-like index (argument
`order='F'`). For a tuple $(i_1,\ldots,i_d)$, we define

$$\overline{i_1,\ldots,i_d} = i_1 + N_1(i_{2-1}-1) + N_1N_2(i_{3-1}-1) + \ldots + N_1\ldots N_{d-1}(i_d-1) .$$

A tensor $X$ is be identified with a vector
$\mathrm{vec}(X)$ whose entries are
$\mathrm{vec}(X)_{\overline{i_1,\ldots,i_d}}$. This vector can be
obtained with the command `X.reshape(N)` with
`N=numpy.prod(X.shape)`.

### $\alpha$-Matricization
For $\alpha \subset \{1,\ldots,d\}$ an its complementary subset
$\alpha^c$ in $\{1,\ldots,d\}$, an
$\alpha$-matricization of a tensor $X$ is a matrix $M$
of size
$(\prod_{i \in \alpha} N_i) \times (\prod_{i \in \alpha^c} N_i)$,
such that
$X_{i_1,\ldots,i_d} = M_{\overline{i_\alpha},\overline{i_{\alpha^c}}}$
with $i_\alpha = (i_\nu)_{\nu\in \alpha}$. It can be obtained with
`X.matricize(alpha)`, which returns a {py:class}`~tensap.FullTensor` or order
$2$. The matricization relies on the method `reshape`.

### Orthogonalization
It is possible to obtain a representation of a tensor $X$ such
that its $\alpha$-matricization is an orthogonal matrix (i.e. with
orthogonal columns) using the method `X.orth(alpha)`.

## Norms and singular-values

### Computing the Frobenius norm of a `FullTensor`
The command `X.norm()` returns the Frobenius norm
$\Vert X \Vert_F$ of $X$, defined by

$$\Vert X \Vert_F^2 = \sum_{i_1}^{N_1} \cdots \sum_{i_d}^{N_d} X_{i_1, \ldots, i_d}^2.$$

### Computing the $\alpha$-singular values and $\alpha$-principal components of a `FullTensor`
For a subset $\alpha \subset \{1, \ldots, d\}$ and its
complementary subset $\alpha^c$, the $\alpha$-matricization
$M$ of $X$ admits a singular value decomposition

$$M_{i_\alpha,i_{\alpha^c}} = \sum_{k} \sigma^k v^k_{i_\alpha} w^k_{i_\alpha^c}$$

where the $\sigma^k$ are the singular values of $M$ and the
$v^k$ the corresponding left singular vectors, or principal
components of $M$. They are respectively called the
$\alpha$-singular values and $\alpha$-principal components
of $X$. The $\alpha$-singular values are obtained with
`X.singular_values()`. The $\alpha$-principal components (and
$\alpha$-singular values) are obtained with
`X.alpha_principal_components(alpha)`, which is equivalent to
`X.matricize(alpha).principal_components()`.

## Operations with `FullTensor`

### Outer product
The outer product $X \circ Y$ of two tensors $X\in {\mathbb{R}}^{N_1 \times \cdots \times N_d}$ and $Y \in {\mathbb{R}}^{\hat N_1 \times \cdots \times \hat N_{\hat d}}$ is a tensor $Z \in {\mathbb{R}}^{N_1 \times \ldots \times N_d \times \hat N_1 \times \cdots \times \hat N_{\hat d}}$ of order $d + \hat d$ with entries

$${Z}_{i_1,\ldots,i_d,j_1,\ldots,j_{\hat d}} =  X_{i_1, \ldots,i_ d} Y_{j_1, \ldots, j_{\hat d}}$$

It is provided by `X.tensordot(Y, 0)`, similarly to numpy's
tensordot function.

### Kronecker product
The Kronecker product $X\otimes Y$ of two tensors $X$ and
$Y$ of the same order $d=\hat d$ is a tensor $Z$ of
size $N_1\hat N_1 \times  \ldots \times N_d \hat N_{\hat d}$ with
entries

$$Z_{\overline{i_1j_1},\ldots,\overline{i_dj_d}} = X_{i_1,\ldots,i_d} Y_{j_1,\ldots,j_d}.$$

It is given by the command `kron`, which is similar to numpy's kron
function, but for arbitrary tensors.

### Hadamard product
The Hadamard (elementwise) product $X \circledast Y$ of two
tensors $X$ and $Y$ of the same order and size is obtained
through the command `__mul__(X,Y)`, which returns a tensor
$Z$ with entries

$$Z_{i_1,\ldots,i_d} = X_{i_1,\ldots,i_d} Y_{i_1,\ldots,i_d}$$

### Contracted product
For $I\subset \{1,\ldots,d\}$ and $J \subset \{1,\ldots,\hat d\}$ with $\#I = \#J$, `Z = X.tensordot(Y, I, J)` performs the mode $(I,J)$-contracted product of $X$ and $Z$ which is a tensor Z of order $d + \hat d - \#I - \#J$ with entries

$$\begin{aligned}
{Z}_{(i_\nu)_{\nu \notin I}, (j_\mu)_{\mu \notin J}} =
            \sum_{\substack{i_\nu=1 \\ \nu \in I}}^{N_\nu} \sum_{\substack{j_\mu=1 \\ \mu \in J}}^{N_\mu} 
            \prod_{\nu \in I} \prod_{\mu \in J} 
            \delta_{i_\nu, j_\mu} X_{i_1, \ldots,i_ d} Y_{j_1, \ldots, j_{\hat d}}
\end{aligned}$$

with $\delta_{i,j}$ the Kronecker delta, that is a contraction
of tensors $X$ and $Y$ along dimensions $I$ of
$X$ and $J$ of $Y$. For example, for
order-$4$ tensors $X$ and $Y$, `Z =
X.tensordot(Y, [0,1], [1,2])` returns a tensor $Z$ or order
$4$ such that

$$Z_{i_3,i_4,j_1,j_4} = \sum_{i_1,i_2} X_{i_1,i_2,i_3,i_4} Y_{j_1,i_1,i_2,j_4} .$$

The method `tensordot_eval_diag` provides the diagonal (or
entries with equal pairs of indices) of the result of the method
`tensor_dot`, but at a cost lower than when using `X.tensordot(Y,
I, J).eval_diag()`.

For example, for order-$4$ tensors $X$ and $Y$, `X.tensordot_eval_diag(Y,[0,1],[1,2],[2,3],[0,3])` returns the diagonal of $Z$, i.e. an order-one tensor $M$ with entries

$$M_k = Z_{k,k,k,k} = \sum_{i_1,i_2} X_{i_1,i_2,k,k} Y_{k,i_1,i_2,k}$$

`X.tensordot_eval_diag(Y,[0,1],[1,2],[2,3],[0,3],diag = True)`
returns a tensor $M$ of order $2$ with entries

$$M_{k_1,k_2} = Z_{k_1,k_2,k_1,k_2} = \sum_{i_1,i_2} X_{i_1,i_2,k_1,k_3} Y_{k,i_1,i_2,k}$$

`X.tensordot_eval_diag(Y,[0,1],[1,2],[2],[0])` returns the
diagonal of $Z$, i.e. a tensor $M$ of order $3$
$v$ with entries

$$M_{k,i_4,j_4} = Z_{k,i_4,k,j_4} = \sum_{i_1,i_2} X_{i_1,i_2,k,i_4} Y_{k,i_1,i_2,j_4}$$

### Dot product
The dot product of two tensors $X$ and $Y$ with same shape
$[N_1, \ldots, N_d]$, defined by

$$\begin{aligned}
( X, Y ) = \sum_{\substack{i_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} X_{i_1, \ldots, i_d} Y_{i_1, \ldots, i_d},
\end{aligned}$$

can be obtained with `X.dot(Y)`. It is equivalent to `X.tensordot(Y,
range(X.order), range(Y.order))`.

### Contractions with matrices or vectors

Given a tensor $X$ and a list of matrices
$M = [M^1, ..., M^d]$, the command `Z =
X.tensor_matrix_product(M)` returns an order-d tensor $Z$ whose
entries are

$$\begin{aligned}
Z_{i_1, \ldots, i_d} = \sum_{\substack{k_\nu = 1 \\ \nu = 1, \ldots, d}}^{N_\nu} X_{k_1, \ldots, k_d} \prod_{\nu = 1}^d M^\nu_{i_\nu, k_\nu}
\end{aligned}$$

The same method exists for vectors instead of matrices:
`tensor_vector_product`. Similarly to `tensordot_eval_diag`, the
method `tensor_matrix_product_eval_diag` evaluates the diagonal of
the result of `tensor_matrix_product`, with a lower cost.
