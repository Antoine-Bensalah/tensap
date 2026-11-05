# Learning

We present in this section some objects implemented in tensap for
learning functions or tensors.

## `(Functional)TensorPrincipalComponentAnalysis`

The objects {py:class}`~tensap.TensorPrincipalComponentAnalysis` (resp.
{py:class}`~tensap.FunctionalTensorPrincipalComponentAnalysis`) implements approximation
methods for algebraic (resp. functional) tensors based on principal
component analysis, using an adaptive sampling of the entries of the
tensor (or the function). See {cite:p}`Nouy2017hopca` for a
description of the algorithms, and for examples of use, see the examples {ref}`sphx_glr_examples_pca_tensor_pca.py`
and {ref}`sphx_glr_examples_pca_functional_tensor_pca.py`.

The difference between the two objects if that {py:class}`~tensap.TensorPrincipalComponentAnalysis`' methods take as first input a function returning components of the algebraic tensor to learn, whereas the methods of {py:class}`~tensap.FunctionalTensorPrincipalComponentAnalysis` take as first input the functional tensor to learn.

Both objects are parameterized by the attributes:

- `pca_sampling_factor`: a factor to determine the number of
  samples $N$ for the estimation of the principal components (1
  by default): if the precision is prescribed,
  $N = \texttt{pca\_sampling\_factor} \times N_\alpha$,
  if the rank is prescribed,
  $N = \texttt{pca\_sampling\_factor} \times t$;
- `pca_adaptive_sampling`: a boolean indicating if adaptive
  sampling is used to determine the principal components with
  prescribed precision;
- `tol`: an array containing the prescribed relative precision; set
  `tol = inf` for prescribing the rank;
- `max_rank`: an array containing the maximum alpha-ranks (the
  length depends on the format). If `len(max_rank) == 1`, uses the
  same value for all alpha; setting `max_rank = inf` prescribes the
  precision.

Furthermore, a {py:class}`~tensap.FunctionalTensorPrincipalComponentAnalysis` is
parameterized by the attributes:

- `bases`: the functional bases used for the projection of the
  function;
- `grid`: the {py:class}`~tensap.FullTensorGrid` used for the projection of the
  function on the functional bases;
- `projection_type`: the type of projection, the default being
  'interpolation'.

Both objects implement four main methods:

- `hopca`: returns the set of $\{\nu\}$-principal components of
  an order $d$ tensor, for all $\nu \in \{1,\ldots,d\}$;
- `tucker_approximation`: returns an approximation of a tensor of
  order $d$ or a function of $d$ variables in Tucker
  format;
- `tree_based_approximation`: provided with a tree and a list of
  active nodes, returns an approximation of a tensor of order $d$
  or a function of $d$ variables in tree-based tensor format;
- `tt_approximation`: returns an approximation of a tensor of order
  $d$ or a function of $d$ variables in tensor-train
  format.

## `LossFunction`

In tensap, a loss function is an object inheriting from {py:class}`~tensap.LossFunction`. Given a function `fun` and a sample as a list used to evaluate the loss function, a {py:class}`~tensap.LossFunction` object $\ell$ can be evaluated with `l.eval(fun, sample)`. The risk associated with `fun` can be evaluated using the sample with `l.risk_estimation(fun, sample)`. Finally, the test error and relative test error (if defined) can be evaluated with `l.test_error(fun, sample)` and `l.relative_test_error(fun, sample)`, respectively.  
Currently, three loss functions are implemented in tensap:

- {py:class}`~tensap.SquareLossFunction`: $\ell(g, (x, y)) = (y - g(x))^2$, used
  for least-squares regression in supervised learning, to construct an
  approximation of a random variable $Y$ as a function of a
  random vector $X$ (a predictive model);
- {py:class}`~tensap.DensityL2LossFunction`: $\ell(g, x) = \|g\|^2 - 2g(x)$, used
  for least-squares density estimation, to approximate the distribution
  of a random variable $X$ from samples of $X$;
- {py:class}`~tensap.CustomLossFunction`: defined by the user as any function defining
  a loss. If the loss is defined using tensorflow operations, then the
  empirical risk can be minimized using tensorflow's automatic
  differentiation capability with a {py:class}`~tensap.LinearModelLearningCustomLoss`
  object, presented in the next section.

## `LinearModelLearning`

Objects inheriting from {py:class}`~tensap.LinearModelLearning` implement the empirical
risk minimization associated with a linear model that writes

$$g(x) = \sum_{i \in I} a_i \phi_i(x),$$

with $\{\phi_i\}_{i \in I}$ a given basis (or a set of features)
and $(a_i)_{i \in I}$ some coefficients, and a loss function,
introduced in the previous section.

In order to perform empirical risk minimization, a
{py:class}`~tensap.LinearModelLearning` object `s` must be provided with a training
sample in `s.training_sample`. In supervised learning, for the
approximation of a random variable $Y$ as a function of $X$,
the training sample is a list `[x, y]`, with
$\texttt{y}$ represents $n$ samples
$\{y_k\}_{k=1}^n$ of $Y$ and $x$ the $n$
corresponding samples $\{x_k\}_{k=1}^n$ of $X$. In density
estimation, the training sample is an array `x` containing samples
$\{x_k\}_{k=1}^n$ from the distribution to estimate.

One must also provide a basis (in `s.basis`) or evaluations of the
basis on the training set (in `s.basis_eval`, in which case the
$x$ are not mandatory in `s.training_sample`). The latter
option allows for providing features $\phi_i(x_k)$ associated with
samples $x^k$, without providing the feature maps $\phi_i$.

One can also provide the {py:class}`~tensap.LinearModelLearning` `s` with a test sample in `s.test_data` to compute a test error.  
Currently in tensap, three different {py:class}`~tensap.LinearModelLearning` objects are implemented:

- {py:class}`~tensap.LinearModelLearningSquareLoss`, to minimize the risk associated
  with a {py:class}`~tensap.SquareLossFunction`;
- {py:class}`~tensap.LinearModelLearningDensityL2`, to minimize the risk associated
  with a {py:class}`~tensap.DensityL2LossFunction`;
- {py:class}`~tensap.LinearModelLearningCustomLoss`, to minimize the risk associated
  with a {py:class}`~tensap.CustomLossFunction`.

### `LinearModelLearningSquareLoss`
A {py:class}`~tensap.LinearModelLearningSquareLoss` object `s` implements three ways
of solving the empirical risk minimization associated with a
{py:class}`~tensap.SquareLossFunction`:

- by default, `s.solve()` solves the ordinary least-squares problem

  $$\min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2;$$

- with the attribute `s.regularization = True`, `s.solve()` solves
  the regularized problem

  $$\min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2 + \lambda \|a\|_p$$

  with $\lambda$ a regularization hyper-parameter, selected with
  a cross-validation estimate of the error and $p$ specified by
  `s.regularization_type` which can be `'l0'` ($p = 0$),
  `'l1'` ($p = 1$) or `'l2'` ($p = 2$);

- let us suppose that we have a collection of candidate sparsity
  patterns $K_\lambda$, $\lambda \in \Lambda$, for the
  parameter $a$: with the attribute `s.basis_adaptation =
  True`, `s.solve()` solves, for all $\lambda\in \Lambda$, the
  problem

  $$\min_{(a_i)_{i \in I}} \frac 1 n \sum_{k=1}^n (y_k - \sum_{i \in I} a_i \phi_i(x_k))^2 \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,$$

  where $\mathrm{support}(a) = \{ k \in K : a_k \neq 0 \}$, and
  selects the optimal sparsity pattern using a cross-validation
  estimate of the error.

### `LinearModelLearningDensityL2`
A {py:class}`~tensap.LinearModelLearningDensityL2` object `s` implements three ways of
solving the empirical risk minimization associated with a
{py:class}`~tensap.DensityL2LossFunction`:

- by default, `s.solve()` solves the minimization problem

  $$\min_{(a_i)_{i \in I}} \| \sum_{i \in I} a_i \phi_i \|_{L^2}^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k);$$

- with the attribute `s.regularization = True`, `s.solve()` solves
  the constrained problem

  $$\min_{(a_i)_{i \in I}} \|\sum_{i \in I} a_i \phi_i \|_{L^2}^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k) \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,$$

  with $K_\lambda$, $\lambda \in \Lambda$, a sequence of
  sets of indices that introduce the coefficients solution of the
  minimization problem without regularization in decreasing order of
  magnitude. The optimal sparsity pattern is determined using a
  cross-validation estimate of the error;

- let us suppose that we have a collection of candidate patterns
  $K_\lambda$, $\lambda \in \Lambda$, for the parameter
  $a$: with the attribute `s.basis_adaptation = True`,
  `s.solve()` solves, for all $\lambda\in \Lambda$, the problem

  $$\min_{(a_i)_{i \in I}} \|\sum_{i \in I} a_i \phi_i \|^2 - \frac 2 n \sum_{k=1}^n \sum_{i \in I} a_i \phi_i(x_k) \quad \text{subject to } \mathrm{support}(a) \subset K_\lambda,$$

  and selects the optimal sparsity pattern using a cross-validation
  estimate of the error.

### `LinearModelLearningCustomLoss`
A {py:class}`~tensap.LinearModelLearningCustomLoss` object `s` implements a way of
solving the empirical risk minimization associated with a
{py:class}`~tensap.CustomLossFunction` using tensorflow's automatic differentiation
capabilities.

By default, the optimizer used is keras' Adam algorithm, which is a
"stochastic gradient descent method that is based on adaptive estimation
of first-order and second-order moments" (per tensorflow's
documentation).

The algorithm requires a starting point, provided in
`s.initial_guess`, and several options can be set:

- `s.options['max_iter']` sets the maximum number of iterations used
  in the optimization algorithm,
- `s.options['stagnation']` sets the stopping tolerance on the
  stagnation between two iterates,
- for the Adam algorithm (and other minimization algorithms provided by
  tensorflow/keras), the learning rate can be provided in
  `s.optimizer.learning_rate`.

(theory-tensor-learning)=
## `TensorLearning`

The package tensap implements algorithms to perform statistical learning
with canonical and tree-based tensor formats. See {cite:p}`Grelier2018`,
{cite:p}`Grelier2019`, {cite:p}`Michel2020` for a detailed presentation
of algorithms and related theory.

For examples, see the examples
{ref}`sphx_glr_examples_learning_canonical_tensor_learning.py`,
{ref}`sphx_glr_examples_learning_tree_based_tensor_learning.py`,
{ref}`sphx_glr_examples_learning_tree_based_tensor_density_learning.py`,
{ref}`sphx_glr_examples_learning_tensorized_function_learning.py`.

These algorithms are implemented in the core object {py:class}`~tensap.TensorLearning`,
common to all the tensor formats, so that implementing such a learning
algorithm for a new tensor format is simple. In tensap are currently
implemented {py:class}`~tensap.CanonicalTensorLearning` for the learning in canonical
tensor format and {py:class}`~tensap.TreeBasedTensorLearning` for the learning in
tree-based tensor format.

Two algorithms are proposed: the standard one, which minimizes an
empirical risk over the set of tensors in a given format thanks to an
alternating minimization over the parameters of the tensors, and the
adaptive one, which returns a sequence of empirical risk minimizers with
adapted rank (for the canonical and tree-based tensor formats) and
adapted tree (for the tree-based tensor format).

In order to perform empirical risk minimization, a {py:class}`~tensap.TensorLearning`
object `s` must be provided with a training sample in
`s.training_sample`. In supervised learning, for the approximation of
a random variable $Y$ as a function of $X$, the training
sample is a list `[x, y]`, with $\texttt{y}$
represents $n$ samples $\{y_k\}_{k=1}^n$ of $Y$ and
$x$ the $n$ corresponding samples
$\{x_k = (x_{k,1},\ldots,x_{k,d})\}_{k=1}^n$ of $X$. In
density estimation, the training sample is an array `x` containing
samples $\{x_k= (x_{k,1},\ldots,x_{k,d})\}_{k=1}^n$ from the
distribution to estimate.

One must also provide bases (in `s.bases`) or evaluations of the bases
on the training set (in `s.bases_eval`, in which case the $x$
are not mandatory in `s.training_sample`). The latter option allows
for providing features $\phi^\nu_i(x_{\nu,k})$,
$1\le \nu\le d$, associated with samples
$x_k = (x_{k,1},\ldots,x_{k,d})$, without providing the feature
maps $\phi^\nu_i$.

One can also provide the {py:class}`~tensap.TensorLearning` `s` with a test sample in
`s.test_data` to compute a test error.

### Rank adaptation
(See {cite:p}`Grelier2018`) The rank adaptation
is enabled by setting `s.rank_adaptation` to `True`.

For tensors in canonical format, the algorithm returns a sequence of
rank-$r$ approximations, with
$r = 1, \ldots, r_{\text{max}}$, $r_{\text{max}}$ being
given by `s.rank_adaptation_options` `['max_iterations']`.

For tensors in tree-based format, the algorithm returns a sequence of
tensors with non-decreasing tree-based rank, obtained by increasing, at
each iterations, the ranks associated with a subset of nodes of the tree
$T$. The number of nodes in this subset is influenced by a
parameter `s.rank_adaptation_options['theta']` in $[0, 1]$,
which is such that the larger it is, the more ranks are increased at
each iteration. The default value of $0.8$.

### Tree adaptation
(See {cite:p}`Grelier2018`) For tree-based
tensor formats, the tree can be adapted at each iteration using the
algorithm mentioned in {ref}`theory-tree-based-tensor`, by setting
`s.tree_adaptation` to `True`. The tolerance for the tree
adaptation is provided by `s.tree_adaptation_options['tolerance']`
and the maximal number of tried trees by
`s.tree_adaptation_options['max_iterations']`.

### Model selection
(See {cite:p}`Michel2020`) At the end of the adaptive
procedure, a model can be selected by setting `s.model_selection` to
`True`, using either a test error (specified by
`s.model_selection_options['type'] = 'test_error'`) or a
cross-validation estimate of the error (specified by
`s.model_selection_options['type'] = 'cv_error'`).

### Example: classification of handwritten digits

The manual ended with a character classification example, which learns a
classifier of handwritten digits with a {py:class}`~tensap.TensorLearning`
object. It is now an example of the gallery:
{ref}`sphx_glr_examples_learning_digits_recognition.py`.
