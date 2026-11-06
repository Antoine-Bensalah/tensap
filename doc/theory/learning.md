(theory-learning)=
# Learning with tensor formats

Learning approximates a function from samples, by minimizing an empirical
risk over a set of functions: linear models on a basis, or functional tensors
in a low-rank format ({ref}`theory-functions`). In a tensor format, the
minimization is done by alternating over the parameters of the format, as for
the solution of equations ({ref}`theory-greedy`), with an adaptation of the
ranks and of the dimension tree.

## Empirical risk minimization

A *loss function* $\ell(g, z)$ measures the error of a function $g$ on a
sample $z$. Given samples $z_1, \dots, z_n$, the empirical risk
$\frac1n \sum_{k=1}^{n} \ell(g, z_k)$ is minimized over a set of functions.
tensap provides

- the **square loss** $\ell(g, (x, y)) = (y - g(x))^2$, for least-squares
  regression: the approximation of a random variable $Y$ as a function of a
  random vector $X$ from samples $(x_k, y_k)$;
- the **$L^2$ density loss** $\ell(g, x) = \|g\|_{L^2}^2 - 2 g(x)$, for
  least-squares density estimation: since $\mathbb{E}(\ell(g, X)) =
  \|g - p\|_{L^2}^2 - \|p\|_{L^2}^2$ for $X$ of density $p$, its risk is
  minimized by $p$;
- custom losses, defined by the user and minimized by a gradient method with
  automatic differentiation.

## Linear models

For a linear model $g(x) = \sum_{i \in I} a(i)\, \varphi_i(x)$ on a basis or
a set of features $\{\varphi_i\}_{i \in I}$, the minimization of the
empirical risk is a linear problem in the coefficients $a$:

- with the square loss, the least-squares problem
  $\min_a \frac1n \sum_{k=1}^{n} \big(y_k - \sum_{i} a(i)\, \varphi_i(x_k)
  \big)^2$;
- with the $L^2$ density loss and an orthonormal basis,
  $\min_a \|a\|^2 - \frac{2}{n} \sum_{k=1}^{n} \sum_{i} a(i)\,
  \varphi_i(x_k)$, whose solution is the empirical mean
  $a(i) = \frac1n \sum_k \varphi_i(x_k)$.

**Sparsity and model selection.** With many basis functions and few
samples, the problem is regularized ($\ell^0$, $\ell^1$ or $\ell^2$
penalization of $a$), or solved on a sequence of candidate supports
$K_\lambda$, nested sets of basis functions, for instance given by a
regularization path or by a sequence of sets of multi-indices
({ref}`theory-multi-indices`). The support is then selected by a
cross-validation estimate of the error, computed from the training samples
only.

(theory-tensor-learning)=
## Learning in tensor formats

For a function in a low-rank tensor format,
$g(x) = (a, \Phi(x))$ with $a$ a canonical or tree-based tensor
({ref}`theory-functions`), the empirical risk is not convex in the
parameters of $a$, but the parametrization is multilinear: with all the
tensors of the network fixed but one, $g$ is a linear model in the remaining
tensor, whose features are the contractions of the other tensors with the
bases ({ref}`theory-tree-based-tensors`). The *alternating minimization*
solves these linear problems one after the other, with the tools of linear
models (regularization, sparse supports, cross-validation) at each step
{cite:p}`Grelier2018`.

**Rank adaptation.** Starting from small ranks, the algorithm returns a
sequence of approximations of increasing ranks: in canonical format, the
rank is increased by one at each iteration; in tree-based format, the ranks
of the nodes with the largest estimated truncation errors are increased, a
parameter $\theta \in [0, 1]$ controlling how many nodes are enriched at
each iteration, as in the anisotropic greedy algorithms
({ref}`theory-greedy`) {cite:p}`Grelier2018`.

**Tree adaptation.** In tree-based format, the dimension tree is adapted
during the iterations, by permutations of nodes that reduce the storage of the
current approximation ({ref}`theory-tree-based-tensors`)
{cite:p}`Grelier2018,Grelier2019`.

**Model selection.** Among the sequence of approximations, the final one is
selected by a test error or by a cross-validation estimate of the error
{cite:p}`Michel2020`.

## Principal component analysis

When the function or the tensor can be evaluated at chosen points, the
approximation can instead be built by principal component analysis: for each
node $\alpha$ of a dimension tree, the $\alpha$-principal subspace of the
tensor is estimated from evaluations on a sample of the other variables, and
the approximation is the projection (or interpolation) on the tensor product
of these subspaces. The higher-order principal component analysis gives the
Tucker format; its tree-based version gives the tree-based and tensor-train
formats, with a prescribed rank or precision and an adaptive sampling
{cite:p}`Nouy2017hopca`.

## In tensap

- Losses: {py:class}`~tensap.SquareLossFunction`,
  {py:class}`~tensap.DensityL2LossFunction`,
  {py:class}`~tensap.CustomLossFunction`.
- Linear models: {py:class}`~tensap.LinearModelLearningSquareLoss`
  (regularization, basis adaptation),
  {py:class}`~tensap.LinearModelLearningDensityL2`,
  {py:class}`~tensap.LinearModelLearningCustomLoss`.
- Tensor formats: {py:class}`~tensap.CanonicalTensorLearning`,
  {py:class}`~tensap.TreeBasedTensorLearning` (options `rank_adaptation`,
  `tree_adaptation`, `model_selection`).
- Principal component analysis:
  {py:class}`~tensap.TensorPrincipalComponentAnalysis`,
  {py:class}`~tensap.FunctionalTensorPrincipalComponentAnalysis`
  (`hopca`, `tucker_approximation`, `tree_based_approximation`,
  `tt_approximation`).
- Examples: {ref}`sphx_glr_examples_learning_canonical_tensor_learning.py`,
  {ref}`sphx_glr_examples_learning_tree_based_tensor_learning.py`,
  {ref}`sphx_glr_examples_learning_tree_based_tensor_density_learning.py`,
  {ref}`sphx_glr_examples_pca_tensor_pca.py`,
  {ref}`sphx_glr_examples_pca_functional_tensor_pca.py`.

