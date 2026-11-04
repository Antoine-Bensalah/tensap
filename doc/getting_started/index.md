(getting-started)=
# Getting Started

This page installs tensap, checks the installation, and runs a first
computation: learning a function of six variables from samples, in a
tree-based tensor format.

## Installation

tensap requires Python 3 with NumPy, SciPy and NetworkX. Install it from PyPI:

```bash
pip install tensap
```

or from conda-forge:

```bash
conda install -c conda-forge tensap
```

To work on tensap itself, install it from the sources, in editable mode, with
the tools to build the documentation (see {ref}`contributing`):

```bash
git clone https://github.com/anthony-nouy/tensap.git
cd tensap
pip install -e ".[doc]"
```

Some features need optional packages:

| Package | Used by |
| --- | --- |
| matplotlib | the plots of tensors, trees and functions (`plot` methods) |
| scikit-learn | Poincaré learning and the digits recognition example |
| TensorFlow | learning with a custom loss function ({py:class}`~tensap.CustomLossFunction`) |

## Checking the installation

```bash
python -c "import tensap; print(tensap.__version__)"
```

prints the version of tensap, for instance `1.8`.

## A first computation

The script below approximates a function $f(X_1, \ldots, X_6)$ of six
independent random variables, uniform on $[-1, 1]$,

$$
f(x) = \sin(x_1 + x_4)\, e^{x_5}\, x_6 + \sin(x_3 x_4)\, x_6,
$$

from 500 evaluations of $f$ at random points. The approximation is a
function in tree-based tensor format,

$$
f(x) \approx \sum_{i_1, \ldots, i_6} a_{i_1 \ldots i_6}\,
\phi^1_{i_1}(x_1) \cdots \phi^6_{i_6}(x_6),
$$

where $\phi^\nu_i$ are orthonormal polynomials of degree at most 8, and the
tensor of coefficients $a$ is in tree-based tensor format: its ranks, and the
dimension tree itself, are adapted to the data.

```{literalinclude} first_computation.py
:language: python
:start-at: import numpy
```

```{code-block} none
:class: sphx-glr-script-out

Ranks of the approximation: [1 1 1 1 2 2 4 3 3 3 1]
Storage complexity: 186 parameters
Relative test error: 1.89e-05
```

With 186 parameters, instead of $9^6 = 531441$ for the full tensor of
coefficients, the relative error on 10000 test points is about $2 \cdot
10^{-5}$. The objects of the script, and where to read more:

| Step | Objects | To read more |
| --- | --- | --- |
| Random variables | {py:class}`~tensap.RandomVector`, {py:class}`~tensap.UniformRandomVariable` | {doc}`/theory/functions` |
| Bases | {py:class}`~tensap.PolynomialFunctionalBasis`, {py:class}`~tensap.FunctionalBases` | {doc}`/theory/functions` |
| Learning | {py:class}`~tensap.TreeBasedTensorLearning`, {py:class}`~tensap.SquareLossFunction` | {ref}`Learning in tensor formats <theory-tensor-learning>` |
| Result | {py:class}`~tensap.FunctionalTensor`, {py:class}`~tensap.TreeBasedTensor` | {ref}`Tree-based tensors <theory-tree-based-tensor>` |

## The main objects

| Object | What it is |
| --- | --- |
| {py:class}`~tensap.FullTensor` | a tensor stored as a multidimensional array |
| {py:class}`~tensap.TreeBasedTensor` | a tensor in tree-based format (tensor train, Tucker, hierarchical Tucker) |
| {py:class}`~tensap.DimensionTree` | the tree of the dimensions of a tree-based tensor |
| {py:class}`~tensap.FunctionalBases` | one basis of functions per variable |
| {py:class}`~tensap.FunctionalTensor` | a function whose coefficients on a tensor product basis form a tensor |
| {py:class}`~tensap.TreeBasedTensorLearning` | learning of a function in tree-based tensor format from samples |

## Going further

- The {doc}`/examples/index` shows, one example at a time, each feature of
  tensap; for instance {ref}`sphx_glr_examples_learning_tree_based_tensor_learning.py`
  gives the options of the learning algorithm used above.
- {doc}`/theory/index` presents the mathematical objects and the methods.
- {doc}`/use_cases/index` treats complete problems.
- The {doc}`/api/index` describes every class and function.
