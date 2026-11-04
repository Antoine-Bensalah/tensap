---
html_theme.sidebar_secondary.remove: true
---

# tensap

```{raw} html
<div class="tensap-hero">
  <img class="tensap-hero__logo only-light" src="_static/tensap_logo.svg" alt="tensap">
  <img class="tensap-hero__logo only-dark" src="_static/tensap_logo_dark.svg" alt="tensap">
  <p class="tensap-hero__tagline">Tensor Approximation Package</p>
  <p class="tensap-hero__text">
    A Python package for the approximation of functions and tensors with
    low-rank tensor formats: canonical, tensor-train and tree-based tensor
    formats (tree tensor networks), sparse tensors and polynomials, with
    approximation methods based on interpolation, least-squares projection
    and statistical learning.
  </p>
  <p class="tensap-hero__buttons">
    <a class="tensap-button tensap-button--primary" href="getting_started/index.html">Getting Started</a>
    <a class="tensap-button" href="examples/index.html">Examples Gallery</a>
    <a class="tensap-button" href="https://github.com/anthony-nouy/tensap">GitHub</a>
  </p>
</div>
```


## What tensap does

::::{grid} 1 2 3 3
:gutter: 3
:class-container: tensap-cards

:::{grid-item-card} {octicon}`apps;1.4em` Tensors and tensor formats
:link: sphx_glr_examples_tensors
:link-type: ref

Full, canonical, sparse and tree-based tensors, dimension trees, truncation.
:::

:::{grid-item-card} {octicon}`graph;1.4em` Functions, measures and bases
:link: sphx_glr_examples_functions
:link-type: ref

Random variables, orthonormal polynomials, B-splines, functional bases and grids.
:::

:::{grid-item-card} {octicon}`pin;1.4em` Interpolation and projection
:link: sphx_glr_examples_interpolation
:link-type: ref

Interpolation, magic points and least-squares approximation in tensor formats.
:::

:::{grid-item-card} {octicon}`stack;1.4em` Principal component analysis
:link: sphx_glr_examples_pca
:link-type: ref

Higher-order PCA of tensors and of multivariate functions.
:::

:::{grid-item-card} {octicon}`cpu;1.4em` Learning in tensor formats
:link: sphx_glr_examples_learning
:link-type: ref

Statistical learning with canonical and tree-based tensor formats, with rank and tree adaptation.
:::

:::{grid-item-card} {octicon}`git-merge;1.4em` Poincaré learning
:link: sphx_glr_examples_poincare
:link-type: ref

Feature maps for dimension reduction, learnt by minimising a Poincaré-based loss, and surrogate models.
:::

::::

## Documentation

::::{grid} 1 2 3 3
:gutter: 3
:class-container: tensap-cards tensap-cards--sections

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc

How do I begin using tensap?
:::

:::{grid-item-card} Theory & Methods
:link: theory/index
:link-type: doc

What are the mathematical concepts, and how do the algorithms work?
:::

:::{grid-item-card} Examples Gallery
:link: examples/index
:link-type: doc

How do I use this particular method or feature?
:::

:::{grid-item-card} Use cases
:link: use_cases/index
:link-type: doc

How can tensap solve a complete scientific problem?
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

What exactly does this software interface do?
:::

:::{grid-item-card} Contributing
:link: contributing/index
:link-type: doc

How do I build the documentation, write an example or propose a change?
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
theory/index
examples/index
use_cases/index
api/index
contributing/index
bibliography
```
