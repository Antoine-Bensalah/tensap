---
html_theme.sidebar_secondary.remove: true
---

# tensap

```{raw} html
<div class="tensap-hero">
  <div class="tensap-hero__main">
    <img class="tensap-hero__logo only-light" src="_static/tensap_logo.svg" alt="tensap">
    <img class="tensap-hero__logo only-dark" src="_static/tensap_logo_dark.svg" alt="tensap">
    <p class="tensap-hero__tagline">Tensor Approximation Package</p>
    <p class="tensap-hero__text">
      Low-rank tensor formats to approximate and learn high-dimensional
      functions and operators.
    </p>
    <p class="tensap-hero__buttons">
      <a class="tensap-button tensap-button--primary" href="getting_started/index.html">Getting Started</a>
      <a class="tensap-button" href="examples/index.html">Examples Gallery</a>
      <a class="tensap-button" href="theory/index.html">Theory &amp; Methods</a>
    </p>
  </div>
  <div class="tensap-hero__picture">
    <img class="only-light" src="_static/tensap_hero.svg" alt="A large tensor approximated by a tree-based tensor format">
    <img class="only-dark" src="_static/tensap_hero_dark.svg" alt="A large tensor approximated by a tree-based tensor format">
  </div>
</div>
```


## What tensap does

::::{grid} 1 2 3 3
:gutter: 2
:class-container: tensap-cards

:::{grid-item-card} {octicon}`apps;1.4em` Tensor formats
:link: sphx_glr_examples_tensors
:link-type: ref

Full, sparse and low-rank tensors, dimension trees.
:::

:::{grid-item-card} {octicon}`graph;1.4em` Functions and bases
:link: sphx_glr_examples_functions
:link-type: ref

Measures, polynomials, B-splines, bases and grids.
:::

:::{grid-item-card} {octicon}`pin;1.4em` Interpolation
:link: sphx_glr_examples_interpolation
:link-type: ref

Interpolation and least squares in tensor formats.
:::

:::{grid-item-card} {octicon}`stack;1.4em` Tensor PCA
:link: sphx_glr_examples_pca
:link-type: ref

Higher-order PCA of tensors and functions.
:::

:::{grid-item-card} {octicon}`cpu;1.4em` Learning
:link: sphx_glr_examples_learning
:link-type: ref

Learning with rank and tree adaptation.
:::

:::{grid-item-card} {octicon}`git-merge;1.4em` Poincaré learning
:link: sphx_glr_examples_poincare
:link-type: ref

Dimension reduction with a Poincaré-based loss.
:::

::::

## Documentation

::::{grid} 1 2 3 3
:gutter: 2
:class-container: tensap-cards tensap-cards--sections

:::{grid-item-card} {octicon}`rocket;1.4em` Getting Started
:link: getting_started/index
:link-type: doc

Install tensap and run a first computation.
:::

:::{grid-item-card} {octicon}`mortar-board;1.4em` Theory & Methods
:link: theory/index
:link-type: doc

Concepts, notations and algorithms.
:::

:::{grid-item-card} {octicon}`image;1.4em` Examples Gallery
:link: examples/index
:link-type: doc

One method or feature per example.
:::

:::{grid-item-card} {octicon}`beaker;1.4em` Use Cases
:link: use_cases/index
:link-type: doc

Complete scientific problems.
:::

:::{grid-item-card} {octicon}`code;1.4em` API Reference
:link: api/index
:link-type: doc

Every class and function.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.4em` Contributing
:link: contributing/index
:link-type: doc

Build the docs, add an example, propose a change.
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
