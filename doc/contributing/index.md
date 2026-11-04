(contributing)=
# Contributing

```{note}
This section is being written. It will explain how to set up a development
environment, write documentation and docstrings, and propose a change.
```

## Building the documentation

From the root of the repository:

```bash
pip install -e ".[doc]"
make html -C doc
```

The pages are written to `doc/_build/html`. Warnings are errors: fix them
before opening a pull request.

## Documenting a public object

Every class or function exported by the `tensap` package must appear in the
API reference, and must have a docstring in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.
Both rules are checked when the documentation is built.

- **Listing the object.** Add its name to the right section of
  `doc/api/<sub-package>.rst`, in an `autosummary` block. The page of the
  object is generated in `doc/api/generated/`, which is not versioned.
  After a renaming, delete this folder before building again.
- **Mathematical description.** Formulas are kept out of the Python
  sources. Write them in `doc/api_notes/<full name>.rst`, for instance
  `doc/api_notes/tensap.FullTensor.rst`; the text is inserted in the API
  page of the object under the rubric "Mathematical description". Follow
  the notation of the Theory & Methods section.

## Adding an example

The examples of the gallery are Python scripts in `examples/<section>/`, in
the format of [sphinx-gallery](https://sphinx-gallery.github.io/): a
docstring with a title and a short introduction, then cells separated by
`# %%` lines, whose first comment lines are rendered as text. Fix the random
seed after the imports. An example is executed when the documentation is
built, unless its name is in `NOT_EXECUTED` in `doc/conf.py`; its position
in its section is set in `doc/_ext/gallery_order.py`.

Use Cases follow the same format, in `use_cases/`.

Code is shown the same way on every page: the style of the code blocks
(padding, line spacing, colours) is set once for the whole site, in
`doc/_static/tensap.css`. To keep it uniform, write code only in the cells of
a sphinx-gallery script, in a fenced block ```` ```python ```` of a MyST
page, or with `.. code-block::` or `.. literalinclude::` in reStructuredText.
Do not paste raw HTML, images of code, or styles specific to one page.

## Writing a theory page

The pages of Theory & Methods are MyST Markdown files in `doc/theory/`.
Write mathematics as in LaTeX, between `$...$` or `$$...$$`. The macros of
`doc/math_notations.sty` (such as `\Rbb`, `\Ebb` or `\Tc`) are also defined
for MathJax in `doc/conf.py`, so that the HTML pages and the LaTeX manual use
the same notations. Link a class or a function to its API page with
`` {py:class}`~tensap.FullTensor` `` (or `{py:func}` for a function), and an
example of the gallery with `` {ref}`sphx_glr_examples_tensors_full_tensor.py` ``.

## Citing a reference

The references are BibTeX entries of `doc/references.bib`, listed on the
{ref}`bibliography` page. Cite one with `` {cite:p}`Grelier2018` ``
(in Markdown) or `` :cite:p:`Grelier2018` `` (in reStructuredText). Add a new
entry to `references.bib` before citing it: a citation to an unknown key
fails the build.
