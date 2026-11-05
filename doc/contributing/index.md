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

## Code in the pages

Code is shown the same way on every page: the style of the code blocks
(padding, line spacing, colours) is set once for the whole site, in
`doc/_static/tensap.css`. To keep it uniform, write code only in the cells of
a fenced block ```` ```python ```` of a MyST
page, or with `.. code-block::` or `.. literalinclude::` in reStructuredText.
Do not paste raw HTML, images of code, or styles specific to one page.

Lines of code are at most 72 characters long (the limit PEP 8 sets for
comments and docstrings), so that a code block fits in the page without
scrolling on a laptop screen, from a window 1280 pixels wide.

## Writing a theory page

The pages of Theory & Methods are MyST Markdown files in `doc/theory/`.
Write mathematics as in LaTeX, between `$...$` or `$$...$$`. The macros of
`doc/math_notations.sty` (such as `\Rbb`, `\Ebb` or `\Tc`) are also defined
for MathJax in `doc/conf.py`, so that the HTML pages and the LaTeX manual use
the same notations. Link a class or a function to its API page with
`` {py:class}`~tensap.FullTensor` `` (or `{py:func}` for a function).

## Citing a reference

The references are BibTeX entries of `doc/references.bib`, listed on the
{ref}`bibliography` page. Cite one with `` {cite:p}`Grelier2018` ``
(in Markdown) or `` :cite:p:`Grelier2018` `` (in reStructuredText). Add a new
entry to `references.bib` before citing it: a citation to an unknown key
fails the build.
