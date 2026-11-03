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
