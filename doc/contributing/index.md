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
