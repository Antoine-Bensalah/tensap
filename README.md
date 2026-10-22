[![Build Status](https://github.com/anthony-nouy/tensap/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/anthony-nouy/tensap/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/anthony-nouy/tensap/branch/master/graph/badge.svg)](https://codecov.io/gh/anthony-nouy/tensap)
[![Doc](https://readthedocs.org/projects/spack/badge/?version=latest)](https://anthony-nouy.github.io/sphinx/tensap/master/)


# tensap (Tensor Approximation Package)

A Python package for the approximation of functions and tensors.

tensap features low-rank tensors (including canonical, tensor-train and tree-based tensor formats or tree tensor networks), sparse tensors, polynomials, and allows the plug-in of other approximation tools. It provides different approximation methods based on interpolation, least-squares projection or statistical learning.

See the tutorials folder and its subfolders for examples of use.

Install from PyPi:

```
pip install tensap
```


Install from conda:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install tensap
```

## Running the tests

Install the test dependencies, then run pytest from the root of the repository:

```
pip install .[test]
pytest              # fast tests only (about 1 min): slow tests are excluded
pytest -n auto      # the same, in parallel (pytest-xdist)
pytest -m slow      # only the slow tests (long tutorials)
pytest -m "" -n auto  # the whole suite, in parallel
```

The tests in `test/test_tutorials.py` run every script of the `tutorials` folder.
Tutorials taking more than about 10 s are listed in `SLOW_TUTORIALS` in that file and
marked `slow`. The ten slowest tests are reported at the end of each run
(`--durations=10`), and a test running for more than 20 min fails (pytest-timeout); see
`[tool.pytest.ini_options]` in `pyproject.toml`.

On GitHub, the fast tests run on every push; the whole suite runs on pull requests,
every night and on demand.
