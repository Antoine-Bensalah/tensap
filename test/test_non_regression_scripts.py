"""Run the non-regression scripts.

The scripts of ``test/non_regression_scripts/`` are the former tutorials of
tensap, kept as they were, with a fixed random seed. Each one must run
without error. They are also the raw material of the Examples Gallery and
of the Use cases of the documentation.
"""

import glob
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "non_regression_scripts")
SCRIPTS = sorted(glob.glob(os.path.join(SCRIPTS_DIR, "**", "*.py"), recursive=True))


def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update(
        {
            "__file__": filepath,
            "__name__": "__main__",
        }
    )

    with open(filepath, "rb") as file:
        exec(compile(file.read(), filepath, "exec"), globals, locals)


@pytest.mark.parametrize("f", SCRIPTS, ids=lambda f: os.path.relpath(f, SCRIPTS_DIR))
def test_non_regression_script(f):
    print(f"-- running {f}", flush=True)
    t0 = time.time()
    execfile(f)
    dt = time.time() - t0
    print(f"-- {f} ran in {dt:.2f} s", flush=True)
    plt.close("all")
