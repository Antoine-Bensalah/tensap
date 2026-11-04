"""Run the first computation of the Getting Started page.

The script ``doc/getting_started/first_computation.py`` is included in the
page, with its output: this test checks that it runs, and that the
approximation is as accurate as the page says.
"""

import os

import matplotlib

matplotlib.use("Agg")

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "doc", "getting_started", "first_computation.py"
)


def test_first_computation():
    namespace = {"__file__": SCRIPT, "__name__": "__main__"}
    with open(SCRIPT, "rb") as file:
        exec(compile(file.read(), SCRIPT, "exec"), namespace)
    assert namespace["test_error"] < 1e-4
