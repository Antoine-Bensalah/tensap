"""Run the non-regression scripts.

The scripts of ``test/non_regression_scripts/`` are the former tutorials of
tensap, kept as they were, with a fixed random seed. Each one must run
without error. They are also the raw material of the Examples Gallery and
of the Use cases of the documentation.
"""

import glob
import importlib.util
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "non_regression_scripts")
SCRIPTS = sorted(glob.glob(os.path.join(SCRIPTS_DIR, "**", "*.py"), recursive=True))

# Scripts taking more than about 10 s each: marked "slow", hence excluded
# from the default run (see [tool.pytest.ini_options] in pyproject.toml).
# Run them with `pytest -m slow`, or everything with `pytest -m ""`.
SLOW_SCRIPTS = {
    "tutorial_tensor_learning_tensorized_function_learning.py",  # ~400 s
    "tutorial_tensor_learning_TreeBasedTensorDensityLearning.py",  # ~290 s
    "tutorial_tensor_learning_TreeBasedTensorLearning.py",  # ~90 s
    "tutorial_tensor_learning_tensor_completion.py",  # ~80 s
    "tutorial_tensor_learning_CanonicalTensorLearning.py",  # ~50 s
    "tutorial_poincare_learning_surrogate_greedy.py",  # ~35 s
    "tutorial_poincare_learning.py",  # ~25 s
    "tutorial_magic_points.py",  # ~20 s
    "tutorial_poincare_learning_surrogate.py",  # ~15 s
    "tutorial_tensor_learning_digits_recognition.py",  # trains a network
    "tutorial_linearsolver_truncation_poisson_solution.py",  # ~35 s
    "tutorial_linearsolver_update_poisson_solution.py",  # ~30 s
    "tutorial_linearsolver_high_dim_poisson.py",  # ~80 s
    "tutorial_qtt_poisson_rectangle.py",  # ~50 s
    "tutorial_qtt_tt_solvers.py",  # ~5 min
}

# Known failures of scripts, with the reason.
# tensorized_function_learning: in the local learning of new_rank_selection
# (rank-one correction), the regression of a node may return zero
# coefficients; TensorLearning._solve_standard prints "returning to the
# previous iteration" but does not restore the previous iterate, the tensor
# ends at zero, and the selection divides its singular values by its norm
# (RuntimeWarning 0/0 in tree_based_tensor_learning.py). Even with the fixed
# seed of the script, whether it happens depends on the round-off (number of
# threads of the linear algebra, platform): the failure is not strict.
KNOWN_FAILURES = {
    "tutorial_tensor_learning_tensorized_function_learning.py": pytest.mark.xfail(
        raises=RuntimeWarning, strict=False,
        reason="zero tensor after a zero local solution in new_rank_selection"),
}

# Scripts that need an optional dependency: skipped when it is missing.
REQUIREMENTS = {
    "tutorial_tensor_learning_digits_recognition.py": "tensorflow",
}


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


def _script_params():
    params = []
    for path in SCRIPTS:
        name = os.path.basename(path)
        marks = [pytest.mark.slow] if name in SLOW_SCRIPTS else []
        if name in KNOWN_FAILURES:
            marks.append(KNOWN_FAILURES[name])
        module = REQUIREMENTS.get(name)
        if module is not None and importlib.util.find_spec(module) is None:
            marks.append(pytest.mark.skip(reason=f"{module} is not installed"))
        params.append(pytest.param(path, marks=marks, id=os.path.relpath(path, SCRIPTS_DIR)))
    return params


@pytest.mark.tutorial
# warnings are errors in the scripts too (an unexpected warning of a solver
# or of a truncation is a failure); the expected ones are listed narrowly:
# tutorial_tensor_truncation truncates a TreeBasedTensor with a tree argument
# on purpose
@pytest.mark.filterwarnings(
    "default:The provided tree and/or is_active_node are not taken into account:UserWarning")
# some scripts open many figures, closed at the end of the test
@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened:RuntimeWarning")
@pytest.mark.parametrize("f", _script_params())
def test_non_regression_script(f):
    print(f"-- running {f}", flush=True)
    t0 = time.time()
    execfile(f)
    dt = time.time() - t0
    print(f"-- {f} ran in {dt:.2f} s", flush=True)
    plt.close("all")
