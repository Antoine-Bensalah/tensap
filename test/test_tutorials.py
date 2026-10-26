import glob
import importlib.util
import os
import pytest
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Tutorials taking more than about 10 s each: marked "slow", hence excluded
# from the default run (see [tool.pytest.ini_options] in pyproject.toml).
# Run them with `pytest -m slow`, or everything with `pytest -m ""`.
SLOW_TUTORIALS = {
    "tutorial_tensor_learning_tensorized_function_learning.py",  # ~400 s
    "tutorial_tensor_learning_TreeBasedTensorDensityLearning.py",  # ~290 s
    "tutorial_tensor_learning_TreeBasedTensorLearning.py",  # ~90 s
    "tutorial_tensor_learning_tensor_completion.py",  # ~80 s
    "tutorial_tensor_learning_CanonicalTensorLearning.py",  # ~50 s
    "tutorial_poincare_learning_surrogate_greedy.py",  # ~35 s
    "tutorial_linearsolver_truncation_poisson_solution.py",  # ~35 s
    "tutorial_qtt_poisson_rectangle.py",  # ~50 s
    "tutorial_staggered_stokes.py",  # ~150 s
    "tutorial_staggered_taylor_green.py",  # ~140 s
    "tutorial_qtt_tt_solvers.py",  # ~5 min (interleaved cube)
    "tutorial_linearsolver_update_poisson_solution.py",  # ~30 s
    "tutorial_linearsolver_high_dim_poisson.py",  # ~80 s (TT and tree-based cores, d = 6)
    "tutorial_poincare_learning.py",  # ~25 s
    "tutorial_magic_points.py",  # ~20 s
    "tutorial_poincare_learning_surrogate.py",  # ~15 s
    "tutorial_tensor_learning_digits_recognition.py",  # trains a network
}

# Tutorials that need an optional dependency: skipped when it is missing.
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


def _tutorial_params():
    params = []
    for path in sorted(glob.glob("tutorials/**/*.py", recursive=True)):
        name = os.path.basename(path)
        marks = [pytest.mark.slow] if name in SLOW_TUTORIALS else []
        module = REQUIREMENTS.get(name)
        if module is not None and importlib.util.find_spec(module) is None:
            marks.append(pytest.mark.skip(reason=f"{module} is not installed"))
        params.append(pytest.param(path, marks=marks, id=name))
    return params


@pytest.mark.tutorial
# warnings are errors in the tutorials too (an unexpected warning of a solver
# or of a truncation is a failed acceptance); the expected ones are listed,
# narrowly: tutorial_tensor_truncation truncates a TreeBasedTensor with a tree
# argument on purpose (the truncation tutorial of the Poisson solution catches
# its own warning)
@pytest.mark.filterwarnings(
    "default:The provided tree and/or is_active_node are not taken into account:UserWarning")
# some tutorials open many figures, closed at the end of the test
@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened:RuntimeWarning")
@pytest.mark.parametrize("f", _tutorial_params())
def test_tutorial(f):
    print(f"-- running {f}", flush=True)
    t0 = time.time()
    execfile(f)
    dt = time.time() - t0
    print(f"-- {f} ran in {dt:.2f} s", flush=True)
    plt.close("all")
