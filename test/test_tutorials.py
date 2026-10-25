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
    "tutorial_qtt_tt_solvers.py",  # ~4 min (interleaved cube)
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
# the tutorials may show the warnings of tensap to the user (as the truncation
# tutorial does): they are displayed, not errors
@pytest.mark.filterwarnings("default::UserWarning:tensap")
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
