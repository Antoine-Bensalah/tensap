"""
Unit tests of the learning of tensors and of the principal component
analysis of tensors.

The target f(x) = x0 x1 + x2^2 + x3 of four uniform variables on [-1, 1] lies
in the span of Legendre polynomials of degree 2, with canonical rank 3 and
ranks 2 in tree-based format on the balanced tree ({0, 1} and {2, 3} each
need two functions): the solvers must recover it up to their tolerances,
with small samples and few iterations.
"""

import contextlib
import io

import numpy as np
import numpy.testing as npt
import pytest
import tensap

D = 4
X = tensap.RandomVector(tensap.UniformRandomVariable(), D)


def _f(x):
    return x[:, 0] * x[:, 1] + x[:, 2] ** 2 + x[:, 3]


@pytest.fixture
def bases():
    return tensap.FunctionalBases(
        [tensap.PolynomialFunctionalBasis(r.orthonormal_polynomials(), range(3))
         for r in X.random_variables])


@pytest.fixture
def data():
    x_train, x_test = X.random(80), X.random(100)
    return [x_train, _f(x_train)], [x_test, _f(x_test)]


def _configure(solver, bases, data, **attributes):
    solver.bases = bases
    solver.training_data = data[0]
    solver.display = False
    for name, value in attributes.items():
        if isinstance(value, dict):
            getattr(solver, name).update(value)
        else:
            setattr(solver, name, value)
    return solver


def _solve(solver):
    with contextlib.redirect_stdout(io.StringIO()):  # solvers print warnings
        return solver.solve()


def _error(f, data):
    return tensap.SquareLossFunction().test_error(f, data[1])


def _tree_solver(bases, data, tree=None, **attributes):
    tree = tensap.DimensionTree.balanced(D) if tree is None else tree
    solver = tensap.TreeBasedTensorLearning(
        tree, np.full(tree.nb_nodes, True), tensap.SquareLossFunction())
    return _configure(solver, bases, data, **attributes)


# --- tree-based tensor learning --------------------------------------------------------


def test_random_exploration_of_the_tree(bases, data):
    # the nodes of each level are updated in a random order: the alternating
    # minimization may stop in a local minimum, only its progress is checked
    solver = _tree_solver(bases, data, rank=2,
                          alternating_minimization_parameters={"random": True})
    f, output = _solve(solver)
    assert _error(f, data) < 1
    assert output["stagnation_criterion"][-1] < output["stagnation_criterion"][0]


# --- canonical tensor learning -------------------------------------------------------------


def test_canonical_learning_with_fixed_rank(bases, data):
    solver = tensap.CanonicalTensorLearning(D, tensap.SquareLossFunction())
    solver = _configure(solver, bases, data, rank=3,
                        alternating_minimization_parameters={"max_iterations": 100,
                                                             "stagnation": 1e-12})
    f, output = _solve(solver)
    assert isinstance(f.tensor, tensap.CanonicalTensor)
    assert f.tensor.core.data.size == 3
    assert _error(f, data) < 1e-6


def test_canonical_learning_one_factor_at_a_time(bases, data):
    solver = tensap.CanonicalTensorLearning(D, tensap.SquareLossFunction())
    solver = _configure(solver, bases, data, rank_adaptation=True,
                        alternating_minimization_parameters={"one_by_one_factor": True})
    f, _ = _solve(solver)
    assert _error(f, data) < 1e-2


def test_greedy_canonical_learning(bases, data):
    # rank-one corrections followed by an update of the coefficients: the
    # error decreases along the sequence (pure greedy, no exact recovery)
    solver = tensap.CanonicalTensorLearning(D, tensap.SquareLossFunction())
    solver = _configure(solver, bases, data, rank=3, algorithm="greedy")
    f, output = _solve(solver)
    errors = [_error(tensap.FunctionalTensor(g, bases), data) for g in output["sequence"]]
    assert np.all(np.diff(errors) < 0)
    assert f.tensor.core.data.size == 3
    npt.assert_allclose(_error(f, data), errors[-1])


# --- principal component analysis ------------------------------------------------------------


SIZE = [5, 6, 4]


def _g(i):
    """A function of indices of Tucker ranks (2, 2, 2) and tree-based ranks 2."""
    i = np.atleast_2d(i) / np.array(SIZE)
    return np.cos(i[:, 0]) * i[:, 1] + i[:, 2] ** 2


def _dense():
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in SIZE], indexing="ij"), -1)
    return _g(grid.reshape(-1, 3)).reshape(SIZE)


@pytest.fixture
def tpca():
    t = tensap.TensorPrincipalComponentAnalysis()
    t.pca_sampling_factor = 1
    t.pca_adaptive_sampling = False
    t.tol = 1e-10
    return t
