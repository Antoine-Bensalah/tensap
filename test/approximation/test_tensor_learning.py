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


@pytest.mark.parametrize("initialization", ["canonical", "random", "ones"])
def test_tree_based_learning_with_fixed_ranks(bases, data, initialization):
    solver = _tree_solver(bases, data, rank=2, initialization_type=initialization)
    f, output = _solve(solver)
    assert isinstance(f, tensap.FunctionalTensor)
    npt.assert_array_equal(f.tensor.ranks, [1, 2, 2, 2, 2, 2, 2])
    assert _error(f, data) < 1e-4
    assert output["flag"] in (0, 1) and len(output["iterates"]) == output["iter"] + 1
    assert np.all(np.diff(output["stagnation_criterion"][1:]) < 0.1)


def test_mean_initialization_requires_rank_one(bases, data):
    solver = _tree_solver(bases, data, rank=2, initialization_type="mean")
    with pytest.raises(NotImplementedError):
        _solve(solver)


def test_random_exploration_of_the_tree(bases, data):
    # the nodes of each level are updated in a random order: the alternating
    # minimization may stop in a local minimum, only its progress is checked
    solver = _tree_solver(bases, data, rank=2,
                          alternating_minimization_parameters={"random": True})
    f, output = _solve(solver)
    assert _error(f, data) < 1
    assert output["stagnation_criterion"][-1] < output["stagnation_criterion"][0]


@pytest.mark.slow  # several seconds
def test_tree_based_learning_with_rank_adaptation(bases, data):
    solver = _tree_solver(bases, data, rank_adaptation=True)
    f, output = _solve(solver)
    assert _error(f, data) < 1e-4
    assert np.all(f.tensor.ranks[1:] >= 2)
    assert len(output["iterates"]) == len(output["enriched_nodes_iterations"])


@pytest.mark.slow  # several seconds
def test_model_selection_with_the_test_error(bases, data):
    solver = _tree_solver(bases, data, rank_adaptation=True, test_error=True,
                          test_data=data[1], model_selection=True)
    f, output = _solve(solver)
    k = output["selected_model_number"]
    npt.assert_allclose(output["test_error"], np.min(output["test_error_iterations"]))
    assert output["test_error"] == output["test_error_iterations"][k] < 1e-4


@pytest.mark.slow  # several seconds
@pytest.mark.parametrize("constructor", ["tensor_train", "tensor_train_tucker"])
def test_tensor_train_formats(bases, data, constructor):
    solver = getattr(tensap.TreeBasedTensorLearning, constructor)(
        D, tensap.SquareLossFunction())
    f, _ = _solve(_configure(solver, bases, data, rank_adaptation=True))
    assert _error(f, data) < 1e-4
    if constructor == "tensor_train":  # the leaves are not active
        assert np.sum(f.tensor.is_active_node) == D


def test_dmrg_rank_adaptation_increases_the_ranks(bases, data):
    solver = tensap.TreeBasedTensorLearning.tensor_train(D, tensap.SquareLossFunction())
    solver = _configure(solver, bases, data, rank=1, rank_adaptation=True,
                        rank_adaptation_options={"type": "dmrg", "max_rank": 3},
                        alternating_minimization_parameters={"max_iterations": 5})
    f, _ = _solve(solver)
    assert np.max(f.tensor.ranks) > 1 and np.max(f.tensor.ranks) <= 3


def test_vector_valued_learning(bases, data):
    y = np.stack([data[0][1], 2 * data[0][1]], 1)
    solver = _tree_solver(bases, [[data[0][0], y], data[1]], rank=2)
    f, _ = _solve(solver)
    npt.assert_allclose(f(data[1][0]), np.stack([data[1][1], 2 * data[1][1]], 1), atol=1e-2)


def test_tensor_learning_errors(bases, data):
    solver = _tree_solver(bases, data, rank_adaptation=True,
                          rank_adaptation_options={"type": 1})
    with pytest.raises(ValueError):
        _solve(solver)
    solver = tensap.CanonicalTensorLearning(D, tensap.SquareLossFunction())
    y = np.stack([data[0][1]] * 2, 1)
    with pytest.raises(NotImplementedError):  # vector-valued outputs
        _solve(_configure(solver, bases, [[data[0][0], y]]))


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


@pytest.mark.parametrize("initialization", ["mean", "greedy", "random", "ones"])
def test_canonical_learning_with_rank_adaptation(bases, data, initialization):
    solver = tensap.CanonicalTensorLearning(D, tensap.SquareLossFunction())
    solver = _configure(solver, bases, data, rank_adaptation=True,
                        initialization_type=initialization)
    f, _ = _solve(solver)
    assert _error(f, data) < 1e-3


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


def test_hopca_finds_the_ranks(tpca):
    bases, outputs = tpca.hopca(_g, SIZE)
    npt.assert_array_equal([b.shape[1] for b in bases], [2, 2, 2])
    assert all(o["number_of_evaluations"] > 0 for o in outputs)


@pytest.mark.parametrize("method", ["tucker_approximation", "tt_approximation"])
def test_tensor_pca_approximations_are_exact(tpca, method):
    f, output = getattr(tpca, method)(_g, SIZE)
    npt.assert_allclose(f.full().numpy(), _dense(), atol=1e-10)
    assert output["number_of_evaluations"] < np.prod(SIZE) * 3


def test_tree_based_pca(tpca):
    tree = tensap.DimensionTree.balanced(3)
    f, _ = tpca.tree_based_approximation(_g, SIZE, tree)
    npt.assert_allclose(f.full().numpy(), _dense(), atol=1e-10)
    tpca.tol = np.inf
    tpca.max_rank = 1
    f, _ = tpca.tucker_approximation(_g, SIZE)
    npt.assert_array_equal(f.ranks[f.tree.dim2ind - 1], [1, 1, 1])


def test_functional_pca(bases):
    fpca = tensap.FunctionalTensorPrincipalComponentAnalysis()
    fpca.bases = bases
    fpca.pca_sampling_factor = 1
    fpca.tol = 1e-10
    sub_bases, _ = fpca.hopca(_f)
    npt.assert_array_equal([b.cardinal() for b in sub_bases], [2, 2, 2, 2])
    x = X.random(20)
    for method, args in [("tucker_approximation", ()), ("tt_approximation", ()),
                         ("tree_based_approximation", (tensap.DimensionTree.balanced(D),))]:
        f, output = getattr(fpca, method)(_f, *args)
        npt.assert_allclose(f(x), _f(x), atol=1e-10)
