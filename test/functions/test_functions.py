"""
Unit tests of the functions of tensap: user-defined, tensorized and
compositional functions, functional tensors, tensor grids.

Functional tensors interpolate f(x) = x0 x1 + x2^2 of three uniform variables
on [-1, 1] with Legendre polynomials of degree at most 2, so that they are
exact and their statistics have closed forms.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap


def _f(x):
    return x[:, 0] * x[:, 1] + x[:, 2] ** 2


@pytest.fixture
def x(rng):
    return rng.uniform(-1, 1, (5, 3))


@pytest.fixture
def bases():
    h = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(3))
    return tensap.FunctionalBases.duplicate(h, 3)


@pytest.fixture(params=["full", "tree-based"])
def g(bases, request):
    g, _ = tensap.FullTensorProductFunctionalBasis(bases).tensor_product_interpolation(_f)
    if request.param == "tree-based":
        tree = tensap.DimensionTree.linear(3)
        g = tensap.FunctionalTensor(tensap.Truncator(1e-12).hsvd(g.tensor, tree), bases)
    return g


# --- user-defined functions -------------------------------------------------------


def test_user_defined_function(x):
    u = tensap.UserDefinedFunction(_f, 3)
    npt.assert_allclose(u(x), _f(x))
    npt.assert_allclose(u.eval(x).ravel(), _f(x))
    v = tensap.UserDefinedFunction(lambda z: np.stack([z[:, 0], 2 * z[:, 1]], 1), 2, [2])
    npt.assert_allclose(v(x[:, :2]), x[:, :2] * [1, 2])


def test_stored_evaluations_are_reused(x):
    calls = []

    def f(z):
        calls.append(z.shape[0])
        return _f(z)

    u = tensap.UserDefinedFunction(f, 3)
    u.store = True
    npt.assert_allclose(u(x), _f(x))
    npt.assert_allclose(u(x[[1, 3]]), _f(x[[1, 3]]))
    npt.assert_allclose(u(np.vstack((x[:2], x[:2] + 1))), _f(np.vstack((x[:2], x[:2] + 1))))
    assert sum(calls) == 7  # only the new points are evaluated
    assert u._x_stored.shape == (7, 3)


def test_random_evaluations(x):
    measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    u = tensap.UserDefinedFunction(_f, 3)
    with pytest.raises(ValueError):
        u.random(3)
    with pytest.raises(ValueError):
        u.random(3, tensap.LebesgueMeasure(0, 1))
    fx, z = u.random(4, measure)
    npt.assert_allclose(np.ravel(fx), _f(z))
    u.measure = measure
    fx, z = u.random(4)
    assert z.shape == (4, 3)


def test_partial_evaluation(x):
    u = tensap.UserDefinedFunction(_f, 3)
    v = u.partial_evaluation([2], np.array([[2.0]]))
    npt.assert_allclose(v(x[:, :2]), x[:, 0] * x[:, 1] + 4)
    u.measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    v = u.partial_evaluation([0], np.array([[0.5]]))
    assert v.measure.ndim() == 2
    npt.assert_allclose(v(x[:, 1:]), 0.5 * x[:, 1] + x[:, 2] ** 2)
    u.dim = []
    with pytest.raises(ValueError):
        u.partial_evaluation([0], np.array([[0.5]]))


def test_evaluations_on_tensor_grids():
    u = tensap.UserDefinedFunction(_f, 3)
    grids = [np.array([0.0, 1.0]), np.array([0.0, 2.0, 3.0]), np.array([1.0, 2.0])]
    y = u.eval_on_tensor_grid(tensap.FullTensorGrid(grids))
    assert isinstance(y, tensap.FullTensor)
    a, b, c = np.meshgrid(*grids, indexing="ij")
    npt.assert_allclose(y.numpy(), a * b + c**2)
    indices = tensap.MultiIndices(np.array([[0, 0, 0], [1, 2, 1]]))
    y = u.eval_on_tensor_grid(tensap.SparseTensorGrid(grids, indices))
    assert isinstance(y, tensap.SparseTensor)
    npt.assert_allclose(y.full().numpy()[1, 2, 1], 1 * 3 + 4)
    with pytest.raises(ValueError):
        u.eval_on_tensor_grid(np.zeros((2, 3)))


def test_test_error(x, g):
    u = tensap.UserDefinedFunction(_f, 3)
    err_l2, err_linf = u.test_error(g, x)
    npt.assert_allclose([err_l2, err_linf], 0, atol=1e-13)
    err_l2, err_linf = u.test_error(2 * _f(x), x)
    npt.assert_allclose([err_l2, err_linf], 0.5)
    u.measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    npt.assert_allclose(u.test_error(g, 20), 0, atol=1e-13)


# --- tensorizer -------------------------------------------------------------------------


def test_base_b_conversions():
    # 0.3 = 0.010011... in base 2: digits 0, 1, 0 and remainder 0.4
    z = tensap.Tensorizer.u2z(np.array([0.3]), 2, 3)
    npt.assert_allclose(z, [[0, 1, 0, 0.4]])
    npt.assert_allclose(tensap.Tensorizer.z2u(z, 2), [0.3])


@pytest.mark.parametrize("ordering", [1, 2])
@pytest.mark.parametrize("b, d, dim", [(2, 4, 1), (3, 2, 2)])
def test_map_and_inverse_map(rng, b, d, dim, ordering):
    t = tensap.Tensorizer(b, d, dim)
    t.ordering_type = ordering
    x = rng.random((6, dim))
    z = t.map(x)
    assert z.shape == (6, (d + 1) * dim)
    assert np.all(z[:, : d * dim] == np.floor(z[:, : d * dim]))  # digits
    npt.assert_allclose(t.inverse_map(z), x, atol=1e-14)
    y, i = t.map(x, nargout=2)
    npt.assert_allclose(np.hstack((i, y)), z)


def test_tensorize(rng):
    t = tensap.Tensorizer(2, 4, 1)
    x = rng.random((5, 1))
    f = t.tensorize(lambda u: np.sin(u[:, 0]))
    assert isinstance(f, tensap.TensorizedFunction)
    npt.assert_allclose(f(t.map(x)), np.sin(x[:, 0]))
    npt.assert_array_equal(f.domain(), [[0, 1]])
    with pytest.raises(ValueError):
        t.tensorize("not a function")


def test_tensorized_function_bases():
    t = tensap.Tensorizer(2, 3, 1)
    H = t.tensorized_function_functional_bases(2)
    assert len(H) == 4
    npt.assert_array_equal(H.cardinals(), [2, 2, 2, 3])
    H = t.tensorized_function_functional_bases()
    npt.assert_array_equal(H.cardinals(), [2, 2, 2, 2])  # degree 1 by default


# --- compositional functions and sets ---------------------------------------------------


def test_compositional_model_function(x):
    tree = tensap.DimensionTree.linear(3)
    measure = tensap.RandomVector(tensap.UniformRandomVariable(), 3)
    f = tensap.CompositionalModelFunction(tree, lambda a, b: a * b, measure)
    npt.assert_allclose(f(x), np.prod(x, 1))
    funs = np.empty(tree.nb_nodes, dtype=object)
    funs[tree.internal_nodes - 1] = lambda a, b: a + b
    f = tensap.CompositionalModelFunction(tree, list(funs), measure)
    npt.assert_allclose(f(x), np.sum(x, 1))


def test_is_in():
    B = np.array([[0.0, 1.0], [1.0, 2.0]])  # [0, 1] x [1, 2]
    x = np.array([[0.5, 1.5], [0.5, 2.5], [-0.1, 1.0], [1.0, 1.0]])
    npt.assert_array_equal(tensap.is_in(B, x), [True, False, False, True])
    npt.assert_array_equal(tensap.is_in([0, 1], np.array([[0.5], [2.0]])), [True, False])


# --- tensor grids ----------------------------------------------------------------------


def test_full_tensor_grid():
    G = tensap.FullTensorGrid([np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])])
    assert G.ndim() == 2 and G.size == 6
    npt.assert_array_equal(G.shape, [2, 3])
    # first dimension varies fastest
    npt.assert_array_equal(G.array()[:3], [[0, 0], [1, 0], [0, 1]])
    npt.assert_array_equal(G.multi_indices().array[:3], [[0, 0], [1, 0], [0, 1]])
    npt.assert_array_equal(G.eval_at_indices(np.array([[1, 2]])), [[1, 2]])
    G = tensap.FullTensorGrid(np.array([0.0, 1.0, 2.0]), 3)
    assert G.ndim() == 3 and G.size == 27
    X = tensap.RandomVector([tensap.UniformRandomVariable(), tensap.NormalRandomVariable()])
    G = tensap.FullTensorGrid.random(X, [2, 4])
    npt.assert_array_equal(G.shape, [2, 4])
    assert np.all(np.abs(G.grids[0]) <= 1)


def test_sparse_tensor_grid():
    G = tensap.SparseTensorGrid(
        [np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])],
        tensap.MultiIndices(np.array([[0, 0], [1, 0], [0, 2]])),
    )
    assert G.size == 3 and G.ndim() == 2
    npt.assert_array_equal(G.array(), [[0, 0], [1, 0], [0, 2]])


# --- functional tensors -------------------------------------------------------------------


def test_functional_tensor_evaluations(g, x, bases):
    npt.assert_allclose(g(x), _f(x), atol=1e-14)
    npt.assert_allclose(g.eval_with_bases_evals(bases.eval(x)), _f(x), atol=1e-14)
    y = g.eval_on_grid([np.linspace(-1, 1, 4)] * 3)
    a, b, c = np.meshgrid(*[np.linspace(-1, 1, 4)] * 3, indexing="ij")
    npt.assert_allclose(y.full().numpy() if hasattr(y, "full") else y.numpy(),
                        a * b + c**2, atol=1e-13)
    fx, z = g.random(4, nargout=2)
    npt.assert_allclose(fx, _f(np.hstack(z)), atol=1e-14)
    assert isinstance(g.get_random_vector(), tensap.RandomVector)
    assert g.storage() == g.tensor.storage()


def test_functional_tensor_statistics(g):
    # E f = 1/3, Var f = 1/5, E f^2 = 14/45
    npt.assert_allclose(g.mean(), 1 / 3)
    npt.assert_allclose(g.expectation(), 1 / 3)
    npt.assert_allclose(g.variance(), 1 / 5)
    npt.assert_allclose(g.std(), np.sqrt(1 / 5))
    npt.assert_allclose(g.dot_product_expectation(g), 14 / 45)
    npt.assert_allclose(g.norm(), np.sqrt(14 / 45))


def test_functional_tensor_conditional_expectations(g, x):
    h = g.conditional_expectation([0, 1])
    npt.assert_allclose(h(x[:, :2]), x[:, 0] * x[:, 1] + 1 / 3, atol=1e-14)
    h = g.conditional_expectation(np.array([False, False, True]))
    npt.assert_allclose(h(x[:, [2]]), x[:, 2] ** 2, atol=1e-14)
    v = g.variance_conditional_expectation(
        np.array([[True, False, False], [True, True, False], [False, False, True]]))
    npt.assert_allclose(np.ravel(v), [0, 1 / 9, 4 / 45], atol=1e-14)


def test_functional_tensor_derivatives(g, x):
    npt.assert_allclose(g.eval_derivative([1, 0, 0], x), x[:, 1], atol=1e-13)
    npt.assert_allclose(g.derivative([0, 0, 1])(x), 2 * x[:, 2], atol=1e-13)
    npt.assert_allclose(g.eval_gradient(x), np.stack([x[:, 1], x[:, 0], 2 * x[:, 2]], 1),
                        atol=1e-13)


def test_functional_tensor_product(g, x):
    h = g * g
    npt.assert_allclose(h(x), _f(x) ** 2, atol=1e-13)
    npt.assert_allclose((g * 2.0)(x), 2 * _f(x), atol=1e-14)


def test_parameter_gradient_euler_identity(bases, x):
    # the function is linear in each node tensor: <gradient, tensor> = f(x)
    g, _ = tensap.FullTensorProductFunctionalBasis(bases).tensor_product_interpolation(_f)
    tree = tensap.DimensionTree.linear(3)
    t = tensap.Truncator(1e-12).hsvd(g.tensor, tree)
    g = tensap.FunctionalTensor(t, bases)
    for alpha in range(1, tree.nb_nodes + 1):
        grad = g.parameter_gradient_eval(alpha, x)
        grad = grad[0] if isinstance(grad, tuple) else grad
        grad = grad.numpy() if hasattr(grad, "numpy") else grad
        core = t.tensors[alpha - 1].numpy()
        value = np.sum(grad * core[None], axis=tuple(range(1, grad.ndim)))
        npt.assert_allclose(value, _f(x), atol=1e-12)
