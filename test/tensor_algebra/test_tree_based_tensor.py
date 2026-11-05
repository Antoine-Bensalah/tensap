"""
Unit tests of tensap.TreeBasedTensor.

The tests run on three dimension trees of order 4 (linear, balanced,
trivial), with all the nodes active or with an inactive leaf. full() is first
checked against explicit contractions; every other operation is then checked
against the same operation on the dense tensor, or through exact identities.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

SHAPE = [3, 4, 2, 3]

TREES = {
    "linear": tensap.DimensionTree.linear(4),
    "balanced": tensap.DimensionTree.balanced(4),
    "trivial": tensap.DimensionTree.trivial(4),
}


def _ranks(tree, value=2):
    ranks = np.full(tree.nb_nodes, value)
    ranks[tree.root - 1] = 1
    return ranks


@pytest.fixture(params=list(TREES))
def tree(request):
    return TREES[request.param]


@pytest.fixture(params=["active", "inactive leaf"])
def x(tree, request):
    """A random tree-based tensor with ranks 2 (np.random is seeded), with all
    the nodes active, or with the leaf of dimension 0 inactive."""
    x = tensap.TreeBasedTensor.randn(tree, _ranks(tree), SHAPE)
    if request.param == "inactive leaf":
        x = x.inactivate_nodes([tree.dim2ind[0]])
    return x


def _random_like(x):
    """A random tensor with the same tree and the same active nodes as x."""
    y = tensap.TreeBasedTensor.randn(x.tree, _ranks(x.tree, 3), SHAPE)
    return y.inactivate_nodes(x.non_active_nodes) if x.non_active_nodes.size else y


# --- construction and full ----------------------------------------------------


def _check_structure(x, shape):
    """The tensors of the nodes are consistent with the ranks, the shape and
    the active nodes of x."""
    tree = x.tree

    def leaf_size(nod):
        return shape[int(np.nonzero(tree.dim2ind == nod)[0][0])]

    npt.assert_array_equal(x.full().shape[: len(shape)], shape)
    for nod in range(1, tree.nb_nodes + 1):
        if not x.is_active_node[nod - 1]:
            assert tree.is_leaf[nod - 1] and x.ranks[nod - 1] == 0
            continue
        if tree.is_leaf[nod - 1]:
            expected = [leaf_size(nod)]
        else:
            expected = [x.ranks[c - 1] if x.is_active_node[c - 1] else leaf_size(c)
                        for c in tree.children(nod)]
        if nod != tree.root or x.ranks[nod - 1] > 1:
            expected.append(x.ranks[nod - 1])
        npt.assert_array_equal(x.tensors[nod - 1].shape, expected)


# --- arithmetic ---------------------------------------------------------------


def test_matrix_times_tensor_acts_on_the_root_rank(tree, rng):
    # (M * x)(i_1, ..., i_d, a) = sum_k M(a, k) x(i_1, ..., i_d, k)
    x = tensap.TreeBasedTensor.randn(tree, np.full(tree.nb_nodes, 2), SHAPE)
    M = rng.standard_normal((5, 2))
    y = M * x
    assert y.ranks[tree.root - 1] == 5
    npt.assert_allclose(y.full().numpy(), np.einsum("ijklr,ar->ijkla", x.full().numpy(), M))
    npt.assert_allclose((np.array([[3.0]]) * x).full().numpy(), 3.0 * x.full().numpy())
    with pytest.raises(ValueError):
        rng.standard_normal((5, 3)) * x


# --- contractions ---------------------------------------------------------------


# --- evaluations ---------------------------------------------------------------


# --- structure ---------------------------------------------------------------------


def test_inactivate_and_activate_leaves(x):
    a = x.full().numpy()
    leaf = x.tree.dim2ind[3]
    y = x.inactivate_nodes([leaf])
    assert not y.is_active_node[leaf - 1]
    assert y.is_active_dim(3) is False
    assert x.is_active_dim(3) is True
    npt.assert_allclose(y.full().numpy(), a, atol=1e-12)
    z = y.activate_nodes([leaf])
    assert z.is_active_node[leaf - 1]
    npt.assert_allclose(z.full().numpy(), a, atol=1e-12)


# === Part 2: operations on the tree ===================================================


def _diag_tensor(tree, inactive_first_leaf=False):
    x = tensap.TreeBasedTensor.randn(tree, _ranks(tree), [3, 3, 3, 3])
    if inactive_first_leaf:
        x = x.inactivate_nodes([tree.dim2ind[0]])
    return x


def _core_in_gradient_order(x, nod):
    """Core of nod with its dimensions ordered as in the gradient: active
    children, then inactive children, then the parent."""
    core = x.tensors[nod - 1].numpy()
    children = list(x.tree.children(nod))
    active = [i for i, c in enumerate(children) if x.is_active_node[c - 1]]
    inactive = [i for i, c in enumerate(children) if not x.is_active_node[c - 1]]
    rest = list(range(len(children), core.ndim))
    return np.transpose(core, active + inactive + rest)


@pytest.mark.parametrize("inactive", [False, True])
def test_parameter_gradient_eval_diag_euler_identity(tree, inactive):
    # the diagonal is linear in each node tensor: f(i) = <grad(i), tensor>
    x = _diag_tensor(tree, inactive)
    d = np.ravel(x.eval_diag())
    for nod in x.active_nodes:
        g = x.parameter_gradient_eval_diag(nod).numpy()
        core = _core_in_gradient_order(x, nod)
        assert g.shape[1:] == core.shape
        npt.assert_allclose(np.tensordot(g, core, axes=core.ndim), d)


def test_parameter_gradient_eval_diag_with_identity_matrices(tree):
    x = _diag_tensor(tree)
    identity = [np.eye(3)] * 4
    for nod in x.active_nodes:
        npt.assert_allclose(
            x.parameter_gradient_eval_diag(nod).numpy(),
            x.parameter_gradient_eval_diag(nod, identity).numpy(),
        )


def _merged_in_gradient_order(x, nod):
    """Contraction of the tensors of nod and of its parent, with its dimensions
    ordered as in the DMRG gradient: active then inactive children of nod,
    then active then inactive other children of the parent, then the parent
    of the parent."""
    tree = x.tree
    parent = tree.parent(nod)
    A = x.tensors[nod - 1].numpy()
    P = x.tensors[parent - 1].numpy()
    position = list(tree.children(parent)).index(nod)
    merged = np.tensordot(A, P, axes=([A.ndim - 1], [position]))

    def split(children, offset):
        active = [offset + i for i, c in enumerate(children) if x.is_active_node[c - 1]]
        inactive = [offset + i for i, c in enumerate(children) if not x.is_active_node[c - 1]]
        return active + inactive

    children_a = list(tree.children(nod)) if not tree.is_leaf[nod - 1] else []
    perm = split(children_a, 0) if children_a else list(range(A.ndim - 1))
    others = [c for c in tree.children(parent) if c != nod]
    perm += split(others, A.ndim - 1)
    perm += list(range(A.ndim - 1 + len(others), merged.ndim))
    return np.transpose(merged, perm)


@pytest.mark.parametrize("inactive", [False, True])
def test_parameter_gradient_eval_diag_dmrg_euler_identity(tree, inactive):
    # the diagonal is linear in the contraction of a node with its parent
    x = _diag_tensor(tree, inactive)
    d = np.ravel(x.eval_diag())
    identity = [np.eye(3)] * 4
    for nod in x.active_nodes:
        if nod == tree.root:
            continue
        merged = _merged_in_gradient_order(x, nod)
        g, g_alpha, g_gamma = x.parameter_gradient_eval_diag_dmrg(nod)
        g = g.numpy()
        npt.assert_allclose(np.tensordot(g, merged, axes=merged.ndim), d)
        npt.assert_allclose(
            g, x.parameter_gradient_eval_diag_dmrg(nod, identity)[0].numpy()
        )


def test_nodes_permutation_cost(tree):
    x = _diag_tensor(tree)
    cost = x.nodes_permutation_cost(tree.dim2ind[0])
    assert np.size(cost) == tree.nb_nodes
    assert np.all(np.asarray(cost) >= 0)


@pytest.mark.parametrize("method", ["optimize_dimension_tree", "optimize_leaves_permutations"])
def test_optimizations_preserve_the_tensor_and_do_not_increase_storage(tree, method):
    x = _diag_tensor(tree)
    a = x.full().numpy()
    y = getattr(x, method)(1e-12, 5)
    npt.assert_allclose(y.full().numpy(), a, atol=1e-9)
    assert y.storage() <= x.storage()
