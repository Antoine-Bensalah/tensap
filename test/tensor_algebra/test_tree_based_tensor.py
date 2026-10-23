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


def test_full_of_a_tucker_tensor(rng):
    # trivial tree: x = C x_1 U1 x_2 U2 x_3 U3
    tree = tensap.DimensionTree.trivial(3)
    C = rng.standard_normal((2, 3, 2))
    U = [rng.standard_normal((n, r)) for n, r in zip((4, 5, 3), (2, 3, 2))]
    x = tensap.TreeBasedTensor([C] + U, tree)
    npt.assert_allclose(x.full().numpy(), np.einsum("abc,ia,jb,kc->ijk", C, *U))
    npt.assert_allclose(x.numpy(), x.full().numpy())


def test_full_of_a_tensor_train(rng):
    G = [
        rng.standard_normal((3, 2)),
        rng.standard_normal((2, 4, 2)),
        rng.standard_normal((2, 5)),
    ]
    x = tensap.TreeBasedTensor.tensor_train([tensap.FullTensor(g) for g in G])
    npt.assert_allclose(np.squeeze(x.full().numpy()), np.einsum("ia,ajb,bk->ijk", *G))


def test_full_tensor_round_trip(rng):
    a = rng.standard_normal(SHAPE)
    x = tensap.FullTensor(a).tree_based_tensor()
    assert x.tree.nb_nodes == 5  # trivial tree
    npt.assert_allclose(x.full().numpy(), a)


def test_properties(x, tree):
    assert x.order == x.ndim == 4
    npt.assert_array_equal(x.shape, SHAPE)
    expected = _ranks(tree)
    expected[x.non_active_nodes - 1] = 0  # inactive nodes have rank 0
    npt.assert_array_equal(x.ranks, expected)
    npt.assert_array_equal(x.representation_rank, x.ranks)
    assert x.is_admissible_rank()
    assert x.storage() == sum(t.storage() for t in x.tensors)
    assert x.sparse_storage() <= x.storage()
    assert x.sparse_leaves_storage() <= x.storage()
    assert "TreeBasedTensor" in repr(x)


def test_copy_constructor(x):
    y = tensap.TreeBasedTensor(x)
    npt.assert_allclose(y.full().numpy(), x.full().numpy())


def test_constructor_rejects_wrong_arguments():
    with pytest.raises(NotImplementedError):
        tensap.TreeBasedTensor([np.ones(2)])


@pytest.mark.parametrize(
    "constructor", [tensap.TreeBasedTensor.zeros, tensap.TreeBasedTensor.ones]
)
def test_constant_constructors(tree, constructor):
    x = constructor(tree, _ranks(tree), SHAPE)
    npt.assert_array_equal(x.shape, SHAPE)
    if constructor is tensap.TreeBasedTensor.zeros:
        npt.assert_array_equal(x.full().numpy(), 0.0)


def test_rand_constructor(tree):
    x = tensap.TreeBasedTensor.rand(tree, _ranks(tree), SHAPE)
    npt.assert_array_equal(x.shape, SHAPE)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ranks": "random"},
        {"ranks": _ranks(TREES["balanced"]), "shape": "random"},
        {"ranks": _ranks(TREES["balanced"]), "shape": SHAPE, "is_active_node": "random"},
        {"ranks": "random", "shape": "random", "is_active_node": "random"},
    ],
)
def test_create_with_random_arguments(kwargs):
    tree = TREES["balanced"]
    for _ in range(10):  # the random choices of the active nodes have 3 cases
        x = tensap.TreeBasedTensor.create(lambda s: np.random.randn(*s), tree, **kwargs)
        assert x.ranks[tree.root - 1] == 1
        assert np.all((x.ranks[x.active_nodes - 1] >= 1) & (x.ranks[x.active_nodes - 1] <= 5))
        assert np.all((x.shape >= 1) & (x.shape <= 10))
        _check_structure(x, x.shape)


def test_create_with_default_arguments_uses_the_ranks_of_the_leaves_as_shape(tree):
    x = tensap.TreeBasedTensor.create(np.ones, tree, ranks=2)
    npt.assert_array_equal(x.shape, [2] * 4)
    npt.assert_array_equal(x.ranks, 2)  # a scalar rank applies to the root too
    _check_structure(x, [2] * 4)


def test_random_active_nodes_keep_the_internal_nodes_active():
    tree = TREES["balanced"]
    seen = set()
    for _ in range(30):
        active = tensap.TreeBasedTensor._random_is_active_node(tree)
        assert np.all(active[~tree.is_leaf])
        seen.add(int(np.sum(~active)))
    assert {0, 4} <= seen and len(seen) > 2  # all active, no leaf, some leaves


def test_create_rejects_an_inactive_internal_node(tree):
    if tree.nb_nodes == 5:
        pytest.skip("the trivial tree has no internal node but the root")
    active = np.full(tree.nb_nodes, True)
    internal = [n for n in range(1, tree.nb_nodes + 1)
                if not tree.is_leaf[n - 1] and n != tree.root][0]
    active[internal - 1] = False
    with pytest.raises(ValueError):
        tensap.TreeBasedTensor.randn(tree, _ranks(tree), SHAPE, active)


# --- arithmetic ---------------------------------------------------------------


def test_arithmetic(x):
    y = _random_like(x)
    a, b = x.full().numpy(), y.full().numpy()
    npt.assert_allclose((x + y).full().numpy(), a + b)
    npt.assert_allclose((x - y).full().numpy(), a - b)
    npt.assert_allclose((-x).full().numpy(), -a)
    npt.assert_allclose((2.0 * x).full().numpy(), 2.0 * a)
    npt.assert_allclose((x * 2.0).full().numpy(), 2.0 * a)
    npt.assert_allclose((x / 2.0).full().numpy(), a / 2.0)


def test_hadamard_product(x):
    y = _random_like(x)
    a, b = x.full().numpy(), y.full().numpy()
    npt.assert_allclose((x * y).full().numpy(), a * b)
    npt.assert_allclose(x.hadamard_product(y).full().numpy(), a * b)


def test_addition_requires_a_tree_based_tensor(x):
    with pytest.raises(NotImplementedError):
        x + tensap.FullTensor(x.full().numpy())


def test_addition_requires_the_same_tree(x):
    other = next(t for t in TREES.values() if not t == x.tree)
    y = tensap.TreeBasedTensor.randn(other, None, SHAPE)
    with pytest.raises(ValueError):
        x + y


def test_hadamard_product_with_root_ranks(tree):
    # with a root rank N, the product is taken for each of the N tensors
    ranks = np.full(tree.nb_nodes, 2)
    x = tensap.TreeBasedTensor.randn(tree, ranks, SHAPE)
    y = tensap.TreeBasedTensor.randn(tree, ranks, SHAPE)
    a, b = x.full().numpy(), y.full().numpy()
    npt.assert_allclose((x * y).full().numpy(), a * b)


def test_hadamard_product_requires_equal_root_ranks(tree):
    x = tensap.TreeBasedTensor.randn(tree, np.full(tree.nb_nodes, 2), SHAPE)
    y = tensap.TreeBasedTensor.randn(tree, _ranks(tree), SHAPE)
    for u, v in ((x, y), (y, x)):
        with pytest.raises(ValueError):
            u * v


def test_hadamard_product_requires_a_tree_based_tensor(x):
    with pytest.raises(NotImplementedError):
        x * tensap.FullTensor(x.full().numpy())


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


def test_dot_and_norm(x):
    y = _random_like(x)
    a, b = x.full().numpy(), y.full().numpy()
    npt.assert_allclose(x.dot(y), np.sum(a * b))
    npt.assert_allclose(x.norm(), np.linalg.norm(a))


def test_dot_with_rank_one_metric(x, rng):
    y = _random_like(x)
    a, b = x.full().numpy(), y.full().numpy()
    M = [rng.standard_normal((n, n)) for n in SHAPE]
    npt.assert_allclose(
        x.dot_with_rank_one_metric(y, M), np.einsum("ijkl,abcd,ia,jb,kc,ld->", a, b, *M)
    )


# --- contractions ---------------------------------------------------------------


def test_tensor_matrix_product(x, rng):
    a = x.full().numpy()
    M = [rng.standard_normal((2, n)) for n in SHAPE]
    npt.assert_allclose(
        x.tensor_matrix_product(M).full().numpy(),
        np.einsum("ijkl,ai,bj,ck,dl->abcd", a, *M),
    )
    npt.assert_allclose(
        x.tensor_matrix_product(M[2], 2).full().numpy(),
        np.einsum("ijkl,ck->ijcl", a, M[2]),
    )


def test_tensor_diagonal_matrix_product(x, rng):
    a = x.full().numpy()
    d = [rng.standard_normal(n) for n in SHAPE]
    npt.assert_allclose(
        x.tensor_diagonal_matrix_product(d).full().numpy(),
        np.einsum("ijkl,i,j,k,l->ijkl", a, *d),
    )


def test_tensor_vector_product(x, rng):
    a = x.full().numpy()
    v = [rng.standard_normal(n) for n in SHAPE]
    npt.assert_allclose(
        np.squeeze(x.tensor_vector_product(v)), np.einsum("ijkl,i,j,k,l->", a, *v)
    )
    y = x.tensor_vector_product([v[0], v[3]], [0, 3])
    npt.assert_allclose(y.full().numpy(), np.einsum("ijkl,i,l->jk", a, v[0], v[3]))


def test_reduce_sum(x):
    a = x.full().numpy()
    npt.assert_allclose(np.squeeze(x.reduce_sum()), a.sum())
    npt.assert_allclose(x.reduce_sum([1]).full().numpy(), a.sum(1))


def test_tensor_matrix_product_eval_diag(x, rng):
    a = x.full().numpy()
    H = [rng.standard_normal((5, n)) for n in SHAPE]
    out = x.tensor_matrix_product_eval_diag(H)
    npt.assert_allclose(
        np.ravel(np.asarray(out.numpy() if hasattr(out, "numpy") else out)),
        np.einsum("ijkl,ai,aj,ak,al->a", a, *H),
    )


@pytest.mark.xfail(
    strict=True,
    reason="TreeBasedTensor.tensordot_matrix_product_except_dim calls the array "
    "tree.dim2ind and a helper that is commented out: it always fails",
)
def test_tensordot_matrix_product_except_dim(x, rng):
    y = _random_like(x)
    a, b = x.full().numpy(), y.full().numpy()
    M = [rng.standard_normal((n, n)) for n in SHAPE]
    out = x.tensordot_matrix_product_except_dim(y, M, 1)
    expected = np.einsum("ijkl,abcd,ia,kc,ld->jb", a, b, M[0], M[2], M[3])
    npt.assert_allclose(np.asarray(out.numpy() if hasattr(out, "numpy") else out), expected)


# --- evaluations ---------------------------------------------------------------


def test_eval_at_indices(x):
    a = x.full().numpy()
    indices = np.array([[0, 0, 0, 0], [2, 3, 1, 2], [1, 2, 0, 1]])
    npt.assert_allclose(np.ravel(x.eval_at_indices(indices)), a[tuple(indices.T)])


@pytest.mark.parametrize("inactive", [False, True])
def test_eval_diag(tree, inactive):
    x = _diag_tensor(tree, inactive)
    a = x.full().numpy()
    npt.assert_allclose(np.ravel(x.eval_diag()), [a[i, i, i, i] for i in range(3)])


def test_sub_tensor(x):
    a = x.full().numpy()
    s = x.sub_tensor([0, 2], ":", [1], [0, 2])
    npt.assert_allclose(s.full().numpy(), a[np.ix_([0, 2], range(4), [1], [0, 2])])


# --- structure ---------------------------------------------------------------------


def test_cat_is_block_diagonal(x):
    y = _random_like(x)
    npt.assert_allclose(
        x.cat(y).full().numpy(), x.full().cat(y.full()).numpy(), atol=1e-12
    )


def test_kron(x):
    y = _random_like(x)
    npt.assert_allclose(x.kron(y).full().numpy(), x.full().kron(y.full()).numpy())


def test_orth_preserves_the_tensor_and_orthonormalizes(x):
    a = x.full().numpy()
    y = x.orth()
    assert y.is_orth
    npt.assert_allclose(y.full().numpy(), a, atol=1e-12)
    npt.assert_allclose(y.norm(), np.linalg.norm(a))
    # the leaf bases are orthonormal
    for nod in y.tree.dim2ind:
        if y.is_active_node[nod - 1]:
            U = y.tensors[nod - 1].numpy()
            npt.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)


def test_orth_at_node(x):
    a = x.full().numpy()
    nod = x.tree.internal_nodes[-1]
    y = x.orth_at_node(nod)
    npt.assert_allclose(y.full().numpy(), a, atol=1e-12)


def test_singular_values_are_those_of_the_matricizations(x):
    a = x.full().numpy()
    sv = x.singular_values()
    tree = x.tree
    for nod in x.active_nodes:
        if nod == tree.root:
            continue
        dims = tree.dims[nod - 1]
        rest = [d for d in range(4) if d not in dims]
        n_rows = int(np.prod([a.shape[d] for d in dims]))
        m = np.transpose(a, list(dims) + rest).reshape(n_rows, -1)
        expected = np.linalg.svd(m, compute_uv=False)
        k = np.size(sv[nod - 1])
        npt.assert_allclose(np.sort(np.ravel(sv[nod - 1]))[::-1], expected[:k], atol=1e-10)


def test_gramians_eigenvalues_are_the_squared_singular_values(x):
    gram, _ = x.gramians()
    sv = x.singular_values()
    for nod in range(1, x.tree.nb_nodes + 1):
        if nod == x.tree.root or gram[nod - 1] is None:
            continue
        eig = np.sort(np.linalg.eigvalsh(gram[nod - 1]))[::-1]
        npt.assert_allclose(eig, np.sort(np.ravel(sv[nod - 1]))[::-1] ** 2, atol=1e-10)


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


def test_change_root(x):
    a = x.full().numpy()
    nod = x.tree.internal_nodes[-1]
    y = x.change_root(nod)
    assert y.tree.root == nod
    npt.assert_allclose(y.full().numpy(), a, atol=1e-12)


def test_squeeze(tree, rng):
    x = tensap.TreeBasedTensor.randn(tree, _ranks(tree, 1), [3, 1, 2, 1])
    a = x.full().numpy()
    npt.assert_allclose(x.squeeze().full().numpy(), np.squeeze(a))


def test_remove_unique_children_preserves_the_tensor(x):
    a = x.full().numpy()
    npt.assert_allclose(x.remove_unique_children().full().numpy(), a, atol=1e-12)


# === Part 2: operations on the tree ===================================================


def _diag_tensor(tree, inactive_first_leaf=False):
    x = tensap.TreeBasedTensor.randn(tree, _ranks(tree), [3, 3, 3, 3])
    if inactive_first_leaf:
        x = x.inactivate_nodes([tree.dim2ind[0]])
    return x


@pytest.mark.parametrize("inactive", [False, True])
def test_eval_diag_below_and_above_factorize_the_diagonal(tree, inactive):
    # for every node beta: f(i) = sum_k v^beta(i, k) w^beta(i, k)
    x = _diag_tensor(tree, inactive)
    d = np.ravel(x.eval_diag())
    below = x.eval_diag_below()
    above = x.eval_diag_above(below)
    for nod in x.active_nodes:
        if nod == tree.root:
            continue
        v = np.asarray(below[nod - 1].numpy())
        w = np.asarray(above[nod - 1].numpy())
        npt.assert_allclose(np.sum(v * w, axis=1), d)


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


def test_permute_leaves_preserves_the_tensor(tree):
    x = _diag_tensor(tree)
    a = x.full().numpy()
    y = x.permute_leaves([1, 0, 3, 2])
    npt.assert_allclose(y.full().numpy(), a, atol=1e-10)


def test_permute_nodes(tree):
    x = _diag_tensor(tree)
    a = x.full().numpy()
    leaves = x.tree.dim2ind
    y = x.permute_nodes([leaves[0], leaves[2]])
    npt.assert_allclose(y.full().numpy(), a, atol=1e-10)


def test_permute_nodes_rejects_an_ascendant():
    tree = TREES["balanced"]
    x = _diag_tensor(tree)
    leaf = tree.dim2ind[0]
    with pytest.raises(ValueError):
        x.permute_nodes([tree.parent(leaf), leaf])


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


def test_plot_requires_igraph(tree):
    pytest.importorskip("igraph")
    _diag_tensor(tree).plot()
