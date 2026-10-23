"""
Unit tests of tensap.DimensionTree.

Nodes are numbered from 1 and dimensions from 0. A dimension tree is checked
through its invariants: the dimensions of an internal node are the union of
those of its children, the leaves partition {0, ..., d-1}, the parent,
children, levels and siblings are consistent with the adjacency matrix.
"""

import numpy as np
import numpy.testing as npt
import pytest
import tensap

D = tensap.DimensionTree


def _dims(tree):
    return [sorted(d.tolist()) for d in tree.dims]


def _check_invariants(tree, order):
    A = tree.adjacency_matrix
    assert tree.nb_nodes == A.shape[0]
    assert np.sum(A[:, tree.root - 1]) == 0
    assert sorted(tree.dim2ind.tolist()) == sorted(set(tree.dim2ind.tolist()))
    assert sorted(tree.dims[tree.root - 1].tolist()) == list(range(order))
    for nod in tree.nodes_indices:
        children = tree.children(nod)
        npt.assert_array_equal(children, np.nonzero(A[nod - 1])[0] + 1)
        assert tree.is_leaf[nod - 1] == (nod in tree.dim2ind)
        if tree.is_leaf[nod - 1]:
            assert children.size == 0
            assert tree.dims[nod - 1].tolist() == [tree.dim2ind.tolist().index(nod)]
        else:
            assert sorted(tree.dims[nod - 1].tolist()) == sorted(
                np.concatenate([tree.dims[c - 1] for c in children]).tolist()
            )
        for k, child in enumerate(children):
            assert tree.parent(child) == nod
            assert tree.child_number(child) == k + 1
            assert tree.level[child - 1] == tree.level[nod - 1] + 1
            npt.assert_array_equal(tree.sibling[: children.size, child - 1], children)
    assert tree.parent(tree.root) == 0 and tree.level[tree.root - 1] == 0
    assert tree.arity == max(tree.children(n).size for n in tree.nodes_indices)
    npt.assert_array_equal(tree.internal_nodes, np.nonzero(~tree.is_leaf)[0] + 1)
    npt.assert_array_equal(
        tree.nodes_parent_of_leaves, np.unique([tree.parent(n) for n in tree.dim2ind])
    )


# --- constructors -------------------------------------------------------------


@pytest.mark.parametrize("constructor", [D.trivial, D.linear, D.balanced, D.random])
@pytest.mark.parametrize("order", [2, 3, 4, 7])
def test_constructors_build_valid_trees(constructor, order):
    _check_invariants(constructor(order), order)


def test_trivial_tree():
    tree = D.trivial(3)
    assert tree.nb_nodes == 4 and tree.arity == 3
    assert _dims(tree) == [[0, 1, 2], [0], [1], [2]]


def test_linear_tree():
    # {0,1,2,3} -> {0,1,2}, {3}; {0,1,2} -> {0,1}, {2}; {0,1} -> {0}, {1}
    tree = D.linear(4)
    assert _dims(tree) == [[0, 1, 2, 3], [0, 1, 2], [3], [0, 1], [2], [0], [1]]
    npt.assert_array_equal(tree.level, [0, 1, 1, 2, 2, 3, 3])


def test_balanced_tree():
    tree = D.balanced(4)
    assert _dims(tree) == [[0, 1, 2, 3], [0, 1], [2, 3], [0], [1], [2], [3]]
    npt.assert_array_equal(tree.level, [0, 1, 1, 2, 2, 2, 2])


@pytest.mark.parametrize("constructor", [D.linear, D.balanced])
def test_constructors_with_an_ordering_of_the_dimensions(constructor):
    # the ordering is given with dimensions numbered from 1
    tree = constructor([2, 1, 3, 4])
    reference = constructor(4)
    npt.assert_array_equal(tree.adjacency_matrix, reference.adjacency_matrix)
    for d in range(4):
        assert tree.dims[tree.dim2ind[d] - 1].tolist() == [d]
    npt.assert_array_equal(tree.dim2ind[[1, 0, 2, 3]], reference.dim2ind)


@pytest.mark.parametrize("arity", [2, 3, [2, 4]])
def test_random_tree_arity(arity):
    for _ in range(20):
        tree = D.random(6, arity)
        _check_invariants(tree, 6)
        assert tree.arity <= np.max(arity)


def test_random_tree_of_order_one():
    tree = D.random(1)
    assert tree.nb_nodes == 1 and tree.is_leaf[0]


def test_constructor_from_an_adjacency_matrix():
    A = np.zeros((5, 5), dtype=int)
    A[0, [1, 2]] = 1
    A[1, [3, 4]] = 1
    tree = D([4, 5, 3], A)  # dims 0, 1 below node 2, dim 2 on node 3
    _check_invariants(tree, 3)
    assert _dims(tree) == [[0, 1, 2], [0, 1], [2], [0], [1]]
    assert "DimensionTree" in repr(tree) and "nb_nodes = 5" in repr(tree)


# --- navigation -------------------------------------------------------------------


def test_ascendants_and_descendants():
    tree = D.balanced(4)
    leaf = tree.dim2ind[0]
    npt.assert_array_equal(tree.ascendants(leaf), [2, 1])
    assert tree.ascendants(tree.root).size == 0
    npt.assert_array_equal(np.sort(tree.descendants(1)), np.arange(2, 8))
    npt.assert_array_equal(tree.descendants(2), [4, 5])
    assert tree.descendants(leaf).size == 0


def test_nodes_with_level():
    tree = D.linear(4)
    npt.assert_array_equal(tree.nodes_with_level(0), [1])
    npt.assert_array_equal(tree.nodes_with_level(3), [6, 7])
    assert tree.nodes_with_level(4).size == 0


def test_node_with_dims():
    tree = D.balanced(4)
    npt.assert_array_equal(tree.node_with_dims([1, 0]), [2])
    npt.assert_array_equal(tree.node_with_dims([3]), [tree.dim2ind[3]])
    assert tree.node_with_dims([0, 2]).size == 0


# --- equality ------------------------------------------------------------------------


def test_equality():
    assert D.balanced(4) == D.balanced(4)
    assert D.linear(4) != D.balanced(4)
    assert D.linear(4) != D.linear(5)
    # the same tree, with other node numbers
    assert D.linear(4) == D.linear(4).change_root(1)[0]
    assert D.balanced(4) == D.balanced([2, 1, 3, 4]) == D.balanced(4).permute([1, 0, 2, 3])
    assert D.balanced(4) != D.balanced([1, 3, 2, 4])


def test_trees_with_different_numbers_of_nodes_per_level_are_different():
    # 6 nodes on 3 levels in both trees, but 3 then 2 nodes at levels 1 and 2,
    # against 2 then 3
    A = np.zeros((6, 6), dtype=int)
    A[0, [1, 2, 3]] = 1  # {0}, {1}, {2,3}
    A[3, [4, 5]] = 1  # {2}, {3}
    t1 = D([2, 3, 5, 6], A)
    B = np.zeros((6, 6), dtype=int)
    B[0, [1, 2]] = 1  # {0,1,2}, {3}
    B[1, [3, 4, 5]] = 1  # {0}, {1}, {2}
    t2 = D([4, 5, 6, 3], B)
    assert t1 != t2 and t2 != t1


# --- transformations ------------------------------------------------------------------


def test_permute_and_ipermute():
    tree = D.balanced(4)
    sigma = [1, 2, 3, 0]
    p = tree.permute(sigma)
    _check_invariants(p, 4)
    # dimension sigma[k] of p sits where dimension k of tree was
    npt.assert_array_equal(p.dim2ind[sigma], tree.dim2ind)
    npt.assert_array_equal(p.ipermute(sigma).dim2ind, tree.dim2ind)
    npt.assert_array_equal(tree.ipermute(sigma).permute(sigma).dim2ind, tree.dim2ind)


@pytest.mark.parametrize("constructor", [D.linear, D.balanced, D.trivial])
def test_change_root_at_an_internal_node(constructor):
    tree = constructor(4)
    nod = tree.children(tree.root)[0]
    if tree.is_leaf[nod - 1]:
        pytest.skip("the root of the trivial tree only has leaves")
    new, modified = tree.change_root(nod)
    _check_invariants(new, 4)
    assert new.root == nod and new.nb_nodes == tree.nb_nodes
    assert new.parent(tree.root) == nod
    npt.assert_array_equal(new.dim2ind, tree.dim2ind)
    assert {nod, tree.root} <= set(modified.tolist())


@pytest.mark.parametrize("constructor", [D.linear, D.balanced, D.trivial])
def test_change_root_at_a_leaf_adds_a_node(constructor):
    tree = constructor(4)
    leaf = tree.dim2ind[0]
    new, _ = tree.change_root(leaf)
    _check_invariants(new, 4)
    assert new.root == leaf and new.nb_nodes == tree.nb_nodes + 1
    assert new.dim2ind[0] == new.nb_nodes and new.parent(new.nb_nodes) == leaf


def test_change_root_at_the_root():
    tree = D.balanced(4)
    new, modified = tree.change_root(tree.root)
    assert new is tree and modified.size == 0


def test_add_child_to_an_internal_node_adds_a_dimension():
    tree = D.balanced(4)
    new = tree.add_child(2)
    _check_invariants(new, 5)
    assert new.dim2ind[4] == 8 and new.parent(8) == 2
    assert _dims(new)[1] == [0, 1, 4]


def test_add_child_to_a_leaf_moves_its_dimension():
    tree = D.balanced(4)
    leaf = tree.dim2ind[0]
    new = tree.add_child(leaf)
    _check_invariants(new, 4)
    assert new.dim2ind[0] == 8 and new.parent(8) == leaf
    assert not new.is_leaf[leaf - 1]


def test_sub_dimension_tree():
    tree = D.linear(4)
    sub, nodes = tree.sub_dimension_tree(2)  # dims {0, 1, 2}
    npt.assert_array_equal(nodes[0], 2)
    assert sorted(nodes.tolist()) == [2, 4, 5, 6, 7]
    _check_invariants(sub, 3)
    # node k of sub is node nodes[k-1] of tree, with the same dimensions
    for k in sub.nodes_indices:
        assert sorted(tree.dims[nodes[k - 1] - 1].tolist()) == sorted(
            sorted(tree.dims[1].tolist())[i] for i in sub.dims[k - 1]
        )


def test_update_dims_from_leaves():
    tree = D.balanced(4)
    tree.dim2ind = tree.dim2ind[[3, 2, 1, 0]]
    tree.update_dims_from_leaves()
    assert _dims(tree) == [[0, 1, 2, 3], [2, 3], [0, 1], [3], [2], [1], [0]]


# --- layout and plots -------------------------------------------------------------------


@pytest.mark.parametrize("alignment", [False, True])
def test_tree_layout(alignment):
    tree = D.linear(4)
    tree.plot_options["level_alignment"] = alignment
    pos = tree.tree_layout()
    assert sorted(pos) == list(range(tree.nb_nodes))
    xy = np.array(list(pos.values()))
    assert np.all((xy > 0) & (xy < 1))
    for nod in tree.nodes_indices:
        children = tree.children(nod)
        if children.size:
            # a parent is above its children, between the first and the last
            assert all(pos[nod - 1][1] > pos[c - 1][1] for c in children)
            x = [pos[c - 1][0] for c in children]
            assert min(x) <= pos[nod - 1][0] <= max(x)
    if alignment:
        for level in range(4):
            y = [pos[n - 1][1] for n in tree.nodes_with_level(level)]
            assert np.allclose(y, y[0])


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


@pytest.mark.parametrize(
    "call",
    [
        lambda t, f: t.plot(fname=f),
        lambda t, f: t.plot(colored_nodes=np.array([2, 3]), fname=f),
        lambda t, f: t.plot(colored_nodes=[2, 3], title="tree", fname=f),
        lambda t, f: t.plot_dims(fname=f),
        lambda t, f: t.plot_dims(nodes=t.dim2ind[[1, 0, 2, 3]], fname=f),
    ],
)
def test_plots_are_saved(plt, tmp_path, call):
    fname = tmp_path / "tree.png"
    call(D.balanced(4), str(fname))
    assert fname.stat().st_size > 0


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_without_a_file_shows_the_figure(plt):
    D.balanced(4).plot()
    assert plt.get_fignums()
