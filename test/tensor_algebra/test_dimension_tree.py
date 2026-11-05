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


# --- navigation -------------------------------------------------------------------


# --- equality ------------------------------------------------------------------------


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


# --- layout and plots -------------------------------------------------------------------


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
