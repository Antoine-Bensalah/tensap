# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module laplacian_like_operator_factory.

Port of ApproximationToolbox/TensorAlgebra/Tools/LaplacianLikeOperatorFactory.m
(canonical, Tucker, tree-based and TT constructions).
"""

import numpy as np
import scipy.sparse as sp
import tensap


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    """Return a dense ndarray copy of matrix, densifying it if sparse."""
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


class LaplacianLikeOperatorFactory:
    """
    Factory building the order-d Laplacian-like operator
    sum_{mu=1}^{d} I x ... x K x ... x I (K applied on dimension mu, I elsewhere)
    in various tensor formats.

    Attributes
    ----------
    K : ndarray or scipy.sparse matrix
        1D stiffness (Laplacian-like) matrix.
    I : ndarray or scipy.sparse matrix
        1D identity matrix, same shape as K.
    order : int
        Number of dimensions d of the operator.
    """

    def __init__(self, K, I, d: int):  # noqa: E741 (notation of the operator)
        self.K = K
        self.I = I  # noqa: E741
        self.order = d

    def make_canonical_tensor(self) -> "tensap.TuckerTensorOperator":
        """Build the operator in canonical format (representation rank d per dimension).

        Dimension mu has a rank-d FrameOperator whose k-th element is K if
        k == mu and I otherwise. The core is a DiagonalTensor of ones, so that
        the operator is sum_{k} A^{(1)}_k x ... x A^{(d)}_k.

        Returns
        -------
        TuckerTensorOperator
        """
        d = self.order
        K = _to_dense(self.K)
        Id = _to_dense(self.I)
        frames = [
            tensap.FrameOperator(np.stack([K if k == mu else Id for k in range(d)]))
            for mu in range(d)
        ]
        tensor_frame = tensap.TFrameOperator(frames)
        core = tensap.DiagonalTensor(np.ones(d), d)

        return tensap.TuckerTensorOperator(core, tensor_frame)

    def make_tucker_tensor(self) -> "tensap.TuckerTensorOperator":
        """Build the operator in Tucker format (representation rank 2 per dimension).

        Each dimension has a rank-2 FrameOperator [K, I]. The core is a
        FullTensor of shape (2, ..., 2) with a 1 at every multi-index where
        exactly one dimension selects K (index 0) and all others select I
        (index 1).

        Returns
        -------
        TuckerTensorOperator
        """
        d = self.order
        tensor_frame = self.make_tensor_frame()

        core_data = np.zeros((2,) * d)
        for mu in range(d):
            index = [1] * d
            index[mu] = 0
            core_data[tuple(index)] = 1.0
        core = tensap.FullTensor(core_data)

        return tensap.TuckerTensorOperator(core, tensor_frame)

    def make_tree_based_tensor(self, tree=None) -> "tensap.TuckerTensorOperator":
        """Build the operator in Tucker format with a tree-based core.

        The frames are [K, I] in every dimension (make_tensor_frame) and the
        core is a TreeBasedTensor with inactive leaves. The index of a node
        tells whether K appears in its subtree (index 0, exactly once) or not
        (index 1): an internal node has the entries (0, 1, 0), (1, 0, 0) and
        (1, 1, 1) (children, then parent), the root the entries (0, 1) and
        (1, 0). All ranks are 2.

        Parameters
        ----------
        tree : DimensionTree, optional
            Binary dimension tree. The default is the balanced tree, as in
            MATLAB.

        Returns
        -------
        TuckerTensorOperator
        """
        if tree is None:
            tree = tensap.DimensionTree.balanced(self.order)
        tensors = []
        for nod in range(1, tree.nb_nodes + 1):
            if tree.is_leaf[nod - 1]:
                tensors.append(tensap.FullTensor([]))
                continue
            if np.count_nonzero(tree.children(nod)) != 2:
                raise ValueError("make_tree_based_tensor requires a binary tree.")
            if nod == tree.root:
                data = np.zeros((2, 2))
                data[0, 1] = data[1, 0] = 1.0
            else:
                data = np.zeros((2, 2, 2))
                data[0, 1, 0] = data[1, 0, 0] = data[1, 1, 1] = 1.0
            tensors.append(tensap.FullTensor(data))
        core = tensap.TreeBasedTensor(tensors, tree)
        return tensap.TuckerTensorOperator(core, self.make_tensor_frame())

    def make_tt_tucker_tensor(self) -> "tensap.TuckerTensorOperator":
        """Build the operator in Tucker format with a TT core (MATLAB
        makeTTTuckerLikeTensor).

        Frames [K, I] in every dimension and a TTTensor core of ranks 2 over
        the indices (0 = K, 1 = I) of the frames. The rank index of a bond
        tells whether K has been placed on its left (0: placed, 1: not yet).
        Entries (left rank, frame index, right rank): first core (0, 0, 0)
        and (0, 1, 1); middle cores (0, 1, 0), (1, 0, 0) and (1, 1, 1); last
        core (0, 1, 0) and (1, 0, 0). K is placed exactly once.

        Returns
        -------
        TuckerTensorOperator
        """
        d = self.order
        if d == 1:
            core = tensap.TTTensor([np.array([1.0, 0.0]).reshape(1, 2, 1)])
            return tensap.TuckerTensorOperator(core, self.make_tensor_frame())
        first = np.zeros((1, 2, 2))
        first[0, 0, 0] = first[0, 1, 1] = 1.0
        middle = np.zeros((2, 2, 2))
        middle[0, 1, 0] = middle[1, 0, 0] = middle[1, 1, 1] = 1.0
        last = np.zeros((2, 2, 1))
        last[0, 1, 0] = last[1, 0, 0] = 1.0
        core = tensap.TTTensor([first] + [middle] * (d - 2) + [last])
        return tensap.TuckerTensorOperator(core, self.make_tensor_frame())

    def make_tt_operator(self) -> "tensap.TTOperator":
        """Build the operator as a TTOperator of ranks 2 (MATLAB
        makeOperatorTTTensor): first core [K, I], middle cores
        [[I, 0], [K, I]], last core [I; K].

        Returns
        -------
        TTOperator
        """
        d = self.order
        K, Id = _to_dense(self.K), _to_dense(self.I)
        n = K.shape[0]
        if d == 1:
            return tensap.TTOperator([K[None, :, :, None]])
        first = np.zeros((1, n, n, 2))
        first[0, :, :, 0], first[0, :, :, 1] = K, Id
        middle = np.zeros((2, n, n, 2))
        middle[0, :, :, 0], middle[1, :, :, 0], middle[1, :, :, 1] = Id, K, Id
        last = np.zeros((2, n, n, 1))
        last[0, :, :, 0], last[1, :, :, 0] = Id, K
        return tensap.TTOperator([first] + [middle] * (d - 2) + [last])

    def make_tensor_frame(self) -> "tensap.TFrameOperator":
        """Frames [K, I] in every dimension (MATLAB makeTSpaceForTuckerLikeTensor).

        Returns
        -------
        TFrameOperator
        """
        stacked = np.stack([_to_dense(self.K), _to_dense(self.I)], axis=0)
        return tensap.TFrameOperator(
            [tensap.FrameOperator(stacked.copy()) for _ in range(self.order)]
        )
