# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module laplacian_like_operator_factory.

Port of ApproximationToolbox/TensorAlgebra/Tools/LaplacianLikeOperatorFactory.m
(currently limited to the Tucker-format construction).
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

    def __init__(self, K, I, d: int):
        self.K = K
        self.I = I
        self.order = d

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
        stacked = np.stack([_to_dense(self.K), _to_dense(self.I)], axis=0)
        frames = [tensap.FrameOperator(stacked.copy()) for _ in range(d)]
        tensor_frame = tensap.TFrameOperator(frames)

        core_data = np.zeros((2,) * d)
        for mu in range(d):
            index = [1] * d
            index[mu] = 0
            core_data[tuple(index)] = 1.0
        core = tensap.FullTensor(core_data)

        return tensap.TuckerTensorOperator(core, tensor_frame)
