# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# This file is part of tensap (tensor approximation package).

# tensap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tensap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with tensap.  If not, see <https://www.gnu.org/licenses/>.

"""
Module truncator.

"""

import warnings
import numpy as np
import tensap


class Truncator:
    """
    Class Truncator.

    Attributes
    ----------
    tolerance : float
        The relative tolerance for the truncation.
    max_rank : int
        The maximum rank for the truncation.
    thresholding_type : str
        The thresholding type ('hard' or 'soft').
    thresholding_parameter : float
        The thresholding parameter.
    greedy_updates : bool
        Truncation of a Tucker tensor with a diagonal core (canonical format):
        if True, each greedy correction is followed by an ALS update of all
        the rank-one factors; if False, pure greedy algorithm (MATLAB).

    """

    def __init__(
        self,
        tolerance=1e-8,
        max_rank=np.inf,
        thresholding_type="hard",
        thresholding_parameter=None,
        greedy_updates=True,
    ):
        """
        Constructor for the class Truncator.

        Parameters
        ----------
        tolerance : float, optional
            The relative tolerance for the truncation. The default is 1e-8.
        max_rank : int, optional
            The maximum rank for the truncation. The default is numpy.inf.
        thresholding_type : str, optional
            The thresholding type ('hard' or 'soft'). The default is 'hard'.
        thresholding_parameter : float, optional
            The thresholding parameter. The default is None.
        greedy_updates : bool, optional
            Update of the factors in the greedy recompression of canonical
            tensors. The default is True (False: pure greedy, as in MATLAB).

        Returns
        -------
        None.

        """
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.thresholding_type = thresholding_type
        self.thresholding_parameter = thresholding_parameter
        self.greedy_updates = greedy_updates
        self._hsvd_type = 2  # 1 for root to leaves, 2 for leaves to root

    def __repr__(self):
        return (
            "<{}:{n}"
            + "{t}tolerance = {},{n}"
            + "{t}max_rank = {},{n}"
            + "{t}thresholding_type = {},{n}"
            + "{t}thresholding_parameter = {},{n}"
            + "{t}greedy_updates = {},{n}"
            + "{t}_hsvd_type = {}>"
        ).format(
            self.__class__.__name__,
            self.tolerance,
            self.max_rank,
            self.thresholding_type,
            self.thresholding_parameter,
            self.greedy_updates,
            self._hsvd_type,
            t="\t",
            n="\n",
        )

    def truncate(self, tensor):
        """
        Compute the truncation of the tensor with relative precision
        self.tolerance and maximal rank self.max_rank.

        Parameters
        ----------
        tensor : numpy.ndarray or tensap.FullTensor or tensorflow.Tensor or
        tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        NotImplementedError
            If the decomposition is not implemented for the tensor format.
        ValueError
            If the tensor is of order 1.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        """

        if isinstance(tensor, tensap.BaseTuckerTensor):
            return self._truncate_tucker(tensor)
        if isinstance(tensor, tensap.BaseTTTensor):
            return self._truncate_tt(tensor)

        if not hasattr(tensor, "order"):
            tensor = tensap.FullTensor(tensor)

        if tensor.order == 2:
            out = self.svd(tensor)
        elif tensor.order > 2:
            if isinstance(tensor, tensap.FullTensor):
                out = self.hosvd(tensor)
            elif isinstance(tensor, tensap.TreeBasedTensor):
                out = self.hsvd(tensor)
            else:
                raise NotImplementedError("Not implemented with this tensor format.")
        else:
            raise ValueError("Wrong tensor order.")
        return out

    def _truncate_tucker(self, tensor):
        """
        Truncate a BaseTuckerTensor (TuckerTensorVector or TuckerTensorOperator).

        As Truncator.truncate for a TuckerLikeTensor in the MATLAB
        ApproximationToolbox: an operator is vectorized, then
        - order 2: truncated SVD of the core of the orthonormalized tensor,
          the result has a diagonal core (the singular values);
        - FullTensor core: HOSVD of the core of the orthonormalized tensor;
        - DiagonalTensor core: recompression by a greedy algorithm with
          rank-one corrections and updates of the factors;
        - TreeBasedTensor core: the frames are absorbed into the leaves and
          the tree-based tensor is truncated by hsvd;
        - TTTensor core: TT rounding of the core, then truncated SVD of each
          frame (see _truncate_tucker_tt).

        Parameters
        ----------
        tensor : tensap.BaseTuckerTensor
            The Tucker tensor to truncate.

        Raises
        ------
        NotImplementedError
            For another type of core.

        Returns
        -------
        tensap.BaseTuckerTensor
            The truncated tensor, of the same subclass as tensor.

        """
        if isinstance(tensor, tensap.TuckerTensorOperator):
            shape = tensor.shape
            vector = self._truncate_tucker(tensor.vectorize())
            return vector.unvectorize([s[0] for s in shape], [s[1] for s in shape])

        if tensor.order == 2:
            return self._truncate_tucker_svd(tensor)
        if isinstance(tensor.core, tensap.FullTensor):
            return self._truncate_tucker_hosvd(tensor)
        if isinstance(tensor.core, tensap.DiagonalTensor):
            return self._truncate_tucker_greedy(tensor)
        if isinstance(tensor.core, tensap.TreeBasedTensor):
            return tensap.TuckerTensorVector.from_tree_based_tensor(
                self.hsvd(tensor.tree_based_tensor())
            )
        if isinstance(tensor.core, tensap.TTTensor):
            return self._truncate_tucker_tt(tensor)
        raise NotImplementedError(
            "Truncation of a BaseTuckerTensor is not implemented for a "
            f"{type(tensor.core).__name__} core."
        )

    def _truncate_tt(self, tensor, local_tolerance=None):
        """
        TT rounding (MATLAB Truncator.ttsvd on a TTTensor): orthogonalization
        with the center on the first core, then a sweep from the left with a
        truncated SVD of each core, of relative precision
        tolerance / sqrt(d - 1). An operator is rounded as its vectorization.

        Parameters
        ----------
        tensor : tensap.TTTensor or tensap.TTOperator
        local_tolerance : float, optional
            Precision of each SVD. The default is tolerance / sqrt(d - 1).

        Returns
        -------
        Same type as tensor, with its orthogonality center on the last core.
        """
        if isinstance(tensor, tensap.TTOperator):
            vector = self._truncate_tt(tensor.vectorize(), local_tolerance)
            return tensap.TTOperator.unvectorize(vector, tensor.shape)
        d = tensor.order
        if d == 1 or tensor.norm() == 0:
            return tensor.orth(d - 1)
        if local_tolerance is None:
            local_tolerance = self.tolerance / np.sqrt(d - 1)
        max_rank = np.atleast_1d(self.max_rank)
        if max_rank.size == 1:
            max_rank = np.repeat(max_rank, d - 1)
        saved_max_rank = self.max_rank
        cores = list(tensor.orth(0).cores)
        try:
            for k in range(d - 1):
                r0, n, r1 = cores[k].shape
                self.max_rank = max_rank[k]
                svd = self.trunc_svd(cores[k].reshape(r0 * n, r1), local_tolerance)
                left, right = svd.space
                cores[k] = left.reshape(r0, n, -1)
                weights = svd.core.data[:, None] * right.T
                cores[k + 1] = np.tensordot(weights, cores[k + 1], axes=(1, 0))
        finally:
            self.max_rank = saved_max_rank
        return tensap.TTTensor(cores, orth_center=d - 1)

    def _truncate_tucker_tt(self, tensor):
        """
        Truncation of a Tucker tensor with a TTTensor core (MATLAB ttsvd on a
        TuckerLikeTensor): orthonormalized frames, TT rounding of the core,
        then for each dimension k, with the center of the core on k, a
        truncated SVD of the matricization of the core along its physical
        axis, whose left factor rotates the frame. The 2d - 1 SVDs have the
        precision tolerance / sqrt(2d - 1).
        """
        d = tensor.order
        local_tolerance = self.tolerance / np.sqrt(2 * d - 1)
        x = tensor.orth()
        core = self._truncate_tt(tensap.TTTensor(x.core.cores), local_tolerance)
        frames = list(x.tensor_frame.frames)
        max_rank = np.atleast_1d(self.max_rank)
        saved_max_rank = self.max_rank
        self.max_rank = np.min(max_rank)
        try:
            for k in range(d):
                core = core.orth(k)
                cores = list(core.cores)
                r0, n, r1 = cores[k].shape
                matrix = cores[k].transpose(1, 0, 2).reshape(n, r0 * r1)
                svd = self.trunc_svd(matrix, local_tolerance)
                left, right = svd.space
                frames[k] = frames[k].space_times_matrix(left)
                rest = svd.core.data[:, None] * right.T
                cores[k] = rest.reshape(-1, r0, r1).transpose(1, 0, 2)
                core = tensap.TTTensor(cores, orth_center=k)
        finally:
            self.max_rank = saved_max_rank
        return type(x)(core, x.tensor_frame.__class__(frames))

    def _truncate_tucker_svd(self, tensor):
        """Truncated SVD of an order-2 Tucker tensor, with a diagonal core."""
        x = tensor.orth()
        saved_max_rank = self.max_rank
        self.max_rank = np.min(self.max_rank)
        try:
            svd = self.trunc_svd(x.core.full().numpy())
        finally:
            self.max_rank = saved_max_rank
        new_frames = x.tensor_frame.space_times_matrix(svd.space)
        return type(x)(tensap.DiagonalTensor(svd.core.data, 2), new_frames)

    def _truncate_tucker_greedy(self, tensor):
        """
        Recompression of a Tucker tensor with a diagonal core (canonical
        format) by the greedy algorithm applied to I y = x: rank-one
        corrections computed by ALS, each followed by an update of all the
        rank-one factors by ALS if self.greedy_updates.

        Differences with MATLAB: by default the factors are updated after each
        correction (MATLAB uses the pure greedy algorithm, greedy_updates=False,
        which converges much more slowly), and the number of corrections is bounded by the rank of x
        (instead of the size of the first dimension), since x itself is an
        exact representation of that rank.
        """
        x_norm = tensor.norm()
        if x_norm == 0:
            return tensor
        identity = tensap.BaseTuckerTensor.eye([s[0] for s in tensor.shape])
        local_solver = tensap.RankOneALSLinearSolver(
            identity, tensor, max_iterations=30, stagnation=1e-5, display=False
        )
        update = None
        if self.greedy_updates:
            update = tensap.TuckerTensorALSLinearSolver(
                identity, tensor, max_iterations_frames=2, display=False
            ).update_frames_by_als
        max_iterations = int(np.min([tensor.core.data.size, np.min(self.max_rank)]))
        solver = tensap.GreedyLinearSolver(
            identity,
            tensor,
            local_solver=local_solver,
            update=update,
            max_iterations=max_iterations,
            tolerance=self.tolerance,
            display=False,
        )
        out = solver.solve()[0]
        if (out - tensor).norm() / x_norm > self.tolerance:
            warnings.warn("Truncator: the approximation is not accurate enough.")
        return out

    def _truncate_tucker_hosvd(self, tensor):
        """
        HOSVD of a Tucker tensor with a FullTensor core: orthonormalize the
        frames, run HOSVD on the (small) core, and push the resulting factor
        matrices back into the frames.
        """
        x = tensor.orth()
        core = x.core
        order = core.order

        max_rank = np.atleast_1d(self.max_rank)
        if max_rank.size == 1:
            max_rank = np.repeat(max_rank, order)
        local_tol = self.tolerance / np.sqrt(order)

        saved_max_rank = self.max_rank
        factors = []
        for dim in range(order):
            self.max_rank = max_rank[dim]
            factors.append(
                self.trunc_svd(core.matricize(dim).numpy(), local_tol).space[0]
            )
        self.max_rank = saved_max_rank

        new_core = core.tensor_matrix_product(factors[0].T, 0)
        for dim in range(1, order):
            new_core = new_core.tensor_matrix_product(factors[dim].T, dim)

        new_frames = [
            frame.space_times_matrix(factor)
            for frame, factor in zip(x.tensor_frame.frames, factors)
        ]
        new_tensor_frame = x.tensor_frame.__class__(new_frames)

        return type(x)(new_core, new_tensor_frame)

    def trunc_svd(self, matrix, tolerance=None, power=2):
        """
        Compute the truncated svd of the matrix x with relative precision
        self.tolerance (or tolerance if provided) in Schatten p-norm (with p
        given by the input power) and maximal rank self.max_rank.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to truncate.
        tolerance : float, optional
            The relative tolerance for the truncation. The default is
            self.tolerance.
        power : int or 'inf' or float('inf') or numpy.inf, optional
            The integer p of the Schatten-p norm (1 <= p <= inf, p = 2 for
            Frobenius). The default is 2.

        Returns
        -------
        tensap.CanonicalTensor
            The truncated matrix.

        """
        if tolerance is None:
            tolerance = self.tolerance

        left, sin_val, right = np.linalg.svd(matrix, full_matrices=False)
        if power in ("inf", float("inf"), np.inf):
            error = sin_val / np.max(sin_val)
        else:
            error = np.power(
                np.flip(np.cumsum(np.flip(np.power(sin_val, power))))
                / np.sum(np.power(sin_val, power)),
                1 / power,
            )
        error = np.concatenate((np.atleast_1d(error[1:]), [0]))
        ind = np.nonzero(error < tolerance)[0]
        if ind.size == 0:
            ind = np.min(matrix.shape) + 1
        else:
            ind = np.min(ind) + 1

        ind = int(np.min([ind, self.max_rank]))
        if self.thresholding_parameter is not None and self.thresholding_parameter != 0:
            if self.thresholding_type == "soft":
                sin_val -= self.thresholding_parameter
                ind = int(np.min([ind, np.nonzero(sin_val >= 0)[0][-1] + 1]))
            elif self.thresholding_type == "hard":
                ind = int(
                    np.min(
                        [
                            ind,
                            np.nonzero(sin_val >= self.thresholding_parameter)[0][-1]
                            + 1,
                        ]
                    )
                )
        left = np.atleast_2d(left[:, :ind])
        sin_val = np.atleast_1d(sin_val[:ind])
        right = np.transpose(np.atleast_2d(right[:ind, :]))

        return tensap.CanonicalTensor([left, right], sin_val)

    def svd(self, tensor):
        """
        Compute the truncated svd of an order-2 tensor.

        Parameters
        ----------
        tensor : numpy.ndarray or tensap.FullTensor or tensorflow.Tensor or
        tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        """
        assert tensor.ndim == 2, "Wrong order."

        if isinstance(tensor, np.ndarray):
            out = self.trunc_svd(tensor)
        elif isinstance(tensor, tensap.FullTensor):
            out = self.trunc_svd(tensor.data)
        elif isinstance(tensor, tensap.TreeBasedTensor):
            out = self.hsvd(tensor)
        elif isinstance(tensor, tensap.CanonicalTensor):
            tensor = tensor.orth()
            out = self.trunc_svd(tensor.core.data)
            out.space[0] = np.matmul(tensor.space[0], out.space[0])
            out.space[1] = np.matmul(tensor.space[1], out.space[1])
            out.shape = tensor.shape
        else:
            raise NotImplementedError("Method not implemented.")
        return out

    def hosvd(self, tensor):
        """
        Compute the truncated hosvd of tensor.

        Parameters
        ----------
        tensor : np.ndarray or tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        ValueError
            If the input tensor is of the wrong type.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        """
        if isinstance(tensor, np.ndarray):
            tensor = tensap.FullTensor(tensor)

        # self.max_rank is set node by node for trunc_svd: restore it after
        max_rank = self.max_rank
        try:
            return self._hosvd(tensor)
        finally:
            self.max_rank = max_rank

    def _hosvd(self, tensor):
        order = tensor.order
        if order == 2:
            out = self.svd(tensor)
        else:
            max_rank = np.atleast_1d(self.max_rank)
            if max_rank.size == 1:
                max_rank = np.repeat(max_rank, order)
            local_tol = self.tolerance / np.sqrt(order)
            if isinstance(tensor, tensap.FullTensor):
                vec = np.empty(order, dtype=object)
                for dim in range(order):
                    self.max_rank = max_rank[dim]
                    vec[dim] = self.trunc_svd(tensor.matricize(dim).numpy(), local_tol)
                    vec[dim] = vec[dim].space[0]
                core = tensor.tensor_matrix_product(np.transpose(vec[0]), 0)
                for dim in np.arange(1, order):
                    core = core.tensor_matrix_product(np.transpose(vec[dim]), dim)
                tensors = [core] + [tensap.FullTensor(x) for x in vec]
                tree = tensap.DimensionTree.trivial(order)
                out = tensap.TreeBasedTensor(tensors, tree)
            else:
                raise ValueError("Wrong type.")
        return out

    def hsvd(self, tensor, tree=None, is_active_node=None):
        """
        Compute the truncated svd in tree-based tensor format of tensor.

        Parameters
        ----------
        tensor : tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.
        tree : tensap.DimensionTree, optional
            The tree of the output tree-based tensor. The default is None,
            indicating if tensor is a tensap.TreeBasedTensor to take
            tensor.tree.
        is_active_node : numpy.ndarray, optional
            Logical array indicating if the nodes are active.. The default is
            None, indicating if tensor is a tensap.TreeBasedTensor to take
            tensor.is_active_node.

        Raises
        ------
        ValueError
            If the wrong value of the atttribude _hsvd_type is provided.
        NotImplementedError
            If the method is not implemented for the format.

        Returns
        -------
        out : tensap.TreeBasedTensor
            The truncated tensor in tree-based tensor format.

        """
        # self.max_rank is set node by node for trunc_svd: restore it after
        max_rank = self.max_rank
        try:
            return self._hsvd(tensor, tree, is_active_node)
        finally:
            self.max_rank = max_rank

    def _hsvd(self, tensor, tree, is_active_node):
        if isinstance(tensor, tensap.TreeBasedTensor):
            if tree is not None or is_active_node is not None:
                warnings.warn(
                    "The provided tree and/or is_active_node "
                    "are not taken into account when x is a "
                    "tensap.TreeBasedTensor."
                )
            is_active_node = tensor.is_active_node
            tree = tensor.tree
        elif is_active_node is None:
            is_active_node = np.full(tree.nb_nodes, True)

        max_rank = np.atleast_1d(self.max_rank)
        if max_rank.size == 1:
            max_rank = np.repeat(max_rank, tree.nb_nodes)
            max_rank[tree.root - 1] = 1

        local_tol = self.tolerance / np.sqrt(np.count_nonzero(is_active_node) - 1)

        if isinstance(tensor, tensap.FullTensor):
            root_rank_greater_than_one = tensor.order == len(tree.dim2ind) + 1

            tensors = np.empty(tree.nb_nodes, dtype=object)
            shape = np.array(tensor.shape)
            nodes_x = tree.dim2ind
            ranks = np.ones(tree.nb_nodes, dtype=int)

            for level in np.arange(np.max(tree.level), 0, -1):
                for nod in tree.nodes_with_level(level):
                    if is_active_node[nod - 1]:
                        if tree.is_leaf[nod - 1]:
                            rep = np.nonzero(nod == nodes_x)[0][0]
                        else:
                            children = tree.children(nod)
                            rep = [
                                np.nonzero(np.isin(nodes_x, x))[0][0] for x in children
                            ]
                        rep_c = tensap.fast_setdiff(np.arange(nodes_x.size), rep)

                        if root_rank_greater_than_one:
                            rep_c = np.concatenate((rep_c, [tensor.order - 1]))

                        self.max_rank = max_rank[nod - 1]
                        tmp = self.trunc_svd(tensor.matricize(rep).numpy(), local_tol)
                        tensors[nod - 1] = tmp.space[0]
                        ranks[nod - 1] = tensors[nod - 1].shape[1]
                        shape_loc = np.hstack((shape[rep], ranks[nod - 1]))
                        tensors[nod - 1] = tensap.FullTensor(
                            tensors[nod - 1], shape=shape_loc
                        )
                        tmp = np.matmul(tmp.space[1], np.diag(tmp.core.data))
                        shape = np.hstack((shape[rep_c], ranks[nod - 1]))
                        tensor = tensap.FullTensor(tmp, shape=shape)

                        if root_rank_greater_than_one:
                            perm = np.concatenate(
                                (
                                    np.arange(tensor.order - 2),
                                    [tensor.order - 1],
                                    [tensor.order - 2],
                                )
                            )
                            tensor = tensor.transpose(perm)
                            shape = shape[perm]
                            rep_c = rep_c[:-1]

                        nodes_x = np.hstack((nodes_x[rep_c], nod))
                    else:
                        tensors[nod - 1] = []

            root_ch = tree.children(tree.root)
            rep = [np.nonzero(np.isin(nodes_x, x))[0][0] for x in root_ch]
            if root_rank_greater_than_one:
                rep = np.concatenate((rep, [tensor.order - 1]))
            tensors[tree.root - 1] = tensor.transpose(rep)
            out = tensap.TreeBasedTensor(tensors, tree)
        elif isinstance(tensor, tensap.TreeBasedTensor):
            if self._hsvd_type == 1:
                out = tensor.orth()
                gram = out.gramians()[0]
                mat = np.empty(gram.shape, dtype=object)
                shape = np.zeros(gram.shape)
                for nod in range(gram.size):
                    # Truncation of the Gramian in trace norm for a control
                    # of Frobenius norm of the tensor
                    if gram[nod] is not None:
                        self.max_rank = max_rank[nod]
                        tmp = self.trunc_svd(gram[nod], tolerance=local_tol**2, power=1)
                        shape[nod] = tmp.core.shape[0]
                        mat[nod] = np.transpose(tmp.space[0])

                # Interior nodes without the root
                for level in np.arange(1, np.max(tree.level)):
                    nod_level = tensap.fast_setdiff(
                        tree.nodes_with_level(level), np.nonzero(tree.is_leaf)[0] + 1
                    )
                    for nod in tree.nodes_indices[nod_level - 1]:
                        order = out.tensors[nod - 1].order
                        out.tensors[nod - 1] = out.tensors[
                            nod - 1
                        ].tensor_matrix_product(mat[nod - 1], order - 1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent - 1] = out.tensors[
                            parent - 1
                        ].tensor_matrix_product(mat[nod - 1], ch_nb - 1)

                # Leaves
                for nod in tree.dim2ind:
                    if out.is_active_node[nod - 1]:
                        order = out.tensors[nod - 1].order
                        out.tensors[nod - 1] = out.tensors[
                            nod - 1
                        ].tensor_matrix_product(mat[nod - 1], order - 1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent - 1] = out.tensors[
                            parent - 1
                        ].tensor_matrix_product(mat[nod - 1], ch_nb - 1)
                # Update the shape
                out = out.update_attributes()
                out.is_orth = False

            elif self._hsvd_type == 2:
                out = tensor.orth()
                gram = out.gramians()[0]
                for level in np.arange(np.max(tree.level), 0, -1):
                    for nod in tensap.fast_intersect(
                        tree.nodes_with_level(level), out.active_nodes
                    ):
                        # Truncation of the Gramian in trace norm for a control
                        # of Frobenius norm of the tensor
                        self.max_rank = max_rank[nod - 1]
                        tmp = self.trunc_svd(gram[nod - 1], local_tol**2, power=1)
                        tmp = np.transpose(tmp.space[0])
                        order = out.tensors[nod - 1].order
                        out.tensors[nod - 1] = out.tensors[
                            nod - 1
                        ].tensor_matrix_product(tmp, order - 1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent - 1] = out.tensors[
                            parent - 1
                        ].tensor_matrix_product(tmp, ch_nb - 1)
                out = out.update_attributes()
                out.is_orth = True
                out.orth_node = tree.root
            else:
                raise ValueError("Wrong value of _hsvd_type.")
        else:
            raise NotImplementedError("Method not implemented.")
        return out

    def ttsvd(self, tensor):
        """
        Compute the truncated svd in tensor-train format of tensor.

        Parameters
        ----------
        tensor : tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.

        Returns
        -------
        tensap.TreeBasedTensor
            The truncated tensor in tree-based tensor format with a linear
            tree.

        """
        tree = tensap.DimensionTree.linear(tensor.order)
        is_active_node = np.full(tree.nb_nodes, True)
        is_active_node[tree.dim2ind[1:] - 1] = False

        return self.hsvd(tensor, tree, is_active_node)
