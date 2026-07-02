"""Tests for BaseTuckerTensor, TuckerTensorVector, TuckerTensorOperator (v2)."""

import numpy as np
import pytest
import tensap
from tensap.tensor_algebra.tensors.frame import Frame, FrameVector, FrameOperator
from tensap.tensor_algebra.tensors.tensor_frame import TFrame, TFrameVector, TFrameOperator
from tensap.tensor_algebra.tensors.full_tensor import FullTensor
from tensap.tensor_algebra.tensors.tree_based_tensor import TreeBasedTensor
from tensap.tensor_algebra.tensors.tucker_like_tensor import (
    BaseTuckerTensor,
    TuckerTensorVector,
    TuckerTensorOperator,
)


def _vector_shape(shape):
    """Convert a list of physical dimensions to the TFrameVector shape convention."""
    return tuple((d, 1) for d in shape)


def _operator_shape(dims1, dims2):
    """Convert two lists of dimensions to the TFrameOperator shape convention."""
    return tuple(zip(dims1, dims2))


class TestConstruction:
    @staticmethod
    def test_construction_from_frames():
        rng = np.random.RandomState(0)
        ranks = [3, 4]
        shape = [5, 6]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        assert t.order == 2
        assert t.shape == _vector_shape(shape)
        assert t.ranks == tuple(ranks)

    @staticmethod
    def test_construction_from_frames_operator():
        rng = np.random.RandomState(0)
        ranks = [2, 3]
        dims1 = [5, 7]
        dims2 = [3, 4]
        frames = TFrameOperator([
            FrameOperator(rng.rand(r, n1, n2))
            for r, n1, n2 in zip(ranks, dims1, dims2)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorOperator(core, frames)
        assert t.order == 2
        assert t.shape == _operator_shape(dims1, dims2)
        assert t.ranks == tuple(ranks)

    @staticmethod
    def test_from_tree_based_tensor():
        rng = np.random.RandomState(0)
        t1 = BaseTuckerTensor.randn([2, 3], [5, 6])
        tbt = t1.tree_based_tensor()
        t2 = BaseTuckerTensor.from_tree_based_tensor(tbt)
        assert isinstance(t2, TuckerTensorVector)
        assert t2.order == 2
        assert np.allclose(t2.full().numpy(), t1.numpy())


class TestConversionDense:
    @staticmethod
    def _make_vector(order=2, seed=0):
        rng = np.random.RandomState(seed)
        ranks = rng.randint(2, 5, size=order)
        shape = rng.randint(5, 10, size=order)
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorVector(core, frames)

    @staticmethod
    def _make_operator(order=2, seed=0):
        rng = np.random.RandomState(seed)
        ranks = rng.randint(2, 5, size=order)
        dims1 = rng.randint(5, 10, size=order)
        dims2 = rng.randint(3, 8, size=order)
        frames = TFrameOperator([
            FrameOperator(rng.rand(r, n1, n2))
            for r, n1, n2 in zip(ranks, dims1, dims2)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorOperator(core, frames)

    def test_full_vector(self):
        t = self._make_vector(2)
        mats = t.tensor_frame.get_vectorize_factors()
        expected = mats[0] @ t.core.data @ mats[1].T
        assert np.allclose(t.full().data, expected)

    def test_full_operator(self):
        t = self._make_operator(2)
        full = t.full()
        assert np.allclose(full.numpy(), t.numpy())

    def test_numpy(self):
        t = self._make_vector(3)
        assert np.allclose(t.numpy(), t.full().numpy())


class TestAlgebra:
    @staticmethod
    def _make_vector(order=2, seed=0):
        rng = np.random.RandomState(seed)
        ranks = rng.randint(2, 5, size=order)
        shape = rng.randint(5, 10, size=order)
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorVector(core, frames)

    def test_dot(self):
        t = self._make_vector(2)
        d = t.dot(t)
        assert np.isclose(d, np.linalg.norm(t.numpy()) ** 2)

    def test_norm(self):
        t = self._make_vector(2)
        assert np.isclose(t.norm(), np.linalg.norm(t.numpy()))

    def test_norm_orth(self):
        t = self._make_vector(2)
        expected = t.norm()
        t_orth = t.orth()
        assert t_orth.tensor_frame.is_orth
        assert np.isclose(t_orth.norm(), expected)

    def test_orth(self):
        t = self._make_vector(2)
        t_orth = t.orth()
        assert t_orth.tensor_frame.is_orth
        assert np.isclose(t_orth.norm(), t.norm())

    def test_neg(self):
        t = self._make_vector(2)
        assert np.allclose((-t).numpy(), -t.numpy())

    def test_mul_scalar(self):
        t = self._make_vector(2)
        assert np.allclose((t * 2.5).numpy(), 2.5 * t.numpy())

    def test_rmul_scalar(self):
        t = self._make_vector(2)
        assert np.allclose((3.0 * t).numpy(), 3.0 * t.numpy())

    def test_truediv_scalar(self):
        t = self._make_vector(2)
        assert np.allclose((t / 2.0).numpy(), t.numpy() / 2.0)

    def test_add(self):
        rng = np.random.RandomState(0)
        shape = [10, 20]
        ranks = [2, 3]
        f1 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        f2 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        c1 = FullTensor(rng.rand(*ranks))
        c2 = FullTensor(rng.rand(*ranks))
        t1 = TuckerTensorVector(c1, f1)
        t2 = TuckerTensorVector(c2, f2)
        result = t1 + t2
        assert np.allclose(result.numpy(), t1.numpy() + t2.numpy())

    def test_sub(self):
        rng = np.random.RandomState(0)
        shape = [10, 20]
        ranks = [2, 3]
        f1 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        f2 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        c1 = FullTensor(rng.rand(*ranks))
        c2 = FullTensor(rng.rand(*ranks))
        t1 = TuckerTensorVector(c1, f1)
        t2 = TuckerTensorVector(c2, f2)
        result = t1 - t2
        assert np.allclose(result.numpy(), t1.numpy() - t2.numpy())

    def test_radd(self):
        t = self._make_vector(2)
        result = 0 + t
        assert np.allclose(result.numpy(), t.numpy())


class TestContractions:
    @staticmethod
    def _make_vector(order=2, seed=0):
        rng = np.random.RandomState(seed)
        ranks = rng.randint(2, 5, size=order)
        shape = rng.randint(5, 10, size=order)
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorVector(core, frames)

    def test_tensor_matrix_product(self):
        t = self._make_vector(2)
        shape = [s[0] for s in t.shape]
        A = [np.random.rand(7, shape[0]), np.random.rand(8, shape[1])]
        result = t.tensor_matrix_product(A)
        expected = A[0] @ t.numpy() @ A[1].T
        assert np.allclose(result.numpy(), expected)

    def test_tensor_vector_product(self):
        t = self._make_vector(2)
        shape = [s[0] for s in t.shape]
        v = [np.random.rand(shape[0]), np.random.rand(shape[1])]
        result = t.tensor_vector_product(v)
        expected = np.tensordot(np.tensordot(t.numpy(), v[0], (0, 0)),
                                v[1], (0, 0))
        assert np.allclose(result, expected)

    def test_tensor_vector_product_partial(self):
        t = self._make_vector(3)
        shape = [s[0] for s in t.shape]
        v = [np.random.rand(shape[0])]
        result = t.tensor_vector_product(v, dims=[0])
        assert isinstance(result, TuckerTensorVector)
        assert result.order == 2

    def test_tensor_diag_matrix_product(self):
        t = self._make_vector(2)
        shape = [s[0] for s in t.shape]
        M = [np.random.rand(shape[0]), np.random.rand(shape[1])]
        result = t.tensor_diag_matrix_product(M)
        expected = np.diag(M[0]) @ t.numpy() @ np.diag(M[1])
        assert np.allclose(result.numpy(), expected)

    def test_dot_with_rank_one_metric(self):
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        c1 = FullTensor(rng.rand(*ranks))
        c2 = FullTensor(rng.rand(*ranks))
        t1 = TuckerTensorVector(c1, frames)
        t2 = TuckerTensorVector(c2, frames)
        matrices = [rng.randn(shape[0], shape[0]), rng.randn(shape[1], shape[1])]
        d = t1.dot_with_rank_one_metric(t2, matrices)
        expected = np.sum(t1.numpy() * (matrices[0] @ t2.numpy() @ matrices[1].T))
        assert np.isclose(d, expected)


class TestSqueezePermuteCatKron:
    @staticmethod
    def _make_vector(order=2, seed=0):
        rng = np.random.RandomState(seed)
        ranks = rng.randint(2, 5, size=order)
        shape = rng.randint(5, 10, size=order)
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorVector(core, frames)

    def test_squeeze(self):
        rng = np.random.RandomState(0)
        ranks = [2, 3, 4]
        shape = [5, 1, 7]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        result = t.squeeze()
        assert result.shape == _vector_shape([5, 7])
        assert np.allclose(result.numpy(), t.numpy().squeeze())

    def test_squeeze_noop(self):
        t = self._make_vector(2)
        result = t.squeeze()
        assert result.shape == t.shape

    def test_permute(self):
        t = self._make_vector(3)
        perm = [2, 0, 1]
        result = t.permute(perm)
        shape_list = [s[0] for s in t.shape]
        assert all(r[0] == shape_list[p] for r, p in zip(result.shape, perm))
        assert np.allclose(result.numpy(), np.transpose(t.numpy(), perm))

    def test_cat(self):
        rng = np.random.RandomState(0)
        s1, s2 = [5, 6], [7, 8]
        ranks = [2, 3]
        f1 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, s1)
        ])
        f2 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, s2)
        ])
        c1 = FullTensor(rng.rand(*ranks))
        c2 = FullTensor(rng.rand(*ranks))
        t1 = TuckerTensorVector(c1, f1)
        t2 = TuckerTensorVector(c2, f2)
        result = t1.cat(t2)
        shape_out = [s[0] for s in result.shape]
        assert shape_out == [12, 14]
        expected = np.zeros([12, 14])
        expected[:5, :6] = t1.numpy()
        expected[5:, 6:] = t2.numpy()
        assert np.allclose(result.numpy(), expected)

    def test_kron(self):
        rng = np.random.RandomState(0)
        s1, s2 = [5, 6], [7, 8]
        ranks = [2, 3]
        f1 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, s1)
        ])
        f2 = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, s2)
        ])
        c1 = FullTensor(rng.rand(*ranks))
        c2 = FullTensor(rng.rand(*ranks))
        t1 = TuckerTensorVector(c1, f1)
        t2 = TuckerTensorVector(c2, f2)
        result = t1.kron(t2)
        shape_out = [s[0] for s in result.shape]
        assert shape_out == [35, 48]
        assert np.allclose(result.numpy(), np.kron(t1.numpy(), t2.numpy()))


class TestStorage:
    @staticmethod
    def _make_vector(seed=0):
        rng = np.random.RandomState(seed)
        ranks = [3, 4]
        shape = [5, 6]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorVector(core, frames)

    def test_storage(self):
        t = self._make_vector()
        assert t.storage() > 0

    def test_sparse_storage(self):
        t = self._make_vector()
        assert t.sparse_storage() >= 0

    def test_representation_rank(self):
        t = self._make_vector()
        assert t.representation_rank() == np.prod(t.ranks)


class TestSubTensorEvalDiag:
    @staticmethod
    def test_sub_tensor():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        original = t.numpy().copy()
        st = t.sub_tensor([np.array([0, 1]), np.array([0, 1, 2])])
        assert np.allclose(st.numpy(), original[:2, :3])

    @staticmethod
    def test_sub_tensor_partial():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        original = t.numpy().copy()
        st = t.sub_tensor([np.array([0, 2])], dims=[1])
        assert np.allclose(st.numpy(), original[:, [0, 2]])

    @staticmethod
    def test_eval_diag():
        rng = np.random.RandomState(0)
        shape = [5, 5]
        ranks = [3, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        d = t.eval_diag()
        assert np.allclose(d, np.diag(t.numpy()))

    @staticmethod
    def test_eval_diag_partial():
        rng = np.random.RandomState(0)
        shape = [5, 5, 6]
        ranks = [2, 2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        result = t.eval_diag(dims=[0, 1])
        shape_tuple = tuple(result.shape)
        assert shape_tuple == (5, 6) or shape_tuple == (5,)


class TestTransposeVectorize:
    @staticmethod
    def _make_operator(seed=0):
        rng = np.random.RandomState(seed)
        ranks = [2, 3]
        dims1 = [5, 7]
        dims2 = [3, 4]
        frames = TFrameOperator([
            FrameOperator(rng.rand(r, n1, n2))
            for r, n1, n2 in zip(ranks, dims1, dims2)
        ])
        core = FullTensor(rng.rand(*ranks))
        return TuckerTensorOperator(core, frames)

    def test_transpose(self):
        t = self._make_operator()
        tt = t.transpose()
        assert isinstance(tt, TuckerTensorOperator)
        shape_in = [s[1] for s in t.shape]
        shape_out = [s[0] for s in t.shape]
        assert [s[0] for s in tt.shape] == shape_in
        assert [s[1] for s in tt.shape] == shape_out

    def test_T(self):
        t = self._make_operator()
        assert [s[0] for s in t.T.shape] == [s[1] for s in t.shape]

    def test_ctranspose(self):
        t = self._make_operator()
        th = t.ctranspose()
        assert isinstance(th, TuckerTensorOperator)
        assert [s[0] for s in th.shape] == [s[1] for s in t.shape]

    def test_H(self):
        t = self._make_operator()
        assert [s[0] for s in t.H.shape] == [s[1] for s in t.shape]

    def test_transpose_error_vector(self):
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        with pytest.raises(AttributeError):
            t.transpose()

    def test_vectorize(self):
        t = self._make_operator()
        vec = t.vectorize()
        assert isinstance(vec, TuckerTensorVector)
        assert vec.order == t.order

    def test_vectorize_unvectorize_roundtrip(self):
        rng = np.random.RandomState(0)
        ranks = [2, 3]
        dims1 = [5, 7]
        dims2 = [3, 4]
        frames = TFrameOperator([
            FrameOperator(rng.rand(r, n1, n2))
            for r, n1, n2 in zip(ranks, dims1, dims2)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorOperator(core, frames)
        vec = t.vectorize()
        t2 = vec.unvectorize(dims1, dims2)
        assert isinstance(t2, TuckerTensorOperator)
        assert np.allclose(t.full().numpy(), t2.full().numpy())

    def test_to_operator(self):
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        op = t.to_operator()
        assert isinstance(op, TuckerTensorOperator)
        d1 = [s[0] for s in op.shape]
        d2 = [s[1] for s in op.shape]
        assert d1 == shape
        assert d2 == shape

    def test_to_operator_error_operator(self):
        rng = np.random.RandomState(0)
        ranks = [2, 3]
        dims1 = [5, 7]
        dims2 = [3, 4]
        frames = TFrameOperator([
            FrameOperator(rng.rand(r, n1, n2))
            for r, n1, n2 in zip(ranks, dims1, dims2)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorOperator(core, frames)
        with pytest.raises(AttributeError):
            t.to_operator()


class TestOperatorProduct:
    @staticmethod
    def test_matmul_operator_vector():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]

        op_frames = TFrameOperator([
            FrameOperator(rng.rand(r, n, n))
            for r, n in zip(ranks, shape)
        ])
        op_core = FullTensor(rng.rand(*ranks))
        op = TuckerTensorOperator(op_core, op_frames)

        vec_frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        vec_core = FullTensor(rng.rand(*ranks))
        vec = TuckerTensorVector(vec_core, vec_frames)

        result = op @ vec
        assert isinstance(result, TuckerTensorVector)

    @staticmethod
    def test_matmul_operator_operator():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks1 = [2, 3]
        ranks2 = [4, 5]

        f1 = TFrameOperator([
            FrameOperator(rng.rand(r, n, n))
            for r, n in zip(ranks1, shape)
        ])
        f2 = TFrameOperator([
            FrameOperator(rng.rand(r, n, n))
            for r, n in zip(ranks2, shape)
        ])
        c1 = FullTensor(rng.rand(*ranks1))
        c2 = FullTensor(rng.rand(*ranks2))
        t1 = TuckerTensorOperator(c1, f1)
        t2 = TuckerTensorOperator(c2, f2)

        result = t1 @ t2
        assert isinstance(result, TuckerTensorOperator)


class TestSingularValuesNormalize:
    @staticmethod
    def test_singular_values_order2():
        rng = np.random.RandomState(0)
        shape = [10, 10]
        ranks = [5, 5]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        sv = t.singular_values()
        assert sv.shape == (5,)
        expected = np.linalg.svd(t.numpy(), compute_uv=False)
        assert np.allclose(sv, expected[:5])

    @staticmethod
    def test_singular_values_order3():
        rng = np.random.RandomState(0)
        shape = [5, 6, 7]
        ranks = [2, 3, 4]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        sv = t.singular_values()
        assert len(sv) == 3
        assert sv[0].shape == (2,)

    @staticmethod
    def test_normalize_basis():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        expected_norm = t.norm()
        t.normalize_basis()
        for mu in range(t.order):
            col_norms = np.linalg.norm(
                t.tensor_frame.frames[mu].data[:, :, 0], axis=1)
            assert np.allclose(col_norms, np.ones(t.ranks[mu]), atol=1e-10)
        assert np.isclose(t.norm(), expected_norm)


class TestStaticConstructors:
    @staticmethod
    def test_static_create_vector():
        t = BaseTuckerTensor.create(
            lambda x: np.ones(x), [2, 3], [5, 6])
        assert isinstance(t, TuckerTensorVector)
        shape_out = [s[0] for s in t.shape]
        assert shape_out == [5, 6]
        assert np.allclose(t.numpy(), 6 * np.ones([5, 6]))

    @staticmethod
    def test_static_create_operator():
        shape = [(5, 3), (7, 4)]
        t = BaseTuckerTensor.create(
            lambda x: np.ones(x), [2, 3], shape)
        assert isinstance(t, TuckerTensorOperator)

    @staticmethod
    def test_static_rand():
        t = BaseTuckerTensor.rand([2, 3], [5, 6])
        assert isinstance(t, TuckerTensorVector)
        assert t.order == 2
        assert [s[0] for s in t.shape] == [5, 6]

    @staticmethod
    def test_zeros():
        t = BaseTuckerTensor.zeros([2, 3], [5, 6])
        assert isinstance(t, TuckerTensorVector)
        assert np.allclose(t.numpy(), np.zeros([5, 6]))

    @staticmethod
    def test_ones():
        t = BaseTuckerTensor.ones([2, 3], [5, 6])
        assert isinstance(t, TuckerTensorVector)
        assert np.allclose(t.numpy(), 6 * np.ones([5, 6]))

    @staticmethod
    def test_eye():
        t = BaseTuckerTensor.eye([5, 6])
        assert isinstance(t, TuckerTensorOperator)
        shape_out = [s[0] for s in t.shape]
        shape_in = [s[1] for s in t.shape]
        assert shape_out == [5, 6]
        assert shape_in == [5, 6]

    @staticmethod
    def test_tree_based_tensor_conversion():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        tbt = t.tree_based_tensor()
        assert isinstance(tbt, TreeBasedTensor)


class TestDisplay:
    @staticmethod
    def test_repr():
        rng = np.random.RandomState(0)
        shape = [5, 6]
        ranks = [2, 3]
        frames = TFrameVector([
            FrameVector(rng.rand(r, n, 1))
            for r, n in zip(ranks, shape)
        ])
        core = FullTensor(rng.rand(*ranks))
        t = TuckerTensorVector(core, frames)
        r = repr(t)
        assert "TuckerTensorVector" in r or "BaseTuckerTensor" in r
        assert "order" in r
