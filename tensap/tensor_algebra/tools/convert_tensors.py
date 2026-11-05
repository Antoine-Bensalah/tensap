"""Tensor type conversion for binary operations.

Aligns two tensors to a common type so that operations like ``+``, ``dot``,
and ``kron`` can proceed.  Based on MATLAB's ``convertTensors``.
"""

import tensap


def _canonical_to_tucker(canonical: "tensap.CanonicalTensor") -> "tensap.TuckerTensorVector":
    """Wrap a CanonicalTensor into a TuckerTensorVector."""
    frames = tensap.TFrameVector([
        tensap.FrameVector(f.T.reshape(f.shape[1], f.shape[0], 1))
        for f in canonical.space
    ])
    return tensap.TuckerTensorVector(canonical.core, frames)


def convert_tensors(x, y):
    """Align two tensors to a common type for binary operations.

    Parameters
    ----------
    x : tensor object
    y : tensor object

    Returns
    -------
    (x_converted, y_converted)
        Tensors with the same type.

    Raises
    ------
    NotImplementedError
        If no conversion path exists between the two types.
    """
    _TYPES = (tensap.FullTensor, tensap.DiagonalTensor, tensap.SparseTensor,
              tensap.CanonicalTensor, tensap.TreeBasedTensor, tensap.BaseTuckerTensor,
              tensap.TTTensor)

    if not isinstance(x, _TYPES) or not isinstance(y, _TYPES):
        raise NotImplementedError(
            f"Cannot convert {type(x).__name__} and {type(y).__name__} "
            "to a common tensor type."
        )

    # BaseTuckerTensor + BaseTuckerTensor — align cores (checked before the
    # same-type shortcut: two Tucker tensors of the same class may have cores
    # of different types)
    if isinstance(x, tensap.BaseTuckerTensor) and isinstance(y, tensap.BaseTuckerTensor):
        cx, cy = convert_tensors(x.core, y.core)
        return type(x)._factory(cx, x.tensor_frame), type(y)._factory(cy, y.tensor_frame)

    if type(x) is type(y):
        return x, y

    # FullTensor + other → other.full()
    if isinstance(x, tensap.FullTensor):
        return x, y.full()
    if isinstance(y, tensap.FullTensor):
        return x.full(), y

    # DiagonalTensor + TreeBasedTensor → tree_based_tensor(diag, tree)
    if isinstance(x, tensap.DiagonalTensor) and isinstance(y, tensap.TreeBasedTensor):
        return x.tree_based_tensor(y.tree), y
    if isinstance(y, tensap.DiagonalTensor) and isinstance(x, tensap.TreeBasedTensor):
        return x, y.tree_based_tensor(x.tree)

    # DiagonalTensor + TTTensor → TT representation of the diagonal tensor
    if isinstance(x, tensap.DiagonalTensor) and isinstance(y, tensap.TTTensor):
        return tensap.TTTensor.from_diagonal_tensor(x), y
    if isinstance(y, tensap.DiagonalTensor) and isinstance(x, tensap.TTTensor):
        return x, tensap.TTTensor.from_diagonal_tensor(y)

    # SparseTensor + DiagonalTensor → sparse(diag)
    if isinstance(x, tensap.SparseTensor) and isinstance(y, tensap.DiagonalTensor):
        return x, y.sparse()
    if isinstance(y, tensap.SparseTensor) and isinstance(x, tensap.DiagonalTensor):
        return x.sparse(), y

    # CanonicalTensor + BaseTuckerTensor → TuckerTensorVector(canonical)
    if isinstance(x, tensap.CanonicalTensor) and isinstance(y, tensap.BaseTuckerTensor):
        x = _canonical_to_tucker(x)
        return convert_tensors(x, y)
    if isinstance(y, tensap.CanonicalTensor) and isinstance(x, tensap.BaseTuckerTensor):
        y = _canonical_to_tucker(y)
        return convert_tensors(x, y)

    raise NotImplementedError(
        f"Cannot convert {type(x).__name__} and {type(y).__name__} "
        "to a common tensor type."
    )
