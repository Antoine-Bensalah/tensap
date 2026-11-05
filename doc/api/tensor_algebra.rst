Tensor algebra
==============

.. currentmodule:: tensap

Tensors in full, sparse and low-rank formats, dimension trees, and their truncation.

Sub-package: ``tensap.tensor_algebra``.

Tensors
-------

.. autosummary::
   :toctree: generated/
   :template: class

   FullTensor
   SparseTensor
   DiagonalTensor
   CanonicalTensor
   TreeBasedTensor
   DimensionTree

Tensor-train format
-------------------

A tensor train (TT) is a chain of cores of order three (vectors) or four
(operators); the cores are owned by the tensor and read-only.

.. autosummary::
   :toctree: generated/
   :template: class

   BaseTTTensor
   TTTensor
   TTOperator

Tucker format
-------------

A Tucker tensor is a core tensor (full, diagonal, tree-based or tensor
train) contracted with one frame per dimension: a set of vectors (frame
vectors) or of matrices (frame operators).

.. autosummary::
   :toctree: generated/
   :template: class

   BaseTuckerTensor
   TuckerTensorVector
   TuckerTensorOperator
   Frame
   FrameVector
   FrameOperator
   TFrame
   TFrameVector
   TFrameOperator

Construction of operators
-------------------------

.. autosummary::
   :toctree: generated/
   :template: class

   LaplacianLikeOperatorFactory

.. autosummary::
   :toctree: generated/
   :template: function

   convert_tensors

Quantized tensor trains
-----------------------

Quantization of arrays, functions and matrices into tensor trains with
modes of size b (module ``tensap.tensor_algebra.tools.qtt``), in the
layout of a :class:`~tensap.Tensorizer`, and the QTT primitives of 1D
operators.

.. autosummary::
   :toctree: generated/
   :template: function

   quantize
   dequantize
   quantize_function
   quantize_operator
   dequantize_operator
   grid_points
   to_functional_tensor
   qtt_layout_product
   qtt_embed
   qtt_insert_axis
   qtt_elementary_vector
   qtt_identity
   qtt_shift
   qtt_elementary_matrix
   qtt_ones_matrix
   qtt_banded
   qtt_corner

Truncation
----------

.. autosummary::
   :toctree: generated/
   :template: class

   Truncator

