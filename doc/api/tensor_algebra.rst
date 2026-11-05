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

Tucker format
-------------

A Tucker tensor is a core tensor (full, diagonal or tree-based)
contracted with one frame per dimension: a set of vectors (frame
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

Truncation
----------

.. autosummary::
   :toctree: generated/
   :template: class

   Truncator

