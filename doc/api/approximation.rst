Approximation
=============

.. currentmodule:: tensap

Functional bases, integration rules, statistical learning, tensor learning and principal component analysis.

Sub-package: ``tensap.approximation``.

Functional bases
----------------

.. autosummary::
   :toctree: generated/
   :template: class

   FunctionalBasis
   FunctionalBases
   FunctionalBasisArray
   PolynomialFunctionalBasis
   PiecewisePolynomialFunctionalBasis
   BSplinesFunctionalBasis
   DilatedBSplines
   DilatedBSplinesFunctionalBasis
   DeltaFunctionalBasis
   SubFunctionalBasis
   UserDefinedFunctionalBasis
   FullTensorProductFunctionalBasis
   SparseTensorProductFunctionalBasis

Integration
-----------

.. autosummary::
   :toctree: generated/
   :template: class

   IntegrationRule
   FullTensorProductIntegrationRule

Learning
--------

.. autosummary::
   :toctree: generated/
   :template: class

   Learning
   LinearModelLearning
   LinearModelLearningSquareLoss
   LinearModelLearningDensityL2
   LinearModelLearningCustomLoss
   LossFunction
   SquareLossFunction
   DensityL2LossFunction
   CustomLossFunction

Tensor learning
---------------

.. autosummary::
   :toctree: generated/
   :template: class

   TensorLearning
   CanonicalTensorLearning
   TreeBasedTensorLearning

Principal component analysis
----------------------------

.. autosummary::
   :toctree: generated/
   :template: class

   TensorPrincipalComponentAnalysis
   FunctionalTensorPrincipalComponentAnalysis

Model selection
---------------

.. autosummary::
   :toctree: generated/
   :template: class

   ModelSelection

