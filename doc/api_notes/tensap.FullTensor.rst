A :class:`~tensap.FullTensor` ``X`` represents an order :math:`d` tensor
:math:`X \in \mathbb{R}^{N_1 \times \cdots \times N_d}`, or
multidimensional array of size :math:`N_1 \times \cdots \times N_d`. Its
entries are :math:`X_{i_1, \ldots, i_d}`, with
:math:`(i_1, \ldots, i_d)` a tuple of indices, where
:math:`i_\nu \in \{0, \ldots, N_\nu - 1\}` is related to the
:math:`\nu`-th mode of the tensor.

For :math:`\alpha \subset \{1, \ldots, d\}` and its complementary subset
:math:`\alpha^c`, the :math:`\alpha`-matricization of :math:`X`
(:meth:`~tensap.FullTensor.matricize`) is the matrix :math:`M` of size
:math:`(\prod_{\nu \in \alpha} N_\nu) \times (\prod_{\nu \in \alpha^c} N_\nu)`
such that
:math:`X_{i_1, \ldots, i_d} = M_{\overline{i_\alpha}, \overline{i_{\alpha^c}}}`,
with :math:`i_\alpha = (i_\nu)_{\nu \in \alpha}`.
