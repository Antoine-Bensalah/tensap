The entries of an order :math:`d` tensor
:math:`X \in \mathbb{R}^{N_1 \times \cdots \times N_d}` in canonical
format are

.. math::

   X_{i_1, \ldots, i_d} = \sum_{k=1}^r C_k \, U^1_{i_1, k} \cdots U^d_{i_d, k},

with :math:`r` the canonical rank, and where the
:math:`U^\nu = (U^\nu_{i_\nu, k})_{1 \le i_\nu \le N_\nu, 1 \le k \le r}`
are order-two tensors (attribute ``space``) and
:math:`C = (C_k)_{1 \le k \le r}` is the core (attribute ``core``).
