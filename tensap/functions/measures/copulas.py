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
Module copulas.

"""

from abc import ABC
import numpy as np


class Copula(ABC):
    """
    Abstract class for copulas.

    A copula is the joint cumulative distribution function of a random vector
    whose marginals are uniform on [0, 1]. Subclasses implement the methods
    pdf and cdf.
    """


class IndependentCopula(Copula):
    """
    Independent copula, whose density is equal to 1 on [0, 1]^d.

    Its cumulative distribution function is the product of its arguments.
    """

    def pdf(self, u):
        return np.ones((np.shape(u)[0], 1))

    def cdf(self, u):
        return np.prod(u, axis=1)
