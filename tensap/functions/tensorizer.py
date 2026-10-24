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
Module tensorizer.

"""

import numpy as np
import tensap


class Tensorizer:
    """
    Class Tensorizer.

    Attributes
    ----------
    b : int
        The base of the map.
    d : int or list of int
        The resolution of the map: the number of digits, the same for all the
        dimensions (int) or one per dimension (list of length dim).
    dim : int
        The input dimension of the function to be tensorized.
    X : tensap.RandomVector
        Random vector used for the map.
    Y : tensap.RandomVector
        Random vector used for the map.

    """

    def __init__(self, b, d, dim=1, X=None, Y=None):
        """
        Constructor for the class Tensorizer.

        If dim == 1, defines a map t from [0,1] to {0,...,b-1}^d x [0,1]
        t(x) = (i_1, ..., i_d, y) with y in [0,1] and i_k in {0, ..., b-1}
        such that x = (i + y)b^(-d) with i in {0, ..., b^d-1}
        having the following representation in base b:
        i = sum_{k=1}^d i_k b^(d-k) in [0,b^d-1].

        If dim != 1, defines a map t from [0,1]^dim to
        {{0, ..., b-1}^d}^dim x [0,1]^dim if property orderingType == 1,
        or from [0,1]^dim to {{0, ..., b-1}^dim}^d x [0,1]^dim if property
        orderingType == 2.

        With one resolution per dimension (d = [d_1, ..., d_dim]), dimension k
        has d_k digits. The interleaved ordering then takes the first digit of
        every dimension, then the second digit of the dimensions that have
        one, and so on (digits aligned on the most significant one).

        Parameters
        ----------
        b : int
        The base of the map.
        d : int or list of int
            The resolution of the map, for all the dimensions or one per
            dimension.
        dim : int, optional
            The input dimension of the function to be tensorized. The default
            is 1.
        X : tensap.RandomVector, optional
            Random vector used for the map. The default is None a uniform
            random vector on [0, 1]^dim.
        Y : tensap.RandomVector, optional
            Random vector used for the map. The default is None a uniform
            random vector on [0, 1]^dim.
        ordering_type : int
            Integer specifying the ordering type of the variables.
            The default is 1.

        Returns
        -------
        None.

        """
        self.dim = dim
        self.b = b
        self.d = d
        if np.ndim(d) != 0 and len(d) != dim:
            raise ValueError("d must be an integer or a list of dim integers.")

        self.ordering_type = 1

        if X is not None:
            if isinstance(X, tensap.RandomVariable):
                X = tensap.RandomVector(X, self.dim)
            self.X = X
        else:
            self.X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), self.dim)

        if Y is not None:
            if isinstance(Y, tensap.RandomVariable):
                Y = tensap.RandomVector(Y, self.dim)
            self.Y = Y
        else:
            self.Y = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), self.dim)

    @property
    def levels(self):
        """Number of digits of each dimension, array of length dim."""
        return np.broadcast_to(np.asarray(self.d, dtype=int), (self.dim,)).copy()

    def digit_positions(self):
        """Positions of the digits of each dimension in z = (i, y).

        Returns
        -------
        list of numpy.ndarray
            For each dimension k, the positions of its digits (most
            significant first) among the sum(levels) digits of z.
        """
        levels = self.levels
        if self.ordering_type == 1:
            offsets = np.concatenate(([0], np.cumsum(levels)))
            return [np.arange(offsets[k], offsets[k + 1]) for k in range(self.dim)]
        if self.ordering_type == 2:
            positions = [[] for _ in range(self.dim)]
            position = 0
            for level in range(levels.max()):
                for k in range(self.dim):
                    if level < levels[k]:
                        positions[k].append(position)
                        position += 1
            return [np.array(p, dtype=int) for p in positions]
        raise ValueError("Wrong ordering_type")

    def map(self, x, nargout=1):
        """
        Evaluate the map at points x, returning y and i such that
        self.map(x) = (i_1, ..., i_d, y).

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the map is to be evaluated.
        nargout : int, optional
            The number of outputs. The default is 1, returning a horizontal
            stack of i and y. Set to 2 to return y and i separately.

        Returns
        -------
        numpy.ndarray
            Either (i_1, ..., i_d, y) if nargout == 1, or y if nargout == 2.
        numpy.ndarray
            (i_1, ..., i_d), if nargout == 2.

        """
        if np.ndim(x) == 1:
            x = np.reshape(x, [-1, 1])

        y = []
        positions = self.digit_positions()
        levels = self.levels
        i = np.zeros((x.shape[0], levels.sum()), dtype=int)
        for k in range(self.dim):
            uk = self.X.random_variables[k].cdf(x[:, k])
            y.append(Tensorizer.u2z(uk, self.b, levels[k]))
            y[k][:, -1] = self.Y.random_variables[k].icdf(y[k][:, -1])
            i[:, positions[k]] = y[k][:, :-1]
            y[k] = y[k][:, -1]
        y = np.transpose(y)

        if nargout == 2:
            return y, i
        return np.hstack((i, np.atleast_2d(y)))

    def inverse_map(self, z):
        """
        Evaluate the map at points z = (i_1, ..., i_d, y), returning x such
        that self.map(x) = z.

        Parameters
        ----------
        z : list or numpy.ndarray
            The points at which the inverse map is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The points x such that self.map(x) = z.

        """
        z = np.atleast_2d(z)
        u = []
        positions = self.digit_positions()
        for k in range(self.dim):
            ik = z[:, positions[k]]
            zk = z[:, -self.dim + k]
            zk = self.Y.random_variables[k].cdf(zk)
            u.append(Tensorizer.z2u(np.hstack((ik, np.reshape(zk, [-1, 1]))), self.b))
            u[k] = self.X.random_variables[k].icdf(u[k])
        return np.transpose(u)

    def tensorize(self, fun):
        """
        Tensorize a provided function defined on self.X.support().

        Parameters
        ----------
        fun : function or tensap.Function
            The function to be tensorized.

        Raises
        ------
        ValueError
            If the provided argument is neither a tensap.Function nor a
            function.

        Returns
        -------
        tensap.TensorizedFunction
            The tensorized function.

        """
        if not isinstance(fun, tensap.Function) and not hasattr(fun, "__call__"):
            raise ValueError("The argument must be a tensap.Function or " + "function.")

        if not isinstance(fun, tensap.Function) and hasattr(fun, "__call__"):
            fun = tensap.UserDefinedFunction(fun, self.dim)
            fun.evaluation_at_multiple_points = True

        f = tensap.UserDefinedFunction(
            lambda z: fun(self.inverse_map(z)), int(self.levels.sum()) + self.dim
        )
        f.evaluation_at_multiple_points = fun.evaluation_at_multiple_points
        return tensap.TensorizedFunction(f, self)

    def tensorized_function_functional_bases(self, h=1):
        """
        Return a tensap.FunctionalBases object associated with the provided
        basis or basis function(s) and the Tensorizer object.

        Parameters
        ----------
        h : tensap.FunctionalBases or tensap.FunctionalBasis or function or
        list or scalar, optional
            The function used to generate the basis. The default is the
            function 1.

        Returns
        -------
        tensap.FunctionalBases
            The functional bases.

        """
        # if isinstance(h, (np.ndarray, list)) or np.isscalar(h):
        #    h = lambda y, h=h: h*np.ones(np.shape(y))

        if hasattr(h, "__call__"):
            h = tensap.UserDefinedFunctionalBasis([h])
            h.measure = self.Y.random_variables[0]

        if isinstance(h, tensap.FunctionalBasis):
            h = tensap.FunctionalBases.duplicate(h, self.dim)

        if np.isscalar(h):
            h = [
                tensap.PolynomialFunctionalBasis(
                    y.orthonormal_polynomials(), range(h + 1)
                )
                for y in self.Y.random_variables
            ]
            h = tensap.FunctionalBases(h)

        assert isinstance(h, tensap.FunctionalBases), "Wrong type of argument for h."

        p = tensap.DiscretePolynomials(
            tensap.DiscreteRandomVariable(np.reshape(np.arange(self.b), [-1, 1]))
        )
        p = tensap.PolynomialFunctionalBasis(p, np.arange(self.b))

        bases = [p] * int(self.levels.sum()) + list(h.bases)
        return tensap.FunctionalBases(bases)

    @staticmethod
    def u2z(u, b, d, nargout=1):
        """
        Return the representation of numbers on [0, 1] in base b with
        resolution d.

        Parameters
        ----------
        u : list or numpy.ndarray
            The inputs to be converted in base b with resolution d.
        b : int
            The base.
        d : int
            The resolution.
        nargout : int, optional
            The number of outputs. The default is 1, returning a horizontal
            stack of i and y. Set to 2 to return y and i separately.

        Returns
        -------
        numpy.ndarray
            Either (i_1, ..., i_d, y) if nargout == 1, or y if nargout == 2.
        numpy.ndarray
            (i_1, ..., i_d), if nargout == 2.

        """
        u = np.ravel(u)
        su = u * (b**d)
        i = np.floor(su)
        i = np.minimum(i, b**d - 1).astype(int)
        y = su - i
        i = tensap.integer2baseb(i, b, d)
        if nargout == 2:
            return y, i
        return np.hstack((i, np.reshape(y, [-1, 1])))

    @staticmethod
    def z2u(z, b):
        """
        Return the representation of numbers in base b in decimal on [0, 1].

        Parameters
        ----------
        z : list or numpy.ndarray
            The numbers in base b.
        b : int
            The base.

        Returns
        -------
        numpy.ndarray
            The decimal representation of the inputs on [0, 1].

        """
        z = np.atleast_2d(z)
        d = z.shape[1] - 1
        y = z[:, -1]
        i = tensap.baseb2integer(z[:, :-1].astype(int), b)
        return (y + i) * b ** (-d)
