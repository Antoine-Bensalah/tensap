"""First computation with tensap: learn a function of 6 variables from
samples, in a tree-based tensor format.

This script is included in the Getting Started page of the documentation,
and run by the tests (test/test_getting_started.py).
"""

import numpy as np
import tensap

np.random.seed(0)

# 1. The function to approximate, of 6 random variables uniform on
# [-1, 1].
d = 6
X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)


def fun(x):
    return (
        np.sin(x[:, 0] + x[:, 3]) * np.exp(x[:, 4]) * x[:, 5]
        + np.sin(x[:, 2] * x[:, 3]) * x[:, 5]
    )


# 2. Training and test samples.
x_train = X.random(500)
y_train = fun(x_train)
x_test = X.random(10000)
y_test = fun(x_test)

# 3. One basis of orthonormal polynomials of degree at most 8 per
# variable.
bases = tensap.FunctionalBases(
    [
        tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(9)
        )
        for x in X.random_variables
    ]
)

# 4. Learning in tree-based tensor format, with adaptation of the ranks.
solver = tensap.TreeBasedTensorLearning.tensor_train_tucker(
    d, tensap.SquareLossFunction()
)
solver.bases = bases
solver.bases_eval = bases.eval(x_train)
solver.training_data = [None, y_train]
solver.rank_adaptation = True
solver.rank_adaptation_options["max_iterations"] = 10
solver.tree_adaptation = True
solver.linear_model_learning.basis_adaptation = True
solver.display = False
solver.alternating_minimization_parameters["display"] = False

f, output = solver.solve()

# 5. Results.
test_error = solver.loss_function.test_error(f, [x_test, y_test])
print("Ranks of the approximation:", f.tensor.ranks)
print("Storage complexity:", f.tensor.storage(), "parameters")
print(f"Relative test error: {test_error:.2e}")
