"""Order of the examples in the Examples Gallery.

Within a section of the gallery, sphinx-gallery sorts the examples with the
key below: the examples listed in EXAMPLES come first, in this order, then
the others by file name.
"""

EXAMPLES = [
    "full_tensor.py",
    "dimension_tree.py",
    "tree_based_tensor.py",
    "tensor_truncation.py",
    "tucker_tensor.py",
    "tt_tensor.py",
    "measures.py",
    "orthonormal_polynomials.py",
    "functions_bases_grids.py",
    "bsplines.py",
    "piecewise_polynomials.py",
    "integration_rules.py",
    "multi_indices.py",
    "tensorized_function.py",
    "conditional_expectation.py",
    "functional_basis_approximation.py",
    "magic_points.py",
    "interpolation_tensor.py",
    "least_squares_tensor.py",
    "tensor_pca.py",
    "functional_tensor_pca.py",
    "tensorized_function_pca.py",
    "canonical_tensor_learning.py",
    "tree_based_tensor_learning.py",
    "tree_based_tensor_density_learning.py",
    "tensorized_function_learning.py",
    "tensor_completion.py",
    "digits_recognition.py",
    "poincare_learning.py",
    "poincare_learning_surrogate.py",
    "poincare_learning_surrogate_greedy.py",
    "poincare_learning_surrogate_collective.py",
    "poisson_tucker.py",
    "greedy_with_updates.py",
    "tt_solvers.py",
    # Tensap tutorials, Linear solvers
    "tutorial_linearsolver_tucker_poisson.py",
    "tutorial_linearsolver_high_dim_poisson.py",
    "tutorial_linearsolver_update_poisson_solution.py",
    "tutorial_linearsolver_truncation_poisson_solution.py",
    "tutorial_linearsolver_parametric_diffusion_reaction.py",
]


class ExampleOrder:
    """Sort key of sphinx-gallery: position of the example in EXAMPLES."""

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        if filename in EXAMPLES:
            return (EXAMPLES.index(filename), filename)
        return (len(EXAMPLES), filename)

    def __repr__(self):
        return "<ExampleOrder>"
