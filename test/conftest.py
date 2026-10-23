import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    """Fix the global numpy seed: tensap draws with np.random."""
    np.random.seed(0)


@pytest.fixture
def rng():
    """Random generator for the test data."""
    return np.random.default_rng(0)
