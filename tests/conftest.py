import os, sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath('./src'))

from modelsight.config import settings


@pytest.fixture
def rng() -> np.random.Generator:
    """Construct a new Random Generator using the seed specified in settings.

    Returns:
        numpy.random.Generator: a Random Generator based on BitGenerator(PCG64)
    """
    return np.random.default_rng(settings.SEED)
