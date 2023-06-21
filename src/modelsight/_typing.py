import sys
import random
import numpy as np
from numpy.random import SeedSequence, BitGenerator, Generator
from typing import Union, List, Tuple, Annotated, Literal, TypeVar

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Union[np.ndarray, List[List[float]]]

SeedType = Union[None, int, ArrayLike, SeedSequence, BitGenerator, Generator]