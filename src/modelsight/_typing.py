import sys
import random
import numpy as np
from dataclasses import dataclass, field
from numpy.random import SeedSequence, BitGenerator, Generator
from typing import Union, List, Tuple, Annotated, Literal, TypeVar, Optional
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.model_selection._split import BaseCrossValidator, _RepeatedSplits

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Union[np.ndarray, List[List[float]]]

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


Estimator = TypeVar("Estimator", bound=BaseEstimator)
SeedType = Union[None, int, ArrayLike, SeedSequence, BitGenerator, Generator]
CVScheme: TypeAlias = TypeVar("CVScheme", BaseCrossValidator, _RepeatedSplits)


@dataclass
class CVModellingOutput:    
    gts_train: ArrayLike
    gts_val: ArrayLike
    gts_train_conc: ArrayLike
    gts_val_conc: ArrayLike

    # predicted probabilities
    probas_train: ArrayLike
    probas_val: ArrayLike
    probas_train_conc: ArrayLike
    probas_val_conc: ArrayLike
    
    # misc
    models: List[Estimator]
    errors: Optional[ArrayLike]
    correct: Optional[ArrayLike]
    features: Optional[ArrayLike]