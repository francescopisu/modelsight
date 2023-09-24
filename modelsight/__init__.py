# read version from installed package
from importlib.metadata import version

from modelsight import (
    calibration,
    curves
)

__all__ = [
    "calibration",
    "curves"
]

# __version__ = version("modelsight")

