import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modelsight.calibration import hosmer_lemeshow_plot


@pytest.fixture
def ground_truths():
    return np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1])


@pytest.fixture
def probas():
    return np.array([0.9, 0.88, 0.91, 0.13, 0.2, 0.85, 0.88, 0.05, 0.1, 0.90,
                     0.1, 0.05, 0.1, 0.99, 0.86])


def test_calibration_plot(ground_truths, probas):
    f, ax = hosmer_lemeshow_plot(ground_truths,
                                 probas,
                                 n_bins=10,
                                 colors=('blue', 'red'),
                                 annotate_bars=True,
                                 title="",
                                 brier_score_annot="",
                                 ax=None
                                 )
