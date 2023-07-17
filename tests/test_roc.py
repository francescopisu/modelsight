import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from modelsight.config import settings
from modelsight.curves import average_roc_curves

# pass fit_and_validate_model fixture as input to test_average_roc_curves
# find a way to pass a parameter to fit_and_validate_model so that
# we can seamlessly switch between loading a pre-existing cv_results
# and executing the entire procedure


@pytest.mark.parametrize('load', [True])
@pytest.mark.parametrize('load_path', [Path(__file__).resolve().parent / f"cv_results/cv_results_1689541635.085708.pkl"])
def test_average_roc_curves(cv_results, load, palette):
    model_names = list(cv_results.keys())
    alpha = 0.05
    alph_str = str(alpha).split(".")[1]
    alpha_formatted = f".{alph_str}"
    roc_symbol = "*"
    n_boot = 100

    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    # kwargs = dict(model_names_in_black=["Model 1: Stenosis"])
    kwargs = dict()

    f, ax, barplot, bars, aucs_cis = average_roc_curves(cv_results,
                                                        colors=palette,
                                                        model_keys=model_names,
                                                        show_ci=True,
                                                        n_boot=n_boot,
                                                        bars_pos=[
                                                            0.5, 0.01, 0.4, 0.1*len(model_names)],
                                                        random_state=settings.misc.seed,
                                                        ax=ax,
                                                        **kwargs)

    median_aucs = []
    low_cis, up_cis = [], []
    for model_name, model_results in cv_results.items():
        gts = model_results.gts_val
        probas = model_results.probas_val

        cv_aucs = []
        for split_gts, split_probas in zip(gts, probas):
            auc_val = roc_auc_score(split_gts, split_probas)
            cv_aucs.append(auc_val)
            
        auc_low, auc_med, auc_up = np.percentile(cv_aucs, [2.5, 50, 97.5])
        
        median_aucs.append(auc_med)
        low_cis.append(auc_low)
        up_cis.append(auc_up)
    
    np.testing.assert_array_equal(bars.datavalues, median_aucs)
    
    
    low_up_cis = np.array(list(zip(low_cis, up_cis)))
    xerrors = bars.errorbar.lines[-1][0]._paths[0].vertices
    xerrors = xerrors[:, 0].reshape(len(model_names), 2)

    np.testing.assert_array_equal(xerrors, low_up_cis)