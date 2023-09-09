import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from modelsight.config import settings
from modelsight.curves import (
    average_roc_curves, 
    roc_single_comparison,
    roc_comparisons,
    add_annotations
)

# pass fit_and_validate_model fixture as input to test_average_roc_curves
# find a way to pass a parameter to fit_and_validate_model so that
# we can seamlessly switch between loading a pre-existing cv_results
# and executing the entire procedure


@pytest.mark.parametrize('load', [True])
@pytest.mark.parametrize('load_path', [Path(__file__).resolve().parent / f"cv_results/cv_results_1694259027.035128.pkl"])
def test_average_roc_curves(cv_results, load, palette):
    model_names = list(cv_results.keys())
    alpha = 0.05
    alph_str = str(alpha).split(".")[1]
    alpha_formatted = f".{alph_str}"
    roc_symbol = "*"
    n_boot = 100

    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    kwargs = dict()

    f, ax, barplot, bars, all_data = average_roc_curves(cv_results,
                                                        colors=palette,
                                                        model_keys=model_names,
                                                        show_ci=True,
                                                        n_boot=n_boot,
                                                        bars_pos=[
                                                            0.3, 0.01, 0.6, 0.075*len(model_names)],
                                                        random_state=settings.misc.seed,
                                                        ax=ax,
                                                        **kwargs)
    
    roc_comparisons_results = roc_comparisons(cv_results, "EBM")
    
    kwargs = dict(space_between_whiskers = 0.07)
    order = [
        ("EBM", "RF"),
        ("EBM", "SVC"),
        ("EBM", "LR"),
        ("EBM", "KNN")
    ]
    ax_annot = add_annotations(roc_comparisons_results, 
                      alpha = 0.05, 
                      bars=bars, 
                      direction = "vertical",
                      order = order,
                      symbol = roc_symbol,
                      symbol_fontsize = 30,
                      voffset = -0.05,
                      ext_voffset=0,
                      ext_hoffset=0,
                      ax=barplot,
                      **kwargs)
    
    # plt.show()

    median_aucs_cis = []
    for model_name, model_results in cv_results.items():
        gts = model_results.gts_val
        probas = model_results.probas_val

        cv_aucs = []
        for split_gts, split_probas in zip(gts, probas):
            auc_val = roc_auc_score(split_gts, split_probas)
            cv_aucs.append(auc_val)
            
        auc_low, auc_med, auc_up = np.percentile(cv_aucs, [2.5, 50, 97.5])
        median_aucs_cis.append((model_name, auc_med, auc_low, auc_up))
    
    barplot_sorted_model_names = [text.get_text() for text in barplot.get_yticklabels()]
    
    # sort median_aucs_cis based on the ordering provided by barplot_sorted_model_names
    d = {model_name: index for index, model_name in enumerate(barplot_sorted_model_names)}
    sorted_aucs_cis = sorted(median_aucs_cis, key=lambda item: d.get(item[0], float('inf')))
    
    np.testing.assert_array_equal(bars.datavalues, 
                                  [t[1] for t in sorted_aucs_cis])
        

    low_up_cis = np.array([(t[2], t[3]) for t in sorted_aucs_cis])
    paths = bars.errorbar.lines[-1][0]._paths
    xerrors = np.array([path.vertices[:, 0] for path in paths])
    np.testing.assert_array_equal(xerrors, low_up_cis)