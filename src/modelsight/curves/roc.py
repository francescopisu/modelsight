import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scikits.bootstrap import bootstrap_indices as scikits_boot_indices

from src.modelsight._typing import CVModellingOutput, SeedType


def average_roc_curves(cv_preds: Dict[str, CVModellingOutput],
                       colors: List[str],
                       model_keys_map: Dict[str, str] = {},
                       show_ci: bool = True,
                       n_boot: int = 1000,
                       bars_pos: Tuple[int, int, int, int] = (0.41, 0.01, 0.53, 0.30),
                       random_state: SeedType = 1234,
                       ax: plt.Axes = None,
                       **kwargs) -> Tuple[plt.Axes,
                                          plt.Axes,
                                          matplotlib.container.BarContainer,
                                          Dict[str, Dict[str, float]]]:
    """
    Generate receiver-operating characteristic curves for each model in cv_preds.

    Parameters
    ----------
    cv_preds: Dict[str, CVModellingOutput]
        A dictionary containing model-specific cross-validation modelling outputs.
    colors: List[str]
        A list of colors that will be used to color both curves and bars.
    model_keys_map: Dict[str, str] (default = {})
        A dictionary mapping model keys to model names.
    show_ci: bool (default = True)
        Whether bootstrapped confidence bands around curves should be shown.
    n_boot: int (default = 1000)
        Number of bootstrap iterations for generating confidence bands.
    bars_pos: Tuple[int, int, int, int]
        A tuple of four integers specifying the shape and position of the bar plot inset.
        (x position, y position, width, height)
    random_state: Seed (default = 1234)
        A seed for reproducibility.
    ax: plt.Axes (default = None)
        Optional Axes to plot curves onto.
    **kwargs:
        model_names_in_black: List[str]
            Names of models to show in black color, default is []

    Returns
    -------
    Tuple[plt.Axes, plt.Axes, matplotlib.container.BarContainer, Dict[str, Dict[str, float]]]
        First: the Axes containing the general plot.
        Second: the axes containing the bar plot inset.
        Third: the actual BarContainer of the bar plot inset.
        Fourth: A dictionary containing median (95%CI) area-under-curve over cross-validation
                for each model.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()
    
    rng = np.random.RandomState(random_state)
    auc_cis = dict()
    for j, (algo_name, outer_cv_data) in enumerate(cv_preds.items()):
        aucs = []

        gts = outer_cv_data.gts_val
        probas = outer_cv_data.probas_val

        for split_gts, split_probas in zip(gts, probas):
            auc_val = roc_auc_score(split_gts, split_probas)
            aucs.append(auc_val)

        fpr, tpr, thresholds = roc_curve(outer_cv_data.gts_val_conc,
                                         outer_cv_data.probas_val_conc)

        auc_low, auc_med, auc_up = np.percentile(aucs, [2.5, 50, 97.5])
        auc_cis[algo_name] = {"auc": auc_med,
                              "ci_low": auc_low, "ci_up": auc_up}

        ax.plot(fpr,
                tpr,
                linestyle='-',
                alpha=1.0,
                linewidth=2,
                color=colors[j])

        if show_ci:
            bootstrap_indices = list(scikits_boot_indices(data=outer_cv_data.probas_val_conc,
                                                          n_samples=n_boot,
                                                          seed=random_state))

            bootstrap_tprs = []
            for i in range(n_boot):
                sample_gt = outer_cv_data.gts_val_conc[bootstrap_indices[i]]
                sample_pred = outer_cv_data.probas_val_conc[bootstrap_indices[i]]

                bootstrap_fpr, bootstrap_tpr, bootstrap_thresholds = roc_curve(
                    sample_gt, sample_pred)

                # interpolate the bootstrapped fpr using the fpr based on accumulated
                # ground-truths and predicted probabilities
                interp_tpr = np.interp(fpr, bootstrap_fpr, bootstrap_tpr)
                interp_tpr[0] = 0.0

                bootstrap_tprs.append(interp_tpr)

            bootstrap_tprs = np.stack(bootstrap_tprs)

            tpr_lower = np.percentile(bootstrap_tprs, 2.5, axis=0)
            tpr_upper = np.percentile(bootstrap_tprs, 97.5, axis=0)

            ax.fill_between(fpr, tpr_lower, tpr_upper,
                            alpha=0.13,
                            color=colors[j])

    ax.set_xlabel('1 - Specificity',
                  fontdict={"weight": "normal", "size": 26},
                  labelpad=20)
    ax.set_ylabel('Sensitivity',
                  fontdict={"weight": "normal", "size": 26},
                  labelpad=20)

    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1,
            color='gray', label="Baseline", alpha=.8)

    ins = ax.inset_axes(bars_pos)
    ins.spines['top'].set_visible(False)
    ins.spines['right'].set_visible(False)
    ins.spines['bottom'].set_visible(False)

    ins.get_xaxis().set_ticks([])

    model_names = [model_keys_map.get(k, k) for k in cv_preds.keys()]
    model_aucs = [v["auc"] for _, v in auc_cis.items()]
    model_cis_low = np.array([v["auc"] - v["ci_low"]
                             for _, v in auc_cis.items()])
    model_cis_up = np.array([v["ci_up"] - v["auc"]
                            for _, v in auc_cis.items()])

    all_data = list(zip(model_names, model_aucs,
                    model_cis_low, model_cis_up, colors))
    all_data.sort(key=lambda x: x[1], reverse=False)

    model_names = [t[0] for t in all_data]
    model_aucs = [t[1] for t in all_data]
    model_cis_low = [t[2] for t in all_data]
    model_cis_up = [t[3] for t in all_data]
    model_colors = [t[4] for t in all_data]

    bars = ins.barh(range(len(model_names)), model_aucs,
                    xerr=[model_cis_low, model_cis_up],
                    align="center",
                    color=model_colors,
                    capsize=3,
                    error_kw=dict(linewidth=1)
                    )
    ins.invert_yaxis()
    ins.set_yticks(range(len(model_names)), labels=model_names, fontsize=16)

    model_names_in_black = kwargs.pop("model_names_in_black", [])
    for i, (name, auc, bar) in enumerate(zip(model_names, model_aucs, ins.patches)):
        ins.annotate(f"{auc:.2f}",
                     (bar.get_width(),
                      bar.get_y() + bar.get_height() / 2), ha='left', va='center',
                     size=20,
                     xytext=(0.04, i), color="#fff" if name not in model_names_in_black else "#000",
                     textcoords='data'
                     )

    ins.set_title("AUC (bars) and 95% CI (whiskers)",
                  fontsize=18,
                  fontweight="bold",
                  position=(0.4, 0.5))

    return fig, ax, ins, bars, all_data
