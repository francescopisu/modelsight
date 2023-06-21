import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from src.modelsight._typing import ArrayLike


def ntile_name(n: int) -> str:
    """Returns the ntile name corresponding to an ntile integer.
    Parameters
    ----------
    n : int
        An ntile integer.
    Returns
    -------
    ntile_name : str
        The corresponding ntile name.
    """
    ntile_names = {
        4: 'Quartile',
        5: 'Quintile',
        6: 'Sextile',
        10: 'Decile',
        12: 'Duodecile',
        20: 'Vigintile',
        100: 'Percentile'
    }
    return ntile_names.get(n, f'{n}-tile')


def make_recarray(y_true: ArrayLike,
                  y_pred: ArrayLike) -> np.recarray:
    """Combines arrays into a recarray.
    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    Returns
    -------
    table : recarray
        A record array with observed label and predicted probability
        columns, sorted by predicted probability.
    """
    recarray = np.recarray(len(y_true), [('y_true', 'u8'), ('y_pred', 'f8')])
    recarray['y_true'] = y_true
    recarray['y_pred'] = y_pred
    recarray.sort(order='y_pred')
    return recarray


def hosmer_lemeshow_table(y_true: ArrayLike,
                          y_pred: ArrayLike,
                          n_bins: int = 10) -> np.recarray:
    """Constructs a Hosmer–Lemeshow table.
    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    n_bins : int, optional
        The number of groups to create. The default value is 10, which
        corresponds to deciles of predicted probabilities.
    Returns
    -------
    table : recarray
        A record array with `n_bins` rows and four columns: Group Size,
        Observed Frequency, Predicted Frequency, and Mean Probability.
    """
    if n_bins < 2:
        raise ValueError('Number of groups must be greater than or equal to 2')

    if n_bins > len(y_true):
        raise ValueError('Number of predictions must exceed number of groups')

    table = make_recarray(y_true, y_pred)

    table = [(len(g), g.y_true.sum(), g.y_pred.sum(), g.y_pred.mean())
             for g in np.array_split(table, n_bins)]
    names = ('group_size', 'obs_freq', 'pred_freq', 'mean_prob')
    table = np.rec.fromrecords(table, names=names)

    return table


def hosmer_lemeshow_plot(y_true: ArrayLike,
                         y_pred: ArrayLike,
                         n_bins: int = 10,
                         colors: Tuple[str, str] = ('blue', 'red'),
                         annotate_bars: bool = True,
                         title: str = "",
                         brier_score_annot: str = "",
                         ax: plt.Axes = None,
                         **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed vs. predicted probabilities (can be risks in case of survival models), groups by
    n-tiles of predicted probabilities.

    Parameters
    ----------
    y_true: ArrayLike
        (n_obs,) shaped array of ground-truth values
    y_pred: ArrayLike
        (n_obs,) shaped array of predicted probabilities
    n_bins: int
        Number of bins to group observed and predicted probabilities into
    colors: Tuple[str, str]
        Pair of colors for observed (line) and predicted (vertical bars) probabilities.
    annotate_bars: bool
        Whether bars should be annotated with the number of observed probabilities in each bin.
    title: str
        Title to display on top of the calibration plot.
    brier_score_annot: str
        Optional brier score (95% CI) annotation on the top-left corner.
    ax: plt.Axes
        A matplotlib Axes object to draw the calibration plot into. If None, an Axes object is created by default.
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]:
        Corresponding figure and Axes
    """
    table = hosmer_lemeshow_table(y_true, y_pred, n_bins)
    # transform observed and predicted frequencies in percentage relative to the bin dimension

    obs_freq = table.obs_freq
    pred_freq = table.pred_freq
    group_size = table.group_size

    trans_obs_freq = []
    trans_pred_freq = []
    for (gs, of, pf) in zip(group_size, obs_freq, pred_freq):
        trans_of = (of / gs) * 100
        trans_pf = (pf / gs) * 100

        trans_obs_freq.append(trans_of)
        trans_pred_freq.append(trans_pf)

    trans_obs_freq = np.array(trans_obs_freq)
    trans_pred_freq = np.array(trans_pred_freq)

    index = np.arange(n_bins)

    width = 0.9

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    barplot = ax.bar(index + 0.08, trans_obs_freq,
                     width,
                     color=colors[1],
                     label='Observed',
                     align="edge")

    if annotate_bars:
        for of, bar in zip(obs_freq, barplot.patches):
            ax.annotate(of,
                        (bar.get_x() + bar.get_width() / 2,
                         bar.get_height()), ha='center', va='center',
                        size=15, xytext=(-2, 10),
                        textcoords='offset points')

    ax2 = ax.twinx()

    line_points = [(index[i] + index[i + 1]) /
                   2 for i in range(len(index) - 1)] + [n_bins - 0.5]
    ax2.plot(line_points, trans_pred_freq / 100,
             color=colors[0], label='Predicted', marker="s")

    ax.set_xlabel('{} of Predicted Probabilities'.format(ntile_name(n_bins)),
                  fontsize=20)
    ax.set_ylabel('Observed: Proportion of Events (%)', fontsize=20)
    # observed probability: how many subjects have positive outcome over all subjects in the bin
    # this is the definition of probability
    # the model is well calibrated if the sum of predicted probability for subjects in each bin
    # is similar to the sum of target (y)

    ax2.set_ylabel('Predicted: ML score', color=colors[0], fontsize=20)
    ax2.yaxis.labelpad = 20

    ax.set_xticks(index, index)

    ax.set_ylim([0, 110])
    ax2.set_ylim([0, 1.10])

    plt.title(title, fontsize=20, pad=20)
    fig.legend(frameon=False, ncol=2, bbox_to_anchor=(0.72, -0.05),
               prop={"size": 18})

    new_xticks = np.append(ax.get_xticks(), n_bins)
    ax.set_xticks(new_xticks, new_xticks, fontsize=14)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 10
    ax.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    ax.text(0, 98, brier_score_annot, clip_on=False, fontsize=20)

    return ax.get_figure(), ax
