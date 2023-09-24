"""
This file deals with the implementation of functions that allow annotating plots 
with statistical tests results between pairs of estimators.
"""

from typing import Callable, Dict, Tuple, List
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt

from modelsight.curves._delong import delong_roc_test
from modelsight._typing import CVModellingOutput

def annot_stat_vertical(text:str, 
                        x: float, 
                        y1: float, y2: float, 
                        ww: float = 0.02, 
                        col: str = 'black', 
                        fontsize: int = 13, 
                        voffset: float = 0, 
                        n_elems: int = None,
                        ax=None,
                        **kwargs):
    """
    Draw a vertical whisker at position `x` that spans through `y1` to `y2` with annotation specified
    by `text`.

    Parameters
    ----------
    text : str
        Annotation for whisker.
    x : float
        x-position the whisker is positioned at.
    y1 :float
        starting y position.
    y2 : float
        ending y position.
    ww : float, optional
        whisker width, by default 0.02
    col : str, optional
        whisker color, by default 'black'
    fontsize : int, optional
        fontsize for the annotation, by default 13
    voffset : float, optional
        vertical offset for the annotation, by default 0.
        Some font families and characters occupy different vertical spaces; 
        this parameter allows compensating for such variations.
    n_elems : int, optional
        number of discrete elements in the y-axis, by default None.
        This value is precomputed by the caller (add_annotations) and passed
        to this function as input.
    ax : plt.Axes, optional
        a pyplot Axes to draw annotations on, by default None
    **kwargs
        rect_h_base: float, optional
            base height of rectangle patch for single-character annotations, by default 0.1
        fontsize_nonsignif, optional
            fontsize for multi-character annotations (here called non significant annotations
            to reflect the fact that single-character annotations most often use some kind
            of symbol to denote statistical significance, e.g. *), by default `fontsize` (i.e., 13)
    """
    ax = plt.gca() if ax is None else ax
    
    # we want the text to be centered on the whisker 
    text_x_pos = x + ww 

    text_y_pos = (y1+y2)/2
    
    # draw whisker from y1 to y2 with width `ww`
    ax.plot([x, x + ww, x + ww, x], [y1, y1, y2, y2], lw=1, c=col)
    
    # this is the case of a whisker being annotated with a single character.
    # by default, symbols do not enforce a white background, hence when
    # superimposed on whiskers the readibility is limited.
    # here we enforce a white rectangle patch beneath the symbol to enhance
    # readibility of annotations.
    # the built-in bbox parameter of pyplot's .text() doesn't produce
    # acceptable results, hence we came up with a custom implementation for
    # single-character annotations.
    if len(text) == 1:
        # draw text at (text_x_pos, (text_y_pos - voffset) + 0.17)
        ax.text(
            text_x_pos, (text_y_pos - voffset) + 0.17, text, 
            ha='center', va='center', color=col,
            size=fontsize, zorder=10
        )

        # Rectangle's props
        rect_h_base = kwargs.get("rect_h_base", 0.1)
        rect_w = 0.05 - (0.375 * 0.05) # on a scale from 0 to 1
        rect_h = rect_h_base * n_elems # transform to scale from 0 to n_elems-1
        rect_x_offset = -0.002
        rect_y_offset = 0.01 # move rectangle to the bottom. (0,0) is top left in the inserted barplot
        
        # draw white rectangle and put it beneath the text 
        # specifying a zorder inferior to that of the text
        rect = patches.Rectangle(
            (
                text_x_pos - (rect_w/2) + rect_x_offset, 
                text_y_pos - (rect_h/2) + rect_y_offset
            ),
            width = rect_w, height = rect_h, 
            linewidth=1, 
            edgecolor='w', 
            facecolor='w',
            zorder=9
        )
        
        ax.add_patch(rect)
    else:
        # this is the case of multi-character annotations.
        # here, we leverage the built-in bbox of pyplot's text method
        # that allows drawing a bounding box beneath the annotation.
        fontsize_nonsignif = kwargs.pop("fontsize_nonsignif", fontsize)
        ax.text(
            text_x_pos, text_y_pos, text, 
            ha='center', va='center', color=col,
            size=fontsize_nonsignif, zorder=10,
            bbox=dict(
                boxstyle='square,pad=0', 
                facecolor="white", 
                edgecolor="white"
            )
        )        
        

def annot_stat_horizontal(text: str, 
                          x1: float, x2: float, 
                          y: float, 
                          wh: float = 0.02, 
                          col: str = "black", 
                          fontsize: int = 13, 
                          voffset: float = 0, 
                          n_elems:int  = None,
                          ax: plt.Axes = None,
                          **kwargs):
    """
    Draw an horizontal whisker at position `y` that spans through `x1` to `x2` with annotation specified
    by `text`.

    Parameters
    ----------
    text : str
        Annotation for whisker.
    x1 : float
        starting x position.
    x2 :float
        ending x position.
    y : float
        y-position the whisker is positioned at.
    wh : float, optional
        whisker height, by default 0.02
    col : str, optional
        whisker color, by default 'black'
    fontsize : int, optional
        fontsize for the annotation, by default 13
    voffset : float, optional
        vertical offset for the annotation, by default 0.
        Some font families and characters occupy different vertical spaces; 
        this parameter allows compensating for such variations.
    n_elems : int, optional
        number of discrete elements in the y-axis, by default None.
        This value is precomputed by the caller (add_annotations) and passed
        to this function as input.
    ax : plt.Axes, optional
        a pyplot Axes to draw annotations on, by default None
    **kwargs
        fontsize_nonsignif, optional
            fontsize for multi-character annotations (here called non significant annotations
            to reflect the fact that single-character annotations most often use some kind
            of symbol to denote statistical significance, e.g. *), by default `fontsize` (i.e., 13)
    """
    ax = plt.gca() if ax is None else ax
    
    # we want the text to be centered on the whisker 
    text_y_pos = y + wh
    #+ 0.01 
    text_x_pos = (x1+x2)/2
    
    # draw whisker from y1 to y2 with width `ww`
    ax.plot([x1, x1, x2, x2], [y, y + wh, y + wh, y], lw=1, c=col,
           clip_on=False)

    # this is the case of a whisker being annotated with a single character.
    # by default, symbols do not enforce a white background, hence when
    # superimposed on whiskers the readibility is limited.
    # here we enforce a white rectangle patch beneath the symbol to enhance
    # readibility of annotations.
    # the built-in bbox parameter of pyplot's .text() doesn't produce
    # acceptable results, hence we came up with a custom implementation for
    # single-character annotations.    
    if len(text) == 1:
        # draw text at (text_x_pos, text_y_pos) # + 0.15
        ax.text(
            text_x_pos, text_y_pos + voffset, text, 
            ha='center', va='center', color=col,
            size=fontsize, zorder=10
        )

        # Rectangle's props
        rect_w = 0.09 # transform to scale from 0 to n_elems-1        
        rect_h = 0.05 - (0.375 * 0.05) # on a scale from 0 to 1
        rect_x_offset = 0.005
        rect_y_offset = -0.001 # move rectangle to the bottom. (0,0) is top left in the inserted barplot
        
        # draw white rectangle and put it beneath the text 
        # specifying a zorder inferior to that of the text
        rect = patches.Rectangle(
            (
                text_x_pos - (rect_w/2) + rect_x_offset, 
                text_y_pos - (rect_h/2) + rect_y_offset
            ),
            width = rect_w, height = rect_h, 
            linewidth=1, 
            edgecolor='w', 
            facecolor='w',
            zorder=9,
            clip_on=False
        )
        
        ax.add_patch(rect)
    else:
        fontsize_nonsignif = kwargs.pop("fontsize_nonsignif", fontsize)
        ax.text(
            text_x_pos, text_y_pos, text, 
            ha='center', va='center', color=col,
            size=fontsize_nonsignif, zorder=10,
            bbox=dict(
                boxstyle='square,pad=0', 
                facecolor="white", 
                edgecolor="white"
            )
        )        
        
        
def add_annotations(comparisons: Dict[str, Tuple[str, str, float]], 
                    alpha: float, 
                    bars: matplotlib.container.BarContainer,
                    direction: str,
                    order: List[Tuple[str, str]],
                    symbol: str = "*",
                    symbol_fontsize: int = 22,
                    voffset: float = 0,
                    ext_voffset: float = 0,
                    ext_hoffset: float = 0,
                    P_val_rounding: int = 2,
                    ax: plt.Axes = None,
                    **kwargs):
    """
    Annotates the specified plot (`ax`) with the provided comparisons results either vertically or horizontally
    depending on the value of `direction`.

    Parameters
    ----------
    comparisons : Dict[str, Tuple[str, str, float]]
        The results of models comparisons.
    alpha : float
        The significance level used for formatting the P value of comparisons.
    bars : matplotlib.container.BarContainer
        A list of matplotlib's bars that is used to access the bar's width or height
        when annotating horizontally and vertically, respectively.
    direction : str
        The direction for annotation. Possible values are "horizontal" and "vertical".
    order : List[Tuple[str, str]]
        The order in which the comparisons should be displayed.
        Each entry of this list is a tuple where elements are algorithm's names.
    symbol : str, optional
        The symbol used in place of the P value when statistical significance is achieved
        accoring to the specified alpha, by default "*".
    symbol_fontsize : int, optional
        Fontsize for the symbol used when statistical significance is achieved, by default 22
    voffset : float, optional
        vertical offset for the annotation, by default 0., by default 0
    ext_voffset : float, optional
        Additional vertical offset for vertical annotations.
        Ignored when direction = "horizontal", by default 0
    ext_hoffset : float, optional
        Additional horizontal offset for horizontal annotations.
        Ignored when direction = "vertical", by default 0
    P_val_rounding : int, optional
        Number of decimal places to round P values at, by default 2
    ax : plt.Axes, optional
        The plot to be annotated, by default None

    Returns
    -------
    ax : plt.Axes
        The annotated plot.

    Raises
    ------
    ValueError
        When ax is None
    ValueError
        Whenever a comparison key doesn't exist.
    """
    if not ax:
        raise ValueError("I need an Axes to draw comparisons on.")
    
    comparisons_list = []
    if order:
        for fst_algo, snd_algo in order:
            cmp_key = f"{fst_algo}_{snd_algo}"
            cmp = comparisons.get(cmp_key, None)
            if not cmp:
                raise ValueError(f"The comparison {cmp_key} does not exist in the order list.")
            comparisons_list.append(cmp)
    else:
        comparisons_list = list(comparisons.values())
        
    
    if direction == "horizontal":
        width = bars[0].get_width()
        entity_labels = ax.get_xticklabels()
        entity_idx = {label.get_text(): (i + 0.03) for i, label in enumerate(entity_labels)}
        
        whisker_y_offset = kwargs.pop("whisker_y_offset", 0)
        y_lim_upper = ax.get_ylim()[1] + 0.05 + whisker_y_offset
        v_offset = 0.07

        for i, (fst_model, snd_model, P) in enumerate(comparisons_list):
            P_str = symbol if P <= alpha else f"{P:.{P_val_rounding}f}"
            annot_stat_horizontal(text=P_str, 
                                x1=entity_idx[fst_model] + width/2, 
                                x2=entity_idx[snd_model] + width/2, 
                                y=(y_lim_upper - 0.17) + (i * v_offset), # overall distance from top of bars and upper limit of y + inter-distance between whiskers
                                wh=0.02,
                                col="black", 
                                fontsize=symbol_fontsize,
                                voffset = voffset, #-0.02
                                ext_offset = ext_hoffset,
                                n_elems = len(entity_labels),
                                ax=ax,
                                **kwargs)
    elif direction == "vertical":
        height = bars[0].get_height()
        entity_labels = ax.get_yticklabels()
        entity_idx = {label.get_text(): (i + 0.03) for i, label in enumerate(entity_labels)}

        space_between_whiskers = kwargs.pop("space_between_whiskers", 0)
        x_lim_upper = ax.get_xlim()[1] + 0
        h_offset = 0.07 + space_between_whiskers

        for i, (fst_model, snd_model, P) in enumerate(comparisons_list):
            P_str = symbol if P <= alpha else f"{P:.{P_val_rounding}f}"
            annot_stat_vertical(text=P_str,
                                x=x_lim_upper + (i * h_offset),  
                                y1=entity_idx[fst_model], 
                                y2=entity_idx[snd_model],                                  
                                ww=0.02,
                                col="black", 
                                fontsize=symbol_fontsize if P_str == "*" else 16,
                                voffset=voffset, 
                                ext_offset = ext_voffset,
                                n_elems = len(entity_labels),
                                ax=ax,
                                **kwargs)        
    
    return ax
        
def roc_single_comparison(cv_preds: CVModellingOutput, 
                          fst_algo: str, 
                          snd_algo: str) -> Dict[str, Tuple[str, str, float]]:
    """Perform a single comparison of two areas under Receiver Operating Characteristic curves
    computed on the same set of data points by the DeLong test.

    Parameters
    ----------
    cv_preds : CVModellingOutput
        The output of a cross-validation process encompassing mulitple (n>=2) models.
    fst_algo : str
        The name of the first algorithm for the comparison.
        Must be an existing key of `cv_preds`.
    snd_algo : str
        The name of the second algorithm for the comparison.
        Must be an existing key of `cv_preds`.

    Returns
    -------
    comparison_result : Dict[str, Tuple[str, str, float]]
        The output of the comparison. This is a dictionary where the key is
        of the form "<fst_algo>_<snd_algo>" and the value is a tuple of three
        elements, the first two are the names of the algorithms being compared
        and the third element is the P value for the null hypothesis that
        the two AUC values are equal.
    """
    ground_truths = cv_preds[fst_algo].gts_val_conc
    fst_algo_probas = cv_preds[fst_algo].probas_val_conc
    snd_algo_probas = cv_preds[snd_algo].probas_val_conc
    
    P = delong_roc_test(ground_truths, fst_algo_probas, snd_algo_probas)
    cmp_key = f"{fst_algo}_{snd_algo}"
    comparison_result = {cmp_key: (fst_algo, snd_algo, P)}
    return comparison_result
        
def roc_comparisons(cv_preds: CVModellingOutput, 
                    target_algo: str):
    """
    Compares the AUC of the specified algorithm with the AUCs of all other algorithms.

    Parameters
    ----------
    cv_preds : CVModellingOutput
        The output of a cross-validation process encompassing mulitple (n>=2) models.
    target_algo : str
        The name of the target algorithm's whose AUC will be compared with all other AUCs.

    Returns
    -------
    comparisons : Dict[str, Tuple[str, str, float]]
        A dictionary containing the results of all comparisons. See output of `roc_single_comparison`.
    """
    comparisons = dict()

    for algo_name in cv_preds.keys():
        if algo_name != target_algo:
            cmp = roc_single_comparison(cv_preds, target_algo, algo_name)
            comparisons = dict(cmp, **comparisons)
            
    return comparisons     