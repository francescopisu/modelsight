from typing import Callable, Dict, Tuple, List
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import average_precision_score

from src.modelsight.curves._delong import delong_roc_test

def annot_stat_vertical(text, x, y1, y2, ww, 
                        col='k', 
                        fontsize=13, 
                        voffset = 0, 
                        n_elems = None,
                        ax=None,
                        **kwargs):
    """
    ww: float
        whisker width
    """
    ax = plt.gca() if ax is None else ax
    
    # we want the text to be centered on the whisker 
    text_x_pos = x + ww 
    #+ 0.01
    text_y_pos = (y1+y2)/2
    
    # draw whisker from y1 to y2 with width `ww`
    ax.plot([x, x + ww, x + ww, x], [y1, y1, y2, y2], lw=1, c=col)
    
    if len(text) == 1:                
        #text_y_pos = (y1+y2)/2

        # draw text at (text_x_pos, text_y_pos) # + 0.15
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
        
from matplotlib import patches
def annot_stat_horizontal(text, x1, x2, y, wh, col='k', fontsize=13, 
                        voffset = 0, 
                        n_elems = None,
                        ax=None,
                        **kwargs):
    """
    ww: float
        whisker width
    """
    ax = plt.gca() if ax is None else ax
    
    # we want the text to be centered on the whisker 
    text_y_pos = y + wh
    #+ 0.01 
    text_x_pos = (x1+x2)/2
    
    # draw whisker from y1 to y2 with width `ww`
    ax.plot([x1, x1, x2, x2], [y, y + wh, y + wh, y], lw=1, c=col,
           clip_on=False)
    
    if len(text) == 1:     
        #text_y_pos = (y1+y2)/2

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
        
        
from typing import Tuple, List, Dict

def add_annotations(comparisons: Dict[str, Tuple[str, str, float]], 
                    alpha: float, 
                    bars: matplotlib.container.BarContainer,
                    direction: str,
                    order: List[Tuple[str, str]],
                    symbol: str,
                    symbol_fontsize: int = 22,
                    voffset: float = 0,
                    ext_voffset: float = 0,
                    ext_hoffset: float = 0,
                    P_val_rounding: int = 2,
                    ax: plt.Axes = None,
                    **kwargs):
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
                                voffset = -0.02,
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
        
def roc_single_comparison(cv_preds, fst_algo, snd_algo):
    ground_truths = cv_preds[fst_algo].gts_val_conc
    fst_algo_probas = cv_preds[fst_algo].probas_val_conc
    snd_algo_probas = cv_preds[snd_algo].probas_val_conc
    
    P = delong_roc_test(ground_truths, fst_algo_probas, snd_algo_probas)
    cmp_key = f"{fst_algo}_{snd_algo}"
    return {cmp_key: (fst_algo, snd_algo, P)}
        
def roc_comparisons(cv_preds, target_algo):
    comparisons = dict()

    for algo_name in cv_preds.keys():
        if algo_name != target_algo:
            cmp = roc_single_comparison(cv_preds, target_algo, algo_name)
            comparisons = dict(cmp, **comparisons)
            
    return comparisons     