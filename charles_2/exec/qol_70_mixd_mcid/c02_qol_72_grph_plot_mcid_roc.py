    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_72_grph_plot_ import PlotTranQOL_72_mcid_list
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import roc_curve, auc

# Main: https://chatgpt.com/c/699b4784-2cc0-832b-959b-6da55f29dd7a
# Annx: https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# Effect Size Plot (Cohen’s d with CI)

def plot_mcid_roc(plot_tran_mixd: PlotTranQOL_72_mcid_list) -> None: 
    # from qol_70_mixd_mcid.c02_qol_72_grph_plot_ import PlotTranQOL_72_mcid_list

    trac = True

    # Data
    # ---- 
    '''
    '''
    df_with_data = plot_tran_mixd.fram_dict["plot_roc_data"]
    meta = plot_tran_mixd.fram_dict["plot_roc_meta"]  
    df_resu_wide = plot_tran_mixd.fram_dict["resu_wide"] 
    df_resu_wide = df_resu_wide.copy()
    if trac:
        print_yes(df_with_data, labl="plot_roc_data")
        print(meta)

    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_mixd.figu
    ax = plot_tran_mixd.axis
    ax.clear()
     
    # Grph
    # ----
    line_widt = plot_tran_mixd.line_widt
    line_styl = plot_tran_mixd.line_styl
    line_colo = plot_tran_mixd.line_colo
    line_alph = plot_tran_mixd.line_alph
    line_labl = plot_tran_mixd.line_labl
    mark_size = plot_tran_mixd.mark_size
    mark_widt = plot_tran_mixd.mark_widt
    mark_colo = plot_tran_mixd.mark_colo
    capp_size = plot_tran_mixd.capp_size
    erro_colo = plot_tran_mixd.erro_colo

  
    """
    Plots the ROC curve and highlights the Youden Index point.
    """
    
    # Plot the ROC curve
    ax.plot(df_with_data['fpr'], df_with_data['tpr'], 
            color='#4C72B0', lw=2, label=f"ROC Curve (AUC = {meta['auc']:.2f})")
    
    # Plot the random chance baseline
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    
    # Highlight the Youden point (Best Threshold)
    ax.scatter(meta['best_fpr'], meta['best_tpr'], 
            color='red', s=100, edgecolors='black', zorder=5,
            label=f"Best Threshold: {meta['threshold']:.2f}")
    
    # Annotate the point with Sensitivity and Specificity
    ax.annotate(f"Sens: {meta['best_tpr']:.2f}\nSpec: {1-meta['best_fpr']:.2f}",
                xy=(meta['best_fpr'], meta['best_tpr']),
                xytext=(meta['best_fpr'] + 0.05, meta['best_tpr'] - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    # Aesthetics
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve: Anchor-based MCID (Youden Method)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
            
    # Axis, Grid
    # ----
    xlab = plot_tran_mixd.xlab # "Timepoint"
    ax.set_xlabel(xlab)
    ax.margins(x=0.10)  # x-axis padding
    #
    #tick_dict = plot_tran_mixd.tick_dict # {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
    #labl_list = [tick_dict.get(cat, cat) for cat in time_list]
    #ax.set_xticks(x)               # numeric positions
    #ax.set_xticklabels(labl_list)  # readable labels

    ylab = plot_tran_mixd.ylab # "VEINES-QOL score"
    ax.set_ylabel(ylab)
    
    grid_alph = plot_tran_mixd.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # Lgnd
    # ---- 
    ax.legend()     
         
    # Titl
    # ----
    titl = plot_tran_mixd.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    # fig.tight_layout(): See 'FiguTran' initiaization : self.fig = plt.figure(layout="constrained")
    # fig.show()
    pass
    
def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        if isinstance(df, pd.DataFrame):
            print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
            print(df.info())
        elif isinstance(df, pd.Series):
            print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}")
            print(df.info())
    pass