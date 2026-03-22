    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_01_grph_plot_ import PlotTranQOL_01_mean
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----
# Timepoint outcomes
# ----
'''
Timepoint                   T0             T1
Outcome
VCSS (max limb)  5.5 [4.8–6.2]  5.5 [5.2–5.8]
VEINES-QOL         56.0 ± 11.0     58.2 ± 4.5
VEINES-QOL         60.8 ± 12.2     62.5 ± 5.4
'''
def exec_plot_mean(plot_tran_mean: PlotTranQOL_01_mean) -> None:
    # from qol.c02_qol_01_grph_plot_ import PlotTranQOL_01_mean
    
    trac = True

    # Data
    # ---- 
    df_fram = plot_tran_mean.fram # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    if trac:
        print_yes(df_fram, labl="df_fram")
              
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_mean.figu
    ax = plot_tran_mean.axis
    ax.clear()
     
    # Grph
    # ----
    line_widt = plot_tran_mean.line_widt
    line_styl = plot_tran_mean.line_styl
    line_colo = plot_tran_mean.line_colo
    line_alph = plot_tran_mean.line_alph
    line_labl = plot_tran_mean.line_labl
    mark_size = plot_tran_mean.mark_size
    capp_size = plot_tran_mean.capp_size
    erro_colo = plot_tran_mean.erro_colo
    match (plot_tran_mean.stra):
        case 'a': # default
            ax.errorbar(df_fram["timepoint"], df_fram["mean"], yerr=1.96 * df_fram["se"], fmt=line_styl, color=line_colo, alpha=line_alph, linewidth=line_widt, markersize=mark_size, capsize=capp_size, ecolor=erro_colo, label=line_labl)
            # Add sample size annotations just below each CI lower bound
            offset = 0.5
            for i, row in df_fram.iterrows():
                ax.text(
                    x=row["timepoint"],
                    y=row["ci_lower"] - offset,
                    s=f"n = {int(row['n'])}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
            pass
        case _:
            raise Exception()  

    # Axis, Grid
    # ----
    xlab = plot_tran_mean.xlab # "Timepoint"
    ax.set_xlabel(xlab)
    ax.margins(x=0.10)  # x-axis padding
    #
    tick_dict = plot_tran_mean.tick_dict # {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
    labl_list = [tick_dict.get(cat, cat) for cat in df_fram["timepoint"].cat.categories]
    ax.set_xticks(df_fram["timepoint"].cat.categories)
    ax.set_xticklabels(labl_list)

    ylab = plot_tran_mean.ylab # "VEINES-QOL score"
    ax.set_ylabel(ylab)
    #
    ymin = df_fram["ci_lower"].min() - 2
    ymax = df_fram["ci_upper"].max() + 2
    ax.set_ylim(ymin, ymax)
    
    grid_alph = plot_tran_mean.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # Lgnd
    # ---- 
    ax.legend()     
         
    # Titl
    # ----
    titl = plot_tran_mean.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    fig.tight_layout()
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
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass