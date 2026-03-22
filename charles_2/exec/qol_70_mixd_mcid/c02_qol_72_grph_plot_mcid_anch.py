    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_72_grph_plot_ import PlotTranQOL_72_mcid_anch
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_mcid_anch(plot_tran_mixd: PlotTranQOL_72_mcid_anch) -> None: 
    # from qol_70_mixd_mcid.c02_qol_72_grph_plot_ import PlotTranQOL_72_mcid_anch

    trac = True

    # Data
    # ---- 
    df_plot_anch = plot_tran_mixd.fram_dict["plot_anch"] 
    df_resu_wide = plot_tran_mixd.fram_dict["resu_wide"] 
    df_resu_wide = df_resu_wide.copy()
    if trac:
        print_yes(df_plot_anch, labl="plot_anch")
        print_yes(df_resu_wide, labl="resu_wide")
        
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
    
    
        # ---------------------------------------------------------
    # PREPARE PLOT
    # ---------------------------------------------------------
    values = df_plot_anch["Value"].values
    labels = df_plot_anch["Label"].values

    # error bars
    ci_lower = values[0] - df_plot_anch.loc[0, "CI_low"]
    ci_upper = df_plot_anch.loc[0, "CI_high"] - values[0]

    yerr = np.array([[ci_lower, 0, 0, 0],
                    [ci_upper, 0, 0, 0]])

    # ---------------------------------------------------------
    # FIGURE
    # ---------------------------------------------------------
    x = np.arange(len(values))

    ax.bar(
        x,
        values,
        yerr=yerr,
        capsize=8,
        color=["#4C72B0"] + ["#55A868"]*3
    )

    ax.set_ylabel("MCID estimate (VEINES-QOL points)")
    ax.set_title("Anchor-based vs Distribution-based MCID")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)

    for i, v in enumerate(values):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=10)

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