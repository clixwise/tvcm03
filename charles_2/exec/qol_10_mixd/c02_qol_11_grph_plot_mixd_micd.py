    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_11_grph_plot_ import PlotTranQOL_11_mixd_fore
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# https://gemini.google.com/app/51e2b25e86000839
# https://www.perplexity.ai/search/a-basic-prospective-veines-qol-rb1f3tm1SwOSmUKsVqFKbQ

def exec_plot_mixd_micd(plot_tran_mixd: PlotTranQOL_11_mixd_fore) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_mixd_fore

    trac = True

    # Data
    # ---- 
    '''
    '''
    df_orig = plot_tran_mixd.fram_dict["copi_resu_fram"] # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    df_fram = plot_tran_mixd.fram_dict["copi_resu_plot"] # "timepoint", "n", "mean", "ci_lower", "ci_upper", "modl_mean", "modl_ci_lower", "modl_ci_upper", ...
    df_orig = df_orig.copy()
    df_fram = df_fram.copy()
    if trac:
        print_yes(df_orig, labl="df_orig")
        print_yes(df_fram, labl="df_fram")

    # Exec
    # ----
    cols_requ = {"timepoint", "mean", "n", "ci_lower", "ci_upper"}
    if not cols_requ.issubset(df_orig.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_fram.columns)}")
    if not cols_requ.issubset(df_fram.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_fram.columns)}")
    #
    time_list = df_fram["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    x = np.arange(len(time_list))
    df_orig = df_orig.set_index("timepoint", drop=False) # -> column is not dropped
    df_fram = df_fram.set_index("timepoint", drop=False) # -> column is not dropped
    #
    if trac:
        print_yes(df_fram, labl="df_fram")  
  
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

    # Stra
    # ----
    # plot_tran_mixd.stra = 'a'
    match (plot_tran_mixd.stra):

        # ====
        case "a":  
        # ====
            y_pos = x
            ax.errorbar(
            df_orig["mean"],
                y_pos,
                xerr=[df_orig["mean"] - df_orig["ci_lower"], df_orig["ci_upper"] - df_orig["mean"]],
                fmt="o",
                color="black",
                ecolor="gray",
                capsize=4,
                markersize=6,
            )
            #
            ax.set_yticks(y_pos, df_orig["timepoint"])
            pass
        
        # ====
        case "b":  
        # ====
    
            mode = 'a' # 'a' is best
            match mode:
                case 'a':
                    y_pos = x

                    # Raw means - Plot this FIRST or give it a lower zorder
                    ax.scatter(
                        df_orig["mean"],
                        y_pos,
                        facecolors="none",
                        edgecolors="blue",
                        s=120,          # Increased size to "enclose" the black dot
                        linewidths=1.0, # Thinner line so it doesn't crowd the center
                        label="Raw mean",
                        zorder=2
                    )

                    # Modeled means - Plot this SECOND or give it a higher zorder
                    ax.errorbar(
                        df_fram["mean"],
                        y_pos,
                        xerr=[df_fram["mean"] - df_fram["ci_lower"], df_fram["ci_upper"] - df_fram["mean"]],
                        fmt="o",
                        color="black",
                        ecolor="gray",
                        capsize=4,
                        markersize=5,   # Slightly smaller to fit inside the blue ring
                        label="Modeled mean (95% CI)",
                        zorder=3        # Ensures it stays on top
                    )  
                    
                    # 4. Set Y-axis labels to T0, T1, T2
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(["T0", "T1", "T2"])

                    # Optional: Add a subtle horizontal line for each timepoint to guide the eye
                    for y in y_pos:
                        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, zorder=0)
                case 'b':
                    # 1. Setup the positions (0, 1, 2) for the 3 timepoints
                    y_pos = np.arange(len(df_fram)) 
                    offset = 0.1  # The "Dodge" amount to distinguish overlapping points

                    # 2. Modeled means (Black dot + Error bars)
                    # Shifted slightly down (- offset)
                    ax.errorbar(
                        df_fram["mean"],
                        y_pos - offset,
                        xerr=[df_fram["mean"] - df_fram["ci_lower"], df_fram["ci_upper"] - df_fram["mean"]],
                        fmt="o",
                        color="black",
                        ecolor="gray",
                        capsize=4,
                        markersize=6, # Slightly smaller for nesting
                        label="Modeled mean (95% CI)",
                        zorder=3
                    )

                    # 3. Raw means (Blue hollow ring)
                    # Shifted slightly up (+ offset)
                    ax.scatter(
                        df_orig["mean"],
                        y_pos + offset,
                        facecolors="none",
                        edgecolors="blue",
                        s=100,        # Slightly larger to "frame" the black dot if they meet
                        linewidths=1.5,
                        label="Raw mean",
                        zorder=2
                    )

                    # 4. Set Y-axis labels to T0, T1, T2
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(["T0", "T1", "T2"])

                    # Optional: Add a subtle horizontal line for each timepoint to guide the eye
                    for y in y_pos:
                        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, zorder=0)

                    # Keep the rest of your formatting (titles, labels, etc.) as is
                
                case _:
                    raise Exception()
        case _:
            raise Exception()
            
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