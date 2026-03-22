    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_52_grph_plot_ import PlotTranQOL_52_cohe
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Main: https://chatgpt.com/c/699b4784-2cc0-832b-959b-6da55f29dd7a
# Annx: https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# Effect Size Plot (Cohen’s d with CI)

def plot_cohe(plot_tran_mixd: PlotTranQOL_52_cohe) -> None: 
    # from qol_70_mixd_mcid.c02_qol_52_grph_plot_ import PlotTranQOL_52_cohe

    trac = True

    # Data
    # ---- 
    '''
    '''
    df_fram = plot_tran_mixd.fram_dict["cohe_fram"] 
    df_fram = df_fram.copy()
    if trac:
        print_yes(df_fram, labl="df_fram:cohe_fram")

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
    
    # Labels (cleaned for display)
    df_fram = df_fram.set_index("contrast")
    labels = (
        df_fram.index
        #.str.replace("_minus_", " – ")
        #.str.replace("_", " ")
        .tolist()
    )
    print_yes(df_fram, labl="df_fram")
    # plot_tran_mixd.stra = 'a'
    match (plot_tran_mixd.stra):

        # ====
        case "a":
        # ====

            # ----
            # Data from df_fram (cohe_fram)
            # ----
            df_orig = df_fram.copy()

            d_values = df_orig["cohen_d"].values
            ci_low   = df_orig["cohen_d_ci_low"].values
            ci_high  = df_orig["cohen_d_ci_high"].values

            # Compute asymmetric error bars
            err_low  = d_values - ci_low
            err_high = ci_high - d_values

            y_pos = np.arange(len(labels))

            # ----
            # Plot effect sizes with CI
            # ----
            ax.errorbar(
                d_values,
                y_pos,
                xerr=[err_low, err_high],
                fmt='o',
                color='navy',
                ecolor='gray',
                elinewidth=2,
                capsize=5,
                label="Cohen's d"
            )

            # ----
            # Reference lines (Cohen benchmarks)
            # ----
            ax.axvline(0,   color='black', linestyle='--', linewidth=1)
            ax.axvline(0.2, color='gray',  linestyle=':',  linewidth=1)
            ax.axvline(0.5, color='gray',  linestyle=':',  linewidth=1)
            ax.axvline(0.8, color='gray',  linestyle=':',  linewidth=1)

            # ----
            # Y-axis labels
            # ----
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)

            # Optional: invert so T1–T0 appears at top
            ax.invert_yaxis()

            # X-axis label specific to Cohen
            ax.set_xlabel("Cohen's d (bootstrap 95% CI)")
            pass

        # ====
        case "b":
        # ====

            df_orig = df_fram.copy()

            d_values = df_orig["cohen_d"].values
            ci_low   = df_orig["cohen_d_ci_low"].values
            ci_high  = df_orig["cohen_d_ci_high"].values
            interpretations = df_orig["interpretation"].values

            # ----
            # Asymmetric CI
            # ----
            err_low  = d_values - ci_low
            err_high = ci_high - d_values

            y_pos = np.arange(len(labels))

            # ----
            # Color mapping by interpretation
            # ----
            color_map = {
                "small":  "#4C72B0",   # blue
                "medium": "#55A868",   # green
                "large":  "#C44E52",   # red
            }

            colors = [
                color_map.get(interp.lower(), "black")
                for interp in interpretations
            ]

            # ----
            # Plot each point individually (for color control)
            # ----
            for i in range(len(d_values)):
                ax.errorbar(
                    d_values[i],
                    y_pos[i],
                    xerr=[[err_low[i]], [err_high[i]]],
                    fmt='o',
                    color=colors[i],
                    ecolor='gray',
                    elinewidth=2,
                    capsize=5,
                )

                # ----
                # Numeric annotation
                # ----
                ax.text(
                    d_values[i] + 0.05 * np.sign(d_values[i] if d_values[i] != 0 else 1),
                    y_pos[i],
                    f"{d_values[i]:.2f}",
                    va='center',
                    fontsize=9,
                    color=colors[i],
                )

            # ----
            # Reference lines (Cohen benchmarks)
            # ----
            ax.axvline(0,   color='black', linestyle='--', linewidth=1)
            ax.axvline(0.2, color='gray',  linestyle=':',  linewidth=1)
            ax.axvline(0.5, color='gray',  linestyle=':',  linewidth=1)
            ax.axvline(0.8, color='gray',  linestyle=':',  linewidth=1)

            # ----
            # Y-axis formatting
            # ----
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()

            # ----
            # Symmetric auto x-limits
            # ----
            max_abs = np.max(np.abs(np.concatenate([ci_low, ci_high])))
            margin = 0.15 * max_abs
            ax.set_xlim(-max_abs - margin, max_abs + margin)

            ax.set_xlabel("Cohen's d (bootstrap 95% CI)")    
    
        # ====
        case "c":
        # ====

            df_orig = df_fram.copy()

            # ----
            # Labels (formatted for publication)
            # ----

            d_values = df_orig["cohen_d"].values
            ci_low   = df_orig["cohen_d_ci_low"].values
            ci_high  = df_orig["cohen_d_ci_high"].values

            err_low  = d_values - ci_low
            err_high = ci_high - d_values

            y_pos = np.arange(len(labels))

            # ----
            # Plot (monochrome, journal style)
            # ----
            ax.errorbar(
                d_values,
                y_pos,
                xerr=[err_low, err_high],
                fmt='o',
                color='black',
                ecolor='black',
                elinewidth=1.2,
                capsize=4,
                markersize=5,
            )

            # ----
            # Numeric annotations (subtle)
            # ----
            for i, d in enumerate(d_values):
                ax.text(
                    d + 0.04,
                    y_pos[i],
                    f"{d:.2f}",
                    va='center',
                    fontsize=9,
                )

            # ----
            # Reference line at 0 (essential)
            # ----
            ax.axvline(0, color='black', linestyle='--', linewidth=1)

            # Optional: remove 0.2/0.5/0.8 lines for journal minimalism
            # Journals typically prefer explanation in caption instead

            # ----
            # Axis formatting
            # ----
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()

            # Symmetric scaling centered at 0
            max_abs = np.max(np.abs(np.concatenate([ci_low, ci_high])))
            margin = 0.10 * max_abs
            ax.set_xlim(-max_abs - margin, max_abs + margin)

            ax.set_xlabel("Cohen’s d (95% bootstrap CI)")
            
        # ====
        case "d":
        # ====

            df_orig = df_fram.copy()

            # ----
            # Labels
            # ----

            d_values = df_orig["cohen_d"].values
            ci_low   = df_orig["cohen_d_ci_low"].values
            ci_high  = df_orig["cohen_d_ci_high"].values

            y_pos = np.arange(len(labels))

            # ----
            # Symmetric x-limits (centered at 0)
            # ----
            max_abs = np.max(np.abs(np.concatenate([ci_low, ci_high])))
            margin = 0.15 * max_abs
            xmin = -max_abs - margin
            xmax =  max_abs + margin
            ax.set_xlim(xmin, xmax)

            # ----
            # Plot CIs (horizontal lines)
            # ----
            for i in range(len(d_values)):
                ax.hlines(
                    y=y_pos[i],
                    xmin=ci_low[i],
                    xmax=ci_high[i],
                    color='black',
                    linewidth=1.5
                )

            # ----
            # Plot point estimates (squares)
            # ----
            ax.scatter(
                d_values,
                y_pos,
                marker='s',
                color='black',
                s=40,
                zorder=3
            )

            # ----
            # Vertical reference line (no effect)
            # ----
            ax.axvline(0, color='black', linestyle='--', linewidth=1)

            # ----
            # Right-side numeric column (no overlap)
            # ----
            text_x = xmax + 0.02 * (xmax - xmin)

            for i in range(len(d_values)):
                ax.text(
                    text_x,
                    y_pos[i],
                    f"{d_values[i]:.2f}  [{ci_low[i]:.2f}, {ci_high[i]:.2f}]",
                    va='center',
                    fontsize=9
                )

            # Expand right margin to fit text column
            ax.set_xlim(xmin, xmax + 0.35 * (xmax - xmin))

            # ----
            # Y-axis formatting
            # ----
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()

            # ----
            # Clean forest style
            # ----
            ax.set_xlabel("Cohen’s d")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            
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