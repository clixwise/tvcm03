    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_vcss_01_grph_plot_ import PlotTranVCSS_01_scat
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker

# ----
# Timepoint outcomes
# ----
# https://gemini.google.com/app/67b6d6f473954249
def exec_plot_scat(plot_tran_pure: PlotTranVCSS_01_scat) -> None:
    # from vcss.c02_vcss_01_grph_plot_ import PlotTranVCSS_01_scat
    
    trac = True

    # Data
    # ---- 
    df_fram = plot_tran_pure.fram # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    if trac:
        print_yes(df_fram, labl="df_fram")
              
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Data
    # ----
    colu_list = plot_tran_pure.colu_list
    # Ensure numeric types for plotting
    non_float_rows = df_fram[~df_fram[colu_list].apply(lambda x: pd.to_numeric(x, errors='coerce').notna().all(), axis=1)]
    if len(non_float_rows) > 0:
        print(non_float_rows)
        raise Exception()
    #
    data_ser1 = df_fram[colu_list[0]]
    data_ser1 = pd.to_numeric(data_ser1, errors='coerce').astype('Int64')
    # print (data_ser1)
    #
    data_ser2 = df_fram[colu_list[1]]
    data_ser2 = pd.to_numeric(data_ser2, errors='coerce').astype('Int64')
    # print (data_ser2)
    max_score = max(data_ser1.max(), data_ser2.max())

    # Figu
    # ----
    fig = plot_tran_pure.figu
    ax = plot_tran_pure.axis
    ax.clear()

    # Grph : scat
    # ----
    scat_size = plot_tran_pure.scat_size
    scat_edge = plot_tran_pure.scat_edge
    scat_colo = plot_tran_pure.scat_colo
    scat_alph = plot_tran_pure.scat_alph
    match (plot_tran_pure.stra):
        case 'a': # default
            ax.scatter(data_ser1, data_ser2, alpha=scat_alph, s=scat_size, c=scat_colo, edgecolors=scat_edge)
            pass
        case 'b': # add a small random offset to each point so ties separate visually.
            jitter = 0.15  # smaller than 0.5 to preserve integer perception
            x = data_ser1 + np.random.uniform(-jitter, jitter, size=len(data_ser1))
            y = data_ser2 + np.random.uniform(-jitter, jitter, size=len(data_ser2))
            ax.scatter(x, y, alpha=scat_alph, s=scat_size, c=scat_colo, edgecolors=scat_edge)
            pass
        case 'c1': # point size (or color) proportional to number of ties.
            # Count ties
            counts = (plot_tran_pure.fram[colu_list].dropna().astype(int).value_counts().reset_index(name='count'))
            x = counts[colu_list[0]]
            y = counts[colu_list[1]]
            sizes = counts['count'] * 40  # scaling factor
            sc = ax.scatter(x, y, s=sizes, c=counts['count'], cmap='viridis', edgecolors=scat_edge, alpha=scat_alph)
            #
            plt.colorbar(sc, ax=ax, label='Number of patients')
            pass
        case 'c2': # point size (or color) proportional to number of ties.
            # Count ties
            counts = (plot_tran_pure.fram[colu_list].dropna().astype(int).value_counts().reset_index(name='count'))
            x = counts[colu_list[0]]
            y = counts[colu_list[1]]
            sizes = counts['count'] * 40  # scaling factor
            sc = ax.scatter(x, y, s=sizes, c=counts['count'], cmap='viridis', edgecolors=scat_edge, alpha=scat_alph)
            #
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Number of patients')
            cbar.locator = MaxNLocator(integer=True)
            cbar.update_ticks()
            pass
        case 'c3': # (best) : point size (or color) proportional to number of ties.
            # Count ties
            counts = (plot_tran_pure.fram[colu_list].dropna().astype(int).value_counts().reset_index(name='count'))
            x = counts[colu_list[0]]
            y = counts[colu_list[1]]
            sizes = counts['count'] * 40  # scaling factor
            bounds = np.arange(counts['count'].min() - 0.5, counts['count'].max() + 1.5)
            norm = BoundaryNorm(bounds, ncolors=plt.cm.viridis.N)
            sc = ax.scatter(x, y, s=sizes, c=counts['count'], cmap='viridis', edgecolors=scat_edge, alpha=scat_alph, norm=norm)
            #
            cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(counts['count'].min(), counts['count'].max() + 1))
            cbar.set_label('Number of patients')
            pass
        case 'd': # table aspect
            ax.scatter(data_ser1, data_ser2, alpha=scat_alph, s=scat_size, c=scat_colo, edgecolors=scat_edge)
            ax.set_xticks(range(0, max_score))
            ax.set_yticks(range(0, max_score))
            ax.set_xlim(-1.0, max_score + 1.0)
            ax.set_ylim(-1.0, max_score + 1.0)
            ax.set_aspect('equal')
            pass
        case _:
            raise Exception()  

    # Grph : line
    # ----
    line_widt = plot_tran_pure.line_widt
    line_styl = plot_tran_pure.line_styl
    line_colo = plot_tran_pure.line_colo
    line_alph = plot_tran_pure.line_alph
    line_labl = plot_tran_pure.line_labl
    line_colo = 'blue'
    line_styl = '--'
    line_widt = 2
    line_labl = 'L,R limb VCSS Symmetry'
    ax.plot([0, max_score], [0, max_score], color=line_colo, linestyle=line_styl, lw=line_widt, label=line_labl)
    
    # Axis, Grid
    # ----
    ax.set_xlabel(plot_tran_pure.xlab)
    ax.set_ylabel(plot_tran_pure.ylab)

    ax.set_xlim(-1.0, max_score + 1.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=plot_tran_pure.axix_intv + 1, integer=True, prune='upper'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    #
    ax.set_ylim(-1.0, max_score + 1.0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=plot_tran_pure.axiy_intv + 1, integer=True, prune='upper'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    #
    grid_alph = plot_tran_pure.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
  
    # Lgnd
    # ---- 
    ax.legend()
    #
    diff = data_ser1 - data_ser2
    ax.text(
        0.98, 0.02, # x, y in axes coordinates (bottom-right)
        f'L > R: {(diff > 0).sum()}\n'
        f'R > L: {(diff < 0).sum()}\n'
        f'L = R: {(diff == 0).sum()}',
        # boxstyle  = 'round', 'square'
        # edgecolor = '0.5', # gray ; 'none'
        # linewidth = 0.8,
        transform=ax.transAxes, va='bottom', ha='right', fontsize=10, bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=0.8)
    )
       
    # Titl
    # ----
    titl = plot_tran_pure.titl
    ax.set_title(titl)
    fig.tight_layout()
    # fig.show()

    if False:
        # Vars
        # ----
        fig = plot_tran_pure.figu
        ax = plot_tran_pure.axis
        ax.clear()
        titl = plot_tran_pure.titl # "VEINES-VCSS over time"
        xlab = plot_tran_pure.xlab # "Timepoint"
        ylab = plot_tran_pure.ylab # "VEINES-VCSS score"
        tick_dict = plot_tran_pure.tick_dict # {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
        
        # Grph
        # ----
        scat_alph=0.6
        scat_size=60
        scat_colo='purple'
        scat_edge='black'
        match (plot_tran_pure.stra):
            case 'a': # default
                ax.errorbar(df_fram["timepoint"], df_fram["mean"], yerr=1.96 * df_fram["se"], fmt="o-", linewidth=2, markersize=6, capsize=4, label="MARC")
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

        # Grph : line
        # ----
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend()

        # Axis & Grid
        # ----
        ax.margins(x=0.10)  # x-axis padding

        labl_list = [tick_dict.get(cat, cat) for cat in df_fram["timepoint"].cat.categories]
        ax.set_xticks(df_fram["timepoint"].cat.categories)
        ax.set_xticklabels(labl_list)

        ymin = df_fram["ci_lower"].min() - 2
        ymax = df_fram["ci_upper"].max() + 2
        ax.set_ylim(ymin, ymax)
        #
        grid_alph = plot_tran_pure.grid_alph
        if grid_alph is not None:
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
            ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
            
        # Titles
        # ------
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