    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_vcss_01_grph_plot_ import PlotTranVCSS_01_tacs
    
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
def exec_plot_tacs(plot_tran_pure: PlotTranVCSS_01_tacs) -> None:
    # from vcss.c02_vcss_01_grph_plot_ import PlotTranVCSS_01_tacs
    
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
    #
    time_list = df_fram["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    
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
        case 'c': # Progress Trajectory with Centroids
            style_map = {
                'T0': {'marker': '^', 'size': scat_size, 'color': '#e74c3c', 'label': 'Baseline (T0)'},
                'T1': {'marker': 's', 'size': scat_size*0.8, 'color': '#3498db', 'label': 'Follow-up (T1)'},
                'T2': {'marker': 'o', 'size': scat_size, 'color': '#2ecc71', 'label': 'Final (T2)'}
            }
            jitter = 0.18
            # Calculate True Means
            centroids = df_fram.groupby('timepoint')[colu_list].mean().reindex(time_list)

            # Individual Patient Lines (Spaghetti)
            pids = df_fram['patient_id'].unique()
            for pid in pids:
                pat = df_fram[df_fram['patient_id'] == pid].sort_values('timepoint')
                # Apply a tiny bit of jitter to the lines themselves for visibility
                jx = pat[colu_list[0]] + np.random.uniform(-jitter, jitter)
                jy = pat[colu_list[1]] + np.random.uniform(-jitter, jitter)
                ax.plot(jx, jy, color='gray', alpha=0.25, lw=0.7, zorder=1)

            # Draw Mean Progress Path
            ax.plot(centroids[colu_list[0]], centroids[colu_list[1]], 
                    color='black', lw=1.5, ls='-', alpha=0.7, zorder=4, label='Mean Trend')

            # Draw Points and Centroids
            for time, style in style_map.items():
                t_data = df_fram[df_fram['timepoint'] == time]
                
                # Individual dots (jittered)
                jx = t_data[colu_list[0]] + np.random.uniform(-jitter, jitter, size=len(t_data))
                jy = t_data[colu_list[1]] + np.random.uniform(-jitter, jitter, size=len(t_data))
                
                ax.scatter(jx, jy,
                        c=style['color'], marker=style['marker'], s=style['size'],linewidths=0.3,
                        edgecolors='darkgray', alpha=0.4, zorder=2)
                
                # Mean markers
                ax.scatter(centroids.loc[time, colu_list[0]], centroids.loc[time, colu_list[1]], 
                        s=scat_size*1.5, c=style['color'], 
                        marker=style['marker'], edgecolors='black', alpha=0.7, linewidth=1, 
                        zorder=5, label=f"Mean {time}")

    # Grph : line
    # ----
    line_widt = plot_tran_pure.line_widt
    line_styl = plot_tran_pure.line_styl
    line_colo = plot_tran_pure.line_colo
    line_alph = plot_tran_pure.line_alph
    line_labl = plot_tran_pure.line_labl
    line_colo = 'blue'
    line_styl = '--'
    line_widt = 1.5
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
    
    # Lgnd : Clean Legend (Only Means and Symmetry)
    # ---- 
    h, l = ax.get_legend_handles_labels()
    keep = ['Mean T0', 'Mean T1', 'Mean T2', 'Group Mean Path', line_labl]
    by_l = dict(zip(l, h))
    ax.legend([by_l[k] for k in keep if k in by_l], [k for k in keep if k in by_l], loc='upper left', fontsize=8, framealpha=0.8)
    # Stats Box (T2 focus)
    df_fina = df_fram[df_fram['timepoint'] == 'T2']
    diff = df_fina[colu_list[0]] - df_fina[colu_list[1]]
    ax.text(
            0.98, 0.02,
            f'At T2:\n'
            f'L > R: {(diff > 0).sum()}\n'
            f'R > L: {(diff < 0).sum()}\n'
            f'L = R: {(diff == 0).sum()}',
            transform=ax.transAxes, va='bottom', ha='right', 
            fontsize=10, bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=0.8)
    )
    # Titl
    # ----
    titl = plot_tran_pure.titl
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