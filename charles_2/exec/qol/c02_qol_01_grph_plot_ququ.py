    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_01_grph_plot_ import PlotTranQOL_01_ququ
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
def exec_plot_ququ(plot_tran_ququ: PlotTranQOL_01_ququ) -> None:
    # from qol.c02_qol_01_grph_plot_ import PlotTranQOL_01_ququ
    
    trac = True

    # Data
    # ---- 
    df_fram = plot_tran_ququ.fram # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    if trac:
        print_yes(df_fram, labl="df_fram")
    se_fram = df_fram['VEINES_QOL_t']
    if trac:
        print (se_fram)
              
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_ququ.figu
    ax = plot_tran_ququ.axis
    ax.clear()
     
    # Grph
    # ----
    scat_size = plot_tran_ququ.scat_size
    scat_widt = plot_tran_ququ.scat_widt
    scat_edge = plot_tran_ququ.scat_edge
    scat_colo = plot_tran_ququ.scat_colo
    scat_alph = plot_tran_ququ.scat_alph
    scat_labl = plot_tran_ququ.scat_labl
        
    line_widt = plot_tran_ququ.line_widt
    line_styl = plot_tran_ququ.line_styl
    line_colo = plot_tran_ququ.line_colo
    line_alph = plot_tran_ququ.line_alph
    line_labl = plot_tran_ququ.line_labl

    # 1. Extract the calculations (plot=None ensures nothing is drawn yet)
    (osm, osr), (slope, intercept, r) = stats.probplot(se_fram, dist="norm", plot=None)
    # 3. Style the Data Points (The "Dots")
    if False:
        ax.scatter(osm, osr, 
            s=50,                         # Size
            facecolors='none',            # Hollow centers
            edgecolors='#1f77b4',         # Muted blue
            linewidths=1.2,               # Thickness of the circle
            alpha=0.6,                    # Transparency
            label='Observed Data',
            zorder=3)
    else:
        ax.scatter(osm, osr, 
                s=scat_size,                   
                facecolors=scat_colo,            
                edgecolors=scat_edge,
                linewidths=scat_widt, 
                alpha=scat_alph,                 
                label=scat_labl,
                zorder=3)                     
    # 4. Style the Reference Line
    # We create a range for the line to ensure it spans the whole plot
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 
            color=line_colo,           
            linestyle=line_styl,               
            linewidth=line_widt, 
            alpha=line_alph,
            label=f"{line_labl} ($R^2={r**2:.3f}$)",
            zorder=2)
    
    # Axis, Grid
    # ----
    xlab = plot_tran_ququ.xlab 
    ax.set_xlabel(xlab)
    ax.margins(x=0.10) 

    ylab = plot_tran_ququ.ylab 
    ax.set_ylabel(ylab)
   
    grid_alph = plot_tran_ququ.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # Lgnd
    # ---- 
    ax.legend(frameon=True, loc='upper left')
    # Removing top and right spines for a modern "clean" look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
         
    # Titl
    # ----
    titl = plot_tran_ququ.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    fig.tight_layout()
    #fig.show()
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