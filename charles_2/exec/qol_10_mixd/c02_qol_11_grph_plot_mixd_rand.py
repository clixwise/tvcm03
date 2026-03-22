    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_11_grph_plot_ import PlotTranQOL_11_rand_ququ, PlotTranQOL_11_rand_hist
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats as sp_stats

# # https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c

# ----
# Histogram of residuals
# ----
def exec_plot_rand_hist(plot_tran_mixd: PlotTranQOL_11_rand_hist) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_rand_hist

    trac = True

    # Data
    # ---- 
    resu = plot_tran_mixd.fram_dict["resu_mixd_resu"]
    random_effects = resu.random_effects
    print (type(random_effects))
    print (type(random_effects.values()))
    print (random_effects.values())
    re_values = [re.iloc[0] for re in random_effects.values()]
    
    # This turns that dictionary of Series into one clean Series of values
    resu = plot_tran_mixd.fram_dict["resu_mixd_resu"]
    random_effects = resu.random_effects
    re_series = pd.concat(random_effects.values())
    re_values = re_series.tolist()
      
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
    
    # Plot 1: Residuals vs Fitted
    ax.hist(re_values, bins=15, edgecolor='black', color='skyblue', alpha=0.7)
    ax.set_xlabel('Random Effect Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Random Effects')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
# ----
# Q-Q plot of random effects
# ----
def exec_plot_rand_ququ(plot_tran_mixd: PlotTranQOL_11_rand_ququ) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_rand_ququ

    trac = True

    # Data
    # ---- 
    resu = plot_tran_mixd.fram_dict["resu_mixd_resu"]
    random_effects = resu.random_effects
    print (type(random_effects))
    print (type(random_effects.values()))
    print (random_effects.values())
    re_values = [re.iloc[0] for re in random_effects.values()]
    
    # This turns that dictionary of Series into one clean Series of values
    resu = plot_tran_mixd.fram_dict["resu_mixd_resu"]
    random_effects = resu.random_effects
    re_series = pd.concat(random_effects.values())
    re_values = re_series.tolist()
       
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
    
    # Plot 1: Residuals vs Fitted
    sp_stats.probplot(re_values, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot (Random Effects)')
    ax.grid(True, alpha=0.3)
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