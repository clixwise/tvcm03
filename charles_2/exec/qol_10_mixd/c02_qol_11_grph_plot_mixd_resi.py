    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_11_grph_plot_ import PlotTranQOL_11_resi_fitt, PlotTranQOL_11_resi_ququ, PlotTranQOL_11_resi_hist, PlotTranQOL_11_resi_rand
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats as sp_stats

# # https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c

# ----
# Residuals vs Fitted
# ----
def exec_plot_resi_fitt(plot_tran_mixd: PlotTranQOL_11_resi_fitt) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_resi_fitt

    trac = True

    # Data
    # ---- 
    '''
    ----
    Fram labl : df_modl
    ----
                 workbook                                                     patient_id        timepoint iter_t  pati_isok  Age  BMI   fitted  residuals
    0            2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0        46.80   True       59   34.7  46.54   0.26
    1            2025-11-01 2025-11-01 T1 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T1        51.48   True       59   34.7  51.54  -0.06
    2            2025-11-01 2025-11-01 T2 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T2        53.82   True       59   34.7  54.04  -0.22
    '''
    df_modl = plot_tran_mixd.fram_dict["resu_mixd_modl"] 
    if trac:
        print_yes(df_modl, labl="df_modl")
    
    # Exec
    # ----
    cols_requ = {"timepoint", "fitted", "residuals"}
    if not cols_requ.issubset(df_modl.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_modl.columns)}")
    #
    df_modl = df_modl.set_index("timepoint", drop=False) # -> column is not dropped
    if trac:
        print_yes(df_modl, labl="df_modl")
      
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
    ax.scatter(df_modl['fitted'], df_modl['residuals'], alpha=0.6, edgecolors='k')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    ax.grid(True, alpha=0.3)
    # plt.show()

# ----
# Q-Q plot of residuals
# ----
def exec_plot_resi_ququ(plot_tran_mixd: PlotTranQOL_11_resi_ququ) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_resi_ququ

    trac = True

    # Data
    # ---- 
    '''
    ----
    Fram labl : df_modl
    ----
                 workbook                                                     patient_id        timepoint iter_t  pati_isok  Age  BMI   fitted  residuals
    0            2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0        46.80   True       59   34.7  46.54   0.26
    1            2025-11-01 2025-11-01 T1 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T1        51.48   True       59   34.7  51.54  -0.06
    2            2025-11-01 2025-11-01 T2 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T2        53.82   True       59   34.7  54.04  -0.22
    '''
    df_modl = plot_tran_mixd.fram_dict["resu_mixd_modl"] 
    if trac:
        print_yes(df_modl, labl="df_modl")
    
    # Exec
    # ----
    cols_requ = {"timepoint", "fitted", "residuals"}
    if not cols_requ.issubset(df_modl.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_modl.columns)}")
    #
    time_list = df_modl["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    x = np.arange(len(time_list))
    df_modl = df_modl.set_index("timepoint", drop=False) # -> column is not dropped
    #
    if trac:
        print_yes(df_modl, labl="df_modl")
      
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
    sp_stats.probplot(df_modl['residuals'], dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot (Residuals)')
    ax.grid(True, alpha=0.3)
    # plt.show()

# ----
# Histogram of residuals
# ----
def exec_plot_resi_hist(plot_tran_mixd: PlotTranQOL_11_resi_hist) -> None:
    # from qol_10.c02_qol_11_grph_plot_ import PlotTranQOL_11_resi_hist

    trac = True

    # Data
    # ---- 
    '''
    ----
    Fram labl : df_modl
    ----
                 workbook                                                     patient_id        timepoint iter_t  pati_isok  Age  BMI   fitted  residuals
    0            2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0        46.80   True       59   34.7  46.54   0.26
    1            2025-11-01 2025-11-01 T1 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T1        51.48   True       59   34.7  51.54  -0.06
    2            2025-11-01 2025-11-01 T2 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T2        53.82   True       59   34.7  54.04  -0.22
    '''
    df_modl = plot_tran_mixd.fram_dict["resu_mixd_modl"] 
    if trac:
        print_yes(df_modl, labl="df_modl")
    
    # Exec
    # ----
    cols_requ = {"timepoint", "fitted", "residuals"}
    if not cols_requ.issubset(df_modl.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_modl.columns)}")
    #
    time_list = df_modl["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    x = np.arange(len(time_list))
    df_modl = df_modl.set_index("timepoint", drop=False) # -> column is not dropped
    #
    if trac:
        print_yes(df_modl, labl="df_modl")
      
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
    ax.hist(df_modl['residuals'], bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Residuals')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    #plt.show()
    
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