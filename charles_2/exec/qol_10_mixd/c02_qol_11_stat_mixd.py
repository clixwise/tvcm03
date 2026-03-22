    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_11_stat_ import StatTranQOL_11_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import patsy
from pandas import DataFrame
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.stats import shapiro, normaltest
from qol_10_mixd.c02_qol_11_stat_mixd_gemi_to_copiTODOINTOMCID import exec_stat_mixd_gemi_to_copi
from qol_10_mixd.c02_qol_11_stat_mixd_gemiOUTDATED import exec_stat_mixd_gemi
from qol_10_mixd.c02_qol_11_stat_mixd_open import exec_stat_mixd_open
from qol_10_mixd.c02_qol_11_stat_mixd_copi import exec_stat_mixd_copi
# from qol_10_mixd.c02_qol_11_stat_mixd_open_20260217 import exec_stat_mixd_open_20260217

# ----
# Timepoint outcomes
# ----

def exec_stat_mixd(stat_tran_adat: StatTranQOL_11_mixd) -> None:
    df_lon1 = exec_stat_mixd_raw(stat_tran_plot) # OK
    exec_stat_mixd_gemi_to_copi(stat_tran_plot, df_lon1)
    #OUTDATED exec_stat_mixd_gemi(stat_tran_plot, df_lon1)
    exec_stat_mixd_copi(stat_tran_plot, df_lon1)
    # exec_stat_mixd_open_20260217(stat_tran_plot, df_lon1)
    exec_stat_mixd_open(stat_tran_plot, df_lon1)

def exec_stat_mixd_raw(stat_tran_adat: StatTranQOL_11_mixd) -> DataFrame:
    from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11_mixd

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, "df_fram")
    
    # Raw  : df_lon1
    # ===
    df_lon1 = (df_fram.groupby("timepoint").agg(mean=("VEINES_QOL_t", "mean"),sd=("VEINES_QOL_t", "std"),n=("VEINES_QOL_t", "count")).reset_index())
    df_lon1["se"] = df_lon1["sd"] / np.sqrt(df_lon1["n"])
    df_lon1["ci_lower"] = df_lon1["mean"] - 1.96 * df_lon1["se"]
    df_lon1["ci_upper"] = df_lon1["mean"] + 1.96 * df_lon1["se"]
    if trac:
        print_yes(df_lon1, labl="df_lon1")
        
    # Exit
    # ----
    return df_lon1

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