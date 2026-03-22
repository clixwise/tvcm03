    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_31_stat_ import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame

# ----
# Timepoint outcomes
# ----
def exec_stat_mixd_mean_merg(stat_tran_adat: StatTranQOL_31_mixd) -> DataFrame:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # Data
    # ---- 
    df_raww_mean = stat_tran_adat.mixd_mean_raww.copy()
    df_modl_mean = stat_tran_adat.mixd_mean_modl.copy()
    df_modl_mean_rena = df_modl_mean.rename(columns={"mean":"modl_mean", "se":"modl_se", "ci_lower":"modl_ci_lower", "ci_upper":"modl_ci_upper"})
    if trac:
        print_yes(df_raww_mean, "df_raww_mean")
        print_yes(df_modl_mean, "df_modl_mean")
        print_yes(df_modl_mean_rena, "df_modl_mean_rena")
  
    # Merg
    # ----
    if len(df_raww_mean) != len(df_modl_mean):
            raise ValueError(f"DataFrames have different number of rows: {len(df_raww_mean)} vs {len(df_modl_mean)}")
    df_merg_mean = pd.merge(df_raww_mean, df_modl_mean_rena, on=['timepoint', 'n'], how='inner')
    df_merg_mean = df_merg_mean[['timepoint', 'n', 'mean', 'sd', 'se', 'ci_lower', 'ci_upper', 'modl_mean', 'modl_se', 'modl_ci_lower', 'modl_ci_upper']]
    if trac:
        print_yes(df_merg_mean, "df_merg_mean")
    '''
            timepoint  n   mean   ci_lower  ci_upper  sd     se    modl_mean  modl_ci_lower  modl_ci_upper
    0  T0        30  53.34  46.87     59.80     18.06  3.30  53.34      43.43          63.25
    1  T1        30  50.00  48.75     51.26      3.50  0.64  50.00      40.09          59.91
    2  T2        30  63.34  47.48     79.19     44.30  8.09  63.34      53.43          73.25
    '''
    # Exit
    # ----    
    stat_tran_adat.mixd_mean_merg = df_merg_mean
    
    '''
    df_emm = df_emm.rename(columns={"modl_mean":"mean", "modl_se":"se", "modl_ci_lower":"ci_lower", "modl_ci_upper":"ci_upper"})
    cate_list = ['T0','T1','T2'] ; df_emm["timepoint"] = pd.Categorical(df_emm["timepoint"], categories=cate_list, ordered=True)
    # 
    df_plot = pd.merge(df_raww_mean, df_modl_mean, on=['timepoint', 'n'], how='inner')
    df_plot = df_plot[['timepoint', 'n', 'mean', 'ci_lower', 'ci_upper', 'sd', 'se', 'modl_mean', 'modl_ci_lower', 'modl_ci_upper']]
    df_plot['timepoint'] = pd.Categorical(df_plot['timepoint'], categories=df_lon1['timepoint'].cat.categories, ordered=df_lon1['timepoint'].cat.ordered)
    if trac:
        print_yes(df_plot, labl="df_plot")
    '''

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