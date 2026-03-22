    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_stat_ import StatTranQOL_01_mean
    
import pandas as pd
import numpy as np

from util.stat_help import summarize_continuous_stru
from qol.c02_qol_01_stat_adat import StatTranQOL_01_mean

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
def exec_stat_mean(stat_tran_mean: StatTranQOL_01_mean) -> None:
    # from qol.c02_qol_01_stat_ import StatTranQOL_01_mean
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_mean.stat_tran.fram # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exec 1 : mode 1
    # ---- 
    '''
      timepoint       mean        sd   n        se   ci_lower   ci_upper
    0        T0  50.027333  3.526386  30  0.643827  48.765432  51.289234
    '''
    df_lon1 = (df_fram.groupby("timepoint").agg(mean=("VEINES_QOL_t", "mean"),sd=("VEINES_QOL_t", "std"),n=("VEINES_QOL_t", "count")).reset_index())
    if trac:
        print_yes(df_lon1, labl="df_lon1")
    
    # Exec 2 = mode 2
    # ---- 
    fram_list = []
    for tipo in ["T0", "T1", "T2"]:
        
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            continue
        #fram_list.append([StatTranQOL_01_plot.__name__, "QOL_none_t", tipo, summarize_continuous_edit(df_tipo["none_t"], "mean_sd")])
        #fram_list.append([StatTranQOL_01_plot.__name__, "QOL_mean_t", tipo, summarize_continuous_edit(df_tipo["mean_t"], "mean_sd")])
        #fram_list.append([StatTranQOL_01_plot.__name__, "QOL_VEINES_QOL_t", tipo, summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "mean_sd")])
        #fram_list.append([StatTranQOL_01_plot.__name__, "QOL_gemi_t", tipo, summarize_continuous_edit(df_tipo["gemi_t"], "mean_sd")])
        fram_list.append(["QOL_gemi_t", tipo, summarize_continuous_stru(df_tipo["VEINES_QOL_t"], "mean_sd"), len(df_tipo)])
    #
    if trac:
        print(fram_list)
    
    df_lon2 = pd.DataFrame(fram_list, columns=["parm", "timepoint", "value", "n"])
    df_lon2[['mean', 'sd']] = df_lon2['value'].apply(lambda t: pd.Series(t))
    df_lon2 = df_lon2.drop('value', axis=1)
    if trac:
        print_yes(df_lon2, labl="df_lon2")
    pass

    # Oupu
    # ----
    # time_order = sorted(df_lon1["timepoint"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x) # ["T0", "T1", "T2"]
    # df_lon1["timepoint"] = pd.Categorical(df_lon1["timepoint"],categories=time_order,ordered=True)
    df_lon1["se"] = df_lon1["sd"] / np.sqrt(df_lon1["n"])
    df_lon1["ci_lower"] = df_lon1["mean"] - 1.96 * df_lon1["se"]
    df_lon1["ci_upper"] = df_lon1["mean"] + 1.96 * df_lon1["se"]
    #
    if trac:
        print_yes(df_lon1, labl="df_lon1")

    # Exit
    # ----
    stat_tran_mean.resu_plot = df_lon1
    
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