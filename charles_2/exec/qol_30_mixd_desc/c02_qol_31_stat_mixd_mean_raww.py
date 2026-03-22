    
    
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
def exec_stat_mixd_mean_raww(stat_tran_adat: StatTranQOL_31_mixd, df_fram) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # Exec  : df_lon1
    # ----
    df_lon1 = (df_fram.groupby("timepoint").agg(mean=("VEINES_QOL_t", "mean"),sd=("VEINES_QOL_t", "std"),n=("VEINES_QOL_t", "count")).reset_index())
    df_lon1["se"] = df_lon1["sd"] / np.sqrt(df_lon1["n"])
    df_lon1["ci_lower"] = df_lon1["mean"] - 1.96 * df_lon1["se"]
    df_lon1["ci_upper"] = df_lon1["mean"] + 1.96 * df_lon1["se"]
    #
    t0_mean = df_lon1.loc[df_lon1['timepoint'] == 'T0', 'mean'].values[0]
    df_lon1["mean change from T0"] = df_lon1["mean"] - t0_mean
    if trac:
        print_yes(df_lon1, labl="df_lon1")
        
    # Exit
    # ----
    stat_tran_adat.mixd_mean_raww = df_lon1

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