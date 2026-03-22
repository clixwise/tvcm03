    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_31_stat_ import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd

# ----
# Timepoint outcomes
# ----
def exec_stat_mixd_assu_merg(stat_tran_adat: StatTranQOL_31_mixd) -> DataFrame:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # Data
    # ----
    df_rand = stat_tran_adat.mixd_assu_rand 
    df_resi = stat_tran_adat.mixd_assu_resi 
    if trac:
        print_yes(df_rand, "df_rand")
        print_yes(df_resi, "df_resi")
  
    # Merg
    # ----
    if len(df_rand) != len(df_resi):
            raise ValueError(f"DataFrames have different number of rows: {len(df_rand)} vs {len(df_resi)}")
    df_merg_assu = pd.concat([df_rand, df_resi], ignore_index=True)

    if trac:
        print_yes(df_merg_assu, "df_merg_assu")
    '''
    '''
    # Exit
    # ----    
    stat_tran_adat.mixd_mean_merg = df_merg_assu

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