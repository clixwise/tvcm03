    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_01_assu_ import AssuTranQOL_01_isok
    
import pandas as pd
import sys
import os  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
# ----
# Assumption : histogram normality thru qq-plot
# https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
# A1 — completeness (pati_isok)
# A2 — score distribution (histogram, Q–Q, skew/kurtosis)
# C1 — ceiling/floor effects
# ----

def exec_assu_isok(assu_tran_isok: AssuTranQOL_01_isok) -> None:
    from qol.c02_qol_01_assu_ import AssuTranQOL_01_isok
    
    trac = True

    # Data
    # ---- 
    df_fram = assu_tran_isok.assu_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")

    # Oupu
    # ----   
    df_resu = df_fram # pd.DataFrame.from_dict(resu_dict, orient='index')
    #
    if trac:
        print_yes(df_resu, labl="df_resu")
    
    # Exit
    # ----
    assu_tran_isok.resu_tech = df_resu
    
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