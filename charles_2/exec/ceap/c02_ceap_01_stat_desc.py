    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_ceap_01_stat_ import StatTranCEAP_01_desc
  
import numpy as np  
import pandas as pd
import sys
import os  
from pprint import pprint
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
        
# ----
# https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4 
# ----
'''
'''
def exec_stat_desc(stat_tran_desc: StatTranCEAP_01_desc) -> None:
    from ceap.c02_ceap_01_stat_ import StatTranCEAP_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram.copy()
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exec
    # ----
    stat_tran_desc.resu_publ_T0 =  exec_stat_desc_T0(df_fram)
    pass

def exec_stat_desc_T0(df_fram) -> pd.DataFrame: 
    from ceap.c02_ceap_01_stat_ import StatTranCEAP_01_desc
     
    trac = True
   
    # Exec
    # ----   
    tipo = "T0"
    df_tipo = df_fram[df_fram["timepoint"] == tipo]
    if df_tipo.empty:
        raise Exception()
    
    # Trac
    # ----
    if trac:
        print_yes(df_tipo, labl="df_tipo")
    
    # Exit
    # ----
    return df_tipo

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