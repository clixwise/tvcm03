    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_exam_01_stat_ import StatTranEXAM_01_pure
  
import numpy as np  
import pandas as pd
import sys
import os  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
        
# ----
# Timepoint outcomes
# ----
'''
'''
def exec_stat_pure(stat_tran_pure: StatTranEXAM_01_pure) -> None:
    from exam.c02_exam_01_stat_ import StatTranEXAM_01_pure
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_pure.stat_tran.fram.copy()
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exec
    # ----
    stat_tran_pure.resu_dict = {}
    #
    stat_tran_pure.resu_ceap_limb = exec_stat_pure_ceap(df_fram)
    stat_tran_pure.resu_dict["ceap_limb"] = stat_tran_pure.resu_ceap_limb
    pass
       
def exec_stat_pure_ceap(df_fram) -> pd.DataFrame: 
    from exam.c02_exam_01_stat_ import StatTranEXAM_01_pure
    
    trac = True
        
    # CEAP
    # ----
    df_stat = df_fram.copy()
    df_stat = df_stat[['workbook', 'patient_id', 'timepoint', 'Age', 'Sexe', 'CEAP_R', 'CEAP_L']]
    df_stat = df_stat.rename(columns={'CEAP_R': 'R', 'CEAP_L': 'L'})
    time_list = ["T0", "T1", "T2"]
    # df_stat["timepoint"] = pd.Categorical(df_stat["timepoint"],categories=time_list,ordered=True)
    #
    resu_dict = {}
    for tipo in time_list:
 
        df_time = df_stat[df_stat["timepoint"] == tipo].copy()
        resu_dict[tipo] = df_time
    
    # Oupu
    # ----   
    df_resu = pd.concat(resu_dict, axis=0)
    #
    if trac:
        print_yes(df_resu, labl="df_resu")

    # Exit
    # ----
    return df_resu

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