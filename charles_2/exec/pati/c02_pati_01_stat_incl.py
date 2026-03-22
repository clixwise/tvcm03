    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_pati_01_stat_ import StatTranPATI_01_desc
  
import numpy as np  
import pandas as pd
import sys
import os  
import re
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
from util.fram_help import fram_prnt
from util.data_51_inpu import inpu_file_exec_xlat_2
from pprint import pprint
from pati.c02_pati_01_stat_adat import StatTranPATI_01_desc
        
# ----
# https://gemini.google.com/app/a19fc679f0d6795d
# ----
'''
'''
def exec_stat_incl(stat_tran_desc: StatTranPATI_01_desc) -> None:
    # from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram.copy()
    mark_dict = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta01_scre_incl
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
    pass

    # Exec
    # ----
    for tipo in ["T0", "T1", "T2"]:
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            continue
        match tipo:
            case 'T0':
                mark_dict['Screened'] = (len(df_tipo), Path(__file__).stem)
                mark_dict['Eligible'] = (len(df_tipo), Path(__file__).stem)
                mark_dict['Underwent RFA'] = (len(df_tipo), Path(__file__).stem)
                mark_dict['T0 (baseline) assessment'] = (len(df_tipo), Path(__file__).stem)
            case 'T1':
                mark_dict['T1 (3 months) completed'] = (len(df_tipo), Path(__file__).stem)
            case 'T2':
                mark_dict['T2 (12 months) completed'] = (len(df_tipo), Path(__file__).stem)
            case _:
                raise Exception()

def print_yes(df, labl=None):
    '''
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
    '''
    fram_prnt(df, labl=labl, trunc= None, head=10)
    pass