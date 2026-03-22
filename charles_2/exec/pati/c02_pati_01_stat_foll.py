    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_pati_01_stat_ import StatTranPATI_01_foll
  
import numpy as np  
import pandas as pd
import sys
import os  
import re
from pathlib import Path
from pprint import pprint
from pandas.api.types import is_numeric_dtype
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
from pati.c02_pati_01_stat_adat import StatTranPATI_01_foll
        
# ----
# Timepoint outcomes
# ----
def exec_stat_foll(stat_tran_foll: StatTranPATI_01_foll) -> None:
    # from pati.c02_pati_01_stat_ import StatTranPATI_01_foll

    # Data
    # ---- 
    df_fram = stat_tran_foll.stat_tran.fram.copy()
    mark_dict = stat_tran_foll.stat_tran.proc_tran.orch_tran.ta01_base_char

    # Exec
    # ----   
    tipo = "T0"
    #
    df_foll = df_fram[df_fram["timepoint"] == tipo]
    df_foll = df_foll[['workbook', 'patient_id', 'timepoint', 'Telephone', 'Commune', 'Revenir', 'Difficulté', 'Charge', 'Satisfaction']]
    df_foll = df_foll[['workbook', 'patient_id', 'timepoint', 'Telephone', 'Commune', 'Revenir']]
  
    stat_tran_foll.resu_foll = exec_stat_foll_resu(df_foll.copy())
    stat_tran_foll.publ_foll = exec_stat_foll_publ(df_foll.copy())
    exec_stat_foll_mark(stat_tran_foll.resu_foll, mark_dict)

def exec_stat_foll_mark(df_fram, mark_dict:dict):
          
    trac = True
    
    # Exec
    # ----
    if trac:
        print_yes (df_fram[['workbook', 'timepoint','Revenir']])
    counts = df_fram['Revenir'].value_counts()
    percentages = df_fram['Revenir'].value_counts(normalize=True) * 100
    revenir_dict = counts.to_dict()
    for category, count in revenir_dict.items():
        pct = percentages[category]
        print(f"Category: {category:10} | Count: {count:5} | Percentage: {pct:.2f}%")
        match category:
            case "Pour prévention":
                info = f'{count} ({pct:.2f}%)'
                mark_dict['.Preventive'] = (info, Path(__file__).stem)
            case "Si problème":
                info = f'{count} ({pct:.2f}%)'
                mark_dict['.Curative'] = (info, Path(__file__).stem)
            case "Non":
                info = f'{count} ({pct:.2f}%)'
                mark_dict['.None intended'] = (info, Path(__file__).stem)
            case _:
                info = f'{count} ({pct:.2f}%)'
                mark_dict['.Missing info'] = (info, Path(__file__).stem)
    pass
    
def exec_stat_foll_publ(df_fram):
    
    trac = True
    
    # Data
    # ----
    what = "Follow-up Adherence"
    text = ("Reactive Care[1]Proactive Care[2]Loss to Follow-up[3]" )
    df_fram = df_fram.rename(columns={'Revenir': what})
    
    # Exec
    # ----
    matches = re.findall(r"(.+?)\s*\[(\d+)\]", text)
    what_dict = {int(code): value.strip() for value, code in matches}
    what_list = [value.strip() for value, _ in matches]
    df_fram[what] = (
        df_fram[what]
        .map(what_dict)
        .astype(pd.CategoricalDtype(categories=what_list, ordered=True))
    )
    #
    df_fram = df_fram.groupby(what).size().reset_index(name='Count')
  
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_foll_publ")
        
    # Exit
    # ----
    return df_fram
    
def exec_stat_foll_resu(df_fram):
    
    trac = True
    
    # Data
    # ----
    text = ("Si problème[1]Pour prévention[2]Non[3]" )
    
    # Exec
    # ----
    what = "Revenir"
    matches = re.findall(r"(.+?)\s*\[(\d+)\]", text)
    what_dict = {int(code): value.strip() for value, code in matches}
    what_list = [value.strip() for value, _ in matches]
    df_fram[what] = (
        df_fram[what]
        .map(what_dict)
        .astype(pd.CategoricalDtype(categories=what_list, ordered=True))
    )
  
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_foll_resu")
        
    # Exit
    # ----
    return df_fram

def inpu_file_exec_xlat_2(df_fram, what, text):
   
    trac = True
        
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
    
    # Exec
    # ----
    matches = re.findall(r"(.+?)\s*\[(\d+)\]", text)
    what_dict = {int(code): value.strip() for value, code in matches}
    what_list = [value.strip() for value, _ in matches]
    df_fram[what] = (
        df_fram[what]
        .map(what_dict)
        .astype(pd.CategoricalDtype(categories=what_list, ordered=True))
    )
        
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exit
    # ----
    return df_fram

def print_yes(df, labl=None):
    
    # Exec
    # ----
    print (f"\n----\nFram labl : {labl}\n----")
    
    # Exec
    # ----
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        
    # Exec
    # ----
    print(df.info())
    
    # Exec : Summary of all columns and their specific underlying types
    # ----
    type_summary = {}
    #
    for col in df.columns:
        # Get the Pandas dtype
        p_dtype = df[col].dtype
        # Get the actual Python type of the first non-null element
        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        actual_type = type(first_valid).__name__ if first_valid is not None else "All Null"
        #
        type_summary[col] = {"Pandas_Dtype": p_dtype, "Actual_Type": actual_type}
    # Convert to DataFrame for a clean view
    df_types = pd.DataFrame(type_summary).T
    print(df_types)
    pass