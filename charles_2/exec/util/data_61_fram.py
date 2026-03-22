import os
import sys
import pandas as pd
from pathlib import Path

# ****
# Clas
# ****

# ----
# Fram
# ----        
class FramTran():
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FramTran.__name__] = self
        self.inpu = None
        #
        self.fram = None
        self.filt = None
        self.pref = None
        self.func = None
        #
        self.frax = None
        self.drop_list = None

    # Mono 'upda' uses 'fram' [all but VCSS]
    def upda(self):
        _upda_fram(self)
    # Dual 'upda' uses 'fram','frax' [only VCSS]    
    def upda_fram(self):
        _upda_fram(self)
    def upda_frax(self):
        _upda_frax(self)

# ====
# Wide : pati rows
# ====
def _upda_fram(framTran:FramTran):   
 
    # Opti
    # ----
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 2000) 
    
    # Exec
    # ----
    fram_inpu = framTran.inpu
    filt_list = framTran.filt
    colu_pref = framTran.pref
    inpu_func = framTran.func
    df_fram = inpu_func(fram_inpu, filt_list, colu_pref)
    framTran.fram = df_fram

def inpu_fram_exec_selc_1_exte(df_full, filt_list, pref):   
    
    trac = True

    # Exec
    # ----
    # df_full = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)
    # df_full.columns = df_full.columns.str.strip()
        
    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")
            
    cols_base = ["workbook", "date1", "date2", "timepoint", "patient_id", "name"]
    cols_base = ["workbook", "patient_id", "name", "timepoint"]
    cols_vari = filt_list
    cols_full = cols_base + cols_vari
    
    # Debu : makes only sense when the 'filt_list' contains the list of columns by extension
    # ----
    debu = True
    if debu:
        df_column_set = set(df_full.columns)
        cols_full_set = set(cols_full)
        coll_miss = cols_full_set - df_column_set
        if coll_miss:
            print(df_full.head(2))
            print (df_full.info())
            print(f"Error: The following required columns are missing from df_full:\n{list(coll_miss)}")
            print(f"df_full currently has these columns (first 10): {list(df_full.columns)[:10]}")
            raise Exception()
    
    # Exec
    # ----
    df_part = df_full[cols_full]
    df_part.columns = df_part.columns.str.strip()
    #
    if pref is not None:
        df_part = df_part.rename(columns=lambda c: c.split("_", 1)[1] if c.startswith(pref) else c)
        
    # Trac
    # ----
    if trac:
       print_yes(df_part, labl="df_part")
    
    # Exit
    # ----
    return df_part

def inpu_fram_exec_selc_1_inte(df_full, filt_list, pref):   
   
    trac = True

    # Exec
    # ----
    # df_full = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)
    # df_full.columns = df_full.columns.str.strip()
    
    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")
    
    cols_base = ["workbook", "date1", "date2", "timepoint", "patient_id", "name"]
    cols_base = ["workbook", "patient_id", "name", "timepoint"]
    cols_vari = [colu for colu in df_full.columns if any(colu.startswith(pref) for pref in filt_list)]
    cols_full = cols_base + cols_vari

    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")
    #
    df_part = df_full[cols_full]
    df_part.columns = df_part.columns.str.strip()
    #
    if pref is not None:
        df_part = df_part.rename(columns=lambda c: c.split("_", 1)[1] if c.startswith(pref) else c)
    
    # Trac
    # ----
    if trac:
        print_yes(df_part, labl="df_part")
    
    # Exit
    # ----
    return df_part

def inpu_fram_exec_selc_1_mixd(df_full, filt_list, pref):   
    
    trac = True

    # Exec
    # ----
    # df_full = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)
    # df_full.columns = df_full.columns.str.strip()
        
    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")
            
    cols_base = ["workbook", "date1", "date2", "timepoint", "patient_id", "name"]
    cols_base = ["workbook", "patient_id", "name", "timepoint"]
    cols_vari = filt_list
    cols_full = cols_base + cols_vari
    
    # Debu : makes only sense when the 'filt_list' contains the list of columns by extension
    # ----
    debu = True
    if debu:
        df_column_set = set(df_full.columns)
        cols_full_set = set(cols_full)
        coll_miss = cols_full_set - df_column_set
        if coll_miss:
            print(df_full.head(2))
            print (df_full.info())
            print(f"Error: The following required columns are missing from df_full:\n{list(coll_miss)}")
            print(f"df_full currently has these columns (first 10): {list(df_full.columns)[:10]}")
            raise Exception()
    
    # Exec
    # ----
    df_part = df_full[cols_full]
    df_part.columns = df_part.columns.str.strip()
    #
    if pref is not None:
        df_part = df_part.rename(columns=lambda c: c.split("_", 1)[1] if c.startswith(pref) else c)
        
    # Trac
    # ----
    if trac:
        print_yes(df_part, labl="df_part")
 
    # Exit
    # ----
    return df_part

# ====
# Wide : limb rows
# ====
def _upda_frax(framTran:FramTran):   
 
    # Opti
    # ----
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 2000) 
    
    # Exec
    # ----
    fram_fram = framTran.fram
    drop_list = framTran.drop_list
    inpu_func = framTran.func
    df_frax = inpu_func(fram_fram, drop_list)
    framTran.frax = df_frax
    
def inpu_frax_exec_selc_1(df_full, drop_list):   
   
    trac = True
    
    # Data
    # ----
    df_full = df_full.copy()
    
    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")

    # Prec
    # ----
    df_full = df_full.drop(drop_list, axis=1)
    
    # Exec
    # ----
    cols_repl = [col for col in df_full.columns if col not in ['R', 'L']]
    df_long = df_full.melt(
        id_vars=cols_repl,
        value_vars=['R', 'L'],
        var_name='Limb',
        value_name='VCSS'
    )
    cols_ordr = cols_repl + ['Limb', 'VCSS']
    df_long = df_long[cols_ordr]
    df_long = df_long.sort_values(by=['workbook', 'Limb']) # note : , adding 'timepoint' would be redundant with 'workbook'
    #
    df_long.columns = df_long.columns.str.strip()

    # Trac
    # ----
    if trac:
        print_yes(df_long, labl="df_long")
    
    # Exit
    # ----
    return df_long

def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass