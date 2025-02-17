import os
import sys
import pandas as pd
from mist_logi01_ import mist_logi01_exec
from util_file_inpu_mbre import inp1
from util_file_mngr import set_file_objc, write
from datetime import datetime

# ----
# Inpu
# ----
def inpu(filt_name, filt_valu, file_path):      
    
    df11, df12, df13 = inp1(file_path, filt_name, filt_valu) 
  
    # Exit
    return df11, df12, df13

# ----
# Stat
# ----

def stat_glob(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name):
    
    date_curr = datetime.now()
    date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} : stat_glob_perp_2025_02_13')
    write (">>> >>> >>>")
    #  
    mist_logi01_exec(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name)
    
    # Perplexity
    # ----------
    date_curr = datetime.now()
    date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} : stat_glob_perp_2024_12_15')
    write (">>> >>> >>>")
    pass

# ----
# Stat
# ----

def stat_glob(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name):
    
    date_curr = datetime.now()
    date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} : stat_glob_perp_2025_02_13')
    write (">>> >>> >>>")
    #  
    mist_logi01_exec(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name)
    
    # Perplexity
    # ----------
    date_curr = datetime.now()
    date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} : stat_glob_perp_2024_12_15')
    write (">>> >>> >>>")
    pass

def ke30_main(ind1_name, ind1_cate_list, ind2_name, ind2_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file: # LOG des résultats dans fichier
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
    
        # Selector
        # --------
        df1, df2, df3 = inp1(file_path, filt_name, filt_valu)  
        
        # Inpu ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # -----------------------------------------------------
        colu_cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']    
        df11, df12, df13 = inpu(filt_name, filt_valu, file_path)
        # Stat
        colu_name = 'ceap'
        what = f"'{ind1_name}''{ind2_name}' '{colu_name}' ; {ind1_cate_list} {ind2_cate_list} {colu_cate_list}"
        df_line = df11
        stat_glob(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name)
        
        # Inpu ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # -----------------------------------------------
        colu_cate_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # ...
        
        # Inpu ['C3', 'C4', 'C5', 'C6']
        # -----------------------------
        colu_cate_list = ['C3', 'C4', 'C5', 'C6']
        # ...
        pass 