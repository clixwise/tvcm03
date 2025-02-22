from datetime import datetime
import os
import sys
import pandas as pd
from perp_ormo01_ import perp_ormo01_exec
from util_file_mngr import write

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
    perp_ormo01_exec(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name)
    
    # Perplexity
    # ----------
    date_curr = datetime.now()
    date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} : stat_glob_perp_2024_12_15')
    write (">>> >>> >>>")
    pass