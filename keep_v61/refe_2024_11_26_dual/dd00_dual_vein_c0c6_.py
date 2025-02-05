import os
import sys
import pandas as pd

from ke00_stat import stat_glob
from util_file_inpu_dual_vein import inp1, inp2, inp3
from util_file_mngr import set_file_objc, write
from datetime import datetime

# ----
# Inpu
# ----
def inpu(file_path, indx_name, indx_cate_list):     
     
    # Step 21
    df11, df12, df13 = inp1(file_path)
    
    # Step 22
    colu_cate_list = ['NA', 'VI']
    colu_name = 'c0c6'
    grop_name = 'mbre'

    df_line21, df_tabl_21 = inp2(df11, indx_cate_list, colu_cate_list, indx_name, colu_name, grop_name)
    df_line22, df_tabl_22 = inp2(df12, indx_cate_list, colu_cate_list, indx_name, colu_name, grop_name)
    df_line23, df_tabl_23 = inp2(df13, indx_cate_list, colu_cate_list, indx_name, colu_name, grop_name)   
    
    indx_name_stra = 'sexe_stra'
    colu_name_ordi = 'age_bin_ordi'
    df_line = inp3(df_line21, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    
    # Exit
    return df_tabl_21, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def ke10_main(file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
            
        # ----
        # VEIN = f(MBRE)
        # ----
        mbre_exec = True
        if mbre_exec:
            write (">>> >>> >>>")
            write (f"VEIN = f(MBRE) :  {date_form}")
            write (">>> >>> >>>")
            # Inpu
            indx_name = 'mbre'
            indx_cate_list = ['G', 'D']
            df_tabl, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
            inpu(file_path, indx_name, indx_cate_list)
            trac = True
            if trac:
                print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
                write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
            # Stat
            what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}" 
            yate = False 
            stat_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, yate)
            pass 
    
        # ----
        # VEIN = f(SEXE)
        # ----
        sexe_exec = True
        if sexe_exec:
            write (">>> >>> >>>")
            write (f"VEIN = f(SEXE) :  {date_form}")
            write (">>> >>> >>>")
            # Inpu
            indx_name = 'sexe'
            indx_cate_list = ['M', 'F']
            df_tabl, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
            inpu(file_path, indx_name, indx_cate_list)
            trac = True
            if trac:
                print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
                write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
            # Stat
            what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}" 
            yate = False 
            stat_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, yate)
            pass    

def dd00_dual_vein_c0c6():

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
        
    # Step 2
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}jrnl.txt')
    ke10_main(file_path, jrnl_file_path)
    
if __name__ == "__main__":
    dd00_dual_vein_c0c6()