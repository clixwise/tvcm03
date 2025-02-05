import os
import sys
import pandas as pd

from je00_stat import snxn_glob
from util_file_inpu_ceap import inp1, inp2, inp3, inp4
from util_file_mngr import set_file_objc, write
from datetime import datetime

'''
Step 2 : df2.size:9 df2.type:<class 'pandas.core.frame.DataFrame'>
      NA  C0  C1  C2   C3  C4  C5   C6  tota
NA     0   0   0   2   18   5   0   22    47
C0     0   0   0   1    8   2   2   14    27
C1     0   0   1   0    3   2   1    0     7
C2     3   1   1  10   14   3   3   18    53
C3    20   9   0  15   80  16   3   14   157
C4     9   2   1   3   17  23   7    5    67
C5     7   5   1   5    4   2   6    8    38
C6    39  29   3  33    9   5   7   21   146
tota  78  46   7  69  153  58  29  102   542
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'tota'], dtype='object')
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'tota'], dtype='object') sum:542
'''
# ----
# Inpu
# ----
def inpu_init(file_path):
    df11, df12, df13 = inp1(file_path)
    return df11, df12, df13

def inpu11(df11, sexe):       
    # ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    indx_cate_list = colu_cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    indx_name = colu_name = 'ceap'  
    df_tabl_21 = inp2(df11, indx_cate_list, colu_cate_list, indx_name, colu_name, sexe)
    indx_name_stra = 'ceaL'
    colu_name_ordi = 'ceaR'
    df_line = inp3(df11, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    # Exit
    return df_tabl_21, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def inpu12(df12, sexe):    
    # ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    indx_cate_list = colu_cate_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    indx_name = colu_name = 'ceap'  
    df_tabl_22 = inp2(df12, indx_cate_list, colu_cate_list, indx_name, colu_name), sexe
    indx_name_stra = 'ceaL'
    colu_name_ordi = 'ceaR'
    df_line = inp3(df12, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)     
    # Exit
    return df_tabl_22, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def inpu13(df13, sexe):
    # ['C3', 'C4', 'C5', 'C6']
    indx_cate_list = colu_cate_list = ['C3', 'C4', 'C5', 'C6'] 
    indx_name = colu_name = 'ceap'  
    df_tabl_23 = inp2(df13, indx_cate_list, colu_cate_list, indx_name, colu_name, sexe)
    indx_name_stra = 'ceaL'
    colu_name_ordi = 'ceaR'
    df_line = inp3(df13, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    # Exit
    return df_tabl_23, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def ke10_main(filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
    
        if filt_name != 'sexe':
            raise Exception()
        sexe = filt_valu 
        # Inpu
        df11, df12, df13 = inpu_init(file_path)
        df_tabl, df_line, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
        inpu11(df11, sexe)
        trac = True
        if trac:
            print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
            write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}")
        indx_name = "Lcea" ; df_tabl = df_tabl.rename_axis(indx_name, axis=0)
        colu_name = "Rcea" ; df_tabl = df_tabl.rename_axis(colu_name, axis=1)
        if trac:
            print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}\nSum:{df_tabl.sum().sum()}")
            write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}\nSum:{df_tabl.sum().sum()}")
          # Stat
        what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}" 
        yate = False # because of desc : repeat expansion 
        snxn_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, yate)
        pass    

def ke53_ceap_ceap_c3c6_full_abso():

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
        
    # Step : sexe None
    filt_name = 'sexe'
    filt_valu = None # 'M' 'F'
    suppress_suffix = ".py"
    script_name_A = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_A}jrnl_{filt_valu}.txt' if filt_valu is not None else f'jrnl_{script_name_A}.txt')
    ke10_main(filt_name, filt_valu, file_path, jrnl_file_path)
    
    # Step : sexe 'M'
    filt_name = 'sexe'
    filt_valu = 'M'
    suppress_suffix = ".py"
    script_name_M = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_M}jrnl_{filt_valu}.txt' if filt_valu is not None else f'jrnl_{script_name_M}.txt')
    ke10_main(filt_name, filt_valu, file_path, jrnl_file_path)
    
    # Step : sexe 'F'
    filt_name = 'sexe'
    filt_valu = 'F'
    suppress_suffix = ".py"
    script_name_F = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_F}jrnl_{filt_valu}.txt' if filt_valu is not None else f'jrnl_{script_name_F}.txt')
    ke10_main(filt_name, filt_valu, file_path, jrnl_file_path)
    
if __name__ == "__main__":
    ke53_ceap_ceap_c3c6_full_abso()