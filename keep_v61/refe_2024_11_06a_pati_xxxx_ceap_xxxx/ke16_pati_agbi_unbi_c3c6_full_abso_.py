import os
import sys
import pandas as pd

from ke00_stat import stat_glob
from util_file_inpu_pati import inp1, inp2, inp3
from util_file_mngr import set_file_objc, write
from datetime import datetime

'''
Step 2 : df2.size:3 df2.type:<class 'pandas.core.frame.DataFrame'>
age_bin  10-19  20-29  30-39  40-49  50-59  60-69  70-79  80-89  90-99  tota
unbi
U            3      5      8     24     25     15     17      8      0   105
B            2      8     25     34     62     66     50      9      1   257
tota         5     13     33     58     87     81     67     17      1   362
:Index(['U', 'B', 'tota'], dtype='object', name='unbi')
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
       '90-99', 'tota'],
      dtype='object', name='age_bin') sum:362
'''

# ----
# Inpu
# ----
def inpu(file_path):     
     
    # Step 21
    df11, df12, df13 = inp1(file_path)
    df11 = df11.drop_duplicates(subset=['doss', 'age_bin', 'unbi'])
    df12 = df12.drop_duplicates(subset=['doss', 'age_bin', 'unbi'])
    df13 = df13.drop_duplicates(subset=['doss', 'age_bin', 'unbi'])
    
    # Step 22
    colu_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    indx_cate_list = ['U', 'B']
    colu_name = 'age_bin'
    indx_name = 'unbi'
    
    df_tabl_21 = inp2(df11, indx_cate_list, colu_cate_list, indx_name, colu_name)
    df_tabl_22 = inp2(df12, indx_cate_list, colu_cate_list, indx_name, colu_name)
    df_tabl_23 = inp2(df13, indx_cate_list, colu_cate_list, indx_name, colu_name)   
    if (not df_tabl_22.equals(df_tabl_21)) or (not df_tabl_23.equals(df_tabl_21)):
        raise Exception()
    
    indx_name_stra = 'unbi_stra'
    colu_name_ordi = 'age_bin_ordi'
    df_line = inp3(df11, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    
    df_norm = pd.DataFrame() # TODO see 'ke10_ceap_xxx.py' and 'ke30_ceap_xxx.py'
    
    # Exit
    return df_tabl_21, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def ke10_main(file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
    
        # Inpu
        df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
        inpu(file_path)
        trac = True
        if trac:
            print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
            write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
            dfT = pd.DataFrame({ indx_name: [df_tabl.loc[indx_cate_list[0]].sum(), df_tabl.loc[indx_cate_list[1]].sum(), df_tabl.loc[indx_cate_list[0]].sum()+df_tabl.loc[indx_cate_list[1]].sum()]}, index=[indx_cate_list[0], indx_cate_list[1], 'T'])
            print(f"\nContingency table  : totals:{dfT.T}")
            write(f"\nContingency table  : totals:{dfT.T}")
            print(f"\nContingency table normalized : df_norm.size:{len(df_norm)} df_norm.type:{type(df_norm)}\n{df_norm}\n:{df_norm.index}\n:{df_norm.columns}")
            write(f"\nContingency table normalized : df_norm.size:{len(df_norm)} df_norm.type:{type(df_norm)}\n{df_norm}\n:{df_norm.index}\n:{df_norm.columns}")

         # Stat
        what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}" 
        yate = False 
        stat_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm, yate)
        pass    

def ke16_pati_agbi_unbi_c3c6_full_abso():

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
    ke16_pati_agbi_unbi_c3c6_full_abso()