import os
import sys
import pandas as pd

from ke00_stat import stat_glob
from util_file_inpu_mbre import inp1, inp2, inp3
from util_file_mngr import set_file_objc, write
from datetime import datetime

'''
df2.size:10 df2.type:<class 'pandas.core.frame.DataFrame'>
sexe       M    F  tota
age_bin
10-19      3    2     5
20-29      6    7    13
30-39     10   23    33
40-49     26   32    58
50-59     41   46    87
60-69     35   46    81
70-79     29   38    67
80-89      6   11    17
90-99      0    1     1
tota     156  206   362
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', 'tota'], dtype='object', name='age_bin')
:Index(['M', 'F', 'tota'], dtype='object', name='sexe') sum:362
'''

# ----
# Inpu
# ----
def inpu(file_path, filt_name, filt_valu):     
     
    # Step 21
    df11, df12, df13 = inp1(file_path, filt_name, filt_valu)
    
    # Step 22
    ceap_mono = False
    colu_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    indx_cate_list = ['U', 'B']
    colu_name = 'age_bin'
    indx_name = 'unbi'
    
    df_tabl_23 = inp2(df13, indx_cate_list, colu_cate_list, indx_name, colu_name, ceap_mono, filt_name, filt_valu)   
   
    indx_name_stra = 'unbi_stra'
    colu_name_ordi = 'age_bin_ordi'
    df_line = inp3(df13, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    
    df_norm = pd.DataFrame() # TODO see 'ke10_ceap_xxx.py' and 'ke30_ceap_xxx.py'
    
    # Exit
    return df_tabl_23, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def ke10_main(file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
    
        # Inpu
        filt_name = 'sexe'
        filt_valu = None
        df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
        inpu(file_path, filt_name, filt_valu)
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

def ke46_pati_agbi_unbi_c3c6_PART_abso():

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
    ke46_pati_agbi_unbi_c3c6_PART_abso()