import os
import sys
import pandas as pd

from ke00_stat import stat_glob
from util_file_inpu_pati import inp1, inp2, inp3
from util_file_mngr import set_file_objc, write
from datetime import datetime
'''
df2.size:10 df2.type:<class 'pandas.core.frame.DataFrame'>
mbre       G    D  tota
age_bin
10-19      5    5    10
20-29     13   13    26
30-39     33   33    66
40-49     58   58   116
50-59     87   87   174
60-69     81   81   162
70-79     67   67   134
80-89     17   17    34
90-99      1    1     2
tota     362  362   724
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', 'tota'], dtype='object', name='age_bin')
:Index(['G', 'D', 'tota'], dtype='object', name='mbre') sum:724
'''
'''
FILTRAGE VIA 'DF12' pour éliminer les 105 'NA' sinon ce programme serait sans objet
puisque il y aurait 724 = 2*362 membres G,D ; on élimine donc 105 membres en 'NA' 
'''

# ----
# Inpu
# ----
def inpu(file_path):     
     
    # Step 21
    df11, df12, df13 = inp1(file_path)
    df11 = df11.drop_duplicates(subset=['doss', 'age_bin', 'mbre'])
    df12 = df12.drop_duplicates(subset=['doss', 'age_bin', 'mbre']) # SELECTION QUI ELIMINE 105 MEMBRES 'NA'
    df13 = df13.drop_duplicates(subset=['doss', 'age_bin', 'mbre'])
    
    # Step 22
    colu_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    indx_cate_list = ['G', 'D']
    colu_name = 'age_bin'
    indx_name = 'mbre'
    
    df_tabl_21 = inp2(df11, indx_cate_list, colu_cate_list, indx_name, colu_name)
    df_tabl_22 = inp2(df12, indx_cate_list, colu_cate_list, indx_name, colu_name) # SELECTION QUI ELIMINE 105 MEMBRES 'NA'
    df_tabl_23 = inp2(df13, indx_cate_list, colu_cate_list, indx_name, colu_name)   
    
    indx_name_stra = 'mbre_stra'
    colu_name_ordi = 'age_bin_ordi'
    df_line = inp3(df12, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)
    
    df_norm = pd.DataFrame() # TODO see 'ke10_ceap_xxx.py' and 'ke30_ceap_xxx.py'
        
    # Exit
    return df_tabl_22, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

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
     
def ke19_pati_agbi_mbre_c3c6_full_abso():

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
    ke19_pati_agbi_mbre_c3c6_full_abso()