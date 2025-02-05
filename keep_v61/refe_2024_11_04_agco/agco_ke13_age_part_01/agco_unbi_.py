from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
from stat_func import stat_exec
from util_file_mngr import set_file_objc, write

def inp1(file_path):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)

    df11 = df1.copy() # keep all
    df12 = df1[~df1['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df1[~df1['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def inpu(file_path, filt_name, filt_valu):
    df11, df12, df13 = inp1(file_path)
    df1 = df11.drop_duplicates(subset=['doss'], keep='first')
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    df3 = df2[['doss', 'age']] 
    print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    print(f"\nStep 1 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
    write(f"\nStep 1 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
    return df2

def agco_unbi():

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
    jrnl_file_path = os.path.join(script_dir, f"{script_name}jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        
        def main(file_path, filt_name, filt_valu):
            
            date_curr = datetime.now()
            date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
            write ("\n")
            write (">>> >>> >>>")
            write (f'AGCO = f({filt_name}={filt_valu}) : {date_form}')
            write (">>> >>> >>>")
            
            df = inpu(file_path, filt_name, filt_valu)
            
            colu_cate_list = ['age']
            indx_name = 'indx'
            colu_name = 'colu'
            what = f"'{indx_name}' '{colu_name}' {colu_cate_list} '{filt_name}'='{filt_valu}'"
            
            '''
            Note : Kolmogorov-Smirnov (K-S) does not add anything in our case
            It is of use when we want to compare to a reference distribution : log-normal, etc.
            '''
            df_resu = stat_exec(what, df, colu_cate_list, indx_name, colu_name)
            
            return df_resu
            
        df_resu_T = main(file_path, 'unbi', None)
        df_resu_M = main(file_path, 'unbi', 'U')
        df_resu_F = main(file_path, 'unbi', 'B')
        df_resu = pd.concat([df_resu_T, df_resu_M, df_resu_F], axis=0)
        df_resu.index = ['T', 'M', 'F']
        print(f"\nStep 1 : df_resu.size:{len(df_resu)} df_resu.type:{type(df_resu)}\n{df_resu}\n:{df_resu.index}\n:{df_resu.columns}")
        # Display the DataFrame
        print(f"\nRESU")
        write(f"\nRESU")
        with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
            print(f"UNBI ages continus\n df_resu.size:{len(df_resu)} df_resu.type:{type(df_resu)}\n{df_resu}\n:{df_resu.index}\n:{df_resu.columns}")
            write(f"UNBI ages continus\n{df_resu}")
            xlsx = True
        if xlsx: 
            file_name = 'agco_unbi.xlsx'
            df_resu.to_excel(file_name, index=False)

if __name__ == "__main__":
    agco_unbi()