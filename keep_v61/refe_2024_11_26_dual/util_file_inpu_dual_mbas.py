import os
import pandas as pd
from util_file_mngr import write

def inp1(file_path):
       
    # Vars
    trac = True
    
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.ixam.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)

    df11 = df1.copy() # keep all
    df12 = df1[~df1['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df1[~df1['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    if trac:
        df = df11
        print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        write(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    # ----
    # Exit
    # ----
    return df11, df12, df13

def inp2(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, grop_name):
    
    # Vars
    trac = True
    
    # ----
    # 1:Create 'inve' = ['NA', 'VI']
    # ----
    df2 = df1.copy()
    #df2['inve'] = df2['ceap'].apply(lambda x: 'VI' if x in cond_list else 'NA')
    df2 = df2.drop(columns=['ceap'])
    df2 = df2.groupby(['doss', grop_name], as_index=False).first() # indx_name = 'mbre'
    print(f"df2:{type(df2)}\n{df2}\n:{df2.index}")
    
    df3 = df2.pivot_table(index=indx_name, columns=colu_name, aggfunc='size', fill_value=0)
    df3 = df3.reindex(index=indx_cate_list, columns=colu_cate_list, fill_value=0)
    df4 = df3.T

    if trac:
        print(f"\nStep 1 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}\nSum:{df3.sum().sum()}")
        write(f"\nStep 1 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}\nSum:{df3.sum().sum()}")
        print(f"\nStep 1 : df4.size:{len(df4)} df4.type:{type(df4)}\n{df4}\n:{df4.index}\n:{df4.columns}\nSum:{df4.sum().sum()}")
        write(f"\nStep 1 : df4.size:{len(df4)} df4.type:{type(df4)}\n{df4}\n:{df4.index}\n:{df4.columns}\nSum:{df4.sum().sum()}")
        
    # ----
    # 2:Chck with row & columnn 'tota'
    # ----
    df9 = df3.copy()
    df9_sum = df9.sum().sum()
    df9['tota'] = df9.sum(axis=1)
    # Calculate total sum across all columns and add as a new row
    total_row = df9.sum(axis=0)
    total_row.name = 'tota'  # Set the name for the total row
    df9 = pd.concat([df9, total_row.to_frame().T])
    df9 = df9.rename_axis(indx_name, axis='index')
    print(f"\nStep 2 : df9.size:{len(df9)} df9.type:{type(df9)}\n{df9}\n:{df9.index}\n:{df9.columns} sum:{df9_sum}")
    write(f"\nStep 2 : df9.size:{len(df9)} df9.type:{type(df9)}\n{df9}\n:{df9.index}\n:{df9.columns} sum:{df9_sum}")
   
    # ----
    # Exit
    # ----
    return df2, df3

def inp3(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi):
           
    # Vars
    trac = True
    
    # ----
    # Exec
    # ----
    df2 = df1.copy()
   
    # ----
    # Reindex [since some duplicates where droped ind 'df1']
    # ----
    df2 = df2.reset_index(drop=True)
    
    # ----
    # Compute 'stra' and 'ordi' indexes
    # ----
    # Convert sexe to ordinal values
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    df2[indx_name_stra] = df2[indx_name].map({indx_cate_nam1: 0, indx_cate_nam2: 1}) # df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})

    # Convert age bins to ordinal values
    age_bin_ordi = {bin: i for i, bin in enumerate(colu_cate_list)}
    df2[colu_name_ordi] = df2[colu_name].map(age_bin_ordi)
    
    if trac:
        print(f"\nStep 0 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
        write(f"\nStep 0 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")

    
    # ----
    # Exit
    # ----
    return df2