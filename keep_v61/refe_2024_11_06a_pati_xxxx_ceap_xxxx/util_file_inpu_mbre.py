from enum import Enum
import os
import numpy as np
import pandas as pd
from util_file_mngr import write

def inp1(file_path, filt_name, filt_valu):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    write(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def in21(df1, colu_cate_list, ceap_mono):
 
    # ----
    # Step 1 : project
    #       doss sexe mbre ceap
    # 0    D9972    F    G   C2
    # 1    D9972    F    G   C6
    # ----
    df2 = df1[['doss', 'age_bin', 'sexe', 'mbre', 'unbi', 'ceap']] 
    print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    
    if not ceap_mono:
        df2 = df1[['doss', 'age_bin', 'sexe', 'mbre', 'unbi', 'ceap']] 
        print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
        write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    else:
        if False:
            data = {
                'doss': ['D9972', 'D9972', 'D9972', 'D9972', 'D9921'],
                'sexe': ['F', 'F', 'F', 'F', 'M'],
                'mbre': ['G', 'G', 'D', 'D', 'G'],
                'ceap': ['C2', 'C5', 'C2', 'C6', 'C3']
            }
            df1 = pd.DataFrame(data)
        rank_dict = {rank: idx for idx, rank in enumerate(colu_cate_list)}
        # Create a new column with the rank of 'ceap'
        df1['ceap_rank'] = df1['ceap'].map(rank_dict)
        # Group by 'doss' and 'mbre', and get the row with the highest 'ceap_rank'
        df2 = df1.loc[df1.groupby(['doss', 'mbre'])['ceap_rank'].idxmax()]
        #
        df2 = df2[['doss', 'age_bin', 'sexe', 'mbre', 'ceap']] 
        print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
        write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    
    # ----
    # Exit
    # ----   
    return df2

def in23(df1, indx_cate_list, colu_cate_list, indx_name, colu_name):

    if False:
        data = {
            'doss': ['D9972', 'D9972', 'D9972', 'D9972', 'D9921'],
            'sexe': ['F', 'F', 'F', 'F', 'M'],
            'mbre': ['G', 'G', 'D', 'D', 'G'],
            'ceap': ['C2', 'C6', 'C2', 'C6', 'C3']
        }
        df = pd.DataFrame(data)
    df2 = pd.DataFrame(0, index=indx_cate_list, columns=colu_cate_list)

    print(f"\ndf1.size:{len(df1)} df1.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")
    write(f"\ndf1.size:{len(df1)} df1.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")

    # Populate the new DataFrame with counts from the original DataFrame
    for idx, row in df1.iterrows():
        mbre = row[indx_name]
        ceap = row[colu_name]
        if ceap in colu_cate_list:
            df2.loc[mbre, ceap] += 1
    print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    xlsx = True
    if xlsx: 
        file_name = 'ceap_age.xlsx'
        df2.to_excel(file_name, index=False)
    
    # ----
    # 2:Chck with row & columnn 'tota'
    # ----
    df2_sum = df2.sum().sum()
    df2['tota'] = df2.sum(axis=1)
    # Calculate total sum across all columns and add as a new row
    total_row = df2.sum(axis=0)
    total_row.name = 'tota'  # Set the name for the total row
    df2 = pd.concat([df2, total_row.to_frame().T])
    df2.index.name = indx_name
    df2.columns.name = colu_name 
    indx_list = indx_cate_list.copy() ; indx_list.append('tota')
    colu_list = colu_cate_list.copy() ; colu_list.append('tota')
    df2.columns = colu_list
    df2.index = indx_list
    print(f"\nStep 2 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns} sum:{df2_sum}")
    write(f"\nStep 2 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")

    # ----
    # 3: Drop row & columnn 'tota'
    # ----
    df2 = df2.drop(index='tota')
    df2 = df2.drop(columns='tota')
    df2 = df2.rename_axis(indx_name, axis='index')
    print(f"\nStep 3 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns} sum:{df2_sum}")
    write(f"\nStep 3 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    
    # ----
    # Step 2
    # ----
    df2.index = indx_cate_list
    df2.columns = colu_cate_list
    df2.index.name = indx_name
    df2.columns.name = colu_name  
    print(f"df2.size{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n:{df2.sum().sum()}")
    write(f"df2.size{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n:{df2.sum().sum()}")
        
    # Exit
    return df2
    
def inp2(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, ceap_mono, filt_name, filt_valu):
    df1 = in21(df1, colu_cate_list, ceap_mono)
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    write(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    print(f"grop_levl : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    df2 = in23(df2, indx_cate_list, colu_cate_list, indx_name, colu_name)
    return df2

def inp3(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_ordi, colu_name_ordi):
           
    # Vars
    trac = True
    
    # ----
    # Exec
    # ----
    df2 = df1.copy()
    
    # ----
    # Reindex [since some duplicates where droped]
    # ----
    df2 = df2.reset_index(drop=True)
    
    # ----
    # Compute 'stra' and 'ordi' indexes
    # ----
    # Convert age bins to ordinal values
    indx_ordi = {bin: i for i, bin in enumerate(indx_cate_list)}
    df2[indx_name_ordi] = df2[indx_name].map(indx_ordi)

    # Convert ceap to ordinal values
    ceap_ordi = {bin: i for i, bin in enumerate(colu_cate_list)}
    df2[colu_name_ordi] = df2[colu_name].map(ceap_ordi)
    if trac:
        print(f"\nStep 0 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
        write(f"\nStep 0 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    
    # ----
    # Exit
    # ----
    return df2

def inp4(df2):
    
    # Exec
    mbrG_marg = df2.sum(axis=1)  # Row-wise sums
    mbrD_marg = df2.sum(axis=0)  # Column-wise sums
    df3 = pd.DataFrame({'mbrG': mbrG_marg, 'mbrD': mbrD_marg})
    df3.columns.name = 'mbre'
    print(f"df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns} {df3.sum().sum()}")   
        
    # Exit
    return df3 

def inp5(df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Trac
    trac = True
        
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    indx_expe_nam1 = f'{indx_cate_nam1}_expe'
    indx_expe_nam2 = f'{indx_cate_nam2}_expe'
    
    # Observed counts from your table
    if False:
        observed_data = {
            'CEAP': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
            'Males': [31, 5, 44, 93, 38, 18, 97],
            'Females': [36, 6, 54, 156, 59, 35, 99]
        }
        df1 = pd.DataFrame(observed_data)  
    print(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    write(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")

    # Exec
    # ----
    if trac:
        dfT = pd.DataFrame({ indx_name: [df.loc[indx_cate_nam1].sum(), df.loc[indx_cate_nam2].sum(), df.loc[indx_cate_nam1].sum()+df.loc[indx_cate_nam2].sum()]}, index=[indx_cate_nam1, indx_cate_nam2, 'T'])
        print(f"\nContingency table  : totals:{dfT.T}")
        write(f"\nContingency table  : totals:{dfT.T}")
    observed_males = df.loc[indx_cate_nam1].sum()
    observed_females = df.loc[indx_cate_nam2].sum()


    #total_males = 378   # Total males in the dataset
    #total_females = 498 # Total females in the dataset
    observed_population = observed_males + observed_females

    # Global sex ratios
    male_ratio = observed_males / observed_population  # ~43%
    female_ratio = observed_females / observed_population  # ~57%

    # Step 1: Compute expected counts based on global proportions
    dfc = pd.DataFrame()
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    dfc[indx_cate_nam1] = df.loc[indx_cate_nam1]
    dfc[indx_cate_nam2] = df.loc[indx_cate_nam2]
    dfc[indx_expe_nam1] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * male_ratio # df['Expected_Males'] = (df['Males'] + df['Females']) * male_ratio
    dfc[indx_expe_nam2] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * female_ratio # df['Expected_Females'] = (df['Males'] + df['Females']) * female_ratio
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")

    # Step 2: Compare observed vs expected counts
    medi = False
    if medi:
        dfc[f'{indx_cate_nam1}_devi'] = df.loc[indx_cate_nam1] - dfc[indx_expe_nam1] # df['Males'] - df['Expected_males']
        dfc[f'{indx_cate_nam2}_devi'] = df.loc[indx_cate_nam2] - dfc[indx_expe_nam2] # df['Females'] - df['Expected_Females']
        print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
        write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    print(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")

    # Exit
    return dfc