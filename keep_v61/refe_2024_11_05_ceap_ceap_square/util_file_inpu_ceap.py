import os
import numpy as np
import pandas as pd
import ast
from util_file_mngr import write

def inp1(file_path):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.ixam.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)

    df11 = df1.copy() # keep all
    df12 = df1[~df1['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df1[~df1['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def in21(df1, indx_cate_list):
 
    # ----
    # Step 1 : project
    #       doss sexe mbre ceap
    # 0    D9972    F    G   C2
    # 1    D9972    F    G   C6
    # ----
    df2 = df1[['doss', 'sexe', 'mbre', 'ceap']] 
    print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    write(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")

    # ----
    # Step 2 : create lisL, R arrays
    #        doss sexe                      lisL                      lisR
    # 0    D10077    F  [1, 0, 0, 0, 0, 0, 0, 0]  [0, 0, 0, 0, 1, 0, 0, 0]
    # 1    D10103    M  [1, 0, 0, 0, 0, 0, 0, 0]  [0, 0, 0, 0, 1, 0, 0, 0]
    # ----
    df3 = df2.copy() # avoids warning
    for ceap_class in indx_cate_list:
        df3.loc[:, ceap_class] = df3['ceap'].apply(lambda x: 1 if x == ceap_class else 0)
    df3 = df3.head(10000)
    # Group by 'doss' and create separate arrays for L and R legs
    df3 = df3.groupby(['doss', 'sexe']).agg(
        lisL=('mbre', lambda x: df3.loc[x.index[x == 'G'], indx_cate_list].sum().astype(int).tolist()),
        lisR=('mbre', lambda x: df3.loc[x.index[x == 'D'], indx_cate_list].sum().astype(int).tolist())
    ).reset_index()
    print(f"Step 2 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
    write(f"Step 2 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
 
    # ----
    # Exit
    # ----   
    return df3

def in22(df1, indx_cate_list, colu_cate_list, indx_name, colu_name):
  
    # ----
    # Prec
    # ----
    row_index = df1.index[0]  # Assuming you want the first row
    # Get the lists from the specified row
    row_lisL = df1.loc[row_index, 'lisL']
    row_lisR = df1.loc[row_index, 'lisR']
    # Get the sizes
    size_lisL = len(row_lisL)
    size_lisR = len(row_lisR)
    if not ((size_lisL == size_lisR) & (size_lisR == len(indx_cate_list))):
        raise Exception()

    # ----
    # Step 1: Create an 8x8 DataFrame
    # ----
    leng = len(indx_cate_list)
    df2 = pd.DataFrame(np.zeros((leng, leng), dtype=int))
    #
    pairs = []
    for index, row in df1.head(10000).iterrows():
        lisL = row['lisL']
        lisR = row['lisR']
        for i in range(leng):
            for j in range(leng):
                if lisL[i] == 1 and lisR[j] == 1:
                    pairs.append((i, j))
                    df2.iloc[i, j] += 1
    # print(f"Row {index}: {pairs}") # to list all pairs
    
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
    write(f"\nStep 2 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns} sum:{df2_sum}")
    xlsx = False
    if xlsx: 
        file_name = 'xlsx_01_.xlsx'
        df2.to_excel(file_name, index=False)

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
    print(f"df2.size\n{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n:{df2.sum().sum()}")
    write(f"df2.size\n{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n:{df2.sum().sum()}")
        
    # Exit
    return df2

def in23(df2):

    df2_sum = df2.sum().sum()
    dfp = df2.applymap(lambda x: round((x / df2_sum) * 100, 2))
    print(f"dfp.size\n{len(dfp)} dfp.type:{type(dfp)}\n{dfp}\n:{dfp.index}\n:{dfp.columns}\n:{dfp.sum().sum()}")  
    dfp = df2.applymap(lambda x: round((x / df2_sum) * 100))
    print(f"dfp.size\n{len(dfp)} dfp.type:{type(dfp)}\n{dfp}\n:{dfp.index}\n:{dfp.columns}\n:{dfp.sum().sum()}") 
    pass     
    
def inp2(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, sexe):
    df1 = in21(df1, indx_cate_list)
    df2 = df1 if sexe is None else df1[df1['sexe'] == sexe]
    print(f"sexe : {sexe} : df1.size={len(df1)} df2.size={len(df2)}")
    write(f"sexe : {sexe} : df1.size={len(df1)} df2.size={len(df2)}")
    df3 = in22(df2, indx_cate_list, colu_cate_list, indx_name, colu_name)
    in23(df3)
    return df3

#
# INSPIRED FROM 'C:\tate01\grph01\gr05\keep_v53\prog\g01_tabl_xxx_v10_inpu.py'
#
def inp3(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_ordi, colu_name_ordi):
 
    # Meth 1
    # ------
    df = df1.copy()
    
    # Step 2: Nemar : create binary indicators for each CEAP class
    # ------
    
    # For each 'pati'' : create NA,...C6 columns
    for ceap_class in colu_cate_list:
        df[ceap_class] = df['ceap'].apply(lambda x: 1 if x == ceap_class else 0)
    print(f"df.size\n{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # For each 'doss' : group by 'doss' and create separate arrays for L and R legs
    ceap_unbi_abso_deta = df.groupby(['doss', 'sexe', 'unbi']).agg(
        L_array=('mbre', lambda x: df.loc[x.index[x == 'G'], colu_cate_list].sum().astype(int).tolist()),
        R_array=('mbre', lambda x: df.loc[x.index[x == 'D'], colu_cate_list].sum().astype(int).tolist())
    ).reset_index()
    ceap_unbi_abso_deta = ceap_unbi_abso_deta.rename(columns={ 'L_array': indx_name_ordi, 'R_array': colu_name_ordi })
    
    # Text variant
    # ------------
    def map_binary_to_list(binary_list, lst):
        return [lst[i] if binary_list[i] == 1 else 'x' for i in range(len(binary_list))]
    ceap_unbi_abso_deta['L'] = ceap_unbi_abso_deta[indx_name_ordi].apply(lambda x: map_binary_to_list(x, indx_cate_list))
    ceap_unbi_abso_deta['R'] = ceap_unbi_abso_deta[colu_name_ordi].apply(lambda x: map_binary_to_list(x, indx_cate_list))
    print(f"ceap_unbi_abso_deta:{type(ceap_unbi_abso_deta)}\n{ceap_unbi_abso_deta}\n:{ceap_unbi_abso_deta.index}")
    write(f"ceap_unbi_abso_deta:{type(ceap_unbi_abso_deta)}\n{ceap_unbi_abso_deta}\n:{ceap_unbi_abso_deta.index}")
   
    # Verification
    # ------------
    sum_lisL = ceap_unbi_abso_deta[indx_name_ordi].apply(pd.Series).sum().tolist()
    sum_lisR = ceap_unbi_abso_deta[colu_name_ordi].apply(pd.Series).sum().tolist()
    sum_list = [l + r for l, r in zip(sum_lisL, sum_lisR)]
    print(f"Total1 '{indx_name_ordi}':{sum_lisL} '{colu_name_ordi}':{sum_lisR} list:{sum_list}")
    print(f"Total2 '{indx_name_ordi}:{sum(sum_lisL)} '{colu_name_ordi}':{sum(sum_lisR)} list:{sum(sum_list)}")
    write(f"Total1 '{indx_name_ordi}':{sum_lisL} '{colu_name_ordi}':{sum_lisR} list:{sum_list}")
    write(f"Total2 '{indx_name_ordi}:{sum(sum_lisL)} '{colu_name_ordi}':{sum(sum_lisR)} list:{sum(sum_list)}")

    # Exit
    # ----
    return ceap_unbi_abso_deta
 
def inp4(df2):
    
    # Exec
    mbrG_marg = df2.sum(axis=1)  # Row-wise sums
    mbrD_marg = df2.sum(axis=0)  # Column-wise sums
    df3 = pd.DataFrame({'mbrG': mbrG_marg, 'mbrD': mbrD_marg})
    df3.columns.name = 'mbre'
    print(f"df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns} {df3.sum().sum()}")   
        
    # Exit
    return df3 