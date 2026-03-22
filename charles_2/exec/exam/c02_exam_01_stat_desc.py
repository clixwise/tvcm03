    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_exam_01_stat_ import StatTranEXAM_01_desc
  
import numpy as np  
import pandas as pd
import sys
import os  
from pprint import pprint
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
        
# ----
# https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4 
# ----
'''
'''
def exec_stat_desc(stat_tran_desc: StatTranEXAM_01_desc) -> None:
    from exam.c02_exam_01_stat_ import StatTranEXAM_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram.copy()
    mark_dic1 = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta01_base_char
    mark_dic2 = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta02_base_char
    mark_dic3 = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta04_ther_adju
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exec
    # ----
    stat_tran_desc.resu_publ_T0 =  exec_stat_desc_T0(df_fram, mark_dic1, mark_dic2)
    stat_tran_desc.resu_publ_TX =  exec_stat_desc_TX(df_fram, mark_dic1, mark_dic3)
    pass

def exec_stat_desc_T0(df_fram, mark_dic1:dict, mark_dic2:dict) -> pd.DataFrame: 
    from exam.c02_exam_01_stat_ import StatTranEXAM_01_desc
     
    trac = True
   
    # Exec
    # ----   
    publ_list = []  
    tipo = "T0"
    df_tipo = df_fram[df_fram["timepoint"] == tipo]
    if df_tipo.empty:
        raise Exception()

    # Count
    # ---
    count_info = len(df_tipo)
    mark_dic1['Patient characteristics'] = (f'(N = {count_info})', Path(__file__).stem) # Age, years 54.8 ± 11.6
    
    # Age
    # ---
    age_info = summarize_continuous_edit(df_tipo["Age"], method="mean_sd")
    publ_list.append([StatTranEXAM_01_desc.__name__, "Age, years", tipo, age_info])
    mark_dic1['Age, years'] = (age_info, Path(__file__).stem) # Age, years 54.8 ± 11.6
    
    # Sex
    # ---
    sex_info = summarize_categorical_edit(df_tipo["Sexe"], nan_labl='-', cate_sort=True)
    for k, v in sex_info.items():
        publ_list.append([StatTranEXAM_01_desc.__name__, f"Sex: {k}", tipo, v])
    mark_dic1['Female, n (%)'] = (sex_info['F'], Path(__file__).stem)
    # print (sex_info)
    
    '''
    M_CEAP_R M_CEAP_L M_CEAP_P M_RANK_P M_LIMB_R  M_LIMB_L  M_LIMB_P M_UNBI M_LATE
          NA       C6       C6        7        0         1         1      U      L
    M_OPER_VEIN_R M_OPER_VEIN_L M_OPER_VEIN_P  M_OPER_VEIN_C  M_OPER_LIMB_R  M_OPER_LIMB_L  M_OPER_LIMB_P M_OPER_UNBI M_OPER_LATE
              NaN           G,A           G,A              2              0              1              1           U           L
   ['M_Age', 'M_Sexe', 'M_BMI', 
    'M_CEAP_R', 'M_CEAP_L', 'M_CEAP_P', 'M_RANK_P', 
    "M_LIMB_R", "M_LIMB_L", "M_LIMB_P", 
    'M_UNBI', 'M_LATE',
    "M_OPER_VEIN_R","M_OPER_VEIN_L","M_OPER_VEIN_P", "M_OPER_VEIN_C",
    "M_OPER_LIMB_R", "M_OPER_LIMB_L", "M_OPER_LIMB_P", 
    "M_OPER_UNBI", "M_OPER_LATE"]
    '''
    # Late
    # ----
    unbi_valid = df_tipo['UNBI'].isin(['U', 'B']).all()
    if not unbi_valid:
        raise ValueError("Column 'UNBI' contains invalid values. Only 'U' or 'B' are allowed.")
    late_valid = df_tipo.loc[df_tipo['UNBI'] == 'U', 'LATE'].isin(['L', 'R']).all()
    if not late_valid:
        raise ValueError("Column 'LATE' contains invalid values for rows where 'UNBI' is 'U'. Only 'L' or 'R' are allowed.")
    # ---
    df_tipo['Laterality'] = df_tipo.apply(
        lambda row: 'Unilateral,Left' if (row['UNBI'] == 'U' and row['LATE'] == 'L') else
                    'Unilateral,Right' if (row['UNBI'] == 'U' and row['LATE'] == 'R') else
                    'Bilateral',
        axis=1
    )
    # print (df_tipo[['UNBI', 'LATE', 'Laterality']])
    df_tipo['Laterality'] = pd.Categorical(df_tipo['Laterality'], categories=['Unilateral,Left', 'Unilateral,Right', 'Bilateral'], ordered=True)
    laterality_info = summarize_categorical_edit(df_tipo["Laterality"], nan_labl='-', cate_sort=True)
    for k, v in laterality_info.items():
            publ_list.append([StatTranEXAM_01_desc.__name__, f"Laterality: {k}", tipo, v])
    # ---
    df_tipo['Laterality_extd'] = df_tipo.apply(
        lambda row: 'Unilateral' if row['UNBI'] == 'U' else
                    'Bilateral',
        axis=1
    )
    # print (df_tipo[['UNBI', 'LATE', 'Laterality_extd']])
    df_tipo['Laterality_extd'] = pd.Categorical(df_tipo['Laterality_extd'], categories=['Unilateral', 'Bilateral'], ordered=True)
    laterality_info = summarize_categorical_edit(df_tipo["Laterality_extd"], nan_labl='-', cate_sort=True)
    mark_dic1['.Unilateral'] = (laterality_info['Unilateral'], Path(__file__).stem)
    mark_dic1['.Bilateral']  = (laterality_info['Bilateral'], Path(__file__).stem)
    
    # Limbs examined
    # --------------
    liex_inf1 = len(df_tipo) * 2
    publ_list.append([StatTranEXAM_01_desc.__name__, "Total limbs examined", tipo, liex_inf1])
    mark_dic2['Limb characteristics'] = (f'(L = {liex_inf1} limbs)', Path(__file__).stem) 
    #
    liex_inf2 = summarize_continuous_edit(df_tipo["LIMB_P"], method="sum")
    publ_list.append([StatTranEXAM_01_desc.__name__, "Limbs with CVD, n", tipo, liex_inf2])
    mark_dic2['Limbs with CVD, n'] = (f'{liex_inf2}', Path(__file__).stem) 
    #
    liex_inf3 = summarize_continuous_edit(df_tipo["LIMB_P"], method="mean_sd")
    publ_list.append([StatTranEXAM_01_desc.__name__, "CVD-affected limbs per patient", tipo, liex_inf3])
    mark_dic2['CVD-affected limbs per patient']  = (liex_inf3, Path(__file__).stem)
    
    # Limbs operated
    # --------------
    liop_inf1 = summarize_continuous_edit(df_tipo["OPER_LIMB_P"], method="sum")
    publ_list.append([StatTranEXAM_01_desc.__name__, "Treated limbs", tipo, liop_inf1])
    mark_dic2['Treated limbs, n'] = (f'{liop_inf1}', Path(__file__).stem) 
    #
    liop_inf2 = summarize_continuous_edit(df_tipo["OPER_LIMB_P"], method="mean_sd")
    publ_list.append([StatTranEXAM_01_desc.__name__, "Treated limbs per patient", tipo, liop_inf2])
    mark_dic2['Treated limbs per patient']  = (liop_inf2, Path(__file__).stem)
    #
    df_tipo['CHIR_TIME'] = pd.to_numeric(df_tipo['CHIR_TIME'], errors='coerce')
    print_yes (df_tipo[["CHIR_TIME"]])
    dura_info = summarize_continuous_edit(df_tipo["CHIR_TIME"], method="mean_sd")
    publ_list.append([StatTranEXAM_01_desc.__name__, "Procedure duration per limb, min", tipo, dura_info])
    mark_dic2['Procedure duration per limb, min'] = (dura_info, Path(__file__).stem)
    
    # Anesthesia
    # ----------
    print(df_tipo["ANES_TYPE"])
    anes_type_inf1 = summarize_categorical_edit(df_tipo["ANES_TYPE"], nan_labl='-', cate_sort=True)
    # print(anes_type_inf1)
    if 'GE' in anes_type_inf1:
        mark_dic2['..General'] = (anes_type_inf1['GE'], Path(__file__).stem)
    if 'LR' in anes_type_inf1:
         mark_dic2['..Regional'] = (anes_type_inf1['LR'], Path(__file__).stem)
    if 'LO' in anes_type_inf1:
         mark_dic2['..Local'] = (anes_type_inf1['LO'], Path(__file__).stem)
    if '-' in anes_type_inf1:
         mark_dic2['..-31 Not specified'] = (anes_type_inf1['LO'], Path(__file__).stem)
    #
    print(df_tipo["ANES_PROD"])
    anes_prod_inf1 = summarize_categorical_edit(df_tipo["ANES_PROD"], nan_labl='-', cate_sort=True)
    # print(anes_prod_inf1)
    if 'TA' in anes_prod_inf1:
        mark_dic2['..Propofol'] = (anes_prod_inf1['TA'], Path(__file__).stem)
    if 'AM' in anes_prod_inf1:
        mark_dic2['..Prilocaine'] = (anes_prod_inf1['AM'], Path(__file__).stem)
    if 'MA' in anes_prod_inf1:
        mark_dic2['..Bupivacaine'] = (anes_prod_inf1['MA'], Path(__file__).stem)
    if '-' in anes_prod_inf1:
        mark_dic2['..-32 Not specified'] = (anes_prod_inf1['-'], Path(__file__).stem)
    #
    if False:
        print(df_tipo["ANES_CONC"])
        anes_conc_inf1 = summarize_continuous_edit(df_tipo["ANES_CONC"], method="sum")
        publ_list.append([StatTranEXAM_01_desc.__name__, "Total limbs treated", tipo, anes_conc_inf1])
        mark_dic2['.Concentration, %'] = (f'{anes_conc_inf1}', Path(__file__).stem) 
    
    # Vein
    # ----
    print_yes(df_tipo[['OPER_VEIN_R','OPER_VEIN_L', 'OPER_VEIN_P', 'OPER_VEIN_C', 'OPER_LIMB_P']], labl=None)
    vein_info = summarize_continuous_edit(df_tipo["OPER_VEIN_C"], method="sum")
    mark_dic2['Target veins treated*, n (%)'] = (f'{vein_info}', Path(__file__).stem)
    
    # 1. Flatten the codes into individual vein entries
    # We split the strings by ',' and 'explode' them into a single Series
    all_veins = df_tipo['OPER_VEIN_P'].str.split(',').explode().str.strip()
    # 2. Convert to Categorical to define the desired order (G, P, A)
    vein_categories = ['G', 'P', 'A']
    all_veins_cat = pd.Categorical(all_veins, categories=vein_categories, ordered=True)
    # 3. Apply your custom summary function
    # We pass the Series created above to get the counts and percentages
    vein_inf1 = summarize_categorical_edit(pd.Series(all_veins_cat), cate_sort=True)
    print(vein_inf1)
    mark_dic2['.Great saphenous vein']  = (vein_inf1['G'], Path(__file__).stem)
    mark_dic2['.Small saphenous vein']  = (vein_inf1['P'], Path(__file__).stem)
    mark_dic2['.Anterior accessory saphenous vein']  = (vein_inf1['A'], Path(__file__).stem)
    if '-' in vein_inf1:
        mark_dic2['.-4 Not specified']  = (vein_inf1['-'], Path(__file__).stem)
    #
    viop_inf2 = summarize_continuous_edit(df_tipo["OPER_VEIN_C"], method="sum")
    mark_dic2['*Data as n (%) of total treated veins']  = (f'({viop_inf2} veins in {liop_inf1} limbs)', Path(__file__).stem)
   
    # Oupu
    # ----
    cols = ['ID', 'Metric', 'Timepoint', 'Value']
    #
    df_publ = pd.DataFrame(publ_list, columns=cols)
    df_publ.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_publ, labl="df_publ")
        pprint(mark_dic1, indent=4, sort_dicts=False)
        pprint(mark_dic2, indent=4, sort_dicts=False)
        
    # Exit
    # ----
    return df_publ
        
def exec_stat_desc_TX(df_fram, mark_dic1:dict, mark_dic3:dict) -> pd.DataFrame: 
    from exam.c02_exam_01_stat_ import StatTranEXAM_01_desc
     
    trac = True
   
    # Exec : tech
    # ---- 
    publ_list = []
    for tipo in ["T0", "T1", "T2"]:
        
        df_tipo = df_fram[df_fram["timepoint"] == tipo].copy()
        if df_tipo.empty:
            continue

        # BMI
        # ---
        bmi_info = summarize_continuous_edit(df_tipo["BMI"], method="mean_sd")
        publ_list.append([StatTranEXAM_01_desc.__name__, "BMI, kg/m²", tipo, bmi_info])
        if tipo == "T0":
            mark_dic1['BMI, kg/m²']  = (bmi_info, Path(__file__).stem)
            
        # CEAP
        # ----
        #ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        #
        def util_publ(df_tipo, what, publ_list, mark_dic1):
            #df_tipo[what] = pd.Categorical(df_tipo[what], categories=ceap_list, ordered=True)
            ceap_info = summarize_categorical_edit(df_tipo[what], nan_labl='-', cate_sort=True)
            for k, v in ceap_info.items():
                publ_list.append([StatTranEXAM_01_desc.__name__, f"{what} {k}", tipo, v])
        def util_mark(df_tipo, what, publ_list, mark_dic1):
            #df_tipo[what] = pd.Categorical(df_tipo[what], categories=ceap_list, ordered=True)
            ceap_info = summarize_categorical_edit(df_tipo[what], nan_labl='-', cate_sort=True)
            for k, v in ceap_info.items():
                print (f'{k}:{v}')
                mark_dic1[f'.{k}'] = (v, Path(__file__).stem)
            # Filters out keys starting with '.C' that have a (None, None) value
            keys_to_remove = [
                k for k, v in mark_dic1.items() 
                if k.startswith('.C') and v == (None, None)
            ]
            for k in keys_to_remove:
                del mark_dic1[k]
        #
        pati_levl = True
        if pati_levl:
            util_publ(df_tipo, "CEAP_P", publ_list, mark_dic1)
        limb_levl = False
        if limb_levl:
            util_publ(df_tipo, "CEAP_R", publ_list, None)
            util_publ(df_tipo, "CEAP_L", publ_list, None)
        #
        if tipo == 'T0':
            util_mark(df_tipo, "CEAP_P", publ_list, mark_dic1)
               
    # Ther
    # ----
    ther_cols = ['THER_UGFS', 'THER_PRP', 'THER_MEDI',
                 'THER_OHB' , 'THER_AGPL', 'THER_CHD',
                 'THER_KINE', 'THER_SPOR']
    df_ther = df_tipo.filter(ther_cols) # regex='^THER_') 
    # print_yes(df_ther, labl="df_ther")   
    def test_data_valu_ok_not(df):
        known_values = {'Oui', 'Non', ''}  # np.nan handled separately
        # Get unique values, flatten, and drop NaN
        all_values = pd.unique(df.to_numpy().ravel())
        non_na_values = [v for v in all_values if not pd.isna(v)]
        # Compute unexpected values
        unexpected = [v for v in non_na_values if v not in known_values]
        if unexpected:
            print(f"❌ Sanity Check Failed! Unexpected values found: {unexpected}")
            raise Exception()
    test_data_valu_ok_not(df_ther)
    df_ther = df_ther.map(lambda x: 1 if x == 'Oui' else 0)
    df_ther['THER_BAS_QOL'] = df_tipo['THER_BAS_QOL']
    print_yes(df_ther, labl="df_ther") 
    
    # UGFS
    ther_info = summarize_continuous_edit(df_ther["THER_UGFS"], method="sum")
    mark_dic3['UGFS'] = (f'{ther_info}', Path(__file__).stem)         
    # PRP
    ther_info = summarize_continuous_edit(df_ther["THER_PRP"], method="sum")
    mark_dic3['PRP'] = (f'{ther_info}', Path(__file__).stem)         
    # MEDI
    ther_info = summarize_continuous_edit(df_ther["THER_MEDI"], method="sum")
    mark_dic3['Veinotropes'] = (f'{ther_info}', Path(__file__).stem)         
    # OHB
    ther_info = summarize_continuous_edit(df_ther["THER_OHB"], method="sum")
    mark_dic3['Hyperbaric oxygen therapy'] = (f'{ther_info}', Path(__file__).stem)         
    # AGPL
    ther_info = summarize_continuous_edit(df_ther["THER_AGPL"], method="sum")
    mark_dic3['GLP-1 receptor agonists'] = (f'{ther_info}', Path(__file__).stem)         
    # CHD
    ther_info = summarize_continuous_edit(df_ther["THER_CHD"], method="sum")
    mark_dic3['Hygiene and nutritional counseling'] = (f'{ther_info}', Path(__file__).stem)               
    # KINE
    ther_info = summarize_continuous_edit(df_ther["THER_KINE"], method="sum")
    mark_dic3['Pressotherapy, bandaging'] = (f'{ther_info}', Path(__file__).stem)               
    # SPOR
    ther_info = summarize_continuous_edit(df_ther["THER_SPOR"], method="sum")
    mark_dic3['Physical activity'] = (f'{ther_info}', Path(__file__).stem)   
    
    # THER_BAS_QOL
    qol_mapping = {
    1: "All of the time",
    2: "Most of the time",
    3: "A good bit of the time",
    4: "Some of the time",
    5: "Rarely",
    6: "Never"
    }
    df_ther['THER_BAS_QOL_TEXT'] = df_ther['THER_BAS_QOL'].map(qol_mapping)
    qol_order = [qol_mapping[k] for k in sorted(qol_mapping.keys())]
    df_ther['THER_BAS_QOL_TEXT'] = pd.Categorical(df_ther['THER_BAS_QOL_TEXT'], categories=qol_order, ordered=True)            
    stoc_info = summarize_categorical_edit(df_ther['THER_BAS_QOL_TEXT'], nan_labl='-', cate_sort=True)
    print(stoc_info)
    mark_dic3['.All of the time']         = (stoc_info.get('All of the time', ''), Path(__file__).stem)
    mark_dic3['.Most of the time']        = (stoc_info.get('Most of the time', ''), Path(__file__).stem)
    mark_dic3['.A good bit of the time']  = (stoc_info.get('A good bit of the time', ''), Path(__file__).stem)
    mark_dic3['.Some of the time']        = (stoc_info.get('Some of the time', ''), Path(__file__).stem)
    mark_dic3['.Rarely']                  = (stoc_info.get('Rarely', ''), Path(__file__).stem)
    mark_dic3['.Never']                   = (stoc_info.get('Never', ''), Path(__file__).stem)
    
    if trac:
        print (publ_list)
        pprint(mark_dic1, indent=4, sort_dicts=False)
        pprint(mark_dic3, indent=4, sort_dicts=False)
    
    # Oupu
    # ----
    cols = ['ID', 'Metric', 'Timepoint', 'Value']
    #
    df_publ = pd.DataFrame(publ_list, columns=cols)
    df_publ.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_publ, labl="df_publ")
    
    # Exit
    # ----
    return df_publ

def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass