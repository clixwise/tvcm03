    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_pati_01_stat_ import StatTranPATI_01_desc
  
import numpy as np  
import pandas as pd
import sys
import os  
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
from util.fram_help import fram_prnt
from util.data_51_inpu import inpu_file_exec_xlat_2
from pati.c02_pati_01_stat_adat import StatTranPATI_01_desc
        
# ----
# Timepoint outcomes
# ----
'''
'''
def exec_stat_desc(stat_tran_desc: StatTranPATI_01_desc) -> None:
    # from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram.copy()
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exec
    # ----
    ques_dict = exec_ques_T0(df_fram)
    stat_tran_desc.stat_tran.ques_dict = ques_dict
        
    # Exec
    # ----
    df_pub1, df_pub2 =  exec_stat_desc_T0(ques_dict)
    stat_tran_desc.resu_publ_T0a = df_pub1
    stat_tran_desc.resu_publ_T0b = df_pub2
    stat_tran_desc.resu_publ_TX =  exec_stat_desc_TX(df_fram)
    pass

def exec_stat_desc_T0(ques_dict:dict) -> pd.DataFrame: 
    from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
     
    trac = True
   
    # Exec
    # ----   
    publ_list = []  
    tipo = "T0"
    #
    for key, value in ques_dict.items():

        what = value['desc'] # 'Acquaintance''
        df_fram = value['df_fram'] # df
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            raise Exception()
        #
        info_dict = summarize_categorical_edit(df_tipo["Resp"], nan_labl='-', cate_sort=True)
        for k, v in info_dict.items():
            publ_list.append([StatTranPATI_01_desc.__name__, f"{what}: {k}", tipo, v])
        print (f"{what}:{info_dict}")
        pass
     
    # Oupu 1
    # ----
    cols = ['ID', 'Metric', 'Timepoint', 'Value']
    #
    df_pub1 = pd.DataFrame(publ_list, columns=cols)
    df_pub1.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_pub1, labl="df_pub1")
    
    # Oupu 2
    # ----
    df_pub2 = df_pub1.copy()
    df_pub2 = df_pub2.reset_index()
    df_pub2[['Category', 'Subcategory']] = df_pub2['Metric'].str.split(':', n=1, expand=True)
    df_pub2.drop(columns=['Metric'], inplace=True)
    df_pub2.set_index(['ID', 'Category', 'Subcategory'], inplace=True)
    if trac:
        print_yes(df_pub2, labl="df_pub2")
    
    # Exit
    # ----
    return df_pub1, df_pub2
        
def exec_stat_desc_TX(df_fram) -> pd.DataFrame: 
    from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
    
    # Exec
    # ----
    df_publ = exec_stat_desc_TX_drop(df_fram)
    
    # Exit
    # ----
    return df_publ

def exec_stat_desc_TX_drop(df_fram) -> pd.DataFrame: 
    from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
    
    trac = True
        
    def util_sta1(ref_pts, target_pts, mode="dropout"):

        # Exec
        # ----
        n_ref = len(ref_pts)
        if n_ref == 0:
            return 0, 0.0

        if mode == "dropout":
            count = len(ref_pts - target_pts)
        else:  # retention
            count = len(ref_pts.intersection(target_pts))
        
        # Exit
        # ----
        perc = (count / n_ref) * 100
        return count, perc
    
    def util_sta2(df_fram, mode="dropout") -> pd.DataFrame:
        
        # Data
        # ----
        publ_list = []
        mtrc = "Dropped Out" if mode == "dropout" else "Retained"
        
        # Exec T0. Define sets
        # ----
        pts_by_tp = { tp: set(df_fram[df_fram["timepoint"] == tp]["patient_id"]) for tp in ["T0", "T1", "T2"] }
        #
        ref_pts = pts_by_tp.get("T0", set())

        # Exec TX. Iterate through T1 and T2
        # ----
        iden = f"{StatTranPATI_01_desc.__name__} {mode}"
        for tp in ["T1", "T2"]:
            target_pts = pts_by_tp.get(tp, set())
            count, perc = util_sta1(ref_pts, target_pts, mode=mode)
            publ_list.append([iden, mtrc, tp, f"{count} ({perc:.1f}%)"])
        #
        cols = ['ID', 'Metric', 'Timepoint', 'Value']
        df_publ = pd.DataFrame(publ_list, columns=cols)
        df_publ.set_index(['ID', 'Metric'], inplace=True)

        # Exit
        # ----
        return df_publ
    
    # Exec
    # ----
    # Dropouts
    df_drop = util_sta2(df_fram, mode="dropout")
    # Retention
    df_rete = util_sta2(df_fram, mode="retention")
    # Combined
    df_publ = pd.concat([df_drop, df_rete])
    
    if trac:
        print_yes(df_publ, labl="df_pub2")
    
    # Exit
    # ----
    return df_publ

# Stat.prec : https://gemini.google.com/app/b6cb5c4bb1a16850
# ----
def exec_ques_T0(df_fram):
    
    # Data
    # ----
    ques_dict = {}

    # Exec
    # ----
    jrnl = True
    '''
    Recruitment Source (How they heard about you)
    Religious Affiliation
    Occupational Status
    Ergonomic Risk Factors (Standing, walking, etc.)
    Prior Healthcare-Seeking Behavior (Traditional vs. Medical)
    Primary Source of Funding (Out-of-pocket vs. Diaspora/Corporate)
    Barriers to Care (Financial, distance, etc.)
    Follow-up Adherence Pattern
    Patient Satisfaction Level
    '''
    # https://gemini.google.com/app/a19fc679f0d6795d
    text = "Kinshasa (Local)[1] DR Congo (National/Provincial)[2] Africa (Regional)[3] International / Global [4]" # "Kinshasa(1) RD Congo(2) Afrique(3) Monde(4)"
    text = "Local: Kinshasa[1] National: Rest of DR Congo[2] International: Africa & World[3, 4]"
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Residence", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Residence", text)
    ques_dict["Residence"] = {'desc':'Residential Origin','df_fram':df_expl}
    #text = "Kinshasa(1) DR Congo(2) Africa(3) World(4)" # "Kinshasa(1) RD Congo(2) Afrique(3) Monde(4)"
    #df_expl = inpu_file_exec_xlat_2 (df_fram, "Residence", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Residence", text)
    #ques_dict["Residence"] = {'desc':'Residence','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Residence"])
    '''
    Local: Kinshasa [1]
    National: Rest of DR Congo [2]
    International: Africa & World [3, 4]
    '''
    # No data(0) !!! < ---
    text = (
        "Family/Relative[1] Social acquaintance[2] Community proximity / Local residency[3] "
        "Public outreach campaign[4] Religious organization / Place of worship[5] "
        "Internet[6] Television[7] Interpersonal referral[8] Other[9]")
    text = (
        "Personal Networks[1,2] "
        "Community & Peer Referral[3,8] "
        "Institutional & Religious[4,5] "
        "Mass Media & Digital[6,7] Other[9]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Connaissance", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Connaissance", text)
    ques_dict["Connaissance"] = {'desc':'Source of Referral','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Connaissance"])
    #
    text = "Primary[1] Secondary[2] Higher[3] Technical[4] Other[5]" # "Primaire(1) Secondaire(2) Supérieur(3) Technique(4) Autre(5)"
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Etude", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Etude", text)
    ques_dict["Etude"] = {'desc':'Education level','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Etude"])
    #
    text = "Married[1] Separated[2] Divorced[3] Widowed[4] Single[5]"
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Matrimonial", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Matrimonial", text)
    ques_dict["Matrimonial"] = {'desc':'Marital status','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Etude"])
    #
    text = (
        "Catholic[1] Protestant[2] Evangelical/Pentecostal/Revivalist[3] "
        "Kimbanguist[4] Traditional African Religions[5] Muslim/Islam[6] No religious affiliation[7] Other[8]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Confession", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Confession", text)
    ques_dict["Confession"] = {'desc':'Confession','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Confession"])
    #
    text = (
        "Agriculture / Farming[1] Fisheries / Fishing[2] Mining / Extraction[3] "
        "Artisanal work / Crafts[4] Commerce / Retail[5] Transportation / Logistics[[6] Domestic services[7] "
        "Public sector / Civil service[8] Private sector employment[9] Homemaker[10] "
        "Clergy / Religious vocation[11] Student[12] Unemployed[13] Other[14]")
    text = (
        "Primary Sector (Agriculture, Fishing, Mining)[1,2,3] "
        "Secondary & Tertiary (Crafts, Trade, Transport, Domestic)[4,5,6,7] "
        "Formal Employment (Civil Service, Private Sector)[8,9] "
        "Non-Economic / Institutional (Student, Clergy/Religious)[11,12] "
        "Economically Inactive (Homemaker, Unemployed)[10,13] Other[14]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Profession", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Profession", text)
    ques_dict["Profession"] = {'desc':'Occupational Status','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Profession"])
    #
    text = (
        "Prolonged standing[1] Manual material handling / Heavy lifting[2] "
        "Prolonged ambulation / Extensive walking[3] Prolonged sitting / Sedentary posture[4]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Regulier", text) # df_fram = inpu_file_exec_xlat_1 (df_fram, "Regulier", text)
    ques_dict["Regulier"] = {'desc':'Occupational Physical Activity','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Regulier"])
    #
    text = (
    "Non-clinical consultation[1] Conventional medical consultation[2] Traditional healer / Ethnomedicine[3] Spiritual/Faith-based healing[4] Self-medication[5] Paramedical/Other practitioner[6] "
    "Traditional scarification[7] Compression therapy [stockings/bandages](8] Physical therapy / Physiotherapy[9] "
    "Kibadi solution[10] Unspecified traditional remedies[11] Pharmacotherapy / Allopathic medicine[12] Surgical intervention[13]")
    text = (
    "Conventional/Modern Pathway[2,6,8,9,12,13] "
    "Traditional/Alternative Pathway[3,4,7,10,11] "
    "Informal/Self-Care[1,5]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Ressources", text)
    ques_dict["Ressources"] = {'desc':'Prior Healthcare-Seeking Behavior','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Ressources"])
    #
    text = (
    "Patient (Out-of-pocket)[1] Local family (DRC)[2] Diaspora / Remittances[3] Social network / Acquaintance[4] "
    "Corporate insurance / Employer-sponsored[5] Social services / NGO assistance[6] TVC[7] Other[8]")
    text = (
    "Direct Out-of-Pocket[1,2] "
    "External/Remittance Funding [3] "
    "Institutional/Third-Party Payer[5,6,7] "
    "Informal Support: Acquaintance [4]"
    )
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Charge", text)
    ques_dict["Charge"] = {'desc':'Primary Payer','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Charge"])
    #
    text = (        
    "Economic/Financial constraints[1] Time constraints / Occupational commitments[2] Comorbidities / Competing health needs[3] "
    "Geographical distance to facility[4] Lack of transportation / Logistical barriers[5] Adverse weather condition[6] "
    "Administrative/Bureaucratic hurdles[7] Insufficient social suppor[8] Other[9]")
    text = (
    "Financial/Economic Barriers[1] "
    "Logistical/Geographical Barriers[4,5,6] "
    "Personal/Social Barriers[2,3,8] "
    "Institutional Barriers[7]"
    )
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Difficulté", text)
    ques_dict["Difficulté"] = {'desc':'Barriers to Treatment Adherence','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Difficulté"])
    #
    text = (
    "Symptom-driven follow-up[1] Routine/Preventative follow-up[2] No intended follow-up[3]")
    text = (
    "Reactive Care[1] Proactive Care[2] Loss to Follow-up[3]")
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Revenir", text)
    ques_dict["Revenir"] = {'desc':'Follow-up Adherence Pattern','df_fram':df_expl}
    if jrnl:
        print(ques_dict["Revenir"])
    #
    text = (
    "Very satisfied(1) Satisfied(2) Neutral(3) Disappointed(4) Very disappointed(5)"  
    )
    text = (
    "Very Positive Perception[1,2] Neutral Perception[3] Negative Perception[4,5]"  
    )
    df_expl = inpu_file_exec_xlat_2 (df_fram, "Satisfaction", text)
    ques_dict["Satisfaction"] = {'desc':'Service Evaluation','df_fram':df_expl} 
    if jrnl:
        print(ques_dict["Satisfaction"]) 
        
    # Chck
    # ----
    cate_cols = df_fram.select_dtypes(include=['category']).columns.tolist()
    print(f"Categorical columns: {cate_cols}")
    orde_cols = [
        colu for colu in df_fram.columns 
        if isinstance(df_fram[colu].dtype, pd.CategoricalDtype) and df_fram[colu].dtype.ordered
    ]
    print(f"Categorical columns [ordered]: {orde_cols}")

    # Exit
    # ----
    return ques_dict

def print_yes(df, labl=None):
    '''
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
    '''
    fram_prnt(df, labl=labl, trunc= None, head=5)
    pass