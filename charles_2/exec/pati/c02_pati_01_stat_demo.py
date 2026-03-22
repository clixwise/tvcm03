    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_pati_01_stat_ import StatTranPATI_01_desc
  
import numpy as np  
import pandas as pd
import sys
import os  
import re
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit, summarize_categorical_edit
from util.fram_help import fram_prnt
from util.data_51_inpu import inpu_file_exec_xlat_2
from pprint import pprint
from pati.c02_pati_01_stat_adat import StatTranPATI_01_demo
        
# ----
# https://gemini.google.com/app/a19fc679f0d6795d
# ----
'''
'''
def exec_stat_demo(stat_tran_desc: StatTranPATI_01_demo) -> None:
    # from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram.copy()
    mark_dict = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta00_base_char
    work_dict = stat_tran_desc.stat_tran.proc_tran.orch_tran.wk00_base_char
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
    pass

    # Exec
    # ----
    tipo = 'T0'
    df_tipo = df_fram[df_fram["timepoint"] == tipo]
    
    def util_work_dict(work_dict, head, df_tipo, what, text, col='Resp', pref="."):
        
        # Exec
        # ----
        df_ques = inpu_file_exec_xlat_2 (df_tipo, what, text)
        
        # Exec
        # ----
        if df_ques is None or df_ques.empty:
            raise Exception()
        # 1. Calculate counts and total
        counts = df_ques[col].str.strip().value_counts()
        totl = len(df_ques)
        # 2. Populate the sub-dictionary
        for labl, coun in counts.items():
            formatted_val = f"{coun} ({coun/totl:.1%})"
            work_dict[head][f"{pref}{labl}"] = (formatted_val, "")
            
        # Exit
        # ----
        if trac:
            pprint(work_dict, indent=4, sort_dicts=False, width=100)
        return work_dict
    
    text = "Kinshasa (Local)[1] DR Congo (National/Provincial)[2] Africa (Regional)[3] International / Global [4]"
    text = "Local: Kinshasa[1] National: Rest of DR Congo[2] International: Africa & World[3, 4]"
    work_dict = util_work_dict(work_dict, 'Residential Origin, n (%)', df_tipo, "Residence", text)
    #
    text = (
    "Family/Relative[1] Social acquaintance[2] Community proximity / Local residency[3] "
    "Public outreach campaign[4] Religious organization / Place of worship[5] "
    "Internet[6] Television[7] Interpersonal referral[8] Other[9]")
    text = "Personal Networks[1,2]Community & Peer Referral[3,8]Institutional & Religious[4,5]Mass Media & Digital[6,7]Other[9]"
    work_dict = util_work_dict(work_dict, 'Source of Referral, n (%)', df_tipo, "Connaissance", text)
    #
    text = "Primary[1]Secondary[2]Higher[3]Technical[4]Other[5]"
    work_dict = util_work_dict(work_dict, 'Education level, n (%)', df_tipo, "Etude", text)
    #
    text = "Married[1]Separated[2]Divorced[3]Widowed[4]Single[5]"
    work_dict = util_work_dict(work_dict, 'Marital status, n (%)', df_tipo, "Matrimonial", text)
    #
    text = "Catholic[1]Protestant[2]Evangelical/Pentecostal/Revivalist[3]Kimbanguist[4]Traditional African Religions[5]Muslim/Islam[6]No religious affiliation[7]Other[8]"
    work_dict = util_work_dict(work_dict, 'Confession, n (%)', df_tipo, "Confession", text)
    #
    text = (
        "Agriculture / Farming[1] Fisheries / Fishing[2] Mining / Extraction[3] "
        "Artisanal work / Crafts[4] Commerce / Retail[5] Transportation / Logistics[[6] Domestic services[7] "
        "Public sector / Civil service[8] Private sector employment[9] Homemaker[10] "
        "Clergy / Religious vocation[11] Student[12] Unemployed[13] Other[14]")
    text = (
        "Primary Sector (Agriculture, Fishing, Mining)[1,2,3]"
        "Secondary & Tertiary (Crafts, Trade, Transport, Domestic)[4,5,6,7]"
        "Formal Employment (Civil Service, Private Sector)[8,9]"
        "Non-Economic / Institutional (Student, Clergy/Religious)[11,12]"
        "Economically Inactive (Homemaker, Unemployed)[10,13]"
        "Other[14]")
    work_dict = util_work_dict(work_dict, 'Occupational Status, n (%)', df_tipo, "Profession", text)
    #
    text = (
        "Prolonged standing[1]Manual material handling / Heavy lifting[2]"
        "Prolonged ambulation / Extensive walking[3]Prolonged sitting / Sedentary posture[4]")
    work_dict = util_work_dict(work_dict, 'Occupational Physical Activity, n (%)', df_tipo, "Regulier", text)
    #
    text = (
    "Non-clinical consultation[1] Conventional medical consultation[2] Traditional healer / Ethnomedicine[3] Spiritual/Faith-based healing[4] Self-medication[5] Paramedical/Other practitioner[6] "
    "Traditional scarification[7] Compression therapy [stockings/bandages](8] Physical therapy / Physiotherapy[9] "
    "Kibadi solution[10] Unspecified traditional remedies[11] Pharmacotherapy / Allopathic medicine[12] Surgical intervention[13]")
    text = (
    "Conventional/Modern Pathway[2,6,8,9,12,13]Traditional/Alternative Pathway[3,4,7,10,11]Informal/Self-Care[1,5]")
    work_dict = util_work_dict(work_dict, 'Prior Healthcare-Seeking Behavior, n (%)', df_tipo, "Ressources", text)
    #
    text = (
    "Patient (Out-of-pocket)[1] Local family (DRC)[2] Diaspora / Remittances[3] Social network / Acquaintance[4] "
    "Corporate insurance / Employer-sponsored[5] Social services / NGO assistance[6] TVC[7] Other[8]")
    text = (
    "Direct Out-of-Pocket[1,2]External/Remittance Funding[3]Institutional/Third-Party Payer[5,6,7]Informal Support: Acquaintance[4]"
    )
    work_dict = util_work_dict(work_dict, 'Primary Payer, n (%)', df_tipo, "Charge", text)
    #
    text = (        
    "Economic/Financial constraints[1] Time constraints / Occupational commitments[2] Comorbidities / Competing health needs[3] "
    "Geographical distance to facility[4] Lack of transportation / Logistical barriers[5] Adverse weather condition[6] "
    "Administrative/Bureaucratic hurdles[7] Insufficient social suppor[8] Other[9]")
    text = (
    "Financial/Economic Barriers[1]"
    "Logistical/Geographical Barriers[4,5,6] "
    "Personal/Social Barriers[2,3,8]"
    "Institutional Barriers[7]"
    )
    work_dict = util_work_dict(work_dict, 'Barriers to Treatment Adherence, n (%)', df_tipo, "Difficulté", text)
    #
    text = ("Symptom-driven follow-up[1] Routine/Preventative follow-up[2] No intended follow-up[3]")
    text = ("Reactive Care[1]Proactive Care[2]Loss to Follow-up[3]")
    work_dict = util_work_dict(work_dict, 'Follow-up Adherence Pattern, n (%)', df_tipo, "Revenir", text)
    #
    text = ("Very satisfied(1) Satisfied(2) Neutral(3) Disappointed(4) Very disappointed(5)")
    text = ("Very Positive Perception[1,2] Neutral Perception[3] Negative Perception[4,5]")
    work_dict = util_work_dict(work_dict, 'Service Evaluation, n (%)', df_tipo, "Satisfaction", text)

    # Exec
    # ----
    for header, sub_dict in work_dict.items():
        mark_dict[header] = ("", Path(__file__).stem)  # The Header Row
        mark_dict.update(sub_dict) # The Data Rows
    if trac:
        pprint(mark_dict, indent=4, sort_dicts=False, width=100)
    
    pass

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
    fram_prnt(df, labl=labl, trunc= None, head=10)
    pass