import sys
import os
import re
import pandas as pd

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pati.c02_pati_01_stat_ import StatTranPATI_01, StatTranPATI_01_foll
from pati.c02_pati_01_stat_ import StatTranPATI_01_desc
from util.data_02_proc import ProcTranStatPATI01   
from util.data_51_inpu import InpuTran, inpu_file_exec_xlat_1, inpu_file_exec_xlat_2, inpu_file_exec_xlat_3
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_mixd 

def main_exec_pati_01_stat(procTran:ProcTranStatPATI01, inpuTran:InpuTran, oupuTran:OupuTranFile):

    # Inpu
    # ----
    def proc_fram_inpu(framTran:FramTran, inpuTran: InpuTran):
        framTran.inpu = inpuTran.fram
        filt_mode = 'exte'
        match filt_mode:
            case 'inte': # by intention
                raise Exception()
            case 'exte': # by extention
                # P_Charge|P_Commune|P_Confession|P_Connaissance|P_Difficulté|P_Etude|P_Matrimonial|
                # P_Profession|P_Regulier|P_Residence|P_Ressources|P_Revenir|P_Satisfaction|P_Connaissance_T|
                # P_Etude_T|P_Matrimonial_T|P_Confession_T|P_Profession_T|P_Regulier_T|P_Ressources_T|
                # P_Charge_T|P_Difficulté_T|P_Revenir_T|P_Satisfaction_T
                framTran.filt = ['P_Telephone', # 'P_Mail', est exclus pcq aucun mail saisi ; donc colonne absente
                                 'P_Etude','P_Residence',
                                 'P_Commune','P_Connaissance', 
                                 'P_Matrimonial', 'P_Confession', 
                                 'P_Profession', 'P_Regulier', 
                                 'P_Ressources', 'P_Charge',
                                 'P_Difficulté', 'P_Revenir',
                                 'P_Satisfaction']
                # framTran.filt = ['P_Ressources']
                framTran.pref = ("P_")
                framTran.func = inpu_fram_exec_selc_1_exte
                framTran.upda()
            case 'mixd':
                raise Exception()
            case _:
                raise Exception()
    
    # Stat.prec : https://gemini.google.com/app/b6cb5c4bb1a16850
    # ----
    def proc_fram_post(statTran:StatTranPATI_01, framTran:FramTran):
        
        # Exec
        # ----
        df_fram = framTran.fram.copy()
        
        # Exit
        # ----
        statTran.fram = df_fram
        
        if False:
            # Data
            # ----
            df_fram = framTran.fram
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
            statTran.fram = df_fram
            statTran.ques_dict = ques_dict
            pass
        
    # Stat.Exec
    # ----
    def proc_stat_exec(statTran:StatTranPATI_01):
        statTran.stra = 'a'
        statTran.upda()
        
    # Stat.Oupu
    # ----
    def proc_stat_oupu(statTran:StatTranPATI_01, oupuTran:OupuTranFile):
        oupuTran.fram_dict[StatTranPATI_01_desc.__name__] =  { 
                    "resu_publ_T0a" : { 'df':statTran.stat_tran_desc.resu_publ_T0a, 'mode':'md' },
                    "resu_publ_T0b" : { 'df':statTran.stat_tran_desc.resu_publ_T0b, 'mode':'md,ft01' }, 
                    "resu_publ_TX"  : { 'df':statTran.stat_tran_desc.resu_publ_TX,  'mode':'md' } }
        oupuTran.fram_dict[StatTranPATI_01_foll.__name__] =  { 
                    "resu_foll" : { 'df':statTran.stat_tran_foll.resu_foll, 'mode':'md,xlsx' },
                    "publ_foll" : { 'df':statTran.stat_tran_foll.publ_foll, 'mode':'md,ft02' } }
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    statTran = StatTranPATI_01(procTran)
    
    # Stat
    # ----
    proc_fram_inpu(framTran, inpuTran)
    proc_fram_post(statTran, framTran)
    proc_stat_exec(statTran)
    proc_stat_oupu(statTran, oupuTran)
    
    # Exit
    # ----
    return statTran

if __name__ == "__main__":
    pass
