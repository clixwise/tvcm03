import sys
import os
import pandas as pd

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exam.c02_exam_01_stat_ import StatTranEXAM_01
from exam.c02_exam_01_stat_ import StatTranEXAM_01_desc
from util.data_02_proc import ProcTranStatEXAM01   
from util.data_51_inpu import InpuTran, inpu_file_exec_xlat_0
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_mixd 

def main_exec_exam_01_stat(procTran:ProcTranStatEXAM01, inpuTran:InpuTran, oupuTran:OupuTranFile):

    # Inpu
    # ----
    def proc_fram_inpu(framTran:FramTran, inpuTran: InpuTran):
        
        # Data
        # ----
        M_data = ['M_Age', 'M_Sexe', 'M_BMI', 
                    'M_CEAP_R', 'M_CEAP_L', 'M_CEAP_P', 'M_RANK_P', 
                    "M_LIMB_R", "M_LIMB_L", "M_LIMB_P", 
                    'M_UNBI', 'M_LATE',
                    "M_OPER_TIME", "M_CHIR_TIME",
                    'M_CHIR', 'M_GREF','M_DEBR','M_NPWT','M_AMPU',
                    "M_OPER_VEIN_R","M_OPER_VEIN_L","M_OPER_VEIN_P", "M_OPER_VEIN_C",
                    "M_OPER_LIMB_R", "M_OPER_LIMB_L", "M_OPER_LIMB_P", 
                    "M_OPER_UNBI", "M_OPER_LATE",
                    'M_ANES_TYPE', 'M_ANES_PROD', 'M_ANES_CONC',
                    'M_THER_UGFS', 'M_THER_PRP' , 'M_THER_MEDI',
                    'M_THER_OHB' , 'M_THER_AGPL', 'M_THER_CHD',
                    'M_THER_KINE', 'M_THER_SPOR', 'M_THER_BAS_QOL']
        
        # Exec
        # ----
        framTran.inpu = inpuTran.fram
        filt_mode = 'exte'
        match filt_mode:
            case 'inte': # by intention
                raise Exception()
            case 'exte': # by extention
                # M_Affection|M_Age|M_Alcoolisme|M_Anesthesie_cc|M_Anesthesie_prod|M_Anesthesie_type|M_BMI|M_Bas
                # |M_CEAP_L|M_CEAP_P|M_CEAP_R|M_CEAP_many_D|M_CEAP_many_G|M_CHD|M_Chirurgie|M_Debridement|M_Examen
                # |M_Greffe|M_Grossesse_A|M_Grossesse_G|M_Grossesse_P|M_Hosp|M_Kine|M_MBASU_famille|M_MBASU_parent
                # |M_Mixte|M_Naissance|M_Obs|M_Operation_fini|M_Operation_init|M_Oxyg|M_Phlebite
                # |M_Phlebite_oui|M_Poids|M_Recidive|M_Saph_D|M_Saph_G|M_Sema
                # |M_Sexe|M_Soins|M_Sport|M_TPN|M_Tabagisme|M_Taille|M_Timepoint
                framTran.filt = M_data
                framTran.pref = ("M_")
                framTran.func = inpu_fram_exec_selc_1_exte
                framTran.upda()
            case 'mixd':
                raise Exception()
            case _:
                raise Exception()
    
    # Stat.prec
    # ----
    def proc_fram_post(statTran:StatTranEXAM_01, framTran:FramTran):
        
        # Data
        # ----
        df_fram = framTran.fram

        # Exec
        # ----
        what = "Sexe"
        cate_list = ['M','F']
        df_fram = inpu_file_exec_xlat_0 (df_fram, what, cate_list)
        #
        cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        df_fram = inpu_file_exec_xlat_0 (df_fram, "CEAP_P", cate_list)
        df_fram = inpu_file_exec_xlat_0 (df_fram, "CEAP_R", cate_list)
        df_fram = inpu_file_exec_xlat_0 (df_fram, "CEAP_L", cate_list)
        #
        cate_list = ['GR', 'LR', 'LO']
        df_fram = inpu_file_exec_xlat_0 (df_fram, "ANES_TYPE", cate_list)
        #
        cate_list = ['TA', 'AM', 'MA']
        df_fram = inpu_file_exec_xlat_0 (df_fram, "ANES_PROD", cate_list)
        #
        cate_list = ['1.5', '2.5', '3']
        df_fram['ANES_CONC'] = df_fram['ANES_CONC'].astype(str).str.replace('.0', '', regex=False)
        # print_yes (df_fram[["ANES_TYPE", "ANES_PROD", "ANES_CONC"]])
        df_fram = inpu_file_exec_xlat_0 (df_fram, "ANES_CONC", cate_list)
        # print_yes (df_fram[["ANES_TYPE", "ANES_PROD", "ANES_CONC"]])
        
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
        pass
            
    # Stat.Exec
    # ----
    def proc_stat_exec(statTran:StatTranEXAM_01):
        statTran.stra = 'a'
        statTran.upda()
        
    # Stat.Oupu
    # ----
    def proc_stat_oupu(statTran:StatTranEXAM_01, oupuTran:OupuTranFile):
        oupuTran.fram_dict[StatTranEXAM_01_desc.__name__] =  { 
            "resu_publ_T0" : { 'df':statTran.stat_tran_desc.resu_publ_T0, 'mode':'md' },  
            "resu_publ_TX" : { 'df':statTran.stat_tran_desc.resu_publ_TX, 'mode':'md' } 
            }
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    statTran = StatTranEXAM_01(procTran)
    
    # Stat
    # ----
    proc_fram_inpu(framTran, inpuTran)
    proc_fram_post(statTran, framTran)
    proc_stat_exec(statTran)
    proc_stat_oupu(statTran, oupuTran)
    
    # Exit
    # ----
    return statTran

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

if __name__ == "__main__":
    pass
