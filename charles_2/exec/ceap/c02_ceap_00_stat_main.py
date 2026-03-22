import sys
import os
import pandas as pd

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ceap.c02_ceap_01_stat_ import StatTranCEAP_01
from ceap.c02_ceap_01_stat_ import StatTranCEAP_01_desc
from util.data_02_proc import ProcTranStatCEAP01   
from util.data_51_inpu import InpuTran, inpu_file_exec_xlat_0
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_mixd 

#
# CLAUDE : https://claude.ai/chat/c88e0315-b71c-4584-bf2b-c6fc448fd2d4
#
# Bilateral relationship VCSS <-> CEAP
# 
def main_exec_ceap_01_stat(procTran:ProcTranStatCEAP01, inpuTran:InpuTran, oupuTran:OupuTranFile):

    # Inpu
    # ----
    def proc_fram_inpu(framTran:FramTran, inpuTran: InpuTran):
        framTran.inpu = inpuTran.fram
        filt_mode = 'exte'
        match filt_mode:
            case 'inte': # by intention
                raise Exception()
            case 'exte': # by extention
                framTran.filt = ['M_CEAP_R_many', 'M_CEAP_L_many', 'M_UNBI', 'M_LATE', "M_OPER_VEIN_R","M_OPER_VEIN_L", 'M_OPER_UNBI', 'M_OPER_LATE', 'M_CEAP_VCSS_R', 'M_CEAP_VCSS_L']
                framTran.pref = ("M_")
                framTran.func = inpu_fram_exec_selc_1_exte
                framTran.upda()
            case 'mixd':
                raise Exception()
            case _:
                raise Exception()
    
    # Stat.prec
    # ----
    def proc_fram_post(statTran:StatTranCEAP_01, framTran:FramTran):
        
        # Data
        # ----
        df_fram = framTran.fram
    
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
    def proc_stat_exec(statTran:StatTranCEAP_01):
        statTran.stra = 'a'
        statTran.upda()
        
    # Stat.Oupu
    # ----
    def proc_stat_oupu(statTran:StatTranCEAP_01, oupuTran:OupuTranFile):
        oupuTran.fram_dict[StatTranCEAP_01_desc.__name__] =  { 
            "resu_publ_T0" : { 'df':statTran.stat_tran_desc.resu_publ_T0, 'mode':'md,xlsx' }
            }
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    statTran = StatTranCEAP_01(procTran)
    
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
