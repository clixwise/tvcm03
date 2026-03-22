import sys
import os
import pandas as pd
# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vcss_10_mixd.c02_vcss_11_stat_ import StatTranVCSS_11, StatTranVCSS_11_mixd
from util.data_02_proc import ProcTranStatVCSS01   
from util.data_51_inpu import InpuTran, inpu_file_exec_xlat_0
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_mixd, inpu_frax_exec_selc_1 

def main_exec_vcss_11_stat(procTran:ProcTranStatVCSS01, inpuTran:InpuTran, oupuTran:OupuTranFile):

    # Inpu
    # ----
    def proc_fram_inpu(framTran:FramTran, inpuTran: InpuTran):
        framTran.inpu = inpuTran.fram
        filt_mode = 'exte'
        match filt_mode:
            case 'inte': # by intention
                framTran.filt = ["V_"] 
                framTran.pref = ("V_")
                framTran.func = inpu_fram_exec_selc_1_inte
                framTran.upda_fram()
            case 'exte': # by extention
                framTran.filt = ['M_Age', 'M_Sexe', 'V_R', 'V_L', 'V_P']
                framTran.pref = ("M_","V_")
                framTran.func = inpu_fram_exec_selc_1_exte
                framTran.upda_fram()
            case _:
                raise Exception()
            
        # Exec
        # ----
        framTran.drop_list = ['P']
        framTran.func = inpu_frax_exec_selc_1
        framTran.upda_frax()

    # Stat.prec
    # ----
    def proc_fram_post_fram(statTran:StatTranVCSS_11, framTran:FramTran):

        # Data
        # ----
        df_fram = framTran.frax

        # Exec
        # ----
        what = "Sexe"
        cate_list = ['M','F']
        df_fram = inpu_file_exec_xlat_0 (df_fram, what, cate_list)
        
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
    
    def proc_fram_post_frax(statTran:StatTranVCSS_11, framTran:FramTran):

        # Data
        # ----
        df_frax = framTran.frax

        # Exec
        # ----
        what = "Sexe"
        cate_list = ['M','F']
        df_frax = inpu_file_exec_xlat_0 (df_frax, what, cate_list)
        #
        what = "Limb"
        cate_list = ['R','L']
        df_frax = inpu_file_exec_xlat_0 (df_frax, what, cate_list)
        
        # Chck
        # ----
        cate_cols = df_frax.select_dtypes(include=['category']).columns.tolist()
        print(f"Categorical columns: {cate_cols}")
        orde_cols = [
            colu for colu in df_frax.columns 
            if isinstance(df_frax[colu].dtype, pd.CategoricalDtype) and df_frax[colu].dtype.ordered
        ]
        print(f"Categorical columns [ordered]: {orde_cols}")

        # Exit
        # ----
        statTran.frax = df_frax
        pass
    
    def proc_fram_post(statTran:StatTranVCSS_11, framTran:FramTran):
        proc_fram_post_fram(statTran, framTran)
        proc_fram_post_frax(statTran, framTran)
                
    # Stat.Exec
    # ----
    def proc_stat_exec(statTran:StatTranVCSS_11):
        statTran.stra = 'a'
        statTran.upda()
        
    # Stat.Oupu
    # ----
    def proc_stat_oupu(statTran:StatTranVCSS_11, oupuTran:OupuTranFile):
        oupuTran.fram_dict[StatTranVCSS_11_mixd.__name__] =  { "resu_fram" : { 'df':statTran.stat_tran_mixd.resu_fram, 'mode':'md' }, "resu_plot" : { 'df':statTran.stat_tran_mixd.resu_plot, 'mode':'md' } }
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    statTran = StatTranVCSS_11(procTran)
    
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
