import sys
import os
import pandas as pd

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_80_mixd_sens.c02_qol_81_stat_ import StatTranQOL_81, StatTranQOL_81_mcid_copi
from util.data_02_proc import ProcTranStatQOL81   
from util.data_51_inpu import InpuTran
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_mixd 

def main_exec_qol_81_stat(procTran:ProcTranStatQOL81, inpuTran:InpuTran, oupuTran:OupuTranFile):

    # Inpu
    # ----
    def proc_fram_inpu(framTran:FramTran, inpuTran: InpuTran):
        framTran.inpu = inpuTran.fram
        filt_mode = 'mixd'
        match filt_mode:
            case 'inte': # by intention
                framTran.filt = ["Q_"] 
                framTran.pref = ("Q_")
                framTran.func = inpu_fram_exec_selc_1_inte
                framTran.upda()
            case 'exte': # by extention
                framTran.filt = ['Q_none_m', 'Q_none_z', 'Q_none_t', 'Q_mean_m', 'Q_mean_z', 'Q_mean_t', 'Q_iter_m', 'Q_iter_z', 'Q_iter_t', 'Q_gemi_z', 'Q_gemi_t', 'Q_copi_z', 'Q_copi_t', 'Q_pati_isok', 'Q_pati_50pc']
                framTran.pref = ("Q_")
                framTran.func = inpu_fram_exec_selc_1_exte
                framTran.upda()
            case 'mixd':
                framTran.filt = ['Q_none_t', 'Q_mean_t', 'Q_iter_t', 'Q_gemi_t', 'Q_copi_t', 'Q_pati_isok', 'Q_pati_50pc', 'C_3'] # 'M_Age', 'M_BMI'
                framTran.pref = ("Q_")
                framTran.func = inpu_fram_exec_selc_1_mixd
                framTran.upda()
            case _:
                raise Exception()
 
    # Stat.prec
    # ----
    def proc_fram_post(statTran:StatTranQOL_81, framTran:FramTran):
          
        # Exec
        # ----
        df_fram = framTran.fram.copy()
        df_fram = df_fram[['workbook', 'patient_id', 'timepoint', 'copi_t', 'pati_isok', 'pati_50pc', 'C_3']]
        df_fram.rename(columns={'copi_t': 'VEINES_QOL_t'}, inplace=True)
        df_fram['C_3'] = pd.to_numeric(df_fram['C_3'], errors='coerce')
        statTran.fram = df_fram
        
        # Trac
        # ----
        trac = True
        if trac:
            print_yes(df_fram, labl="statTran.fram")
        pass

                
    # Stat.Exec
    # ----
    def proc_stat_exec(statTran:StatTranQOL_81):
        statTran.stra = 'a'
        statTran.upda()
        
    # Stat.Oupu
    # ----
    def proc_stat_oupu(statTran:StatTranQOL_81, oupuTran:OupuTranFile):
        oupuTran.fram_dict[StatTranQOL_81.__name__] = { "orig_fram" : { 'df':statTran.fram, 'mode':'md' } }
        oupuTran.fram_dict[StatTranQOL_81_mcid_copi.__name__] = {}
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    statTran = StatTranQOL_81(procTran)
    
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
            'display.max_rows', None,          # Show more rows before truncating
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
