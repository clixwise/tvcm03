import sys
import os
import pandas as pd

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_70_mixd_mcid.c02_qol_71_assu_ import AssuTranQOL_71, AssuTranQOL_71_dist
from util.data_02_proc import ProcTranAssuQOL71   
from util.data_51_inpu import InpuTran
from util.data_52_oupu import OupuTranFile
from util.data_61_fram import FramTran, inpu_fram_exec_selc_1_exte, inpu_fram_exec_selc_1_inte, inpu_fram_exec_selc_1_mixd 

def main_exec_qol_71_assu(procTran:ProcTranAssuQOL71, inpuTran:InpuTran, oupuTran:OupuTranFile):

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
                framTran.filt = ['Q_none_t', 'Q_mean_t', 'Q_iter_t', 'Q_gemi_t', 'Q_copi_t', 'Q_pati_isok', 'Q_pati_50pc', 'C_3']
                framTran.pref = ("Q_")
                framTran.func = inpu_fram_exec_selc_1_mixd
                framTran.upda()
            case _:
                raise Exception()

    # Assu.prec
    # ----
    def proc_fram_post(assuTran:AssuTranQOL_71, framTran:FramTran):

        # Exec
        # ----
        df_fram = framTran.fram.copy()
        df_fram = df_fram[['workbook', 'patient_id', 'timepoint', 'copi_t', 'pati_isok', 'pati_50pc', 'C_3']]
        df_fram.rename(columns={'copi_t': 'VEINES_QOL_t'}, inplace=True)
        df_fram['C_3'] = pd.to_numeric(df_fram['C_3'], errors='coerce')
        assuTran.fram = df_fram
        
        # Trac
        # ----
        trac = True
        if trac:
            print_yes(df_fram, labl="assuTran.fram")
        pass

    # Assu.Exec
    # ----
    def proc_assu_exec(assuTran:AssuTranQOL_71):
        assuTran.stra = 'a'
        assuTran.upda()
        
    # Assu.Oupu
    # ----
    def proc_assu_oupu(assuTran:AssuTranQOL_71, oupuTran:OupuTranFile):
        oupuTran.fram_dict[AssuTranQOL_71_dist.__name__] =  { "resu_dist" : { 'df':assuTran.assu_tran_dist.resu_dist, 'mode':'md' } }
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    framTran = FramTran(procTran)
    assuTran = AssuTranQOL_71(procTran)
    
    # Assu
    # ----
    proc_fram_inpu(framTran, inpuTran)
    proc_fram_post(assuTran, framTran)
    proc_assu_exec(assuTran)
    proc_assu_oupu(assuTran, oupuTran)
    
    # Exit
    # ----
    return assuTran

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

if __name__ == "__main__":
    pass
