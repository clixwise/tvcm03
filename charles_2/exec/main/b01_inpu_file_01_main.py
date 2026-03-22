import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_02_proc import ProcTranStatInpu01
from util.data_51_inpu import InpuTran, inpu_file_exec_selc_0, inpu_file_exec_xlat_0

def main_exec_inpu_01(procTran:ProcTranStatInpu01) -> InpuTran:

    # Inpu
    # ----
    def proc_exec(inpuTran:InpuTran):
        inpuTran.file = "../../../../data_qol_00_data/03_equal/2099-01-01 2099-01-01 TX/09_QUES_FRAME_DETA/df_pati,qol,sym,mcid,vcss,exam nois incr.csv" # T0, T1, T2
        inpuTran.file = "../../data/df_pati,qol,sym,mcid,vcss,exam nois incr.csv" # T0, T1, T2
        inpuTran.filt = None
        inpuTran.func = inpu_file_exec_selc_0
        inpuTran.upda()
    
    def proc_chck(inpuTran:InpuTran):
           
        # Data
        # ----
        df_fram = inpuTran.fram
                   
        # Exec (for info, when > 15pc)
        # ----
        Q_pati_isok = df_fram[df_fram['Q_pati_isok'] == False]
        if len(Q_pati_isok) > 0:
            Q_pati_isok = Q_pati_isok[['workbook','Q_pati_isok','Q_pati_50pc']]
            print_yes(Q_pati_isok, labl = 'Q_pati_isok')
            pass # since we accept 15%
        #
        S_pati_isok = df_fram[df_fram['S_pati_isok'] == False]
        if len(S_pati_isok) > 0:
            S_pati_isok = S_pati_isok[['workbook','S_pati_isok','S_pati_50pc']]
            print_yes(S_pati_isok, labl = 'S_pati_isok')
            pass # since we accept 15%
        
        # Exec (for excp, when > 50pc)
        # ----
        S_pati_50pc = df_fram[df_fram['S_pati_50pc'] == False]
        if len(S_pati_50pc) > 0:
            print_yes(S_pati_50pc, labl = 'S_pati_50pc')
            raise Exception()
        #
        Q_pati_50pc = df_fram[df_fram['Q_pati_50pc'] == False]
        if len(Q_pati_50pc) > 0:
            print_yes(Q_pati_50pc, labl = 'Q_pati_50pc')
            raise Exception()
        
    def proc_post(inpuTran:InpuTran):
        
        trac = True
           
        # Data
        # ----
        df_fram = inpuTran.fram
                   
        # Frmt
        # ----
        # df_fram['Age'] = pd.to_numeric(df_fram['M_Age'], errors='coerce') # not necessary
        
        # Cate
        # ----
        cate_list = ['T0','T1','T2']
        df_fram = inpu_file_exec_xlat_0 (df_fram, "timepoint", cate_list)
                
        # Anon
        # ----
        # 1. Create a mapping of unique patient_ids to a new "P999" format
        unique_ids = df_fram['patient_id'].unique()
        id_map = {old_id: f'P{i+1:03}' for i, old_id in enumerate(unique_ids)}
        # 2. Map the new IDs to a new column
        df_fram['anon_id'] = df_fram['patient_id'].map(id_map)
        # 3. Anonymize by dropping the sensitive columns
        # We drop 'patient_id' and 'workbook' (which contains names)
        # df_anon = df_fram.drop(columns=['workbook', 'patient_id'])
        # Reorder columns to put 'id' first (optional but cleaner)
        # cols = ['patient'] + [c for c in df_fram.columns if c != 'patient']
        # df_fram = df_fram[cols]
        #
        if trac:
            print_yes(df_fram.iloc[9:12], labl='df_fram.iloc[9:12]')
        
        # Data
        # ----
        inpuTran.fram = df_fram

    # Exec
    # ----
    inpuTran = InpuTran(procTran)
    proc_exec(inpuTran)
    proc_chck(inpuTran)
    proc_post(inpuTran)
    
    # Exit
    # ----
    return inpuTran

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
    
    # Exec
    # ----
    main_exec_inpu_01()
    pass
