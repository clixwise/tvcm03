import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_60_mixd_effe.c02_qol_61_stat_ import StatTranQOL_61, StatTranQOL_61_cohe
from qol_60_mixd_effe.c02_qol_60_grph_main_cohe import main_cohe
from util.data_02_proc import ProcTranStatQOL61  

from util.data_52_oupu import OupuTranGrph

def main_exec_qol_61_grph_stat(procTran:ProcTranStatQOL61, statTran:StatTranQOL_61, oupuTran:OupuTranGrph):
    
    # Data
    # ----
    main_cohe(procTran, statTran.stat_tran_cohe, oupuTran) # Cohen for 'raww data' ; yse 'synt' for 'modl data'
    
    # Exec
    # ----
    pass