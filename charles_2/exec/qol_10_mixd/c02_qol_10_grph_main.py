import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11
from qol_10_mixd.c02_qol_10_grph_main_mcid import mcid_exec_qol_11_grph_stat
from qol_10_mixd.c02_qol_10_grph_main_resi import resi_exec_qol_11_grph_stat
from qol_10_mixd.c02_qol_10_grph_main_rand import rand_exec_qol_11_grph_stat
from util.data_02_proc import ProcTranStatQOL11   
from util.data_52_oupu import OupuTranGrph

def main_exec_qol_11_grph_stat(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
    mcid_exec_qol_11_grph_stat(procTran, statTran, oupuTran)
    rand_exec_qol_11_grph_stat(procTran, statTran, oupuTran)
    resi_exec_qol_11_grph_stat(procTran, statTran, oupuTran)
    pass
