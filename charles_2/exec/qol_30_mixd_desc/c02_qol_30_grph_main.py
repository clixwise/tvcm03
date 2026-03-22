import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31, StatTranQOL_31_mixd
from qol_30_mixd_desc.c02_qol_30_grph_main_assu_rand import main_assu_rand
from qol_30_mixd_desc.c02_qol_30_grph_main_assu_resi_1 import main_assu_resi_1
from qol_30_mixd_desc.c02_qol_30_grph_main_assu_resi_2 import main_assu_resi_2
from qol_30_mixd_desc.c02_qol_30_grph_main_mean_abso_mono_hori import main_mean_abso_mono_hori
from qol_30_mixd_desc.c02_qol_30_grph_main_mean_abso_dual_hori import main_mean_abso_dual_hori
from qol_30_mixd_desc.c02_qol_30_grph_main_mean_abso_dual_vert import main_mean_abso_dual_vert
from util.data_02_proc import ProcTranAssuQOL31, ProcTranStatQOL31  

from util.data_02_proc import ProcTranStatQOL11   
from util.data_52_oupu import OupuTranGrph

#def main_exec_qol_31_grph_assu(procTran:ProcTranAssuQOL31, assuTran:AssuTranQOL_31, oupuTran:OupuTranGrph):
def main_exec_qol_31_grph_assu(procTran:ProcTranAssuQOL31, assuTran, oupuTran:OupuTranGrph):

    # see : main_exec_qol_01_grph_assu
    
    raise Exception()

def main_exec_qol_30_grph_stat(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31, oupuTran:OupuTranGrph):
    
    # Data
    # ----
    stat_tran_mixd: StatTranQOL_31_mixd = statTran.stat_tran_mixd
    
    # Exec
    # ----
    main_assu_resi_1(procTran, stat_tran_mixd, oupuTran)
    main_assu_resi_2(procTran, stat_tran_mixd, oupuTran)
    main_assu_rand(procTran, stat_tran_mixd, oupuTran)
    
    # Exec
    # ----
    main_mean_abso_mono_hori(procTran, stat_tran_mixd, oupuTran)
    main_mean_abso_dual_hori(procTran, stat_tran_mixd, oupuTran)
    main_mean_abso_dual_vert(procTran, stat_tran_mixd, oupuTran)
    pass
