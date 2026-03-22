import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_11_stat_ import StatTranQOL_11_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from qol_60_mixd_effe.c02_qol_61_stat_effe_raww_gemi_2026_03_12 import exec_stat_effe_raww
from qol_60_mixd_effe.c02_qol_61_stat_effe_modl_gemi_2026_03_12 import exec_stat_effe_modl
from util.data_02_proc import ProcTran
from qol_60_mixd_effe.c02_qol_61_stat_adat import StatTranQOL_61_cohe 


class StatTranQOL_61(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranQOL_61.__name__, proc_tran)
        #
        self.stat_tran_cohe = StatTranQOL_61_cohe(self)
        
    def upda(self):
        #
        exec_stat_effe_raww(self.stat_tran_cohe)
        exec_stat_effe_modl(self.stat_tran_cohe)