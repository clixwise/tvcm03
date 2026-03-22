import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_11_stat_ import StatTranQOL_11_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from qol_30_mixd_desc.c02_qol_31_stat_mixd import exec_stat_mixd
from util.data_02_proc import ProcTran
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd 

class StatTranQOL_31(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranQOL_31.__name__, proc_tran)
        #
        self.stat_tran_mixd = StatTranQOL_31_mixd(self)
        
    def upda(self):
        # OLD exec_stat_mixd(self.stat_tran_mixd)
        exec_stat_mixd(self.stat_tran_mixd)
