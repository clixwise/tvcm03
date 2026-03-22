import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_stat_ import StatTranQOL_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from qol.c02_qol_01_stat_desc import exec_stat_desc
from qol.c02_qol_01_stat_desc import StatTranQOL_01_desc
from qol.c02_qol_01_stat_mean import exec_stat_mean
from qol.c02_qol_01_stat_mean import StatTranQOL_01_mean
from qol.c02_qol_01_stat_mixd import exec_stat_mixd
from qol.c02_qol_01_stat_mixd import StatTranQOL_01_mixd
from util.data_02_proc import ProcTran

class StatTranQOL_01(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranQOL_01.__name__, proc_tran)
        #
        self.stat_tran_desc = StatTranQOL_01_desc(self)
        self.stat_tran_mean = StatTranQOL_01_mean(self)
        self.stat_tran_mixd = StatTranQOL_01_mixd(self)
        
    def upda(self):
        exec_stat_desc(self.stat_tran_desc)
        exec_stat_mean(self.stat_tran_mean)
        exec_stat_mixd(self.stat_tran_mixd)