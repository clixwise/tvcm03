import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_11_stat_ import StatTranQOL_11_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from qol_80_mixd_sens.c02_qol_81_stat_mcid_copi_1 import exec_stat_mcid_copi_1
from qol_80_mixd_sens.c02_qol_81_stat_mcid_copi_2 import exec_stat_mcid_copi_2
from util.data_02_proc import ProcTran
from qol_80_mixd_sens.c02_qol_81_stat_adat import StatTranQOL_81_mcid_copi 

class StatTranQOL_81(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranQOL_81.__name__, proc_tran)
        #
        self.stat_tran_mcid_copi = StatTranQOL_81_mcid_copi(self)
       
    def upda(self):

        exec_stat_mcid_copi_1(self.stat_tran_mcid_copi)
        exec_stat_mcid_copi_2(self.stat_tran_mcid_copi)
        pass
'''     
class StatTranQOL_81_mcid_copi():
    def __init__(self, stat_tran:StatTranQOL_81):
        self.stat_tran = stat_tran
        
        self.resu_wide = None
        self.resu_1_anch_mean_change = None
        self.resu_2_anch_roc = None
        self.resu_3_dist = None
        self.resu_4_variability = None
        self.resu_synt = None
        
        self.plot_anch = None
        self.plot_roc_data = None
        self.plot_roc_meta = None
'''
