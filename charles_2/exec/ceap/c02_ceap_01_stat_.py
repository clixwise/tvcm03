import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranCEAP_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from ceap.c02_ceap_01_stat_desc import exec_stat_desc
from util.data_02_proc import ProcTran

class StatTranCEAP_01(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranCEAP_01.__name__, proc_tran)
        #
        self.stat_tran_desc = StatTranCEAP_01_desc(self)
        
    def upda(self):
        exec_stat_desc(self.stat_tran_desc)
        
class StatTranCEAP_01_desc():
    def __init__(self, stat_tran:StatTranCEAP_01):
        self.stat_tran = stat_tran
        #
        self.resu_publ_T0 = None