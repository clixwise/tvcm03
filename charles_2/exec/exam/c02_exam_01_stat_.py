import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranEXAM_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from exam.c02_exam_01_stat_desc import exec_stat_desc
from exam.c02_exam_01_stat_pure import exec_stat_pure
from util.data_02_proc import ProcTran

class StatTranEXAM_01(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranEXAM_01.__name__, proc_tran)
        #
        self.stat_tran_desc = StatTranEXAM_01_desc(self)
        self.stat_tran_pure = StatTranEXAM_01_pure(self)
        
    def upda(self):
        exec_stat_desc(self.stat_tran_desc)
        exec_stat_pure(self.stat_tran_pure)
        
class StatTranEXAM_01_desc():
    def __init__(self, stat_tran:StatTranEXAM_01):
        self.stat_tran = stat_tran
        #
        self.resu_publ_T0 = None
        self.resu_publ_TX = None
        
class StatTranEXAM_01_pure():
    def __init__(self, stat_tran:StatTranEXAM_01):
        self.stat_tran = stat_tran
        #
        self.resu_dict = None
        self.resu_ceap_limb = None