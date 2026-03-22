import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_11_stat_ import StatTranVCSS_11_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from vcss_10_mixd.c02_vcss_11_stat_mixd import exec_stat_mixd
from vcss_10_mixd.c02_vcss_11_stat_mixd import StatTranVCSS_11_mixd
from util.data_02_proc import ProcTran

class StatTranVCSS_11(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranVCSS_11.__name__, proc_tran)
        #
        self.stat_tran_mixd = StatTranVCSS_11_mixd(self)
        
    def upda(self):
        exec_stat_mixd(self.stat_tran_mixd)