import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranVCSS_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from vcss.c02_vcss_01_stat_desc import exec_stat_desc
from vcss.c02_vcss_01_stat_pure import exec_stat_pure
from vcss.c02_vcss_01_stat_mixd import exec_stat_mixd
from util.data_02_proc import ProcTran
from vcss.c02_vcss_01_stat_desc import StatTranVCSS_01_desc 
from vcss.c02_vcss_01_stat_pure import StatTranVCSS_01_pure 
from vcss.c02_vcss_01_stat_mixd import StatTranVCSS_01_mixd 

class StatTranVCSS_01(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranVCSS_01.__name__, proc_tran)
        #
        self.stat_tran_desc = StatTranVCSS_01_desc(self)
        self.stat_tran_pure = StatTranVCSS_01_pure(self)
        self.stat_tran_mixd = StatTranVCSS_01_mixd(self)
        
    def upda(self):
        exec_stat_desc(self.stat_tran_desc)
        exec_stat_pure(self.stat_tran_pure)
        exec_stat_mixd(self.stat_tran_mixd)