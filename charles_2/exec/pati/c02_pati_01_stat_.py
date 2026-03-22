import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranPATI_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from pati.c02_pati_01_stat_incl import exec_stat_incl
from pati.c02_pati_01_stat_demo import exec_stat_demo
from pati.c02_pati_01_stat_desc import exec_stat_desc
from pati.c02_pati_01_stat_foll import exec_stat_foll
from util.data_02_proc import ProcTran
from pati.c02_pati_01_stat_adat import StatTranPATI_01_incl 
from pati.c02_pati_01_stat_adat import StatTranPATI_01_demo 
from pati.c02_pati_01_stat_adat import StatTranPATI_01_desc 
from pati.c02_pati_01_stat_adat import StatTranPATI_01_foll 

class StatTranPATI_01(StatTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(StatTranPATI_01.__name__, proc_tran)
        #
        self.ques_dict = None
        #
        self.stat_tran_incl = StatTranPATI_01_incl(self)
        self.stat_tran_demo = StatTranPATI_01_demo(self)
        self.stat_tran_desc = StatTranPATI_01_desc(self)
        self.stat_tran_foll = StatTranPATI_01_foll(self)
        
    def upda(self):
        exec_stat_incl(self.stat_tran_incl)
        exec_stat_demo(self.stat_tran_desc)
        exec_stat_desc(self.stat_tran_desc)
        exec_stat_foll(self.stat_tran_foll)