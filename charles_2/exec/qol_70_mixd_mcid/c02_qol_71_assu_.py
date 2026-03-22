import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_assu_ import AssuTranQOL_71_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_20_assu import AssuTran
from qol_70_mixd_mcid.c02_qol_71_assu_hiqq import exec_assu_hiqq
from qol_70_mixd_mcid.c02_qol_71_assu_dist import exec_assu_dist
from util.data_02_proc import ProcTran

class AssuTranQOL_71(AssuTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(AssuTranQOL_71.__name__, proc_tran)
        #
        self.assu_tran_hiqq = AssuTranQOL_71_hiqq(self)
        self.assu_tran_dist = AssuTranQOL_71_dist(self)
        
    def upda(self):
        exec_assu_hiqq(self.assu_tran_hiqq)
        exec_assu_dist(self.assu_tran_dist)
              
class AssuTranQOL_71_hiqq():
    def __init__(self, assu_tran:AssuTranQOL_71):
        self.assu_tran = assu_tran
        #
        self.resu_plot = None

class AssuTranQOL_71_dist():
    def __init__(self, assu_tran:AssuTranQOL_71):
        self.assu_tran = assu_tran
        #
        self.resu_wide = None
        self.resu_dist = None