import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_assu_ import AssuTranQOL_01_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_20_assu import AssuTran
from qol.c02_qol_01_assu_hiqq import exec_assu_hiqq
from qol.c02_qol_01_assu_summ import exec_assu_summ
from qol.c02_qol_01_assu_ceil import exec_assu_ceil
from qol.c02_qol_01_assu_isok import exec_assu_isok
from qol.c02_qol_01_assu_synt import exec_assu_synt
from util.data_02_proc import ProcTran

class AssuTranQOL_01(AssuTran):
    def __init__(self, proc_tran:ProcTran):
        super().__init__(AssuTranQOL_01.__name__, proc_tran)
        #
        self.assu_tran_hiqq = AssuTranQOL_01_hiqq(self)
        self.assu_tran_summ = AssuTranQOL_01_summ(self)
        self.assu_tran_ceil = AssuTranQOL_01_ceil(self)
        self.assu_tran_isok = AssuTranQOL_01_isok(self)
        self.assu_tran_synt = AssuTranQOL_01_synt(self)
        
    def upda(self):
        exec_assu_hiqq(self.assu_tran_hiqq)
        exec_assu_isok(self.assu_tran_isok)
        exec_assu_summ(self.assu_tran_summ)
        exec_assu_ceil(self.assu_tran_ceil)
        exec_assu_synt(self.assu_tran_synt, self.assu_tran_isok, self.assu_tran_summ, self.assu_tran_ceil)
              
class AssuTranQOL_01_hiqq():
    def __init__(self, assu_tran:AssuTranQOL_01):
        self.assu_tran = assu_tran
        #
        self.resu_plot = None

class AssuTranQOL_01_isok():
    def __init__(self, assu_tran:AssuTranQOL_01):
        self.assu_tran = assu_tran
        #
        self.resu_tech = None
                      
class AssuTranQOL_01_summ():
    def __init__(self, assu_tran:AssuTranQOL_01):
        self.assu_tran = assu_tran
        #
        self.resu_tech = None
                
class AssuTranQOL_01_ceil():
    def __init__(self, assu_tran:AssuTranQOL_01):
        self.assu_tran = assu_tran
        #
        self.resu_tech = None
                
class AssuTranQOL_01_synt():
    def __init__(self, assu_tran:AssuTranQOL_01):
        self.assu_tran = assu_tran
        #
        self.resu_synt = None
        self.resu_warn = None