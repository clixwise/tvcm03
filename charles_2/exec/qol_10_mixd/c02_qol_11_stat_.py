import pandas as pd
import sys  
import os 
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_11_stat_ import StatTranQOL_11_desc
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
from qol_10_mixd.c02_qol_11_stat_mixd import exec_stat_mixd

class StatTranQOL_11(StatTran):
    def __init__(self, proc_tran):
        super().__init__()
        proc_tran.dict[StatTranQOL_11.__name__] = self
        #
        self.stat_tran_mixd = StatTranQOL_11_mixd(self)
        
    def upda(self):
        exec_stat_mixd(self.stat_tran_mixd)

class StatTranQOL_11_mixd():
    def __init__(self, stat_tran:StatTranQOL_11):
        self.stat_tran = stat_tran
        #
        self.open_resu_fram = None
        self.open_resu_plot = None
        self.open_resu_plot_raw = None
        self.open_resu_plot_lme = None
        
        self.clau_mixd_modl = None
        self.clau_mixd_resu = None
        self.clau_mixd_modl_mean = None
        self.clau_mixd_modl_delt = None
        self.clau_mixd_resi_desc = None
        self.clau_mixd_resi_stat = None
        
        self.copi_mixd_raww_mean = None
        self.copi_mixd_modl_mean = None
        self.copi_mixd_modl_emms = None
        self.copi_mixd_modl_pair = None
        
        self.gemi_mixd_mcid = None
        self.gemi_mixd_mcid_grop = None
        self.gemi_mixd_mcid_pat1 = None
        self.gemi_mixd_mcid_pat2 = None
        self.gemi_mixd_mcid_anal = None
        self.gemi_mixd_mcid_popu_delt = None
        self.gemi_mixd_mcid_effe_size = None