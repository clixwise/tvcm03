import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
    
class StatTranQOL_61_cohe():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        
        self.resu_cohe_raww_stat = None
        self.resu_cohe_raww_plot = None

        self.resu_cohe_modl_plot = None
        self.resu_cohe_modl_stat = None