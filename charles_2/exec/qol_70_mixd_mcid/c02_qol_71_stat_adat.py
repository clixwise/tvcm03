import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
        
class StatTranQOL_71_mcid_copi():
    def __init__(self, stat_tran:StatTran): # stat_tran:StatTranQOL_71):
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
