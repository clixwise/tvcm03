import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran

class StatTranVCSS_11_mixd():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu_fram = None
        self.resu_glob = None
        self.resu_deta = None
        self.resu_plot = None
        self.resu_plot_raw = None
        self.resu_plot_lme = None