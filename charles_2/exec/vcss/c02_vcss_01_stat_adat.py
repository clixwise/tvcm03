import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran
        
class StatTranVCSS_01_desc():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu_tech = None
        self.resu_publ = None
        
class StatTranVCSS_01_pure():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu = None
        
class StatTranVCSS_01_mixd():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu_glob = None
        self.resu_deta = None
        self.resu_plot = None