import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran

class StatTranPATI_01_incl():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
     
class StatTranPATI_01_demo():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
         
class StatTranPATI_01_desc():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu_publ_T0a = None
        self.resu_publ_T0b = None
        self.resu_publ_TX = None
        
class StatTranPATI_01_foll():
    def __init__(self, stat_tran:StatTran):
        self.stat_tran = stat_tran
        #
        self.resu_foll = None
        self.publ_foll = None