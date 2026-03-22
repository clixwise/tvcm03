import pandas as pd
import sys  
import os 
import pandas as pd
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#    from figu_grph import PlotTran  # ONLY for type checkers / IDEs
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_31_figu import PlotTran
from qol_60_mixd_effe.c02_qol_62_grph_plot_cohe import plot_cohe

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranQOL_62(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError
     
class PlotTranQOL_62_cohe(PlotTranQOL_62):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_62_cohe.__name__] = self
                      
        # Data
        # ----        
        self.fram_dict = None
        
        # Figu
        # ----
        self.figu = None
        self.axis = None
        self.stra = None
    
        # Grph
        # ----
        self.line_widt = 2
        self.line_styl = "o-"
        self.line_colo = 'royalblue'
        self.line_alph = 1.0 
        self.line_labl = "GRPH_STRA"
        self.mark_size = 6
        self.mark_widt = 1.5
        self.mark_colo = 'white'
        self.capp_size = 4
        self.erro_colo = 'darkgray'

        # Axis, Grid
        # ----
        self.xlab = "Timepoint"
        self.ylab = "GRPH_STRA"
        self.tick_dict = {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = "GRPH_STRA"
        
    def upda(self):
        plot_cohe(self)
        pass 