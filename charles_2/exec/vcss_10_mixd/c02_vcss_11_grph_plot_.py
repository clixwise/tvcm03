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
from plot.c02_anom_01_grph_plot_ import PlotConfANOM_01_hist, PlotTranANOM_01_hist
from vcss_10_mixd.c02_vcss_11_grph_plot_mixd_mono import exec_plot_mixd_mono

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranVCSS_11(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError
        
class PlotTranVCSS_11_mixd_mono(PlotTranVCSS_11):
    
    def __init__(self):
        super().__init__()
                      
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
        exec_plot_mixd_mono(self)
        pass 
        
class PlotTranVCSS_11_mixd_mono_a(PlotTranVCSS_11_mixd_mono):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[type(self).__name__] = self
        
        # Figu
        # ----
        self.stra = 'a'
     
class PlotTranVCSS_11_mixd_mono_b(PlotTranVCSS_11_mixd_mono):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[type(self).__name__] = self
            
        # Figu
        # ----
        self.stra = 'b'
        
class PlotTranVCSS_11_mixd_mono_c(PlotTranVCSS_11_mixd_mono):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[type(self).__name__] = self
                
        # Figu
        # ----
        self.stra = 'c'
         
class PlotTranVCSS_11_mixd_mono_d(PlotTranVCSS_11_mixd_mono):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[type(self).__name__] = self
        
        # Figu
        # ----
        self.stra = 'd'