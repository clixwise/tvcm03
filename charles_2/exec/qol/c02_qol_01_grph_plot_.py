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
from qol.c02_qol_01_grph_plot_ququ import exec_plot_ququ
from qol.c02_qol_01_grph_plot_mean import exec_plot_mean

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranQOL_01(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError

class PlotTranQOL_01_hist(PlotTranANOM_01_hist):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[PlotTranQOL_01_hist.__name__] = self 
        #
        self.fram = None
        self.figu = None
        self.axis = None
        self.stra = None
        self.titl = None
    
    def parm(self): 
           
        # Exec You only define what is UNIQUE to QOL. Everything else (stra_hist, norm_widt, etc.) stays at the agnostic defaults defined in the dataclass.
        # ----
        self.conf = PlotConfANOM_01_hist(
            
        # Data
        # ----        
        fram = self.fram,
        colu = 'VEINES_QOL_t',
        
        # Figu
        # ----
        figu = self.figu,
        axis = self.axis,
        stra = self.stra,
        # 
        stra_hist = True, 
        stra_kdee = True, 
        stra_norm = True, 
        stra_cdff = True, 
        stra_quar = True,
        stra_medi = True, 
        stra_madd = True, 
        stra_mean = True, 
        stra_mode = True,

        # Grph
        # ----
        hist_widt = '0.5',
        hist_styl = '-',
        hist_colo = 'royalblue',
        hist_alph = None,
        binn_size = 1 ,
        
        norm_widt = '0.5',
        norm_styl = '-',
        norm_colo = 'green',
        
        cdff_widt = '0.5',
        cdff_styl = '-',
        cdff_colo = 'C0', # Use the first default color

        # Axis, Grid
        # ----
        xlab = "QOL Total Score",
        yla1 = "Frequency", 
        yla2 = "Cumulative Distribution (normalized)", 
        grid_alph = None,
        
        # Titl, Lgnd
        # ----
        titl = self.titl
        )
        
    def upda(self):
        super().upda()
        pass 
                
class PlotTranQOL_01_ququ(PlotTranQOL_01):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_01_ququ.__name__] = self        
                
        # Data
        # ----        
        self.fram = None
        self.colu = None
        
        # Figu
        # ----
        self.figu = None
        self.axis = None
        self.stra = None

        # Grph
        # ----
        self.scat_size = 50
        self.scat_widt = 1.2, # Thickness of the circle
        self.scat_edge = '#1f77b4' # Muted blue
        self.scat_colo = 'none' # Hollow centers
        self.scat_alph = 0.6 
        self.scat_labl = 'Observed Data'
        
        self.line_widt = 2
        self.line_styl = '--'
        self.line_colo = '#d62728' # Muted red
        self.line_alph = 0.8 
        self.line_labl = f'Theoretical Normal'

        # Axis, Grid
        # ----
        self.xlab = "Theoretical Quantiles"
        self.ylab = "Ordered Values" 
        self.axix_intv = 4
        self.axiy_intv = 4
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = None
        
    def upda(self):
        exec_plot_ququ(self)
        pass 
        
class PlotTranQOL_01_mean(PlotTranQOL_01):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_01_mean.__name__] = self
                      
        # Data
        # ----        
        self.fram = None
        
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
        self.line_labl = 'Score Mean, SE'
        self.mark_size = 6
        self.capp_size = 4
        self.erro_colo = 'darkgray'

        # Axis, Grid
        # ----
        self.xlab = "Timepoint"
        self.ylab = "VEINES-QOL score"
        self.tick_dict = {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = "VEINES-QOL over time"
        
    def upda(self):
        exec_plot_mean(self)
        pass 