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
from vcss.c02_vcss_01_grph_plot_scat import exec_plot_scat

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranPATI_01(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError
        
class PlotTranPATI_01_hist(PlotTranANOM_01_hist):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[PlotTranPATI_01_hist.__name__] = self  
        #
        self.fram = None
        self.colu = None
        self.figu = None
        self.axis = None
        self.hist_colo = None
        self.stra = None
        self.titl = None
    
    def parm(self):
        
        # Exec You only define what is UNIQUE to QOL. Everything else (stra_hist, norm_widt, etc.) stays at the agnostic defaults defined in the dataclass.
        # ----
        self.conf = PlotConfANOM_01_hist(
            
        # Data
        # ----        
        fram = self.fram,
        colu = self.colu,
        
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
        hist_colo = self.hist_colo,
        hist_alph = None,
        binn_size = 1,
        
        norm_widt = '0.5',
        norm_styl = '-',
        norm_colo = 'green',
        
        cdff_widt = '0.5',
        cdff_styl = '-',
        cdff_colo = 'C0', # Use the first default color

        # Axis, Grid
        # ----
        xlab = "PATI Total Score",
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
            
class PlotTranPATI_01_scat(PlotTranPATI_01):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranPATI_01_scat.__name__] = self
        
        # Data
        # ----        
        self.fram = None
        self.colu_list = ['L','R']
        
        # Figu
        # ----
        self.figu = None
        self.axis = None
        self.stra = None

        # Grph
        # ----
        self.scat_size = 60
        self.scat_edge = 'black'
        self.scat_colo = 'purple'
        self.scat_alph = 0.6 
        
        self.line_widt = 2
        self.line_styl = '--'
        self.line_colo = 'blue'
        self.line_alph = 1.0 
        self.line_labl = 'Diagonal Symmetry'

        # Axis, Grid
        # ----
        self.xlab = "PATI_L"
        self.ylab = "PATI_R" 
        self.axix_intv = 4
        self.axiy_intv = 4
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = "PATI R vs L"

    def upda(self):
        exec_plot_scat(self)
        pass 