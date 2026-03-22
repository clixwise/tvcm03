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
from qol_10_mixd.c02_qol_11_grph_plot_mixd_micd import exec_plot_mixd_micd
from qol_10_mixd.c02_qol_11_grph_plot_mixd_resi import exec_plot_resi_fitt
from qol_10_mixd.c02_qol_11_grph_plot_mixd_resi import exec_plot_resi_ququ
from qol_10_mixd.c02_qol_11_grph_plot_mixd_resi import exec_plot_resi_hist
from qol_10_mixd.c02_qol_11_grph_plot_mixd_rand import exec_plot_rand_ququ
from qol_10_mixd.c02_qol_11_grph_plot_mixd_rand import exec_plot_rand_hist

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranQOL_11(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError

class PlotTranQOL_11_mixd_mcid(PlotTranQOL_11):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_mcid.__name__] = self
                      
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
        self.line_labl = "GRPH_STRA"
 
        self.mark_size = 6
        self.mark_widt = 1.5
        self.mark_colo = 'white'
        self.capp_size = 4
        self.erro_colo = 'darkgray'

        # Axis, Grid
        # ----
        self.xlab = "VEINES-QOL T-score"
        self.ylab = "Timepoints"
        self.tick_dict = {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = "GRPH_STRA"
        
    def upda(self):
        exec_plot_mixd_micd(self)
        pass  
class PlotTranQOL_11_mixd_fore(PlotTranQOL_11):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_fore.__name__] = self
                      
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
        self.xlab = "VEINES-QOL T-score"
        self.ylab = "Timepoints"
        self.tick_dict = {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
        self.grid_alph = 0.4
        
        # Titl, Lgnd
        # ----
        self.titl = "GRPH_STRA"
        
    def upda(self):
        #exec_plot_mixd_fore(self)
        pass
         
class PlotTranQOL_11_mixd_mono(PlotTranQOL_11):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_mono.__name__] = self
                      
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
        #exec_plot_mixd_evol_mono(self)
        pass 
       
class PlotTranQOL_11_mixd_dual(PlotTranQOL_11):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_dual.__name__] = self
                      
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
        #exec_plot_mixd_evol_dual(self)
        pass
     
class PlotTranQOL_11_mixd_resi(PlotTranQOL_11):
    
    def __init__(self):
        super().__init__()
                      
        # Data
        # ----        
        self.fram_dict = None
        
        # Figu
        # ----
        self.figu = None
        self.axis = None

class PlotTranQOL_11_mixd_resi_fitt(PlotTranQOL_11_mixd_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_resi_fitt.__name__] = self
    
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
        exec_plot_resi_fitt(self)
        pass 
    
class PlotTranQOL_11_mixd_resi_ququ(PlotTranQOL_11_mixd_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_resi_ququ.__name__] = self

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
        exec_plot_resi_ququ(self)
        pass 
           
class PlotTranQOL_11_mixd_resi_hist(PlotTranQOL_11_mixd_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_resi_hist.__name__] = self

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
        exec_plot_resi_hist(self)
        pass 

     
class PlotTranQOL_11_mixd_rand(PlotTranQOL_11):
    
    def __init__(self):
        super().__init__()
                      
        # Data
        # ----        
        self.fram_dict = None
        
        # Figu
        # ----
        self.figu = None
        self.axis = None
    
class PlotTranQOL_11_mixd_rand_ququ(PlotTranQOL_11_mixd_rand):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_rand_ququ.__name__] = self

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
        exec_plot_rand_ququ(self)
        pass 
           
class PlotTranQOL_11_mixd_rand_hist(PlotTranQOL_11_mixd_rand):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_11_mixd_rand_hist.__name__] = self

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
        exec_plot_rand_hist(self)
        pass 
    