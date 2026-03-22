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
from qol_30_mixd_desc.c02_qol_32_grph_plot_mean_abso_dual_vert import plot_mean_abso_dual_vert
from qol_30_mixd_desc.c02_qol_32_grph_plot_mean_abso_mono_hori import plot_mean_abso_mono_hori
from qol_30_mixd_desc.c02_qol_32_grph_plot_mean_abso_dual_hori import plot_mean_abso_dual_hori
from qol_30_mixd_desc.c02_qol_32_grph_plot_assu_rand import plot_assu_rand_hist, plot_assu_rand_ququ
from qol_30_mixd_desc.c02_qol_32_grph_plot_assu_resi_1 import plot_assu_resi_hist, plot_assu_resi_ququ
from qol_30_mixd_desc.c02_qol_32_grph_plot_assu_resi_2 import plot_assu_resi_cook, plot_assu_resi_fitt, plot_assu_resi_scal  

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranQOL_32(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError

class PlotTranQOL_32_mean_abso_dual_vert(PlotTranQOL_32):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_mean_abso_dual_vert.__name__] = self
                      
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
        plot_mean_abso_dual_vert(self)
        pass  

class PlotTranQOL_32_mean_abso_dual_hori(PlotTranQOL_32):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_mean_abso_dual_vert.__name__] = self
                      
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
        plot_mean_abso_dual_hori(self)
        pass 

class PlotTranQOL_32_assu_rand(PlotTranQOL_32):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_rand.__name__] = self
        
class PlotTranQOL_32_assu_rand_hist(PlotTranQOL_32_assu_rand):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_rand_hist.__name__] = self
                      
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
        plot_assu_rand_hist(self)
        pass

class PlotTranQOL_32_assu_rand_ququ(PlotTranQOL_32_assu_rand):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_rand_ququ.__name__] = self
                      
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
        plot_assu_rand_ququ(self)
        pass


class PlotTranQOL_32_assu_resi(PlotTranQOL_32):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi.__name__] = self
        
class PlotTranQOL_32_assu_resi_hist(PlotTranQOL_32_assu_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi_hist.__name__] = self
                      
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
        plot_assu_resi_hist(self)
        pass

class PlotTranQOL_32_assu_resi_ququ(PlotTranQOL_32_assu_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi_ququ.__name__] = self
                      
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
        plot_assu_resi_ququ(self)
        pass
    
class PlotTranQOL_32_assu_resi_cook(PlotTranQOL_32_assu_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi_cook.__name__] = self
                      
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
        plot_assu_resi_cook(self)
        pass
 
class PlotTranQOL_32_assu_resi_fitt(PlotTranQOL_32_assu_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi_fitt.__name__] = self
                      
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
        plot_assu_resi_fitt(self)
        pass
    
class PlotTranQOL_32_assu_resi_scal(PlotTranQOL_32_assu_resi):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__(procTran) #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_assu_resi_scal.__name__] = self
                      
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
        plot_assu_resi_scal(self)
        pass
       
class PlotTranQOL_32_mean_abso_mono_hori(PlotTranQOL_32):
    
    def __init__(self, procTran): #, stat_tran, figu_tran, figu_axis):
        super().__init__() #(stat_tran, figu_tran, figu_axis
        procTran.dict[PlotTranQOL_32_mean_abso_dual_vert.__name__] = self
                      
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
        plot_mean_abso_mono_hori(self)
        pass
         
class PlotTranQOL_11_mixd_mono(PlotTranQOL_32):
    
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
       
class PlotTranQOL_11_mixd_dual(PlotTranQOL_32):
    
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
