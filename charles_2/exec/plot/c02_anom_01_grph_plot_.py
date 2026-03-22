import pandas as pd
import sys  
import os 
from dataclasses import dataclass, field
from typing import Optional, Any
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#    from figu_grph import PlotTran  # ONLY for type checkers / IDEs
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_31_figu import PlotTran
from plot.c02_anom_01_grph_plot_hist import exec_plot_hist
from plot.c02_anom_01_grph_plot_boxx import exec_plot_boxx

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class PlotTranANOM_01(PlotTran):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        raise NotImplementedError

# ----
# Hist
# ----
@dataclass
class PlotConfANOM_01_hist:
    
    # Data
    # ---- 
    fram: Optional[pd.DataFrame] = None
    colu: str = None
    
    # Figu
    # ----
    figu: Any = None
    axis: Any = None
    stra: Any = None
    # 
    stra_hist: bool = True 
    stra_kdee: bool = True 
    stra_norm: bool = True 
    stra_cdff: bool = True 
    stra_quar: bool = True 
    stra_medi: bool = True 
    stra_madd: bool = True 
    stra_mean: bool = True 
    stra_mode: bool = True

    # Grph
    # ----
    hist_widt: str = None
    hist_styl: str = None
    hist_colo: str = None
    hist_alph: Optional[float] = None
    binn_size: int = None 
    
    norm_widt: str = None
    norm_styl: str = None
    norm_colo: str = None
    
    cdff_widt: str = None
    cdff_styl: str = None
    cdff_colo: str = None

    # Axis, Grid
    # ----
    xlab: str = None
    yla1: str = None 
    yla2: str = None
    grid_alph: Optional[float] = None
    
    # Titl, Lgnd
    # ----
    titl: Optional[str] = None

class PlotTranANOM_01_hist(PlotTranANOM_01):
    
    def __init__(self):
        super().__init__()
        # We declare that a 'conf' will exist, but we don't fill it with QOL values here.
        self.conf: PlotConfANOM_01_hist = None
        
    def upda(self):
        exec_plot_hist(self)
        pass 

# ----
# Boxx
# ----
@dataclass
class PlotConfANOM_01_boxx:
    
    # Data
    # ---- 
    fram: Optional[pd.DataFrame] = None
    colu: str = None
    
    # Figu
    # ----
    figu: Any = None
    axis: Any = None
    stra: Any = None
    # 
    stra_hist: bool = True 
    stra_kdee: bool = True 
    stra_norm: bool = True 
    stra_cdff: bool = True 
    stra_quar: bool = True 
    stra_medi: bool = True 
    stra_madd: bool = True 
    stra_mean: bool = True 
    stra_mode: bool = True

    # Grph
    # ----
    hist_widt: str = None
    hist_styl: str = None
    hist_colo: str = None
    hist_alph: Optional[float] = None
    binn_size: int = None 
    
    norm_widt: str = None
    norm_styl: str = None
    norm_colo: str = None
    
    cdff_widt: str = None
    cdff_styl: str = None
    cdff_colo: str = None

    # Axis, Grid
    # ----
    xlab: str = None
    yla1: str = None 
    yla2: str = None
    grid_alph: Optional[float] = None
    
    # Titl, Lgnd
    # ----
    titl: Optional[str] = None
    
class PlotTranANOM_01_boxx(PlotTranANOM_01):
    
    def __init__(self):
        super().__init__()
        # We declare that a 'conf' will exist, but we don't fill it with QOL values here.
        self.conf: PlotConfANOM_01_boxx = None
        
    def upda(self):
        exec_plot_boxx(self)
        pass 