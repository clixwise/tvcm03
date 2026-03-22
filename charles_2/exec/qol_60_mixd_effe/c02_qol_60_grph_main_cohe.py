import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_60_mixd_effe.c02_qol_61_stat_ import StatTranQOL_61_cohe
from qol_60_mixd_effe.c02_qol_62_grph_figu_ import FiguTranQOL_61a_cohe
from qol_60_mixd_effe.c02_qol_62_grph_plot_ import PlotTranQOL_62_cohe
from util.data_02_proc import ProcTranStatQOL61   
from util.data_52_oupu import OupuTranGrph

def main_cohe(procTran:ProcTranStatQOL61, statTran:StatTranQOL_61_cohe, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def mono_figu_exec(figu_tran:FiguTranQOL_61a_cohe):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}.{figu_tran.modl}]'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def mono_figu_ex10(figu_tran:FiguTranQOL_61a_cohe, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        mono_figu_exec(figu_tran)
    
    # ----
    # Plot https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
    # ----
    def mono_plot_ex11(plot_qol_11_mixd:PlotTranQOL_62_cohe, figuTran:FiguTranQOL_61a_cohe, statTran:StatTranQOL_61_cohe, modl:str):
        plot_qol_11_mixd.stra = 'd'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Effet Size Raw"
        #
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["cohe_fram"] = statTran.resu_cohe_raww_plot 
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def mono_plot_ex21(plot_qol_11_mixd:PlotTranQOL_62_cohe, figuTran:FiguTranQOL_61a_cohe, statTran:StatTranQOL_61_cohe, modl:str):
        plot_qol_11_mixd.stra = 'd'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Effet Size Model"
        #
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["cohe_fram"] = statTran.resu_cohe_modl_plot 
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    
    # ----
    # Grph
    # ----
    def cohe_grph_exec(procTran:ProcTranStatQOL61, statTran:StatTranQOL_61_cohe, oupuTran:OupuTranGrph, modl:str):
        
        # Oupu
        # ----
        def mono_grph_oupu(figuTran:FiguTranQOL_61a_cohe, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_61a_cohe.__name__} [{figuTran.stra}.{figuTran.modl}]"] = figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = f"{main_cohe.__name__} ex10"
            figuTran = FiguTranQOL_61a_cohe(procTran)
            mono_figu_ex10(figuTran, stra, modl)
            #
            plotTran = PlotTranQOL_62_cohe(procTran)
            mono_plot_ex11(plotTran, figuTran, statTran, modl)
            plotTran = PlotTranQOL_62_cohe(procTran)
            mono_plot_ex21(plotTran, figuTran, statTran, modl)
            #
            mono_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    cohe_grph_exec(procTran, statTran, oupuTran, modl='raw')
    
    # Exit
    # ----
    return None