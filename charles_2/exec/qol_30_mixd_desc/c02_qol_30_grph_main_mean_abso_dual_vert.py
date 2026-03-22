import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd
from qol_30_mixd_desc.c02_qol_32_grph_figu_ import FiguTranQOL_32_mean
from qol_30_mixd_desc.c02_qol_32_grph_plot_ import PlotTranQOL_32_mean_abso_dual_vert
from util.data_02_proc import ProcTranStatQOL31   
from util.data_52_oupu import OupuTranGrph

def main_mean_abso_dual_vert(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):

    # ---- 
    # Figu
    # ----
    def fore_figu_exec(figu_tran:FiguTranQOL_32_mean):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}]\nObserved means & Model estimates'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def fore_figu_ex10(figu_tran:FiguTranQOL_32_mean, stra):
        figu_tran.stra = stra
        fore_figu_exec(figu_tran)
    
    # ----
    # Plot  https://copilot.microsoft.com/shares/Q5DZZFvNPuaU7NF4xrwn6 for a discussion about Forest Plot
    # ----
    def fore_plot_exec(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_dual_vert, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["raww_mean_fram"] = statTran.mixd_mean_raww
        plot_qol_11_mixd.fram_dict["modl_mean_fram"] = statTran.mixd_mean_modl
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def fore_plot_ex11(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_dual_vert, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.titl = "VEINES-QOL Forest Plot : Modeled Mean (95% CI)"
        fore_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    def fore_plot_ex12(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_dual_vert, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.stra = 'b'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.titl = "VEINES-QOL Forest Plot : Raw Mean vs Modeled Mean (95% CI)"
        fore_plot_exec(plot_qol_11_mixd, figuTran, statTran)

    # ----
    # Grph
    # ----
    def fore_grph_exec(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def fore_grph_oupu(figuTran:FiguTranQOL_32_mean, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_32_mean.__name__} [{figuTran.stra}]"] =  figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = f"{main_mean_abso_dual_vert.__name__} ex10"
            figuTran = FiguTranQOL_32_mean(procTran)
            fore_figu_ex10(figuTran, stra)
            #
            plotTran = PlotTranQOL_32_mean_abso_dual_vert(procTran)
            fore_plot_ex11(plotTran, figuTran, statTran)
            plotTran = PlotTranQOL_32_mean_abso_dual_vert(procTran)
            fore_plot_ex12(plotTran, figuTran, statTran)
            #
            fore_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    fore_grph_exec(procTran, statTran, oupuTran) # data linear mixed-effect
    
    # Exit
    # ----
    return None