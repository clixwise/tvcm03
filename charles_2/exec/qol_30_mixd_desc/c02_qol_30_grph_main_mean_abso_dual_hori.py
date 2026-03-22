import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd
from qol_30_mixd_desc.c02_qol_32_grph_figu_ import FiguTranQOL_32_mean
from qol_30_mixd_desc.c02_qol_32_grph_plot_ import PlotTranQOL_32_mean_abso_dual_hori
from util.data_02_proc import ProcTranStatQOL31   
from util.data_52_oupu import OupuTranGrph

def main_mean_abso_dual_hori(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):

    # ---- 
    # Figu
    # ----
    def dual_figu_exec(figu_tran:FiguTranQOL_32_mean):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}]\nObserved means & Model estimates'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def dual_figu_ex10(figu_tran:FiguTranQOL_32_mean, stra):
        figu_tran.stra = stra
        dual_figu_exec(figu_tran)
    
    # ----
    # Plot https://gemini.google.com/app/f11fdb0ca1fc6a70 for a discussion about raw vs modl
    # ----
    def dual_plot_exec(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_dual_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["orig_fram"] = statTran.stat_tran.fram
        plot_qol_11_mixd.fram_dict["mean_fram"] = statTran.mixd_mean_merg
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def dual_plot_ex11(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_dual_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.stra = 'e'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        dual_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    
    # ----
    # Grph
    # ----
    def dual_grph_exec(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def dual_grph_oupu(figuTran:FiguTranQOL_32_mean, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_32_mean.__name__} [{figuTran.stra}]"] = figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = f"{main_mean_abso_dual_hori.__name__} ex10"
            figuTran = FiguTranQOL_32_mean(procTran)
            dual_figu_ex10(figuTran, stra)
            #
            plotTran = PlotTranQOL_32_mean_abso_dual_hori(procTran)
            dual_plot_ex11(plotTran, figuTran, statTran)
            #
            dual_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    dual_grph_exec(procTran, statTran, oupuTran) # data linear mixed-effect
    
    # Exit
    # ----
    return None