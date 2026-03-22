import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd
from qol_30_mixd_desc.c02_qol_32_grph_figu_ import FiguTranQOL_32_assu_resi_1
from qol_30_mixd_desc.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi, PlotTranQOL_32_assu_resi_hist, PlotTranQOL_32_assu_resi_ququ
from util.data_02_proc import ProcTranStatQOL31   
from util.data_52_oupu import OupuTranGrph

def main_assu_resi_1(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def figu_exec(figu_tran:FiguTranQOL_32_assu_resi_1):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Assu Random Effects [{figu_tran.stra}'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def figu_ex10(figu_tran:FiguTranQOL_32_assu_resi_1, stra):
        figu_tran.stra = stra
        figu_exec(figu_tran)

    # ----
    # Plot https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
    # ----
    def plot_exec(plot_qol_11_mixd:PlotTranQOL_32_assu_resi, figuTran:FiguTranQOL_32_assu_resi_1, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["orig_fram"] = statTran.mixd_assu_resi_plot
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def plot_ex11(plot_qol_11_mixd:PlotTranQOL_32_assu_resi_hist, figuTran:FiguTranQOL_32_assu_resi_1, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "Assu Residuals with Normal Curve Overlay"
        plot_exec(plot_qol_11_mixd, figuTran, statTran)
    def plot_ex12(plot_qol_11_mixd:PlotTranQOL_32_assu_resi_ququ, figuTran:FiguTranQOL_32_assu_resi_1, statTran:StatTranQOL_31_mixd):
        plot_qol_11_mixd.stra = 'b'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "Assu Residuals Q–Q Plot (Normality Check)"
        plot_exec(plot_qol_11_mixd, figuTran, statTran)
    
    # ----
    # Grph
    # ----
    def grph_exec(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def grph_oupu(figuTran:FiguTranQOL_32_assu_resi_1, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_32_assu_resi_1.__name__} [{figuTran.stra}"] = figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        stra = f"{main_assu_resi_1.__name__}"
        figuTran = FiguTranQOL_32_assu_resi_1(procTran)
        figu_ex10(figuTran, stra)
        #
        plotTran = PlotTranQOL_32_assu_resi_hist(procTran)
        plot_ex11(plotTran, figuTran, statTran)
        plotTran = PlotTranQOL_32_assu_resi_ququ(procTran)
        plot_ex12(plotTran, figuTran, statTran)
        #
        grph_oupu(figuTran, oupuTran)
            
    # ----
    # Main
    # ----
    grph_exec(procTran, statTran, oupuTran)
    
    # Exit
    # ----
    return None