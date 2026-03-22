import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11
from qol_10_mixd.c02_qol_11_grph_figu_ import FiguTranQOL_11d_mixd_resi
from qol_10_mixd.c02_qol_11_grph_plot_ import PlotTranQOL_11_mixd_resi, PlotTranQOL_11_mixd_resi_fitt, PlotTranQOL_11_mixd_resi_ququ, PlotTranQOL_11_mixd_resi_hist
from util.data_02_proc import ProcTranStatQOL11   
from util.data_52_oupu import OupuTranGrph

def resi_exec_qol_11_grph_stat(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def resi_figu_exec(figu_tran:FiguTranQOL_11d_mixd_resi):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def resi_figu_ex10(figu_tran:FiguTranQOL_11d_mixd_resi):
        resi_figu_exec(figu_tran)
    
    # ----
    # Plot https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c
    # ----
    def resi_plot_exec(plot_qol_11_mixd:PlotTranQOL_11_mixd_resi, figuTran:FiguTranQOL_11d_mixd_resi, statTran:StatTranQOL_11):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["resu_mixd_modl"] = statTran.stat_tran_mixd.clau_mixd_modl
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def resi_plot_ex00(plot_qol_11_mixd:PlotTranQOL_11_mixd_resi_fitt, figuTran:FiguTranQOL_11d_mixd_resi, statTran:StatTranQOL_11):
        plot_qol_11_mixd.axis = figuTran.ax00
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        resi_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    def resi_plot_ex01(plot_qol_11_mixd:PlotTranQOL_11_mixd_resi_ququ, figuTran:FiguTranQOL_11d_mixd_resi, statTran:StatTranQOL_11):
        plot_qol_11_mixd.axis = figuTran.ax01
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        resi_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    def resi_plot_ex10(plot_qol_11_mixd:PlotTranQOL_11_mixd_resi_hist, figuTran:FiguTranQOL_11d_mixd_resi, statTran:StatTranQOL_11):
        plot_qol_11_mixd.axis = figuTran.ax10
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        resi_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    
    # ----
    # Grph
    # ----
    def resi_grph_exec(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def resi_grph_oupu(figuTran:FiguTranQOL_11d_mixd_resi, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_11d_mixd_resi.__name__}"] =  figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        figuTran = FiguTranQOL_11d_mixd_resi(procTran)
        resi_figu_ex10(figuTran)
        #
        plotTran = PlotTranQOL_11_mixd_resi_fitt(procTran)
        resi_plot_ex00(plotTran, figuTran, statTran)
        plotTran = PlotTranQOL_11_mixd_resi_ququ(procTran)
        resi_plot_ex01(plotTran, figuTran, statTran)
        plotTran = PlotTranQOL_11_mixd_resi_hist(procTran)
        resi_plot_ex10(plotTran, figuTran, statTran)
        #
        resi_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    resi_grph_exec(procTran, statTran, oupuTran)
    
    # Exit
    # ----
    return None