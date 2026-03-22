import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11
from qol_10_mixd.c02_qol_11_grph_figu_ import FiguTranQOL_11d_mixd_rand
from qol_10_mixd.c02_qol_11_grph_plot_ import PlotTranQOL_11_mixd_rand, PlotTranQOL_11_mixd_rand_ququ, PlotTranQOL_11_mixd_rand_hist
from util.data_02_proc import ProcTranStatQOL11   
from util.data_52_oupu import OupuTranGrph

def rand_exec_qol_11_grph_stat(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def rand_figu_exec(figu_tran:FiguTranQOL_11d_mixd_rand):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def rand_figu_ex10(figu_tran:FiguTranQOL_11d_mixd_rand):
        rand_figu_exec(figu_tran)
    
    # ----
    # Plot https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c
    # ----
    def rand_plot_exec(plot_qol_11_mixd:PlotTranQOL_11_mixd_rand, figuTran:FiguTranQOL_11d_mixd_rand, statTran:StatTranQOL_11):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["resu_mixd_modl"] = statTran.stat_tran_mixd.clau_mixd_modl
        plot_qol_11_mixd.fram_dict["resu_mixd_resu"] = statTran.stat_tran_mixd.clau_mixd_resu
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def rand_plot_ex01(plot_qol_11_mixd:PlotTranQOL_11_mixd_rand_ququ, figuTran:FiguTranQOL_11d_mixd_rand, statTran:StatTranQOL_11):
        plot_qol_11_mixd.axis = figuTran.ax01
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        rand_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    def rand_plot_ex10(plot_qol_11_mixd:PlotTranQOL_11_mixd_rand_hist, figuTran:FiguTranQOL_11d_mixd_rand, statTran:StatTranQOL_11):
        plot_qol_11_mixd.axis = figuTran.ax10
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        rand_plot_exec(plot_qol_11_mixd, figuTran, statTran)

    # ----
    # Grph
    # ----
    def rand_grph_exec(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def rand_grph_oupu(figuTran:FiguTranQOL_11d_mixd_rand, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_11d_mixd_rand.__name__}"] =  figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        figuTran = FiguTranQOL_11d_mixd_rand(procTran)
        rand_figu_ex10(figuTran)
        #
        plotTran = PlotTranQOL_11_mixd_rand_ququ(procTran)
        rand_plot_ex01(plotTran, figuTran, statTran)
        plotTran = PlotTranQOL_11_mixd_rand_hist(procTran)
        rand_plot_ex10(plotTran, figuTran, statTran)
        #
        rand_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    rand_grph_exec(procTran, statTran, oupuTran)
    
    # Exit
    # ----
    return None