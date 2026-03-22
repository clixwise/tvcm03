import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd
from qol_30_mixd_desc.c02_qol_32_grph_figu_ import FiguTranQOL_32_mean
from qol_30_mixd_desc.c02_qol_32_grph_plot_ import PlotTranQOL_32_mean_abso_mono_hori
from util.data_02_proc import ProcTranStatQOL31   
from util.data_52_oupu import OupuTranGrph

def main_mean_abso_mono_hori(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def mono_figu_exec(figu_tran:FiguTranQOL_32_mean):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}.{figu_tran.modl}]'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def mono_figu_ex10(figu_tran:FiguTranQOL_32_mean, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        mono_figu_exec(figu_tran)
    def mono_figu_ex20(figu_tran:FiguTranQOL_32_mean, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        mono_figu_exec(figu_tran)
    
    # ----
    # Plot https://gemini.google.com/app/f11fdb0ca1fc6a70 for a discussion about raw vs modl
    # ----
    def mono_plot_exec(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_mono_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd, modl:str):
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["orig_fram"] = statTran.stat_tran.fram
        match modl: 
            case 'lme':
                plot_qol_11_mixd.fram_dict["mean_fram"] = statTran.mixd_mean_modl # open_resu_plot_lme [=mixd_open_modl_mean] 
            case 'raw':
                plot_qol_11_mixd.fram_dict["mean_fram"] = statTran.mixd_mean_raww # open_resu_plot_raw
            case _:
                raise Exception()
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def mono_plot_ex11(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_mono_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd, modl:str):
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        mono_plot_exec(plot_qol_11_mixd, figuTran, statTran, modl)
    def mono_plot_ex12(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_mono_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd, modl:str):
        plot_qol_11_mixd.stra = 'b'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        mono_plot_exec(plot_qol_11_mixd, figuTran, statTran, modl)
    def mono_plot_ex21(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_mono_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd, modl:str):
        plot_qol_11_mixd.stra = 'c'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Δ from Baseline"
        plot_qol_11_mixd.ylab = "Score Δ"
        plot_qol_11_mixd.titl = "VEINES-QOL Score Δ from baseline over time"
        mono_plot_exec(plot_qol_11_mixd, figuTran, statTran, modl)
    def mono_plot_ex22(plot_qol_11_mixd:PlotTranQOL_32_mean_abso_mono_hori, figuTran:FiguTranQOL_32_mean, statTran:StatTranQOL_31_mixd, modl:str):
        plot_qol_11_mixd.stra = 'd'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.line_labl = "Mean"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Individual and Mean scores over time"
        mono_plot_exec(plot_qol_11_mixd, figuTran, statTran, modl)
    
    # ----
    # Grph
    # ----
    def mono_grph_exec(procTran:ProcTranStatQOL31, statTran:StatTranQOL_31_mixd, oupuTran:OupuTranGrph, modl:str):
        
        # Oupu
        # ----
        def mono_grph_oupu(figuTran:FiguTranQOL_32_mean, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_32_mean.__name__} [{figuTran.stra}.{figuTran.modl}]"] = figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = f"{main_mean_abso_mono_hori.__name__} ex10"
            figuTran = FiguTranQOL_32_mean(procTran)
            mono_figu_ex10(figuTran, stra, modl)
            #
            plotTran = PlotTranQOL_32_mean_abso_mono_hori(procTran)
            mono_plot_ex11(plotTran, figuTran, statTran, modl)
            plotTran = PlotTranQOL_32_mean_abso_mono_hori(procTran)
            mono_plot_ex12(plotTran, figuTran, statTran, modl)
            #
            mono_grph_oupu(figuTran, oupuTran)
        #
        ex20 = True
        if ex20:
            stra = f"{main_mean_abso_mono_hori.__name__} ex20"
            figuTran = FiguTranQOL_32_mean(procTran)
            mono_figu_ex20(figuTran, stra, modl)
            #
            plotTran = PlotTranQOL_32_mean_abso_mono_hori(procTran)
            mono_plot_ex21(plotTran, figuTran, statTran, modl)
            plotTran = PlotTranQOL_32_mean_abso_mono_hori(procTran)
            mono_plot_ex22(plotTran, figuTran, statTran, modl)
            #
            mono_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    mono_grph_exec(procTran, statTran, oupuTran, modl='raw') # data raw
    mono_grph_exec(procTran, statTran, oupuTran, modl='lme') # data linear mixed-effect
    
    # Exit
    # ----
    return None