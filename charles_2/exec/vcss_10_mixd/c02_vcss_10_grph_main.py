import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vcss_10_mixd.c02_vcss_11_stat_ import StatTranVCSS_11
from vcss_10_mixd.c02_vcss_11_grph_figu_ import FiguTranVCSS_11c
from vcss_10_mixd.c02_vcss_11_grph_plot_ import PlotTranVCSS_11_mixd_mono, PlotTranVCSS_11_mixd_mono_a, PlotTranVCSS_11_mixd_mono_b, PlotTranVCSS_11_mixd_mono_c, PlotTranVCSS_11_mixd_mono_d
from util.data_02_proc import ProcTranStatVCSS01   
from util.data_52_oupu import OupuTranGrph

def main_exec_vcss_11_grph_stat(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_11, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranVCSS_11c):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines VCSS over time [{figu_tran.stra}.{figu_tran.modl}]'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def proc_figu_ex10(figu_tran:FiguTranVCSS_11c, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        proc_figu_exec(figu_tran)
    def proc_figu_ex20(figu_tran:FiguTranVCSS_11c, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        proc_figu_exec(figu_tran)
    
    # https://gemini.google.com/app/f11fdb0ca1fc6a70 for a discussion about raw vs modl
    # https://chatgpt.com/c/696dd048-75ec-832d-92eb-2a1356f33040 for a vcss oriented chat
    def proc_plot_exec(plot_vcss_11_mixd:PlotTranVCSS_11_mixd_mono, figuTran:FiguTranVCSS_11c, statTran:StatTranVCSS_11, modl:str):
        plot_vcss_11_mixd.fram_dict = {}
        plot_vcss_11_mixd.fram_dict["resu_fram"] = statTran.stat_tran_mixd.resu_fram
        match modl: 
            case 'lme':
                plot_vcss_11_mixd.fram_dict["resu_plot"] = statTran.stat_tran_mixd.resu_plot_lme
            case 'raw':
                plot_vcss_11_mixd.fram_dict["resu_plot"] = statTran.stat_tran_mixd.resu_plot_raw
            case _:
                raise Exception()
        plot_vcss_11_mixd.figu = figuTran.fig
        plot_vcss_11_mixd.upda()
    def proc_plot_ex11(plot_vcss_11_mixd:PlotTranVCSS_11_mixd_mono, figuTran:FiguTranVCSS_11c, statTran:StatTranVCSS_11, modl:str):
        # plot_vcss_11_mixd.stra = 'a'
        plot_vcss_11_mixd.axis = figuTran.ax1
        plot_vcss_11_mixd.line_labl = "Mean, 95% CI"
        plot_vcss_11_mixd.ylab = "Score"
        plot_vcss_11_mixd.titl = f"VCSS Score over time [{plot_vcss_11_mixd.stra}]"
        proc_plot_exec(plot_vcss_11_mixd, figuTran, statTran, modl)
    def proc_plot_ex12(plot_vcss_11_mixd:PlotTranVCSS_11_mixd_mono, figuTran:FiguTranVCSS_11c, statTran:StatTranVCSS_11, modl:str):
        # plot_vcss_11_mixd.stra = 'b'
        plot_vcss_11_mixd.axis = figuTran.ax2
        plot_vcss_11_mixd.line_labl = "Mean, 95% CI"
        plot_vcss_11_mixd.ylab = "Score"
        plot_vcss_11_mixd.titl = f"VCSS Score over time [{plot_vcss_11_mixd.stra}]"
        proc_plot_exec(plot_vcss_11_mixd, figuTran, statTran, modl)
    def proc_plot_ex21(plot_vcss_11_mixd:PlotTranVCSS_11_mixd_mono, figuTran:FiguTranVCSS_11c, statTran:StatTranVCSS_11, modl:str):
        # plot_vcss_11_mixd.stra = 'c'
        plot_vcss_11_mixd.axis = figuTran.ax1
        plot_vcss_11_mixd.line_labl = "Δ from Baseline"
        plot_vcss_11_mixd.ylab = "Score Δ"
        plot_vcss_11_mixd.titl = f"VCSS Score Δ from baseline over time [{plot_vcss_11_mixd.stra}]"
        proc_plot_exec(plot_vcss_11_mixd, figuTran, statTran, modl)
    def proc_plot_ex22(plot_vcss_11_mixd:PlotTranVCSS_11_mixd_mono, figuTran:FiguTranVCSS_11c, statTran:StatTranVCSS_11, modl:str):
        # plot_vcss_11_mixd.stra = 'd'
        plot_vcss_11_mixd.axis = figuTran.ax2
        plot_vcss_11_mixd.line_labl = "Mean"
        plot_vcss_11_mixd.ylab = "Score"
        plot_vcss_11_mixd.titl = f"VCSS Individual and Mean scores over time [{plot_vcss_11_mixd.stra}]"
        proc_plot_exec(plot_vcss_11_mixd, figuTran, statTran, modl)

    # Grph.Oupu
    # ----
    def proc_grph_ou10(figuTran:FiguTranVCSS_11c, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[f"{FiguTranVCSS_11c.__name__} [{figuTran.stra}.{figuTran.modl}]"] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}
        pass
    def proc_grph_ou20(figuTran:FiguTranVCSS_11c, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[f"{FiguTranVCSS_11c.__name__} [{figuTran.stra}.{figuTran.modl}]"] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}
        pass

    # Grph
    # ----
    def plot(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_11, oupuTran:OupuTranGrph, modl:str):
        ex10 = True
        if ex10:
            stra = 'ex10'
            figuTran = FiguTranVCSS_11c(procTran)
            proc_figu_ex10(figuTran, stra, modl)
            #
            plotTran = PlotTranVCSS_11_mixd_mono_a(procTran)
            proc_plot_ex11(plotTran, figuTran, statTran, modl)
            plotTran = PlotTranVCSS_11_mixd_mono_b(procTran)
            proc_plot_ex12(plotTran, figuTran, statTran, modl)
            #
            proc_grph_ou10(figuTran, oupuTran)
        #
        ex20 = True
        if ex20:
            stra = 'ex20'
            figuTran = FiguTranVCSS_11c(procTran)
            proc_figu_ex20(figuTran, stra, modl)
            #
            plotTran = PlotTranVCSS_11_mixd_mono_c(procTran)
            proc_plot_ex21(plotTran, figuTran, statTran, modl)
            plotTran = PlotTranVCSS_11_mixd_mono_d(procTran)
            proc_plot_ex22(plotTran, figuTran, statTran, modl)
            #
            proc_grph_ou20(figuTran, oupuTran)

    # Exec
    # ----
    plot(procTran, statTran, oupuTran, modl='raw') # data raw
    # VCSS MODEL TO BE CHECKED : 'df_emm' is not correctly extracted by openai
    # also : this is a SECONDARY FIGURE
    # plot(procTran, statTran, oupuTran, modl='lme') # data linear mixed-effect
    
    # Exit
    # ----
    return None

if __name__ == "__main__":
    pass
