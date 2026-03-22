import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vcss.c02_vcss_01_stat_ import StatTranVCSS_01
from vcss.c02_vcss_01_grph_figu_ import FiguTranVCSS_01a, FiguTranVCSS_01b, FiguTranVCSS_01c, FiguTranVCSS_01d
from vcss.c02_vcss_01_grph_plot_ import PlotTranVCSS_01_boxx, PlotTranVCSS_01_hist
from vcss.c02_vcss_01_grph_plot_ import PlotTranVCSS_01_scat, PlotTranVCSS_01_tacs
from util.data_02_proc import ProcTranStatVCSS01   
from util.data_52_oupu import OupuTranGrph

# ----
# Hist
# ----
def main_exec_vcss_01_grph_hist(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranVCSS_01a):
        figu_tran.size = (6, 8)
        figu_tran.titl = f'VCSS Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec_R(plot_vcss_01_pure:PlotTranVCSS_01_hist, figuTran:FiguTranVCSS_01a, statTran:StatTranVCSS_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.colu = 'R'
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax1
        plot_vcss_01_pure.hist_colo = 'royalblue'
        plot_vcss_01_pure.titl = "VCSS R"
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.parm()
        plot_vcss_01_pure.upda()

    def proc_plot_exec_L(plot_vcss_01_pure:PlotTranVCSS_01_hist, figuTran:FiguTranVCSS_01a, statTran:StatTranVCSS_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.colu = 'L'
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax2
        plot_vcss_01_pure.hist_colo = 'orange'
        plot_vcss_01_pure.titl = "VCSS L"
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.parm()
        plot_vcss_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranVCSS_01a, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranVCSS_01a.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}

    # Data
    # ----
    figuTran = FiguTranVCSS_01a(procTran)
    plotTranPure_R = PlotTranVCSS_01_hist(procTran)    #
    plotTranPure_L = PlotTranVCSS_01_hist(procTran)

    # Grph
    # ----
    proc_figu_exec(figuTran)
    proc_plot_exec_R(plotTranPure_R, figuTran, statTran)
    proc_plot_exec_L(plotTranPure_L, figuTran, statTran)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

# ----
# Scat
# ----
def main_exec_vcss_01_grph_scat(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranVCSS_01b):
        figu_tran.size = (12, 5)
        figu_tran.titl = f'VCSS Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_vcss_01_pure:PlotTranVCSS_01_scat, figuTran:FiguTranVCSS_01b, indx, tipo, df_time):
        plot_vcss_01_pure.fram = df_time
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.axis_list[indx]
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.titl = f"Scatter plot for '{tipo}'"
        plot_vcss_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranVCSS_01b, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranVCSS_01b.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}

    # Exec
    # ----
    figuTran = FiguTranVCSS_01b(procTran)
    proc_figu_exec(figuTran)
    #
    df_time_list = statTran.stat_tran_pure.resu
    for indx, tipo in enumerate(df_time_list.index.levels[0]):
        df_time = df_time_list.loc[tipo]
        plotTranPure = PlotTranVCSS_01_scat(procTran)
        proc_plot_exec(plotTranPure, figuTran, indx, tipo, df_time)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None 

# ----
# Tacs
# ----
def main_exec_vcss_01_grph_tacs(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranVCSS_01c):
        figu_tran.size = (6,6)
        figu_tran.titl = f'VCSS Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_vcss_01_pure:PlotTranVCSS_01_tacs, figuTran:FiguTranVCSS_01c, statTran:StatTranVCSS_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax1
        plot_vcss_01_pure.stra = 'c'
        plot_vcss_01_pure.titl = f"Scatter plot"
        plot_vcss_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranVCSS_01c, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranVCSS_01c.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}
    
    # Data
    # ----
    figuTran = FiguTranVCSS_01c(procTran)
    plotTranPure = PlotTranVCSS_01_tacs(procTran) 

    # Grph
    # ----
    proc_figu_exec(figuTran)
    proc_plot_exec(plotTranPure, figuTran, statTran)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None        

# ----
# Boxx
# ----
def main_exec_vcss_01_grph_boxx(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranVCSS_01d):
        figu_tran.size = (14, 6)
        figu_tran.titl = f'VCSS Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_vcss_01_pure:PlotTranVCSS_01_boxx, figuTran:FiguTranVCSS_01d, statTran:StatTranVCSS_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.colu = 'R'
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax1
        plot_vcss_01_pure.hist_colo = 'royalblue'
        plot_vcss_01_pure.titl = "VCSS R"
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.parm()
        plot_vcss_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranVCSS_01d, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranVCSS_01d.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}

    # Data
    # ----
    figuTran = FiguTranVCSS_01d(procTran)
    plotTranPure = PlotTranVCSS_01_boxx(procTran) 

    # Grph
    # ----
    proc_figu_exec(figuTran)
    proc_plot_exec(plotTranPure, figuTran, statTran)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

# ----
# Main
# ----
def main_exec_vcss_01_grph(procTran:ProcTranStatVCSS01, statTran:StatTranVCSS_01, oupuTran:OupuTranGrph):
    main_exec_vcss_01_grph_scat(procTran, statTran, oupuTran)
    main_exec_vcss_01_grph_tacs(procTran, statTran, oupuTran)
    main_exec_vcss_01_grph_boxx(procTran, statTran, oupuTran)
    main_exec_vcss_01_grph_hist(procTran, statTran, oupuTran)

if __name__ == "__main__":
    pass
