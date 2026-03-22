import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pati.c02_pati_01_stat_ import StatTranPATI_01
from pati.c02_pati_01_grph_figu_ import FiguTranPATI_01a
from pati.c02_pati_01_grph_plot_ import PlotTranPATI_01_hist
from util.data_02_proc import ProcTranStatPATI01   
from util.data_52_oupu import OupuTranGrph

def main_exec_pati_01_grph(procTran:ProcTranStatPATI01, statTran:StatTranPATI_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranPATI_01a):
        figu_tran.size = (6, 8)
        figu_tran.titl = f'PATI Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec_R(plot_vcss_01_pure:PlotTranPATI_01_hist, figuTran:FiguTranPATI_01a, statTran:StatTranPATI_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.colu = 'R'
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax1
        plot_vcss_01_pure.hist_colo = 'royalblue'
        plot_vcss_01_pure.titl = "PATI R"
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.parm()
        plot_vcss_01_pure.upda()

    def proc_plot_exec_L(plot_vcss_01_pure:PlotTranPATI_01_hist, figuTran:FiguTranPATI_01a, statTran:StatTranPATI_01):
        plot_vcss_01_pure.fram = statTran.stat_tran_pure.resu
        plot_vcss_01_pure.colu = 'L'
        plot_vcss_01_pure.figu = figuTran.fig
        plot_vcss_01_pure.axis = figuTran.ax2
        plot_vcss_01_pure.hist_colo = 'orange'
        plot_vcss_01_pure.titl = "PATI L"
        plot_vcss_01_pure.stra = 'a'
        plot_vcss_01_pure.parm()
        plot_vcss_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranPATI_01a, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranPATI_01a.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Data
    # ----
    figuTran = FiguTranPATI_01a(procTran)
    plotTranPure_R = PlotTranPATI_01_hist(procTran)    #
    plotTranPure_L = PlotTranPATI_01_hist(procTran)

    # Grph
    # ----
    proc_figu_exec(figuTran)
    proc_plot_exec_R(plotTranPure_R, figuTran, statTran)
    proc_plot_exec_L(plotTranPure_L, figuTran, statTran)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

if __name__ == "__main__":
    pass
