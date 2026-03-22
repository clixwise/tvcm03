import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exam.c02_exam_01_stat_ import StatTranEXAM_01
from exam.c02_exam_01_grph_figu_ import FiguTranEXAM_01a, FiguTranEXAM_01b
from exam.c02_exam_01_grph_plot_ import PlotTranEXAM_01_heat, PlotTranEXAM_01_hist, PlotTranEXAM_01_scat
from util.data_02_proc import ProcTranStatEXAM01   
from util.data_52_oupu import OupuTranGrph

def main_exec_exam_01_grph_heat(procTran:ProcTranStatEXAM01, statTran:StatTranEXAM_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranEXAM_01a):
        figu_tran.size = (12, 5)
        figu_tran.titl = f'CEAP Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_ceap_01_pure:PlotTranEXAM_01_heat, figuTran:FiguTranEXAM_01a, indx, tipo, df_time):
        plot_ceap_01_pure.fram = df_time
        plot_ceap_01_pure.figu = figuTran.fig
        plot_ceap_01_pure.axis = figuTran.axis_list[indx]
        plot_ceap_01_pure.stra = 'd'
        plot_ceap_01_pure.titl = f"Heatmap for '{tipo}'"
        plot_ceap_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranEXAM_01a, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranEXAM_01a.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Exec
    # ----
    figuTran = FiguTranEXAM_01a(procTran)
    proc_figu_exec(figuTran)
    #
    df_time_list = statTran.stat_tran_pure.resu_ceap_limb
    for indx, tipo in enumerate(df_time_list.index.levels[0]):
        df_time = df_time_list.loc[tipo]
        plotTranPure = PlotTranEXAM_01_heat(procTran)
        proc_plot_exec(plotTranPure, figuTran, indx, tipo, df_time)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

def main_exec_exam_01_grph_scat(procTran:ProcTranStatEXAM01, statTran:StatTranEXAM_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranEXAM_01b):
        figu_tran.size = (12, 5)
        figu_tran.titl = f'CEAP Supertitle'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_ceap_01_pure:PlotTranEXAM_01_scat, figuTran:FiguTranEXAM_01b, indx, tipo, df_time):
        plot_ceap_01_pure.fram = df_time
        plot_ceap_01_pure.figu = figuTran.fig
        plot_ceap_01_pure.axis = figuTran.axis_list[indx]
        plot_ceap_01_pure.stra = 'd'
        plot_ceap_01_pure.titl = f"Scatter plot for '{tipo}'"
        plot_ceap_01_pure.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranEXAM_01b, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranEXAM_01b.__name__] =  figuTran
        oupuTran.upda()
        oupuTran.fram_dict = {}

    # Exec
    # ----
    figuTran = FiguTranEXAM_01b(procTran)
    proc_figu_exec(figuTran)
    #
    df_time_list = statTran.stat_tran_pure.resu_ceap_limb
    for indx, tipo in enumerate(df_time_list.index.levels[0]):
        df_time = df_time_list.loc[tipo]
        plotTranPure = PlotTranEXAM_01_scat(procTran)
        proc_plot_exec(plotTranPure, figuTran, indx, tipo, df_time)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

# ----
# Main
# ----
def main_exec_exam_01_grph(procTran:ProcTranStatEXAM01, statTran:StatTranEXAM_01, oupuTran:OupuTranGrph):
    main_exec_exam_01_grph_heat(procTran, statTran, oupuTran)
    main_exec_exam_01_grph_scat(procTran, statTran, oupuTran)
    #main_exec_exam_01_grph_tacs(procTran, statTran, oupuTran)
    
if __name__ == "__main__":
    pass
