import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol.c02_qol_01_stat_ import StatTranQOL_01
from qol.c02_qol_01_assu_ import AssuTranQOL_01
from qol.c02_qol_01_grph_figu_ import FiguTranQOL_01a, FiguTranQOL_01b, FiguTranQOL_01c
from qol.c02_qol_01_grph_plot_ import PlotTranQOL_01_hist, PlotTranQOL_01_ququ
from qol.c02_qol_01_grph_plot_ import PlotTranQOL_01_mean
from util.data_02_proc import ProcTranStatQOL01   
from util.data_02_proc import ProcTranAssuQOL01  
from util.data_52_oupu import OupuTranGrph

def main_exec_qol_01_grph_assu_hist(procTran:ProcTranAssuQOL01, assuTran:AssuTranQOL_01, oupuTran:OupuTranGrph):
    
    # Grph.Exec
    # ----
    def proc_hist_figu_exec(figu_tran:FiguTranQOL_01a):
        figu_tran.size = (6, 8)
        figu_tran.titl = f'Veines QOL Histogram over time'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
        
    # Grph.Exec
    # ----   
    def proc_hist_plot_exec(plot_qol_01_hist:PlotTranQOL_01_hist, figuTran:FiguTranQOL_01a, indx, tipo, df_time):
        # print(f"Data for {tipo} has {len(df_time)} rows.")
        #  
        plot_qol_01_hist.fram = df_time
        plot_qol_01_hist.figu = figuTran.fig
        plot_qol_01_hist.axis = figuTran.axis_list[indx]
        plot_qol_01_hist.stra = 'a'
        plot_qol_01_hist.titl = f"Histogram for '{tipo}'"
        plot_qol_01_hist.parm()
        plot_qol_01_hist.upda()
    
    # Grph.Oupu
    # ----
    def proc_hist_grph_oupu(figuTran:FiguTranQOL_01a, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranQOL_01a.__name__] = figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}
        
    # Exec
    # ----
    figuTran = FiguTranQOL_01a(procTran)
    proc_hist_figu_exec(figuTran)
    #
    df_time_list = assuTran.assu_tran_hiqq.resu_plot
    for indx, tipo in enumerate(df_time_list.index.levels[0]):
        df_time = df_time_list.loc[tipo]
        plotTranQuqu = PlotTranQOL_01_hist(procTran)
        proc_hist_plot_exec(plotTranQuqu, figuTran, indx, tipo, df_time)
    proc_hist_grph_oupu(figuTran, oupuTran)
            
def main_exec_qol_01_grph_assu_ququ(procTran:ProcTranAssuQOL01, assuTran:AssuTranQOL_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_ququ_figu_exec(figu_tran:FiguTranQOL_01b):
        figu_tran.size = (6, 8)
        figu_tran.titl = f'Veines Q-Q Plot over time'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
         
    # Grph.Exec
    # ----   
    def proc_ququ_plot_exec(plot_qol_01_ququ:PlotTranQOL_01_ququ, figuTran:FiguTranQOL_01a, indx, tipo, df_time):
        # print(f"Data for {tipo} has {len(df_time)} rows.")
        #  
        plot_qol_01_ququ.fram = df_time
        plot_qol_01_ququ.figu = figuTran.fig
        plot_qol_01_ququ.axis = figuTran.axis_list[indx]
        plot_qol_01_ququ.stra = 'a'
        plot_qol_01_ququ.titl = f"Q-Q Plot for '{tipo}'"
        plot_qol_01_ququ.upda()
    
    # Grph.Oupu
    # ----
    def proc_ququ_grph_oupu(figuTran:FiguTranQOL_01b, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranQOL_01b.__name__] = figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}
            
    # Exec
    # ----
    figuTran = FiguTranQOL_01b(procTran)
    proc_ququ_figu_exec(figuTran)
    #
    df_time_list = assuTran.assu_tran_hiqq.resu_plot
    for indx, tipo in enumerate(df_time_list.index.levels[0]):
        df_time = df_time_list.loc[tipo]
        plotTranQuqu = PlotTranQOL_01_ququ(procTran)
        proc_ququ_plot_exec(plotTranQuqu, figuTran, indx, tipo, df_time)
    proc_ququ_grph_oupu(figuTran, oupuTran)

def main_exec_qol_01_grph_assu(procTran:ProcTranAssuQOL01, assuTran:AssuTranQOL_01, oupuTran:OupuTranGrph):
    
    # Exec
    # ----
    main_exec_qol_01_grph_assu_hist(procTran, assuTran, oupuTran)
    main_exec_qol_01_grph_assu_ququ(procTran, assuTran, oupuTran)
    
    # Exit
    # ----
    return None

def main_exec_qol_01_grph_stat(procTran:ProcTranStatQOL01, statTran:StatTranQOL_01, oupuTran:OupuTranGrph):
        
    # Grph.Exec
    # ----
    def proc_figu_exec(figu_tran:FiguTranQOL_01c):
        figu_tran.size = (6, 8)
        figu_tran.titl = f'Veines QOL over time'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    
    def proc_plot_exec(plot_qol_01_mean:PlotTranQOL_01_mean, figuTran:FiguTranQOL_01c, statTran:StatTranQOL_01):
        plot_qol_01_mean.fram = statTran.stat_tran_mean.resu_plot
        plot_qol_01_mean.figu = figuTran.fig
        plot_qol_01_mean.axis = figuTran.ax1
        plot_qol_01_mean.stra = 'a'
        plot_qol_01_mean.upda()

    # Grph.Oupu
    # ----
    def proc_grph_oupu(figuTran:FiguTranQOL_01c, oupuTran:OupuTranGrph):
        oupuTran.figu_dict[FiguTranQOL_01c.__name__] = figuTran
        oupuTran.upda()
        oupuTran.figu_dict = {}

    # Data
    # ----
    figuTran = FiguTranQOL_01c(procTran)
    plotTranMean = PlotTranQOL_01_mean(procTran)

    # Grph
    # ----
    proc_figu_exec(figuTran)
    proc_plot_exec(plotTranMean, figuTran, statTran)
    proc_grph_oupu(figuTran, oupuTran)
    
    # Exit
    # ----
    return None

if __name__ == "__main__":
    pass
