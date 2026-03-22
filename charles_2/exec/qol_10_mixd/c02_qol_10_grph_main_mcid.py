import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11
from qol_10_mixd.c02_qol_11_grph_figu_ import FiguTranQOL_11f_mixd_mcid
from qol_10_mixd.c02_qol_11_grph_plot_ import PlotTranQOL_11_mixd_mcid
from util.data_02_proc import ProcTranStatQOL11   
from util.data_52_oupu import OupuTranGrph

def mcid_exec_qol_11_grph_stat(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):

    # ---- 
    # Figu
    # ----
    def mcid_figu_exec(figu_tran:FiguTranQOL_11f_mixd_mcid):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}]\nObserved means & Model estimates'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def mcid_figu_ex10(figu_tran:FiguTranQOL_11f_mixd_mcid, stra):
        figu_tran.stra = stra
        mcid_figu_exec(figu_tran)    
    # ----
    # Plot  https://gemini.google.com/app/0f865d976b405499 for a discussion about MCID
    # ----
    def mcid_plot_exec(plot_qol_11_mixd:PlotTranQOL_11_mixd_mcid, figuTran:FiguTranQOL_11f_mixd_mcid, statTran:StatTranQOL_11):
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def mcid_plot_ex11(plot_qol_11_mixd:PlotTranQOL_11_mixd_mcid, figuTran:FiguTranQOL_11f_mixd_mcid, statTran:StatTranQOL_11):
        '''
        self.gemi_mixd_mcid = None
        self.gemi_mixd_mcid_grop = None
        self.gemi_mixd_mcid_pat1 = None
        self.gemi_mixd_mcid_pat2 = None
        self.gemi_mixd_mcid_anal = None
        self.gemi_mixd_mcid_popu_delt = None
        self.gemi_mixd_mcid_effe_size = None
        '''
        plot_qol_11_mixd.fram = statTran.stat_tran_mixd.gemi_mixd_mcid_grop
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax01
        plot_qol_11_mixd.titl = f"VEINES-QOL {plot_qol_11_mixd.stra}"
        mcid_plot_exec(plot_qol_11_mixd, figuTran, statTran)
    
    # ----
    # Grph
    # ----
    def mcid_grph_exec(procTran:ProcTranStatQOL11, statTran:StatTranQOL_11, oupuTran:OupuTranGrph):
        
        # Oupu
        # ----
        def mcid_grph_oupu(figuTran:FiguTranQOL_11f_mixd_mcid, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_11f_mixd_mcid.__name__} [{figuTran.stra}]"] =  figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = 'ex10'
            figuTran = FiguTranQOL_11f_mixd_mcid(procTran)
            mcid_figu_ex10(figuTran, stra)
            #
            plotTran = PlotTranQOL_11_mixd_mcid(procTran)
            mcid_plot_ex11(plotTran, figuTran, statTran)
            #
            mcid_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    mcid_grph_exec(procTran, statTran, oupuTran) # data linear mixed-effect
    
    # Exit
    # ----
    return None