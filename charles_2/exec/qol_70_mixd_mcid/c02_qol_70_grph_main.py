import sys
import os

# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qol_70_mixd_mcid.c02_qol_71_stat_ import StatTranQOL_71
from qol_70_mixd_mcid.c02_qol_71_assu_ import AssuTranQOL_71
from util.data_02_proc import ProcTranStatQOL71  

from util.data_52_oupu import OupuTranGrph

from qol_70_mixd_mcid.c02_qol_71_stat_ import StatTranQOL_71_mcid_copi
from qol_70_mixd_mcid.c02_qol_72_grph_figu_ import FiguTranQOL_71a_mcid
from qol_70_mixd_mcid.c02_qol_72_grph_plot_ import PlotTranQOL_72_mcid_anch, PlotTranQOL_72_mcid_roc
from util.data_02_proc import ProcTranStatQOL71   
from util.data_52_oupu import OupuTranGrph

def main_mcid_stat(procTran:ProcTranStatQOL71, statTran:StatTranQOL_71_mcid_copi, oupuTran:OupuTranGrph):
       
    # ---- 
    # Figu
    # ----
    def mono_figu_exec(figu_tran:FiguTranQOL_71a_mcid):
        figu_tran.size = (8, 8)
        figu_tran.titl = f'Veines QOL over time [{figu_tran.stra}.{figu_tran.modl}]'
        figu_tran.hspa = 0.4
        figu_tran.vspa = 0.2
        figu_tran.upda()
    def mono_figu_ex10(figu_tran:FiguTranQOL_71a_mcid, stra, modl):
        figu_tran.stra = stra
        figu_tran.modl = modl
        mono_figu_exec(figu_tran)
    
    # ----
    # Plot https://chatgpt.com/c/699b4784-2cc0-832b-959b-6da55f29dd7a
    # ----
    def mono_plot_ex11(plot_qol_11_mixd:PlotTranQOL_72_mcid_anch, figuTran:FiguTranQOL_71a_mcid, statTran:StatTranQOL_71_mcid_copi, modl:str):
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax1
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        #
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["resu_wide"] = statTran.resu_wide
        plot_qol_11_mixd.fram_dict["plot_anch"] = statTran.plot_anch
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    def mono_plot_ex12(plot_qol_11_mixd:PlotTranQOL_72_mcid_roc, figuTran:FiguTranQOL_71a_mcid, statTran:StatTranQOL_71_mcid_copi, modl:str):
        plot_qol_11_mixd.stra = 'a'
        plot_qol_11_mixd.axis = figuTran.ax2
        plot_qol_11_mixd.line_labl = "Mean, 95% CI"
        plot_qol_11_mixd.ylab = "Score"
        plot_qol_11_mixd.titl = "VEINES-QOL Score over time"
        #
        plot_qol_11_mixd.fram_dict = {}
        plot_qol_11_mixd.fram_dict["resu_wide"] = statTran.resu_wide
        plot_qol_11_mixd.fram_dict["plot_roc_data"] = statTran.plot_roc_data
        plot_qol_11_mixd.fram_dict["plot_roc_meta"] = statTran.plot_roc_meta
        plot_qol_11_mixd.figu = figuTran.fig
        plot_qol_11_mixd.upda()
    
    # ----
    # Grph
    # ----
    def mcid_stat_grph_exec(procTran:ProcTranStatQOL71, statTran:StatTranQOL_71_mcid_copi, oupuTran:OupuTranGrph, modl:str):
        
        # Oupu
        # ----
        def mono_grph_oupu(figuTran:FiguTranQOL_71a_mcid, oupuTran:OupuTranGrph):
            oupuTran.figu_dict[f"{FiguTranQOL_71a_mcid.__name__} [{figuTran.stra}]"] = figuTran
            oupuTran.upda()
            oupuTran.figu_dict = {}
            pass
        
        # Exec
        # ----
        ex10 = True
        if ex10:
            stra = f"{main_mcid_stat.__name__} ex10"
            figuTran = FiguTranQOL_71a_mcid(procTran)
            mono_figu_ex10(figuTran, stra, modl)
            #
            plotTran = PlotTranQOL_72_mcid_anch(procTran)
            mono_plot_ex11(plotTran, figuTran, statTran, modl)
            #
            plotTran = PlotTranQOL_72_mcid_roc(procTran)
            mono_plot_ex12(plotTran, figuTran, statTran, modl)
            #
            mono_grph_oupu(figuTran, oupuTran)

    # ----
    # Main
    # ----
    mcid_stat_grph_exec(procTran, statTran, oupuTran, modl='raw')
    
    # Exit
    # ----
    return None

def main_exec_qol_71_grph_stat(procTran:ProcTranStatQOL71, statTran:StatTranQOL_71, oupuTran:OupuTranGrph):
    
    # Data
    # ----
    main_mcid_stat(procTran, statTran.stat_tran_mcid_copi, oupuTran)
    
    # Exec
    # ----
    pass

def main_exec_qol_71_grph_assu(procTran:ProcTranStatQOL71, statTran:AssuTranQOL_71, oupuTran:OupuTranGrph):
    
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print ("TOOOOOOOOOOOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    '''
    LE BUT : POUR MCID, les distributions doivent etre normale
    EN PARTICLUIER lA DISTRIBUTION EN T0 [baseline_scores]
    # STEP 4 — Visual profile (3 panels)
    # ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("T0 Baseline VEINES-QOL — Cohort Profile (n=30)", fontweight="bold")

    # Panel A : distribution
    axes[0].hist(baseline_scores, bins=8, edgecolor="white", color="#4C72B0")
    axes[0].axvline(baseline_scores.mean(), color="red",    linestyle="--", label=f"Mean {baseline_scores.mean():.1f}")
    axes[0].axvline(baseline_scores.median(), color="orange", linestyle="--", label=f"Median {baseline_scores.median():.1f}")
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("VEINES-QOL score")
    axes[0].legend(fontsize=8)

    # Panel B : boxplot with individual points
    axes[1].boxplot(baseline_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.4))
    axes[1].scatter([1]*len(baseline_scores), baseline_scores, alpha=0.6, color="#4C72B0", zorder=3)
    axes[1].set_title("Spread & Outliers")
    axes[1].set_ylabel("VEINES-QOL score")
    axes[1].set_xticks([])

    # Panel C : Q-Q plot
    stats.probplot(baseline_scores, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot (Normality)")

    plt.tight_layout()
    plt.savefig("T0_profile.png", dpi=150)
    plt.show()
    '''