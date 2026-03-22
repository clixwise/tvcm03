import sys
import os
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from b01_inpu_file_01_main import main_exec_inpu_01
from b02_oupu_stat_01_main import OupuTranFile, main_exec_oupu_stat_01
from b02_oupu_grph_01_main import main_exec_oupu_grph_01

from exam.c02_exam_00_stat_main import main_exec_exam_01_stat
from exam.c02_exam_00_grph_main import main_exec_exam_01_grph
from exam.c02_exam_01_stat_ import StatTranEXAM_01

from qol.c02_qol_00_assu_main import ProcTranAssuQOL01, main_exec_qol_01_assu
from qol.c02_qol_00_stat_main import main_exec_qol_01_stat
from qol.c02_qol_00_grph_main import main_exec_qol_01_grph_assu, main_exec_qol_01_grph_stat
from qol.c02_qol_01_assu_ import AssuTranQOL_01
from qol.c02_qol_01_stat_ import StatTranQOL_01

from vcss.c02_vcss_00_stat_main import main_exec_vcss_01_stat
from vcss.c02_vcss_00_grph_main import main_exec_vcss_01_grph
from vcss.c02_vcss_01_stat_ import StatTranVCSS_01

from pati.c02_pati_00_stat_main import main_exec_pati_01_stat
from pati.c02_pati_00_grph_main import main_exec_pati_01_grph
from pati.c02_pati_01_stat_ import StatTranPATI_01

from qol_10_mixd.c02_qol_10_stat_main import main_exec_qol_11_stat
from qol_10_mixd.c02_qol_10_grph_main import main_exec_qol_11_grph_stat
from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11

from qol_30_mixd_desc.c02_qol_30_stat_main import main_exec_qol_31_stat
from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31
from qol_30_mixd_desc.c02_qol_30_grph_main import main_exec_qol_31_grph_assu, main_exec_qol_30_grph_stat

from qol_70_mixd_mcid.c02_qol_71_assu_ import AssuTranQOL_71
from qol_70_mixd_mcid.c02_qol_70_stat_main import main_exec_qol_71_stat
from qol_70_mixd_mcid.c02_qol_71_stat_ import StatTranQOL_71
from qol_70_mixd_mcid.c02_qol_70_grph_main import main_exec_qol_71_grph_assu, main_exec_qol_71_grph_stat
from qol_70_mixd_mcid.c02_qol_70_assu_main import main_exec_qol_71_assu

from qol_80_mixd_sens.c02_qol_80_stat_main import ProcTranStatQOL81, main_exec_qol_81_stat
from qol_80_mixd_sens.c02_qol_81_stat_ import StatTranQOL_81

from qol_60_mixd_effe.c02_qol_61_stat_ import StatTranQOL_61
from qol_60_mixd_effe.c02_qol_60_stat_main import main_exec_qol_61_stat
from qol_60_mixd_effe.c02_qol_60_grph_main import main_exec_qol_61_grph_stat

from vcss_10_mixd.c02_vcss_10_stat_main import main_exec_vcss_11_stat
from vcss_10_mixd.c02_vcss_10_grph_main import main_exec_vcss_11_grph_stat
from vcss_10_mixd.c02_vcss_11_stat_ import StatTranVCSS_11

from ceap.c02_ceap_00_stat_main import main_exec_ceap_01_stat
from ceap.c02_ceap_01_stat_ import StatTranCEAP_01

from util.data_01_orch import OrchTranStat
from util.data_02_proc import ProcTranAssuQOL31, ProcTranAssuQOL71, ProcTranStatCEAP01, ProcTranStatEXAM01, ProcTranStatInpu01, ProcTranStatOupu01, ProcTranStatPATI01, ProcTranStatQOL01, ProcTranStatQOL71, ProcTranStatQOL61, ProcTranStatVCSS01
from util.data_02_proc import ProcTranStatQOL11, ProcTranStatQOL31, ProcTranStatVCSS11

# Step 1: https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
# Step 2: https://chatgpt.com/c/69667f25-5780-8330-88c0-fdb7649b4a96
# Step 3: https://chatgpt.com/c/696dd048-75ec-832d-92eb-2a1356f33040 VCSS mimics QOL
# 2026-01-14 https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c
# 2026-01-25 https://chat.mistral.ai/chat/8a997b72-eed0-4629-b55e-9d3aa089ae45
# Step 5 : https://copilot.microsoft.com/shares/Q5DZZFvNPuaU7NF4xrwn6
# Publ : https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4

# Philosophy : 2026-0-28 : https://chatgpt.com/c/6979a38f-fafc-8333-9058-0791dd18cd51

'''
OrchTranStat [1-n] ProcTran <1-1> ProcTranStatQOL01, Pati01, etc.

ProcTran <1-1> ProcTranStatFull01 [1-1] InpuTran <1-1> InpuTran

ProcTran <1-1> ProcTranStatQOL01  [1-1] StatTran <1-1> StatTranQOL_01
ProcTran <1-1> ProcTranStatQOL01  [1-1] InpuTran <1-1> FramTran
'''
'''
TODO : Filter 'pati_isok' = True
TODO :C:\tate01\grph01\gr05\keep_v63_qol\exec_qol_13_publ_01_result_intro_2025_12_14_plot_01
'''
   
def main_exec():

    # Orch
    # ----
    orchTran = OrchTranStat()

    # Inpu
    # ----
    procTran = ProcTranStatInpu01(orchTran)
    inpuTran = main_exec_inpu_01(procTran)
    procTran = ProcTranStatOupu01(orchTran)
    oupuTranFile = main_exec_oupu_stat_01(procTran)
    oupuTranGrph = main_exec_oupu_grph_01(procTran)
    
    # Exec
    # ----
    execFull = True
    
    # Exec : Patient+Examen
    # ----
    def exec_pati_exam(orchTran, inpuTran, oupuTranFile):
        
        patiExec = execFull
        if patiExec:
            procTran = ProcTranStatPATI01(orchTran)
            statTran:StatTranPATI_01 = main_exec_pati_01_stat(procTran, inpuTran, oupuTranFile)
            # Stub that leads to generic 'hist' ; not used
            # main_exec_pati_01_grph(procTran, statTran, oupuTranGrph)
        #
        examExec = execFull
        if examExec:
            procTran = ProcTranStatEXAM01(orchTran)
            statTran:StatTranEXAM_01 = main_exec_exam_01_stat(procTran, inpuTran, oupuTranFile)
            main_exec_exam_01_grph(procTran, statTran, oupuTranGrph)        

    # Exec 1 : QOL
    # ----
    def exec_qol(orchTran, inpuTran, oupuTranFile):
        
        qolExec = execFull
        if qolExec:
            qolAssuExec = qolExec
            if qolAssuExec:
                procTran = ProcTranAssuQOL01(orchTran)
                assuTran:AssuTranQOL_01 = main_exec_qol_01_assu(procTran, inpuTran, oupuTranFile)
                main_exec_qol_01_grph_assu(procTran, assuTran, oupuTranGrph)
            qolStatExec = qolExec
            if qolStatExec:
                procTran = ProcTranStatQOL01(orchTran)
                statTran:StatTranQOL_01 = main_exec_qol_01_stat(procTran, inpuTran, oupuTranFile)
                main_exec_qol_01_grph_stat(procTran, statTran, oupuTranGrph)
        
        # mixd_desc
        # ---------
        qolExec = execFull
        if qolExec:
            procTran = ProcTranStatQOL31(orchTran)
            statTran:StatTranQOL_31 = main_exec_qol_31_stat(procTran, inpuTran, oupuTranFile)
            main_exec_qol_30_grph_stat(procTran, statTran, oupuTranGrph)
        
        # mixd_effe
        # ---------
        qolExec = execFull
        if qolExec:
            qolStatExec = qolExec
            if qolStatExec:
                procTran = ProcTranStatQOL61(orchTran)
                statTran:StatTranQOL_61 = main_exec_qol_61_stat(procTran, inpuTran, oupuTranFile)
                main_exec_qol_61_grph_stat(procTran, statTran, oupuTranGrph)
        
        # mixd_mcid
        # ---------
        qolExec = execFull
        if qolExec:
            qolAssuExec = execFull
            if qolAssuExec:
                procTran = ProcTranAssuQOL71(orchTran)
                assuTran:AssuTranQOL_71 = main_exec_qol_71_assu(procTran, inpuTran, oupuTranFile)
                main_exec_qol_71_grph_assu(procTran, assuTran, oupuTranGrph)
            qolStatExec = execFull
            if qolStatExec:
                procTran = ProcTranStatQOL71(orchTran)
                statTran:StatTranQOL_71 = main_exec_qol_71_stat(procTran, inpuTran, oupuTranFile)
                main_exec_qol_71_grph_stat(procTran, statTran, oupuTranGrph)
        
        # sensitivity
        # -----------
        qolStatExec = False # TODO
        if qolStatExec:
            procTran = ProcTranStatQOL81(orchTran)
            statTran:StatTranQOL_81 = main_exec_qol_81_stat(procTran, inpuTran, oupuTranFile)
            main_exec_qol_81_stat(procTran, statTran, oupuTranGrph)
            
        # $$$$
        # This is outdated ; it has been moved from '11' to '31'
        # Still to do : récupérer 'RESI' et 'RAND'
        # $$$$
        qolExec = False # outdated
        if qolExec:
            procTran = ProcTranStatQOL11(orchTran)
            statTran:StatTranQOL_11 = main_exec_qol_11_stat(procTran, inpuTran, oupuTranFile)
            main_exec_qol_11_grph_stat(procTran, statTran, oupuTranGrph)

    # Exec : VCSS
    # ----
    def exec_vcss(orchTran, inpuTran, oupuTranFile):
        
        vcssExec = execFull
        if vcssExec:
            procTran = ProcTranStatVCSS01(orchTran)
            statTran:StatTranVCSS_01 = main_exec_vcss_01_stat(procTran, inpuTran, oupuTranFile)
            main_exec_vcss_01_grph(procTran, statTran, oupuTranGrph)
            #        
            procTran = ProcTranStatVCSS11(orchTran)
            statTran:StatTranVCSS_11 = main_exec_vcss_11_stat(procTran, inpuTran, oupuTranFile)
            main_exec_vcss_11_grph_stat(procTran, statTran, oupuTranGrph)

    # Exec : CEAP
    # ----
    def exec_ceap(orchTran, inpuTran, oupuTranFile):
        
        ceapExec = execFull
        if ceapExec:
            procTran = ProcTranStatCEAP01(orchTran)
            statTran:StatTranCEAP_01 = main_exec_ceap_01_stat(procTran, inpuTran, oupuTranFile)      

    # Orch.Main
    # ---- 
    def orch_exec(orchTran:OrchTranStat):
        
        if True: 
            exec_pati_exam(orchTran, inpuTran, oupuTranFile)
        if True:  
            exec_qol(orchTran, inpuTran, oupuTranFile)
        if True:  
            exec_vcss(orchTran, inpuTran, oupuTranFile)
        if True:  
            exec_ceap(orchTran, inpuTran, oupuTranFile)
        #
        orchTran.upda()
        
    # Orch.Oupu
    # ----
    def orch_oupu(orchTran:OrchTranStat, oupuTran:OupuTranFile):
        mani = {
            "df_ta00_base_char": (orchTran.df_ta00_base_char, 'md'),
            "df_ta01_scre_incl": (orchTran.df_ta01_scre_incl, 'md'),
            "df_ta01_base_char": (orchTran.df_ta01_base_char, 'md'),
            "df_ta02_base_char": (orchTran.df_ta02_base_char, 'md'),
            "df_ta04_ther_adju": (orchTran.df_ta04_ther_adju, 'md'),
            "df_ta03_endp_prim_raww": (orchTran.df_ta03_endp_prim_raww, 'md'),
            "df_ta04_endp_prim_modl": (orchTran.df_ta04_endp_prim_modl, 'md'),
            "df_ta05_endp_prim_modl": (orchTran.df_ta05_endp_prim_modl, 'md'),
            "df_ta06_endp_effe_size": (orchTran.df_ta06_endp_effe_size, 'md'),
            "df_ta07_mcid"          : (orchTran.df_ta07_mcid, 'md'),
        }
        oupuTran.fram_dict[OrchTranStat.__name__] = {
            key: {'df': df, 'mode': mode}
            for key, (df, mode) in mani.items()
            if df is not None
        }
        oupuTran.upda()
        oupuTran.fram_dict = {}
    
    # Exec
    # ----
    orch_exec(orchTran)
    orch_oupu(orchTran, oupuTranFile)
    pass
      
if __name__ == "__main__":
    main_exec()
