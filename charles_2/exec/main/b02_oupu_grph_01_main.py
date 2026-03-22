import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_02_proc import ProcTranStatOupu01
from util.data_52_oupu import OupuTranGrph, oupu_grph_exec

def main_exec_oupu_grph_01(procTran:ProcTranStatOupu01) -> OupuTranGrph:

    # Inpu
    # ----
    def proc_exec(oupuTran:OupuTranGrph):
        
        # Exec
        # ----    
        scri_path = os.path.abspath(__file__)
        scri_dire = os.path.dirname(scri_path)
        pare_dire = os.path.dirname(scri_dire)
        #
        path_oupu = os.path.join(pare_dire, f"../resu_figu")
        path_oupu = os.path.normpath(path_oupu)
        oupuTran.dire = path_oupu
        oupuTran.func = oupu_grph_exec

    # Orch
    # ----
    oupuTran = OupuTranGrph(procTran)
    proc_exec(oupuTran)
    
    # Exit
    # ----
    return oupuTran


if __name__ == "__main__":
    
    # Exec
    # ----
    main_exec_oupu_grph_01()
    pass
