import os
import sys
import pandas as pd
from pathlib import Path
from util.data_01_orch import OrchTran
# ****
# Clas
# ****
class ProcTran:
    def __init__(self, name, orch_tran:OrchTran):
        self.orch_tran = orch_tran
        orch_tran.dict[name] = self
        self.dict = {}
        #
        self.titl = None
        self.inpu = None
        self.stat = None
        self.plot = None
        self.oupu = None

class ProcTranStat(ProcTran):
    def __init__(self, name, orch_tran:OrchTran):
        super().__init__(name, orch_tran)
        
class ProcTranStatInpu01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatInpu01.__name__, orch_tran)
class ProcTranStatOupu01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatOupu01.__name__, orch_tran)
        
class ProcTranStatPATI01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatPATI01.__name__, orch_tran)
class ProcTranAssuQOL01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranAssuQOL01.__name__, orch_tran)
class ProcTranStatQOL01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL01.__name__, orch_tran)
class ProcTranStatSym01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatSym01.__name__, orch_tran)
class ProcTranStatVCSS01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatVCSS01.__name__, orch_tran)
class ProcTranStatEXAM01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatEXAM01.__name__, orch_tran)
        
class ProcTranStatQOL11(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL11.__name__, orch_tran)
class ProcTranStatVCSS11(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatVCSS11.__name__, orch_tran)

class ProcTranAssuQOL31(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranAssuQOL31.__name__, orch_tran)      
class ProcTranStatQOL31(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL31.__name__, orch_tran)

class ProcTranStatQOL81(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL81.__name__, orch_tran)
        
class ProcTranAssuQOL71(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranAssuQOL71.__name__, orch_tran)      
class ProcTranStatQOL71(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL71.__name__, orch_tran)
        
class ProcTranStatQOL61(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatQOL61.__name__, orch_tran)
        
class ProcTranStatCEAP01(ProcTranStat):
    def __init__(self, orch_tran):
        super().__init__(ProcTranStatCEAP01.__name__, orch_tran)