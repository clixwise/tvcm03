import os
import sys
import pandas as pd
from pathlib import Path
from util.data_02_proc import ProcTran

# ****
# Clas
# ****
        
class StatTran:
    def __init__(self, name, proc_tran:ProcTran):
        self.proc_tran = proc_tran
        proc_tran.dict[name] = self
        self.dict = {}
        #
        self.stra = None
        self.fram = None
        self.frax = None

    def upda(self):
        raise NotImplementedError