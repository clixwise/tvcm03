import pandas as pd
import sys  
import os 
import pandas as pd
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#    from figu_grph import FiguTran  # ONLY for type checkers / IDEs
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_31_figu import FiguTran2x1

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class FiguTranPATI_01(FiguTran2x1):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        super().upda()
        
class FiguTranPATI_01a(FiguTranPATI_01):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranPATI_01a.__name__] = self
        #
        self.name = FiguTranPATI_01a.__name__
        self.stra = None

    def upda(self):
        super().upda()