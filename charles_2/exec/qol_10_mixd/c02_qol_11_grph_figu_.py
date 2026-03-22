import pandas as pd
import sys  
import os 
import pandas as pd
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#    from figu_grph import FiguTran  # ONLY for type checkers / IDEs
#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_31_figu import FiguTran2x1, FiguTran3x1, FiguTran2x2

# https://gemini.google.com/app/2bf3d6757a7f52d0 [the Scenario approach]

class FiguTranQOL_11_2x1(FiguTran2x1):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        super().upda()
        
class FiguTranQOL_11_3x1(FiguTran3x1):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        super().upda() 
        
class FiguTranQOL_11_2x2(FiguTran2x2):
    
    def __init__(self):
        super().__init__()
        
    def upda(self):
        super().upda()
'''   
class FiguTranQOL_11a(FiguTranQOL_11_3x1):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11a.__name__] = self
        #
        self.name = FiguTranQOL_11a.__name__
        self.stra = None

    def upda(self):
        super().upda()
    
class FiguTranQOL_11b(FiguTranQOL_11_3x1):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11b.__name__] = self
        #
        self.name = FiguTranQOL_11b.__name__
        self.stra = None

    def upda(self):
        super().upda()
'''        
class FiguTranQOL_11c_mixd_mono(FiguTranQOL_11_2x1):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11c_mixd_mono.__name__] = self
        #
        self.name = FiguTranQOL_11c_mixd_mono.__name__
        self.stra = None
        self.modl = None

    def upda(self):
        super().upda()
        
class FiguTranQOL_11c_mixd_dual(FiguTranQOL_11_2x1):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11c_mixd_dual.__name__] = self
        #
        self.name = FiguTranQOL_11c_mixd_dual.__name__
        self.stra = None

    def upda(self):
        super().upda()
        
class FiguTranQOL_11d_mixd_resi(FiguTranQOL_11_2x2):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11d_mixd_resi.__name__] = self
        #
        self.name = FiguTranQOL_11d_mixd_resi.__name__

    def upda(self):
        super().upda()

class FiguTranQOL_11d_mixd_rand(FiguTranQOL_11_2x2):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11d_mixd_rand.__name__] = self
        #
        self.name = FiguTranQOL_11d_mixd_rand.__name__

    def upda(self):
        super().upda()
        
class FiguTranQOL_11e_mixd_fore(FiguTranQOL_11_2x1):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11e_mixd_fore.__name__] = self
        #
        self.name = FiguTranQOL_11e_mixd_fore.__name__
        self.stra = None

    def upda(self):
        super().upda()
       
class FiguTranQOL_11f_mixd_mcid(FiguTranQOL_11_2x2):
    
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[FiguTranQOL_11f_mixd_mcid.__name__] = self
        #
        self.name = FiguTranQOL_11f_mixd_mcid.__name__
        self.stra = None

    def upda(self):
        super().upda()