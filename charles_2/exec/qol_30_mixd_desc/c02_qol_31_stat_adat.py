import sys  
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_21_stat import StatTran

class StatTranQOL_31_mixd():
    def __init__(self, stat_tran:StatTran):  # stat_tran:StatTranQOL_31):
        self.stat_tran = stat_tran
        #
        '''
        
        self.open_resu_plot = None
        self.open_resu_plot_raw = None
        self.open_resu_plot_lme = None
        
        self.clau_mixd_modl = None
        self.clau_mixd_resu = None
        self.clau_mixd_modl_mean = None
        self.clau_mixd_modl_delt = None
        self.clau_mixd_resi_desc = None
        self.clau_mixd_resi_stat = None
        
        

        
        self.gemi_mixd_mcid = None
        self.gemi_mixd_mcid_grop = None
        self.gemi_mixd_mcid_pat1 = None
        self.gemi_mixd_mcid_pat2 = None
        self.gemi_mixd_mcid_anal = None
        self.gemi_mixd_mcid_popu_delt = None
        self.gemi_mixd_mcid_effe_size = None
        '''
        
        self.mixd_assu_rand = None
        self.mixd_assu_resi = None
        self.mixd_assu_merg = None
        #
        self.mixd_assu_rand_plot = None
        self.mixd_assu_resi_plot = None
        self.mixd_assu_resi_resu = None
        
        self.mixd_mean_raww = None
        self.mixd_mean_modl = None
        self.mixd_mean_merg = None
        self.mixd_emms_modl = None
        self.mixd_pair_modl_1 = None
        self.mixd_pair_modl_2 = None
        
        self.mixd_mist_modl_cont = None
        self.mixd_mist_modl_resi_desc = None
        self.mixd_mist_modl_resi_stat = None
