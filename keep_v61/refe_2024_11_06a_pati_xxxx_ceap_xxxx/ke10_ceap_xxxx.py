import os
import sys
import pandas as pd

from ke00_stat import stat_glob
from zz12_xxx_331_fish_mist import fish_mist
from zz12_xxx_321_kruswall_clau import kruswall_clau
from zz12_xxx_311_Cochran_Mantel_Haenszel import cochmanthaen
from zz12_xxx_301_ANOVA import anov
from zz12_xxx_251_catx_logi import catx_logi
from zz12_xxx_251_catx_prop import catx_prop
from zz12_xxx_251_catx_bino import catx_bino
from zz12_xxx_251_catx_fish_odds import catx_fish_odds
from zz12_xxx_231_goodkruslamb_tabl import goodkruslamb_tabl
from zz12_xxx_201_cram_clau import cram_clau
from zz12_xxx_201_cram_mist import cram_mist
from zz12_xxx_201_cram_perp import cram_perp
from zz12_xxx_120_muin import muin
from zz12_xxx_090_thei import thei
from zz12_xxx_080_spea import spea
from zz12_xxx_070_goodkrusgamm import goodkrusgamm
from zz12_xxx_060_somd import somd
from zz12_xxx_050_kendtaub import kendtaub
from zz12_xxx_040_cocharmi import cocharmi
from zz12_xxx_030_joncterp import joncterp
from zz12_xxx_024_mannwhit_clau import mannwhit_clau
from zz12_xxx_024_mannwhit_perp import mannwhit_perp
from zz12_xxx_022_stuamaxw_clau import stuamaxw_clau
from zz12_xxx_022_stuamaxw_perp import stuamaxw_perp
from zz12_xxx_020_kolmsmir import kolmsmir
from zz12_xxx_001b_dist_var3 import dist_var3
from zz12_xxx_001b_dist_var2 import dist_var2
from zz12_xxx_001b_dist_var1 import dist_var1
from zz12_xxx_001a_dist_mean import dist_mean
from zz11_xxx_resi import resi
from zz11_xxx_chi2 import chi2

from util_file_inpu_mbre import inp1, inp2, inp3, inp5
from util_file_mngr import set_file_objc, write
from datetime import datetime

'''
pati : 362
mbre : 724 dont C0...C6:619 ; NA:105
ceap : 876 dont C0...C6:771 ; NA:105
ceap_pair : 524
'''
# ----
# Inpu
# ----
def inpu(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path):      
        # Step 21
        df1, df2, df3 = inp1(file_path, filt_name, filt_valu)  
        if False:
            colu_name = 'ceap'

            colu_cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']      
            # ...
            
            colu_cate_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
            # ...
                
            colu_cate_list = ['C3', 'C4', 'C5', 'C6']
        # !!!   
        # df3 = we only retain C3_C6 ; we do the 'age_bin' analysis based on CEAP io PATI
        # !!!
        colu_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
        colu_name = 'age_bin'
        #    
        df_tabl = inp2(df3, indx_cate_list, colu_cate_list, indx_name, colu_name, ceap_mono, filt_name, filt_valu)
        
        indx_name_stra = f'{indx_name}_stra'
        colu_name_ordi = 'age_bin_ordi'
        df_line = inp3(df3, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi)  
    
        df_norm = inp5(df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) 
        
        # Exit
        return df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi
        
def ke10_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path):

    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
        
        # Inpu
        df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
        inpu(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path)
        trac = True
        if trac:
            print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
            write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
            dfT = pd.DataFrame({ indx_name: [df_tabl.loc[indx_cate_list[0]].sum(), df_tabl.loc[indx_cate_list[1]].sum(), df_tabl.loc[indx_cate_list[0]].sum()+df_tabl.loc[indx_cate_list[1]].sum()]}, index=[indx_cate_list[0], indx_cate_list[1], 'T'])
            print(f"\nContingency table  : totals:{dfT.T}")
            write(f"\nContingency table  : totals:{dfT.T}")
            print(f"\nContingency table normalized : df_norm.size:{len(df_norm)} df_norm.type:{type(df_norm)}\n{df_norm}\n:{df_norm.index}\n:{df_norm.columns}")
            write(f"\nContingency table normalized : df_norm.size:{len(df_norm)} df_norm.type:{type(df_norm)}\n{df_norm}\n:{df_norm.index}\n:{df_norm.columns}")
        # dg1 = dg1.astype(int)
        # print(dg1.dtypes)
        # Stat
        what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}"
        yate = False
        stat_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm, yate)
        pass