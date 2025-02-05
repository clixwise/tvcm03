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

def stat_glob_perpOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    yate = True
    if not yate:
        df_tabl = dg1
    else:
        df_tabl = dg1.where(dg1 != 0, dg1 + 0.5) # yates correction
    
    # if True:
    if len(indx_cate_list) != 2:
        
        chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
 
    else:
        
        dist_mean(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        dist_var1(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        dist_var2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        dist_var3(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)       
        '''
        1. Tests primarily for Independence/Association (A):
        - Chi-2
        - Mutual Information
        - Cramer V
        - Goodman and Kruskal's Lambda
        - Goodman and Kruskal's Gamma
        - Polychoric correlation [NEW]
        '''
        chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        muin(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        cram_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        cram_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        cram_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        goodkruslamb_tabl(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        goodkrusgamm(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)      
        '''
        2. Tests for Distribution Comparison (B):
        - Kolmogorov-Smirnov
        - Theil U
        - Kruskal-Wallis [TODO]
        '''
        kolmsmir(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        thei(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        '''
        3. Tests for Location/Central Tendency (C):
        - Mann-Whitney U
        - ANOVA
        - Wilcoxon signed-rank [for Larray, Rarray]
        - Friedman test [only for 3 groups of line-line patients comparisons]
        '''
        mannwhit_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        mannwhit_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        anov(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        '''
        4. Tests for Trend/Progression (D):
        - Jonckheere-Terpstra
        - Cochran-Armitage
        - Kendall Tau [2 variables MUST BE Ordinal]
        - Spearman's Rank [2 variables MUST BE Ordinal]
        - Page's trend test [TODO]
        - Ordinal logistic regression [TODO]
        '''
        joncterp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        cocharmi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        '''
        5. Tests for Agreement/Concordance (E):
        - Cohen's kappa [TODO]
        - Somer's D
        - McNemar's test [TODO]
        - Cochran Mantel Haenszel
        '''
        somd(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        cochmanthaen(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        '''
        6. Tests for Symmetry/Marginal Homogeneity (F):
        - Stuart-Maxwell
        '''
        stuamaxw_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        stuamaxw_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)

        #
        catx_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_fish_odds(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_logi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
       
    pass

def stat_glob_clauOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    # Yates correction
    yate = False
    if not yate:
        df_tabl = dg1
    else:
        df_tabl = dg1.where(dg1 != 0, dg1 + 0.5) # yates correction    
     
    '''
    0. Descriptive
    '''
    dist_mean(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    dist_var1(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    dist_var2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    dist_var3(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)   
    
    '''
    1. Tests for Independence/Association (A):
    Check for independence, association between gender and age group
    Input : contingency table
    * Chi-Square
        - H0: The gender and age group variables are independent.
        - Ha: The gender and age group variables are not independent, i.e., they are associated.
    * Cramer's V [DONE]
        - H0: There is no association between the gender and age group variables.
        - Ha: There is an association between the gender and age group variables.
    * Goodman and Kruskal's Lambda
        - H0: The gender variable does not provide any information about the age group variable, or vice versa.
        - Ha: The gender variable provides information about the age group variable, or vice versa.
    '''
    chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    goodkruslamb_tabl(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    '''
    2. Tests for Distribution Comparison (B):
    Compare the age group distributions between genders
    Input : individual patient rows
        - male_ages = df[df['Gender'] == 'Male']['Age_Ordinal']
        - female_ages = df[df['Gender'] == 'Female']['Age_Ordinal']
    * Kruskal-Wallis [DONE]
        - kruskal_stat, kruskal_p = kruskal_wallis(male_ages, female_ages)
        - H0: The age group distributions are the same across genders.
        - Ha: The age group distributions differ across genders.
    * Kolmogorov-Smirnov (perp) [DONE]
        Can be added since it compares two distributions with respect to age_bin
    '''
    kruswall_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # TODO
    kolmsmir(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    '''
    3. Tests for Distribution Location/Central Tendency (C):
    * Mann-Whitney U [DONE]
        - u_stat, mw_p = mannwhitneyu(male_ages, female_ages, alternative='two-sided')
        - H0: The age group distributions have the same central tendency (median) across genders.
        - Ha: The age group distributions have different central tendencies (medians) across genders.
    '''
    mannwhit_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    mannwhit_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    '''
    6. Tests for Symmetry/Marginal Homogeneity (F):
    * Stuart-Maxwell [DONE]
        - H0: The marginal probabilities (row and column totals) of the gender-age group contingency table are equal.
        - Ha: The marginal probabilities (row and column totals) of the gender-age group contingency table are not equal.
    '''
    stuamaxw_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    stuamaxw_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    
    '''
    7. Stratification
    '''
    catx_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_fish_odds(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_logi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    pass


def snxn_glob_clauOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    # Yates correction
    yate = False
    if not yate:
        df_tabl = dg1
    else:
        df_tabl = dg1.where(dg1 != 0, dg1 + 0.5) # yates correction    
    
    # Table structure
    chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)

def stat_globOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
       
    # Exec
    if len(indx_cate_list) != 2:       
        snxn_glob_clauOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    else: 
        # stat_glob_perp(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        stat_glob_clauOLD(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    pass

# ----
# Inpu
# ----
def inpu(df1, indx_name, indx_cate_list, colu_cate_list, ceap_mono, filt_name, filt_valu, file_path):      
    
    colu_name = 'ceap'  
    df_tabl = inp2(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, ceap_mono, filt_name, filt_valu)
    
    indx_name_stra = f'{indx_name}_stra'
    colu_name_ordi = 'ceap_ordi'
    df_line = inp3(df1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi) 
    
    df_norm = inp5(df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) 
     
    # Exit
    return df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi

def ke30_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
    
        # Selector
        # --------
        df1, df2, df3 = inp1(file_path, filt_name, filt_valu)  
        
        # Inpu ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # -----------------------------------------------------
        colu_cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']    
        df_tabl, df_line, df_norm, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi = \
        inpu(df1, indx_name, indx_cate_list, colu_cate_list, ceap_mono, filt_name, filt_valu, file_path)
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
        yate = True
        stat_glob(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm, yate)
        
        # Inpu ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # -----------------------------------------------
        colu_cate_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        # ...
        
        # Inpu ['C3', 'C4', 'C5', 'C6']
        # -----------------------------
        colu_cate_list = ['C3', 'C4', 'C5', 'C6']
        # ...
        pass 