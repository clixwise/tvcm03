import os
import sys
import pandas as pd

from zz11_xxx_fish import fishfreehalt
from zz12_xxx_096_good_gamm import good_gamm
from zz12_xxx_097_kend_tau_ import kend_tau_
from zz12_xxx_094_pear_perp import pear_perp
from zz12_xxx_094_tsch_perp import tsch_perp
from zz12_xxx_095_symmetry_perp import symm_perp
from zz12_xxx_094_cram_perp import cram_perp
from zz12_xxx_094_cram_mist import cram_mist
from zz12_xxx_092_mann_whitney import mann_with
from zz12_xxx_022_stuamaxw_clau import stuamaxw_clau
from zz12_xxx_022_stuamaxw_perp import stuamaxw_perp
from zz12_xxx_091_mc_nemar import nema
from zz12_xxx_090_jacc import jacc
from zz12_xxx_085_logi_regr import logi_regr
from zz12_xxx_084_coch import coch
from zz12_xxx_083_cohen_kappa import cohe_kapp
from zz12_xxx_082_cohen_d import cohe
from zz12_xxx_081_wilk_rank import wilk_rank
from zz12_xxx_080_spea import spea
from zz12_xxx_020_kolmsmir import kolmsmir
from zz12_xxx_001b_dist_var3 import dist_var3
from zz12_xxx_001b_dist_var2 import dist_var2
from zz12_xxx_001b_dist_var1 import dist_var1
from zz12_xxx_001a_dist_mean import dist_mean
from zz11_xxx_resi import resi
from zz11_xxx_chi2 import chi2

# ----
# Stat
# ----
def snxn_glob_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    # Table structure
    if len(indx_cate_list) != len(colu_cate_list):       
        raise Exception()
    
    # Exec
    exec = True
    '''
    0. Descriptive
    '''
    if exec:
        dist_mean(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        dist_var1(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        dist_var2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        dist_var3(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    '''
    1. Tests for Independence,Association (A):
    Check for independence, association between gender and age group
    Input : contingency table
    * Chi-Square
        - H0: The gender and age group variables are independent.
        - Ha: The gender and age group variables are not independent, i.e., they are associated.
    * Fisher Exact
        - H0: The gender and age group variables are independent.
        - Ha: The gender and age group variables are not independent, i.e., they are associated.
    * Cramer's V [Datab 194 : ok]
        - H0: There is no association between the gender and age group variables.
        - Ha: There is an association between the gender and age group variables.
    * Goodman and Kruskal's Lambda
        - H0: The gender variable does not provide any information about the age group variable, or vice versa.
        - Ha: The gender variable provides information about the age group variable, or vice versa.
    '''
    if exec:
        chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02
        resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02
        # 2025_02_02 : ce test [Variante Fisher nxn] est à revoir # 2025_02_02
        # integrer 4 tests [qui donnent tous pval=1 -> fiable ?]
        fish = False
        if fish:
            fishfreehalt(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) 
        cram_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02
        cram_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02
        pear_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02
        tsch_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_02_02 [idem Cramer V si carré ; better than Cramer V for rectangular table]
        # ignore [2024-11-23] goodkruslamb(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    '''
    2. Tests for Distribution Comparison (B):
    Compare the age group distributions between genders
    Input : individual patient rows
        - male_ages = df[df['Gender'] == 'Male']['Age_Ordinal']
        - female_ages = df[df['Gender'] == 'Female']['Age_Ordinal']
    * Mann-Whitney U [Datatab 164 : ok]
        - u_stat, mw_p = mannwhitneyu(male_ages, female_ages, alternative='two-sided')
        - H0: The age group distributions have the same central tendency (median) across genders.
        - Ha: The age group distributions have different central tendencies (medians) across genders.
    * Kruskal-Wallis H [Datatab 244 : for 3 or more groups]
        - kruskal_stat, kruskal_p = kruskal_wallis(male_ages, female_ages)
        - H0: The age group distributions are the same across genders.
        - Ha: The age group distributions differ across genders.
    * Kolmogorov-Smirnov (perp) [Datattab : -]
        Can be added since it compares two distributions with respect to age_bin
    * Wilcoxon signed-rank [for Larray, Rarray]
    '''
    if exec:
        mann_with(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        # Applies only for continuous variables 2025_01_20
        # kolmsmir(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)

        cohe(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        jacc(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) 
    '''
    3. Tests for Location/Central Tendency (C):
    - Mann-Whitney U [MOVED TO 2.]
    - ANOVA
    - Wilcoxon signed-rank [for Larray, Rarray] [MOVED TO 2.]
    - Friedman test [only for 3 groups of line-line patients comparisons]
    '''
    if exec:
        # ignore [2024-11-23] anov(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        pass
    '''
    4. Tests for Trend/Progression (D):
    - Jonckheere-Terpstra
    - Cochran-Armitage
    - Kendall Tau [Datatab 365] [CEAP G,D ranking][also Fleiss Kappa]
    - Spearman's Rank [ERROR : BOTH VARIABLES MUST BE ORDINAL ; M,F and G,D are not]
    - Page's trend test [TODO]
    - Ordinal logistic regression [TODO]
    '''
    if exec:
        # ignore [2024-11-23] joncterp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        # ignore [2024-11-23] cocharmi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
        # ignore [2024-11-23] kendtaub(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        
        # Only when 
        # 1/ both variable are ordinal eg 'ceap & age' ; 'ceap & ag_bin' ; 'ceap_L and ceap_R' ; ...
        # 2/ implying they can be ranked
        spea(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) # 2025_02_02 : évaluer si correct
        kend_tau_(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_01_20 integrates 'rank' generation
        good_gamm(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # 2025_01_20 integrates 'rank' generation
        wilk_rank(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) # note : requires a concept of 'difference' which 2*CEAP has
        # ignore [2024-11-23] : for 3 or more groups
        # kruswall_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    ''' 
    5. Tests for Agreement/Concordance (E):
    - Cohen's kappa [CEAP G,D ranking]
    - Somer's D [CEAP ranking]
    - McNemar's test [TODO]
    - Cochran Mantel Haenszel
    '''
    if exec:
        cohe_kapp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)     
        # ignore [2024-11-23] somd(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        # ignore [2024-11-23] cochmanthaen(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        nema(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) 
    '''
    6. Tests for Symmetry/Marginal Homogeneity (F):
    * Stuart-Maxwell [DONE]
        - H0: The marginal probabilities (row and column totals) of the gender-age group contingency table are equal.
        - Ha: The marginal probabilities (row and column totals) of the gender-age group contingency table are not equal.
    '''
    if exec:
        stuamaxw_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        stuamaxw_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)  
        symm_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
 
    if False: # TODO     
        coch(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        logi_regr(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)        

def snxn_glob(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, yate):

    # Yates correction
    if not yate:
        df_tabl = dg1
    else:
        df_tabl = dg1.where(dg1 != 0, dg1 + 0.5) # yates correction  
    
    # Exec    
    snxn_glob_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    pass