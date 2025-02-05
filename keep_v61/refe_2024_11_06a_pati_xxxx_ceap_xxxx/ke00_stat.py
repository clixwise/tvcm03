from datetime import datetime
import os
import sys
import pandas as pd

from zz12_xxx_097_kend_tau_ import kend_tau_
from zz12_xxx_096_good_gamm import good_gamm
from util_file_mngr import write
from zz12_yyy_251_caty_prop import caty_prop
from zz12_yyy_251_caty_bino import caty_bino
from zz12_yyy_251_caty_fish import caty_fish
from zz12_yyy_251_caty_chi2 import caty_chi2
from zz12_xxx_251_catx_wils import catx_wils
from zz12_xxx_331_fish_mist import fish_mist
from zz12_xxx_321_kruswall_clau import kruswall_clau
from zz12_xxx_311_Cochran_Mantel_Haenszel import cochmanthaen
from zz12_xxx_301_ANOVA import anov
from zz12_xxx_251_catx_logi import catx_logi
from zz12_xxx_251_catx_prop import catx_prop
from zz12_xxx_251_catx_bino import catx_bino
from zz12_xxx_251_catx_fish_odds import catx_fish_odds
from zz12_xxx_251_catx_chi2 import catx_chi2
from zz12_xxx_231_goodkruslamb_line import goodkruslamb_line
from zz12_xxx_231_goodkruslamb_tabl import goodkruslamb_tabl
from zz12_xxx_201_cram_clau import cram_clau
from zz12_xxx_201_cram_mist import cram_mist
from zz12_xxx_201_cram_perp import cram_perp
from zz12_xxx_120_muin import muin
from zz12_xxx_090_thei import thei
from zz12_xxx_080_spea import spea
from zz12_xxx_070_goodkrusgamm import goodkrusgamm
from zz12_xxx_065_cohekapp import cohekapp
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
from zz11_xxx_fish import fishfreehalt
from zz11_xxx_resi import resi
from zz11_xxx_chi2 import chi2

# ----
# Stat
# ----

# In 2024_12_15, we reconsidered all the tests for vertical stratification eg by age, by C(EAP): binomial, proportionality, etc. 
# Hence we created : 'stat_glob_perp_2024_12_15' ; see 'ke37_ceap_sexe_c3c6_full_abso_trac.explain_11_perplexity.py' onwards
def stat_glob_perp_2024_12_15(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm):
   
    '''
    7. Stratification
    '''
    caty_chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL -> DF_VERT
    # caty_fish_odds == catx_fish_odds
    caty_fish(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL -> DF_VERT
    caty_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL -> DF_VERT
    caty_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL -> DF_VERT
    catx_wils(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL -> DF_VERT
    pass
# This is an 'old version' that was created simultaneaously with the 'stat_glob_clau' version
# In 2024_12_15, we reconsidered all the tests for vertical stratification eg by age, by C(EAP): binomial, proportionality, etc. 
# Hence we created : 'stat_glob_perp_2024_12_15'
def stat_glob_perp_2024_10_01(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm):
    
    fish_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # TODO
    kruswall_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # TODO
    
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
    - Goodman and Kruskal's Gamma [CEAP concordance]
    - Polychoric correlation [NEW]
    '''
    chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    muin(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    cram_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    goodkruslamb_line(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    goodkruslamb_tabl(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    goodkrusgamm(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)      
    '''
    2. Tests for Distribution Comparison (B):
    - Kolmogorov-Smirnov
    - Theil U [CEAP ranking]
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
    - Kendall Tau [Datatab 365] [CEAP G,D ranking][also Fleiss Kappa]
    - Spearman's Rank [ERROR : BOTH VARIABLES MUST BE ORDINAL ; M,F and G,D are not]
    - Page's trend test [TODO]
    - Ordinal logistic regression [TODO]
    '''
    joncterp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    cocharmi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line)
    kendtaub(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    spea(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    '''
    5. Tests for Agreement/Concordance (E):
    - Cohen's kappa [CEAP G,D ranking]
    - Somer's D [CEAP ranking]
    - McNemar's test [TODO]
    - Cochran Mantel Haenszel
    '''
    cohekapp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    somd(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    cochmanthaen(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    '''
    6. Tests for Symmetry/Marginal Homogeneity (F):
    - Stuart-Maxwell
    '''
    stuamaxw_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    stuamaxw_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    '''
    7. Stratification
    '''
    catx_chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_fish_odds(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_wils(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    catx_logi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    
    pass

def stat_glob_clau_2024_10_01(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm):
    
    # Table structure
    if len(indx_cate_list) != 2:       
        raise Exception()
    '''
    0. Descriptive
    '''
    dist_mean(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_TABL-> DF_DIST
    dist_var1(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_TABL-> DF_DIST
    dist_var2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_TABL-> DF_DIST
    dist_var3(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_TABL-> DF_DIST     
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
    chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    fishfreehalt(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    cram_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    cram_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    cram_mist(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # STRA DF_TABL
    good_line = False
    if good_line: # The results are ok from the stat ; yet the permutation test tells pval = 1.0 ; So, its better to use the '..._tabl' version
        goodkruslamb_line(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) # STRA DF_LINE [OK 2025_01_12]
    goodkruslamb_tabl(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line) # STRA DF_TABL [OK 2025_01_12]
    #
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
    '''
    mannwhit_perp(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_LINE [OK 2025_01_12]
    mannwhit_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # STRA DF_LINE [OK 2025_01_12]
    krus = False
    if krus: # Kruskal-Wallis H test is an extension of the Mann-Whitney U test for comparing more than two independent groups.
        kruswall_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # RESERVED [OK 2025_01_12]
    kolm = False
    if kolm: # Same philisophy as Mann Whitney (see comment inside file) ; rather used for continuous variables (eg: ages)
        kolmsmir(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df_line) # RESERVED [OK 2025_01_12]
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
    stra = False
    if stra: # is replaced by 'stat_glob_perp_2024_12_15'
        catx_chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        # catx_fish_odds == caty_fish_odds
        catx_fish_odds(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_wils(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
        catx_logi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    pass

def stat_glob_mist_2025_01_20(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm):
    # Only when 
    # 1/ both variable are ordinal eg 'ceap & age' ; 'ceap & ag_bin' ; 'ceap_L and ceap_R' ; ...
    # 2/ implying they can be ranked
    spea(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    kend_tau_(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # integrates 'rank' generation
    good_gamm(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name) # integrates 'rank' generation
    chck_old = False 
    if chck_old: # pre 2025_01_20 version (keep to check with new ones)
        kendtaub(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        
def snxn_glob_clau(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm):
    
    '''
    1. Tests for Independence,Association (A):
    Check for independence, association between gender and age group
    Input : contingency table
    * Chi-Square
        - H0: The gender and age group variables are independent.
        - Ha: The gender and age group variables are not independent, i.e., they are associated.
    '''
    chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    resi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    '''
    7. Stratification [to reconsider]
    '''
    #  [to reconsider] catx_bino(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    #  [to reconsider] catx_prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    #  [to reconsider] catx_fish_odds(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pass

def stat_glob(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm, yate):

    # Yates correction
    if not yate:
        df_tabl = dg1
    else:
        df_tabl = dg1.where(dg1 != 0, dg1 + 0.5) # yates correction  
    
    # Exec  
    # Only when 
    # 1/ both variable are ordinal eg 'ceap & age' ; 'ceap & ag_bin' ; 'ceap_L and ceap_R' ; ...
    # 2/ implying they can be ranked  
    if len(indx_cate_list) != 2: # This is a rude test to exclude non-ordinal variables aka 'sexe', 'mbre' ; and keep aka 'age_bin'
         
        # Claude
        # ------     
        snxn_glob_clau(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm)
        
        # Mistral 2025_01_20
        # -------
        stat_glob_mist_2025_01_20(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm)
        
    else:
        
        # Perplexity
        # ----------
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} : stat_glob_perp_2024_12_15')
        write (">>> >>> >>>")

        # ... OLD and NEVER USED : because we it is replaced by 'stat_glob_clau'
        # stat_glob_perp_2024_10_01(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
        # ... NEW : all binom..., prop..., etc. reconsidered 
        stat_glob_perp_2024_12_15(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm)
        
        # Claude
        # ------
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} : stat_glob_clau_2024_10_01')
        write (">>> >>> >>>")
        stat_glob_clau_2024_10_01(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, df_norm)
    pass
'''
Comparing Central Tendency (Medians) : Assess whether the central tendency of two or more groups differs.
Mood's Median Test : Compare medians between groups : for ordinal or non-normally distributed data.
Mann-Whitney U Test : Compare medians of two independent groups : non-parametric test.
'''
'''
The statistical tests you mentioned can be grouped based on their primary purposes and the type of data they are designed to analyze. Here’s a proposed regrouping:

---

### **1. Tests for Comparing Central Tendency (Medians)**
These tests assess whether the central tendency of two or more groups differs.

- **Mood's Median Test**: Compares medians between groups; useful for ordinal or non-normally distributed data.
- **Mann-Whitney U Test**: Non-parametric test for comparing medians of two independent groups.

---

### **2. Tests for Assessing Variance (Spread)**
These tests evaluate differences in the variability or spread of data between groups.

- **Brown-Forsythe Test**: Tests equality of variances between groups; robust to deviations from normality.
- **Fligner-Killeen Test**: Non-parametric test for equality of variances; handles skewed data well.

---

### **3. Tests for Distributional Similarity**
These assess whether two distributions are similar beyond just central tendency or variance.

- **Ansari-Bradley Test**: Compares dispersion between two independent samples; sensitive to differences in scale.

---

### **4. Tests for Categorical Data Relationships**
These are used to assess associations or dependencies between categorical variables.

- **Chi-Square Test**: Evaluates the independence or association between two categorical variables.
- **Fisher's Exact Test**: Exact test for small sample sizes in contingency tables; assesses association between categorical variables.

---

### **5. Measures of Effect Size or Association Strength**
These quantify the strength of relationships or associations between variables.

- **Cramér's V**: Measures the strength of association between two nominal variables; derived from the Chi-Square test.
- **Goodman-Kruskal Lambda**: Measures the predictive strength of one categorical variable on another.

---

### Why This Grouping Makes Sense:
- It clusters tests by **purpose** (e.g., central tendency, variance, association).
- It considers the **type of data** (e.g., ordinal, categorical).
- It aids in selecting the most appropriate test based on the research question.

Let me know if you'd like further clarification or refinements!

'''