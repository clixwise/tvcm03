import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

# -------------------------------
# Variance Test of Independence
# -------------------------------
# Assesses equality of variances
# Brown-Forsythe test:
# This is a modification of Levene's test that uses the median instead of the mean, making it more robust for ordinal data. 
# It can be used to compare variances across two or more groups.
def dist_var1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
 
    # Trac
    trac = False
        
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not   
    indx_list_stra = df1[indx_name_stra].apply(ceap_to_numeric) # 'ceaL'
    colu_list_ordi = df1[colu_name_ordi].apply(ceap_to_numeric) # 'ceaR'
    if trac:
        print(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Step 2: Perform the Brown-Forsythe test
    stat, pval = stats.levene(indx_list_stra, colu_list_ordi, center='median')
    # Calculate effect size (eta-squared)
    def eta_squared(f_stat, df1, df2):
        return (f_stat * df1) / (f_stat * df1 + df2)
    effect_size = eta_squared(stat, 1, len(indx_list_stra) + len(colu_list_ordi) - 2) # print(f"\nEffect Size (η²): {effect_size}")

    # Interpret effect size
    if effect_size < 0.01:
        effect_interpretation = "negligible"
    elif effect_size < 0.06:
        effect_interpretation = "small"
    elif effect_size < 0.14:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effect_size_form = f"{effect_size:.3e}" if effect_size < 0.001 else f"{effect_size:.3f}"
    print(f"\nData : {what}\nVariance : Brown-Forsythe : Stat:{stat_form} Pval:{pval_form} Effect Size (Eta squared) : {effect_size_form} ({effect_interpretation})")
    write(f"\nData : {what}\nVariance : Brown-Forsythe : Stat:{stat_form} Pval:{pval_form} Effect Size (Eta squared) : {effect_size_form} ({effect_interpretation})")

    # Intp
    H0 = f"H0 : There is no difference in variances of '{indx_name}' and '{colu_name}' values"
    Ha = f"Ha : There is a difference in variances of '{indx_name}' and '{colu_name}' values"
    alpha = 0.05
    if pval < alpha:
        print(f"Variance : Brown-Forsythe : Reject the null hypothesis:\n{Ha}")
        write(f"Variance : Brown-Forsythe : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Variance : Brown-Forsythe : Fail to reject the null hypothesis:\n{H0}")
        write(f"Variance : Brown-Forsythe : Fail to reject the null hypothesis:\n{H0}") 
    pass