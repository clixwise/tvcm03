import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Variance Test of Independence
# -------------------------------
# Assesses equality of variances
# Brown-Forsythe test:
# This is a modification of Levene's test that uses the median instead of the mean, making it more robust for ordinal data. 
# It can be used to compare variances across two or more groups.
def dist_var1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    trac = False
    
    # Prec : expand df_tabl -> df_dist
    ind1_name = indx_cate_list[0] # 'M'
    ind2_name = indx_cate_list[1] # 'F'
    left_leg = np.repeat(range(len(colu_cate_list)), df.loc[ind1_name].values)
    righ_leg = np.repeat(range(len(colu_cate_list)), df.loc[ind2_name].values)
    if trac:
        print (df)
        print(left_leg)
        print(righ_leg)
 
    # Exec Brown-Forsythe test
    stat, pval = stats.levene(left_leg, righ_leg, center='median')
    # Calculate effect size (eta-squared)
    def eta_squared(f_stat, df1, df2):
        return (f_stat * df1) / (f_stat * df1 + df2)
    effect_size = eta_squared(stat, 1, len(left_leg) + len(righ_leg) - 2) # print(f"\nEffect Size (η²): {effect_size}")

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
    H0 = f"H0 : There is no difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    Ha = f"Ha : There is a difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    alpha = 0.05
    if pval < alpha:
        print(f"Variance : Brown-Forsythe : Reject the null hypothesis:\n{Ha}")
        write(f"Variance : Brown-Forsythe : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Variance : Brown-Forsythe : Fail to reject the null hypothesis:\n{H0}")
        write(f"Variance : Brown-Forsythe : Fail to reject the null hypothesis:\n{H0}") 
    pass