import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Variance Test of Independence
# -------------------------------
# Assesses equality of variances
# Fligner-Killeen test:
# This is a non-parametric test for homogeneity of variances that is particularly robust against departures from normality. 
# It can be applied to ordinal data and is considered more appropriate than Levene's test when dealing with highly skewed distributions.
def dist_var2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
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

    # Exec Fligner-Killeen test
    stat, pval = stats.fligner(left_leg, righ_leg)
    # Calculate effect size (Cramer's V)
    def cramers_v(chi2, n, min_dim):
        return np.sqrt(chi2 / (n * (min_dim - 1)))
    n = len(left_leg) + len(righ_leg)
    min_dim = 2  # For two groups
    effect_size = cramers_v(stat, n, min_dim) # print(f"\nEffect Size (Cramer's V): {effect_size}")
    # Interpret effect size
    if effect_size < 0.1:
        effect_interpretation = "negligible"
    elif effect_size < 0.3:
        effect_interpretation = "small"
    elif effect_size < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effect_size_form = f"{effect_size:.3e}" if effect_size < 0.001 else f"{effect_size:.3f}"
    print(f"\nData : {what}\nVariance : Fligner-Killeen : Stat:{stat_form} Pval:{pval_form} Effect Size (Cramer's V) : {effect_size_form} ({effect_interpretation})")
    write(f"\nData : {what}\nVariance : Fligner-Killeen : Stat:{stat_form} Pval:{pval_form} Effect Size (Cramer's V) : {effect_size_form} ({effect_interpretation})")

    # Intp
    H0 = f"H0 : There is no difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    Ha = f"Ha : There is a difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    alpha = 0.05
    if pval < alpha:
        print(f"Fligner-Killeen : Reject the null hypothesis:\n{Ha}")
        write(f"Fligner-Killeen : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Fligner-Killeen : Fail to reject the null hypothesis:\n{H0}")
        write(f"Fligner-Killeen : Fail to reject the null hypothesis:\n{H0}")
    pass