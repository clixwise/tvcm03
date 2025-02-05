import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway
from util_file_mngr import write
from scipy.stats import chi2

# -------------------------------
# Cochran Mantel Haenszel Test of Independence
# -------------------------------

def cochmanthaen(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1] 
    
    # Initialize variables
    sum_a = sum_b = sum_c = sum_d = 0
    sum_r = sum_s = sum_n = sum_m = 0

    # Total counts
    total_m = df.loc[indx_cate_nam1].sum()
    total_f = df.loc[indx_cate_nam2].sum()

    # Iterate through each age bin
    for age_bin in df.columns:
        a = df.loc[indx_cate_nam1, age_bin]
        b = df.loc[indx_cate_nam2, age_bin]
        c = total_m - a
        d = total_f - b
        n = a + b + c + d

        # Calculate components for CMH statistic
        sum_a += a
        sum_r += (a + b) * (a + c) / n
        sum_s += (a + b) * (a + c) * (c + d) / (n * n)
        sum_n += n

    # Calculate CMH statistic
    stat = (abs(sum_a - sum_r) - 0.5)**2 / sum_s

    # Calculate p-value
    pval = 1 - chi2.cdf(stat, 1)

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nCochran Mantel Haenszel : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nCochran Mantel Haenszel : Stat:{stat_form} Pval:{pval_form}")

    # Intp
    # Mistral
    # H0 = "H0 : ???"
    # Ha = "Ha : ???"
    H0 = f"H0 : There is no association between '{indx_name}' groups across '{colu_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an association between '{indx_name}' groups across '{colu_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Cochran Mantel Haenszel : Reject the null hypothesis:\n{Ha}")
        write(f"Cochran Mantel Haenszel : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Cochran Mantel Haenszel : Fail to reject the null hypothesis:\n{H0}")
        write(f"Cochran Mantel Haenszel : Fail to reject the null hypothesis:\n{H0}")
    pass
