import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Variance Test of Independence
# -------------------------------
# Ansari-Bradley test:
# This test is specifically designed to compare the dispersion (spread) of two samples. It can be used with ordinal data and doesn't assume normality.
def dist_var3(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
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

    # Exec Ansari-Bradley test
    stat, pval = stats.ansari(left_leg, righ_leg)
    # Calculate effect size (Cliff's delta)
    def cliffs_delta(x, y):
        return (np.sum(np.sign(x[:, None] - y)) / (len(x) * len(y)))
    effect_size = cliffs_delta(left_leg, righ_leg)

    # Interpret effect size
    if abs(effect_size) < 0.147:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.33:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.474:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effect_size_form = f"{effect_size:.3e}" if effect_size < 0.001 else f"{effect_size:.3f}"
    print(f"\nData : {what}\nVariance : Ansari-Bradley : Stat:{stat_form} Pval:{pval_form} Effect Size (Cliff's delta) : {effect_size_form} ({effect_interpretation})")
    write(f"\nData : {what}\nVariance : Ansari-Bradley : Stat:{stat_form} Pval:{pval_form} Effect Size (Cliff's delta) : {effect_size_form} ({effect_interpretation})")

    # Intp
    H0 = f"H0 : There is no difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    Ha = f"Ha : There is a difference in the dispersion (spread) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    alpha = 0.05
    if pval < alpha:
        print(f"Ansari-Bradley : Reject the null hypothesis:\n{Ha}")
        write(f"Ansari-Bradley : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Ansari-Bradley : Fail to reject the null hypothesis:\n{H0}")
        write(f"Ansari-Bradley : Fail to reject the null hypothesis:\n{H0}")
    pass