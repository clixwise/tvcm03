import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Median Test of Independence
# -------------------------------
def dist_mean(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
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

    # Exec
    stat, pval, medi, tabl = stats.median_test(left_leg, righ_leg)
    left_medi = np.median(left_leg)
    righ_medi = np.median(righ_leg)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    flat = [item for sublist in tabl for item in sublist]
    flat = f"[[{flat[0]}, {flat[1]}], [{flat[2]}, {flat[3]}]]"
    print(f"\nData : {what}\nMedian Test : Stat:{stat_form} Pval:{pval_form} Left median: {left_medi} Median: {medi} Righ median: {righ_medi} Contingency table:{flat}")
    write(f"\nData : {what}\nMedian Test : Stat:{stat_form} Pval:{pval_form} Left median: {left_medi} Median: {medi} Righ median: {righ_medi} Contingency table:{flat}")

    # Intp
    H0 = f"H0 : There is no difference in the central tendency (median) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    Ha = f"Ha : There is a difference in the central tendency (median) of '{colu_name}' values between '{ind1_name}' and '{ind2_name}'"
    alpha = 0.05
    if pval < alpha:
        print(f"Median Test : Reject the null hypothesis:\n{Ha}")
        write(f"Median Test : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Median Test : Fail to reject the null hypothesis:\n{H0}")
        write(f"Median Test : Fail to reject the null hypothesis:\n{H0}")
    pass