import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Stuart-Maxwell (Marginal Homogeneity Test)
# -------------------------------
def stuamaxw_clau(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    # Prec 
    is_square = df.shape[0] == df.shape[1]
    if not (df.shape[0] == df.shape[1]):
        print(f"\nData : {what}\n(perp) Stuart-Maxwell : table must be square")
        write(f"\nData : {what}\n(perp) Stuart-Maxwell : table must be square")
        return
    
    # Exec
    resu = SquareTable(df).homogeneity()
    stat = resu.statistic
    pval = resu.pvalue
  
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\n(clau) Stuart-Maxwell : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\n(clau) Stuart-Maxwell : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    H0 = f"(clau) H0: The marginal probabilities ('{indx_name}' and '{colu_name}' totals) are equal\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(clau) Ha: The marginal probabilities ('{indx_name}' and '{colu_name}' totals) are not equal\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"(clau) Stuart-Maxwell : Reject the null hypothesis:\n{Ha}")
        write(f"(clau) Stuart-Maxwell : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(clau) Stuart-Maxwell : Fail to reject the null hypothesis:\n{H0}")
        write(f"(clau) Stuart-Maxwell : Fail to reject the null hypothesis:\n{H0}")
    pass