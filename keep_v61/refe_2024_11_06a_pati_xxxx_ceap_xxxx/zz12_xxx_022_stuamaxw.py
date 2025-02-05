import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Stuart-Maxwell Test of Independence (Marginal Homogeneity Test)
# -------------------------------
def stuamaxw(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    # Exec
    resu = SquareTable(df).homogeneity()
    stat = resu.statistic
    pval = resu.pvalue
  
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nStuart-Maxwell : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nStuart-Maxwell : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    # Mistral
    # H0 = "H0 : the distributions of the severity scores for the left and right sides are the same."
    # Ha = "Ha : the distributions of the severity scores for the left and right sides are different."
    H0 = f"H0 : There is no difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Stuart-Maxwell : Reject the null hypothesis:\n{Ha}")
        write(f"Stuart-Maxwell : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Stuart-Maxwell : Fail to reject the null hypothesis:\n{H0}")
        write(f"Stuart-Maxwell : Fail to reject the null hypothesis:\n{H0}")
    pass