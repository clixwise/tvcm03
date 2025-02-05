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
# Ansari-Bradley test:
# This test is specifically designed to compare the dispersion (spread) of two samples. It can be used with ordinal data and doesn't assume normality.
def dist_var3(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
      
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

    # Step 2: Perform the Ansari-Bradley test
    stat, pval = stats.ansari(indx_list_stra, colu_list_ordi)

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nVariance : (prep) Ansari-Bradley : Stat:{stat_form} Pval:{pval_form}")
    print(f"(prep) Ansari-Bradley : Stat:{stat_form} indicates the degree of difference in variances.")
    print(f"(prep) Ansari-Bradley : A larger test statistic suggests greater differences in variability between the two groups.")
    write(f"\nData : {what}\nVariance : (prep) Ansari-Bradley : Stat:{stat_form} Pval:{pval_form}")
    write(f"(prep) Ansari-Bradley : Stat:{stat_form} indicates the degree of difference in variances.")
    write(f"(prep) Ansari-Bradley : A larger test statistic suggests greater differences in variability between the two groups.")

    # Intp
    H0 = f"H0 : There is no difference in variances of '{indx_name}' and '{colu_name}' values"
    Ha = f"Ha : There is a difference in variances of '{indx_name}' and '{colu_name}' values"
    alpha = 0.05
    if pval < alpha:
        print(f"Ansari-Bradley : Reject the null hypothesis:\n{Ha}")
        write(f"Ansari-Bradley : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Ansari-Bradley : Fail to reject the null hypothesis:\n{H0}")
        write(f"Ansari-Bradley : Fail to reject the null hypothesis:\n{H0}")
    pass