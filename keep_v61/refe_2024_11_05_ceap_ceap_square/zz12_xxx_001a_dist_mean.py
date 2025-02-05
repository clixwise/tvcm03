import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

# -------------------------------
# Median Test of Independence
# -------------------------------
def dist_mean(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):

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

    # Exec
    stat, pval, grand_median, table = stats.median_test(indx_list_stra, colu_list_ordi) # Perform Mood's median test
    male_median = np.median(indx_list_stra) # male_ages
    female_median = np.median(colu_list_ordi) # female_ages

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    male_median_form = f"{male_median:.3e}" if male_median < 0.001 else f"{male_median:.3f}"
    female_median_form = f"{female_median:.3e}" if female_median < 0.001 else f"{female_median:.3f}"
    print(f"\nData : {what}\n(perp) Mood's Median : Stat:{stat_form} Pval:{pval_form}")
    print(f"(perp) Mood's Median : Median for '{indx_name}' groups : {indx_name}={male_median_form},{colu_name}={female_median_form}")
    print(f"(perp) Mood's Median : The Stat:'{stat_form}' follows a chi-square distribution with 1 degree of freedom.")
    print("A larger test statistic indicates a greater difference between the medians.")
    print(f"(perp) Mood's Median : assesses whether the distribution of CEAP classifications differs significantly between left and right legs, without assuming normality.")
    write(f"\nData : {what}\n(perp) Mood's Median : Stat:{stat_form} Pval:{pval_form}")
    write(f"(perp) Mood's Median : Median for '{indx_name}' groups : {indx_name}={male_median_form},{colu_name}={female_median_form}")
    write(f"(perp) Mood's Median : The Stat:'{stat_form}' follows a chi-square distribution with 1 degree of freedom.")
    write("A larger test statistic indicates a greater difference between the medians.")
    write(f"(perp) Mood's Median : assesses whether the distribution of CEAP classifications differs significantly between left and right legs, without assuming normality.")

    # Intp
    H0 = f"H0 : There is no difference in the median CEAP classifications between left and right legs."
    Ha = f"Ha : There is a difference in the median CEAP classifications between left and right legs."
    alpha = 0.05
    if pval < alpha:
        print(f"(perp) Mood's Median : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Mood's Median : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Mood's Median : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Mood's Median : Fail to reject the null hypothesis:\n{H0}")
    pass