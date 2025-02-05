import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Mann-Whitney U
# -------------------------------
# Note : can be used for unequal group sizes
def mannwhit_clau(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    # Trac
    trac = True

    # 'df_tabl' only 
    if df1 is None:
        print(f"\nData : {what}\n(df_table) : Mann-Whitney U : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Mann-Whitney U : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        return
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    colu_list_ord1 = df1[df1[indx_name] == indx_cate_nam1][colu_name_ordi] # male_ages = df[df['Gender'] == 'Male']['Age_Ordinal']
    colu_list_ord2 = df1[df1[indx_name] == indx_cate_nam2][colu_name_ordi] # female_ages = df[df['Gender'] == 'Female']['Age_Ordinal']
    if trac:
        print(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        print(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")
        write(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        write(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")

    # Exec
    stat, pval = stats.mannwhitneyu(colu_list_ord1, colu_list_ord2, alternative='two-sided')
    
    # Calculate effect size (r)
    n1, n2 = len(colu_list_ord1), len(colu_list_ord2) # len(male_ages), len(female_ages)
    z_score = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    effe = abs(z_score) / np.sqrt(n1 + n2)
    # Interpret effect size
    if abs(effe) < 0.3:
        effe_size = "small"
    elif abs(effe) < 0.5:
        effe_size = "medium"
    else:
        effe_size = "large"
    # Calculate and print median ages for each group
    male_median = np.median(colu_list_ord1) # male_ages
    female_median = np.median(colu_list_ord2) # female_ages

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effe_form = f"{effe:.3e}" if effe < 0.001 else f"{effe:.3f}"
    male_median_form = f"{male_median:.3e}" if male_median < 0.001 else f"{male_median:.3f}"
    female_median_form = f"{female_median:.3e}" if female_median < 0.001 else f"{female_median:.3f}"
    print(f"\nData : {what}\n(perp) Mann-Whitney U : Stat:{stat_form} Pval:{pval_form}")
    print(f"Effect size:{effe_form}({effe_size}) ; Median for '{indx_name}' groups : {indx_cate_nam1}={male_median_form},{indx_cate_nam2}={female_median_form}")
    write(f"\nData : {what}\n(perp) Mann-Whitney U : Stat:{stat_form} Pval:{pval_form}")
    write(f"Effect size:{effe_form}({effe_size}) ; Median for '{indx_name}' groups : {indx_cate_nam1}={male_median_form},{indx_cate_nam2}={female_median_form}")

    # Intp
    H0 = f"H0 : H0: The '{colu_name}' distributions have the same central tendency (median) across '{indx_name}'.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : H0: The '{colu_name}' distributions have different central tendency (median) across '{indx_name}'.\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"(perp) Mann-Whitney U : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Mann-Whitney U : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Mann-Whitney U : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Mann-Whitney U : Fail to reject the null hypothesis:\n{H0}")
    pass
    pass