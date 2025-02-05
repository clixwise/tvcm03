import numpy as np
import pandas as pd
from util_file_mngr import write
import numpy as np
from scipy import stats
    
def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

# -------------------------------
# Mc Nemar Test of Independence
# -------------------------------
def mann_with(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):

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
    stat, pval = stats.mannwhitneyu(indx_list_stra, colu_list_ordi, alternative='two-sided')
    
    # Effect size calculation (r)
    n1, n2 = len(indx_list_stra), len(colu_list_ordi)
    z_score = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    effe = abs(z_score) / np.sqrt(n1 + n2)
    # Interpret effect size
    if abs(effe) < 0.1:
        effe_size = "negligible"
    elif abs(effe) < 0.3:
        effe_size = "small"
    elif abs(effe) < 0.5:
        effe_size = "medium"
    else:
        effe_size = "large"
    # Calculate and print median ages for each group
    male_median = np.median(indx_list_stra) # male_ages
    female_median = np.median(colu_list_ordi) # female_ages
    if male_median > female_median:
        com1_form = f"{indx_name}(r) tends to have higher CEAP classifications."
    else:
        com1_form = f"{colu_name}(l) tends to have higher CEAP classifications."

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effe_form = f"{effe:.3e}" if effe < 0.001 else f"{effe:.3f}"
    male_median_form = f"{male_median:.3e}" if male_median < 0.001 else f"{male_median:.3f}"
    female_median_form = f"{female_median:.3e}" if female_median < 0.001 else f"{female_median:.3f}"
    print(f"\nData : {what}\n(perp) Mann-Whitney U : Stat:{stat_form} Pval:{pval_form} Effect size(0<..<1):{effe_form}({effe_size})")
    print(f"(perp) Mann-Whitney U : Median for '{indx_name}' groups : {indx_name}={male_median_form},{colu_name}={female_median_form} : {com1_form}")
    write(f"\nData : {what}\n(perp) Mann-Whitney U : Stat:{stat_form} Pval:{pval_form} Effect size:{effe_form}({effe_size})")
    write(f"(perp) Mann-Whitney U : Median for '{indx_name}' groups : {indx_name}={male_median_form},{colu_name}={female_median_form} : {com1_form}")

    print(f"(perp) Mann-Whitney U : assesses whether the distribution of CEAP classifications differs significantly between left and right legs, without assuming normality.")
    write(f"(perp) Mann-Whitney U : assesses whether the distribution of CEAP classifications differs significantly between left and right legs, without assuming normality.")

    # Intp
    # Mistral
    # H0 : there is no difference between the distributions of the severity scores for the left and right sides.
    # Ha : there is a difference between the distributions of the severity scores for the left and right sides.
    H0 = f"H0 : There is no difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"(perp) Mann-Whitney U : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Mann-Whitney U : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Mann-Whitney U : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Mann-Whitney U : Fail to reject the null hypothesis:\n{H0}")
    pass