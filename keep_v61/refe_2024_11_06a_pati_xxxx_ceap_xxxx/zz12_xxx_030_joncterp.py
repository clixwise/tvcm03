import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

# -------------------------------
# Jonckheere-Terpstra Test of Independence
# -------------------------------
def jonckheere_terpstra(x, y):
    """
    Perform Jonckheere-Terpstra test
    x: array of group labels
    y: array of ordinal values
    """
    unique_groups = np.unique(x)
    n_groups = len(unique_groups)
    counts = [np.sum(x == group) for group in unique_groups]
    
    S = 0
    for i in range(n_groups - 1):
        for j in range(i + 1, n_groups):
            S += stats.mannwhitneyu(y[x == unique_groups[i]], 
                                    y[x == unique_groups[j]],
                                    alternative='two-sided').statistic
    
    N = sum(counts)
    mean_S = N * (N - 1) / 4
    var_S = (N * (N - 1) * (2 * N + 5) - sum([c * (c - 1) * (2 * c + 5) for c in counts])) / 72
    
    z = (S - mean_S) / np.sqrt(var_S)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

def joncterp(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    # Trac
    trac = False
        
    # Prec
    indx_list_stra = df1[indx_name]
    colu_list_ordi = df1[colu_name_ordi] # colu_list_ordi ordinal [0...n] : ceap series for F ages
    if trac:
        print(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Exec
    #print (df1)
    stat, pval = jonckheere_terpstra(indx_list_stra, colu_list_ordi) # stat, pval = jonckheere_terpstra(df['Gender'], df['Age_Ordinal'])
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nJonckheere-Terpstra : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nJonckheere-Terpstra : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    # Mistral
    # H0 = "H0 : there is no trend in the severity scores across the different categories for the left and right sides."
    # Ha = "Ha : there is a trend in the severity scores across the different categories for the left and right sides."
    H0 = f"H0 : There is no trend or ordered difference in '{colu_name}' distribution between '{indx_name}' groups\nThe group distributions are the same\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a trend or ordered difference in '{colu_name}' distribution between '{indx_name}' groups\nThe group distributions are ordered in a specific way : increasing or decreasing\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Jonckheere-Terpstra : Reject the null hypothesis:\n{Ha}")
        write(f"Jonckheere-Terpstra : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Jonckheere-Terpstra : Fail to reject the null hypothesis:\n{H0}")
        write(f"Jonckheere-Terpstra : Fail to reject the null hypothesis:\n{H0}")
    pass
