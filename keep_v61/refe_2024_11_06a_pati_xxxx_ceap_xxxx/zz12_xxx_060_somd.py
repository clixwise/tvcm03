import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau

# -------------------------------
# Somers' D Test of Independence
# -------------------------------

def somers_d(x, y):
    tau, _ = kendalltau(x, y)
    concordant = tau * (len(x) * (len(x) - 1) / 4)
    discordant = (len(x) * (len(x) - 1) / 4) - concordant
    ties_x = len(x) * (len(x) - 1) / 2 - concordant - discordant
    d_xy = (concordant - discordant) / np.sqrt((concordant + discordant + ties_x) * (concordant + discordant))
    return d_xy

def permutation_test(x, y, statistic, n_permutations=10000):
    observed = statistic(x, y)
    permuted_stats = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_stats.append(statistic(x, y_permuted))
    
    p_value = np.mean([abs(stat) >= abs(observed) for stat in permuted_stats])
    return observed, p_value

def somd(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = False
        
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not  
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Exec : Use the permutation test for Somers' D
    perm = 5000
    stat, pval = permutation_test(indx_list_stra, colu_list_ordi, somers_d, perm) # df['Gender_num'], df['Age_Ordinal']
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nSomers' D : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nSomers' D : Stat:{stat_form} Pval:{pval_form}")  
   
    # Intp
    # Mistral
    # H0 = "H0 : There is no significant association between the severity scores for the left and right sides."
    # Ha = "Ha : There is a significant association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no ordinal association between '{indx_name}' and '{colu_name}' : D eq 0\nThe variables are independant.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an ordinal association between '{indx_name}' and '{colu_name}' : D ne 0\nThe variables are positively or negatively associated.\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Somers' D : Reject the null hypothesis:\n{Ha}")
        write(f"Somers' D : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Somers' D : Fail to reject the null hypothesis:\n{H0}")
        write(f"Somers' D : Fail to reject the null hypothesis:\n{H0}")
    pass
