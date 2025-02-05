import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# -------------------------------
# Mutual Information Test of Independence
# -------------------------------

def mutual_info_pvalue(x, y, n_permutations=10000):
    observed_mi = mutual_info_score(x, y)
    permuted_mis = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_mi = mutual_info_score(x, y_permuted)
        permuted_mis.append(permuted_mi)
    
    p_value = np.mean(np.array(permuted_mis) >= observed_mi)
    return observed_mi, p_value

def interpret_nmi(nmi):
    if 0 <= nmi < 0.10:
        return "negligible"
    elif 0.10 <= nmi < 0.30:
        return "weak"
    elif 0.30 <= nmi < 0.50:
        return "moderate"
    elif 0.50 <= nmi <= 1.00:
        return "strong"
    else:
        return "invalid"

def muin(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
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
 
    # ----
    # Exec 1
    # ----
    stat_norm_not = mutual_info_score(indx_list_stra, colu_list_ordi)

    # Resu
    if np.isnan(stat_norm_not):
        raise Exception("Stat is NaN")
    stat_norm_not_form = f"{stat_norm_not:.3e}" if stat_norm_not < 0.001 else f"{stat_norm_not:.3f}"
    print(f"\nData : {what}\nMutual Information (1) Mutual Information (not normalized) : Stat:{stat_norm_not_form}")
    write(f"\nData : {what}\nMutual Information (1) Mutual Information (not normalized) : Stat:{stat_norm_not_form}")
 
    # ----
    # Exec 2
    # ----   
    stat_norm_yes = normalized_mutual_info_score(indx_list_stra, colu_list_ordi)
    
    # Resu
    if np.isnan(stat_norm_yes):
        raise Exception("Stat is NaN")
    stat_norm_yes_form = f"{stat_norm_yes:.3e}" if stat_norm_yes < 0.001 else f"{stat_norm_yes:.3f}"
    print(f"\nData : {what}\nMutual Information (2) Mutual Information (normalized) : Stat:{stat_norm_yes_form}")
    write(f"\nData : {what}\nMutual Information (2) Mutual Information (normalized) : Stat:{stat_norm_yes_form}")
          
    # ----
    # Exec 3
    # ----
    perm = 5000
    stat, pval = mutual_info_pvalue(indx_list_stra, colu_list_ordi, perm)
    intp = interpret_nmi(stat)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nMutual Information (not normalized) : Stat:{stat_norm_not_form} Mutual Information (normalized) : Stat:{stat_norm_yes_form}")
    print(f"Intp:asso strength:{intp} Pval:{pval_form}")
    write(f"\nData : {what}\nMutual Information (not normalized) : Stat:{stat_norm_not_form} Mutual Information (normalized) : Stat:{stat_norm_yes_form}")
    write(f"Intp:asso strength:{intp} Pval:{pval_form}")
 
    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no association between the '{colu_name}' and the counts for '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an association between the '{colu_name}' and the counts for '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Mutual Information : Reject the null hypothesis:\n{Ha}")
        write(f"Mutual Information : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Mutual Information : Fail to reject the null hypothesis:\n{H0}")
        write(f"Mutual Information : Fail to reject the null hypothesis:\n{H0}")
    pass
