import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from sklearn.metrics import mutual_info_score

def theil_u(x, y):
    contingency = pd.crosstab(x, y)
    n = contingency.sum().sum()
    py = contingency.sum(axis=0) / n
    px = contingency.sum(axis=1) / n
    pxy = contingency / n
    MI = mutual_info_score(None, None, contingency=contingency)
    Hx = -np.sum(px * np.log(px))
    Hy = -np.sum(py * np.log(py))
    return MI / Hy, MI / Hx

# Permutation test function
def permutation_test(x, y, statistic_func, n_permutations=10000):
    observed = statistic_func(x, y)
    permuted_stats = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_stats.append(statistic_func(x, y_permuted))
    
    p_values = [np.mean([stat[i] >= observed[i] for stat in permuted_stats]) for i in range(len(observed))]
    return observed, p_values

def res(what, stat, pval, parm, colu_name, indx_name, colu_cate_list, indx_cate_list):
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nTheil's U : U {parm} : Stat: {stat_form}, Pval: {pval_form}")
    write(f"\nData : {what}\nTheil's U : U {parm} : Stat: {stat_form}, Pval: {pval_form}")

    # Intp
    H0 = f"H0 : There is no association between '{indx_name}' and '{colu_name}' : U({indx_name}|{colu_name}) = 0 and U({colu_name}|{indx_name}) = 0\nKnowing one variable does not reduce uncertainty about the other.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an association between '{indx_name}' and '{colu_name}' : U({indx_name}|{colu_name}) > 0 or U({colu_name}|{indx_name}) > 0\nKnowing one variable reduces uncertainty about the other.\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Theil's U : Reject the null hypothesis:\n{Ha}")
        write(f"Theil's U : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Theil's U : Fail to reject the null hypothesis:\n{H0}")
        write(f"Theil's U : Fail to reject the null hypothesis:\n{H0}")
    
def thei(what, df21, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Prec
    indx_list = df1[indx_name]
    colu_list = df1[colu_name]
    
    # Exec : without pval
    stat_u_gender_age, stat_u_age_gender = theil_u(indx_list, colu_list)
    
    # Resu
    if np.isnan(stat_u_gender_age) or np.isnan(stat_u_age_gender):
        raise Exception("Stat or Pval are NaN")
    stat_u_gender_age_form = f"{stat_u_gender_age:.3e}" if stat_u_gender_age < 0.001 else f"{stat_u_gender_age:.3f}"
    stat_u_age_gender_form = f"{stat_u_age_gender:.3e}" if stat_u_age_gender < 0.001 else f"{stat_u_age_gender:.3f}"
    print(f"\nData : {what}\nTheil's U : Stat: U ({indx_name}|{colu_name}): {stat_u_gender_age_form} U ({colu_name}|{indx_name}): {stat_u_age_gender_form}")
    write(f"\nData : {what}\nTheil's U : Stat: U ({indx_name}|{colu_name}): {stat_u_gender_age_form} U ({colu_name}|{indx_name}): {stat_u_age_gender_form}")
    
    # Exec : with pval
    perm = 5000
    obsv_list, pval_list = permutation_test(indx_list, colu_list, theil_u, perm)
    stat = obsv_list[0] ; pval = pval_list[0]
    res(what, stat, pval, f"({indx_name}|{colu_name})", indx_name, colu_name, indx_cate_list, colu_cate_list)
    stat = obsv_list[1] ; pval = pval_list[1]
    res(what, stat, pval, f"({colu_name}|{indx_name})", colu_name, indx_name, colu_cate_list, indx_cate_list)
    pass