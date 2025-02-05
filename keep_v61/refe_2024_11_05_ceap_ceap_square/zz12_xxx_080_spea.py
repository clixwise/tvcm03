import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import spearmanr

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

def interpret_effect_size(correlation):
    abs_corr = abs(correlation)
    if abs_corr < 0.20:
        return "very weak"
    elif abs_corr < 0.40:
        return "weak"
    elif abs_corr < 0.60:
        return "moderate"
    elif abs_corr < 0.80:
        return "strong"
    else:
        return "very strong"
    
# -------------------------------
# Spearman's Rank Test of Independence
# -------------------------------
def spea(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
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
    stat, pval = spearmanr(indx_list_stra, colu_list_ordi)
    correlation = stat
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    asso_form = f"'positive LE 1 (increase in {indx_name_stra}, increase in {colu_name_ordi})'" if stat > 0 else f"'negative GE -1 (increase in {indx_name_stra}, decrease in {colu_name_ordi})'" if stat < 0 else "none"
    effe_size = interpret_effect_size(correlation)
    print(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form}")  
    print(f"Spearman's Rank : Asso:{asso_form} Effe:{effe_size}")
    write(f"Spearman's Rank : Asso:{asso_form} Effe:{effe_size}")
    
    # Resu : Calculate the percentage of cases where severity is equal, higher in left, or higher in right
    both_equal = np.sum(indx_list_stra == colu_list_ordi) / len(indx_list_stra) * 100
    left_highr = np.sum(indx_list_stra > colu_list_ordi) / len(indx_list_stra) * 100
    righ_highr = np.sum(indx_list_stra < colu_list_ordi) / len(indx_list_stra) * 100
    both_equal_form = f"{both_equal:.1f}"
    left_highr_form = f"{left_highr:.1f}"
    righ_highr_form = f"{righ_highr:.1f}"
    print(f"Spearman's Rank : Additional Insights:\n")
    print(f"- In {both_equal_form}% of cases, the severity is equal in both {indx_name_stra} and {colu_name_ordi}.")
    print(f"- In {left_highr_form}% of cases, the {indx_name_stra} shows higher severity.")
    print(f"- In {righ_highr_form}% of cases, the {colu_name_ordi} higher severity.")
    write(f"Spearman's Rank : Additional Insights:\n")
    write(f"- In {both_equal_form}% of cases, the severity is equal in both {indx_name_stra} and {colu_name_ordi}.")
    write(f"- In {left_highr_form}% of cases, the {indx_name_stra} shows higher severity.")
    write(f"- In {righ_highr_form}% of cases, the {colu_name_ordi} higher severity.")

    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho EQ 0."
    Ha = f"Ha : There is a monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho NE 0."
    alpha = 0.05
    if pval < alpha:
        print(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
        write(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
        write(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
    pass
