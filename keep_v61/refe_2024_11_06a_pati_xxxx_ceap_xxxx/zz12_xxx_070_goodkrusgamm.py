import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau

# -------------------------------
# Goodman and Kruskal's Gamma Test of Independence
# -------------------------------

# Calculate Goodman and Kruskal's Gamma
def goodman_kruskal_gamma(x, y):
    concordant = 0
    discordant = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):        
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1
    gamma = (concordant - discordant) / (concordant + discordant)
    return gamma
    
def goodkrusgamm(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = False
        
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # in both cases : Exception has occurred: TypeError : cannot unpack non-iterable rv_continuous_frozen object
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Exec
    stat = goodman_kruskal_gamma(indx_list_stra, colu_list_ordi) # Calculate Gamma
    _, pval = kendalltau(indx_list_stra, colu_list_ordi) # Calculate p-value using Kendall's Tau (as an approximation)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    asso_form = "positive" if stat > 0 else "negative" if stat < 0 else "none"
    print(f"\nData : {what}\nGoodman and Kruskal's Gamma : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")
    write(f"\nData : {what}\nGoodman and Kruskal's Gamma : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}") 
   
    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no association between '{indx_name}' and '{colu_name}' : Gamma eq 0.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an association between '{indx_name}' and '{colu_name}' : Gamma ne 0.\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Goodman and Kruskal's Gamma : Reject the null hypothesis:\n{Ha}")
        write(f"Goodman and Kruskal's Gamma : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Goodman and Kruskal's Gamma : Fail to reject the null hypothesis:\n{H0}")
        write(f"Goodman and Kruskal's Gamma : Fail to reject the null hypothesis:\n{H0}")
    pass
