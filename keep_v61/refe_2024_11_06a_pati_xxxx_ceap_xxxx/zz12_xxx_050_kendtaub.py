import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau

# -------------------------------
# Kendall Tau Test of Independence
# tuto :
# https://datatab.net/tutorial/kendalls-tau
# -------------------------------

def kendtaub(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
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

    # Exec
    #print (df1)
    stat, pval = kendalltau(indx_list_stra, colu_list_ordi)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nKendall Tau : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nKendall Tau : Stat:{stat_form} Pval:{pval_form}")  
   
    # Intp
    # Mistral
    # H0 = " H0 : there is no significant association between the severity scores for the left and right sides."
    # Ha = " Ha : there is a significant association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no association between the '{colu_name}' and the counts for '{indx_name}'groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is an association between the '{colu_name}' and the counts for '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Kendall Tau : Reject the null hypothesis:\n{Ha}")
        write(f"Kendall Tau : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Kendall Tau : Fail to reject the null hypothesis:\n{H0}")
        write(f"Kendall Tau : Fail to reject the null hypothesis:\n{H0}")
    pass
