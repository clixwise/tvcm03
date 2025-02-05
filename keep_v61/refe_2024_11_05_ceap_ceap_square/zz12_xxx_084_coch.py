import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy import stats

# -------------------------------
# Cochran Q Test of Independence
# -------------------------------

def coch(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True


    def ceap_to_numeric(ceap_list):
        return max([i for i, x in enumerate(ceap_list) if x == 1])
    # Assuming df1 is your DataFrame with 'ceaL' and 'ceaR' columns
    left_severity = df1['ceaL'].apply(ceap_to_numeric)
    right_severity = df1['ceaR'].apply(ceap_to_numeric)
    print (df1['ceaL'])
    print (df1['ceaR'])
    print (left_severity)
    print (right_severity)
    # Assuming left_severity and right_severity are already defined
    # We need to convert the severity scores to binary (0 or 1)
    # Let's consider severity > 0 as 1, and 0 as 0
    binary_left = (left_severity > 0).astype(int)
    binary_right = (right_severity > 0).astype(int)
    print (binary_left)
    print (binary_right)

    # Perform Cochran's Q test
    # NOT PRESENT q_statistic, p_value = cochrans_q(binary_left, binary_right)


    def cochrans_q(*arrays):
        k = len(arrays)
        n = len(arrays[0])
        
        Q = (k-1) * (k * sum([sum(x)**2 for x in arrays]) - sum([sum(x) for x in arrays])**2)
        Q /= (k * sum([sum(x) for x in arrays]) - sum([sum(x**2) for x in arrays]))
        
        df = k - 1
        p_value = 1 - chi2.cdf(Q, df)
        
        return Q, p_value

    # Use the function as before
    q_statistic, p_value = cochrans_q(binary_left, binary_right)

    print("\nCochran's Q Test Results:")
    print(f"Q statistic: {q_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

    print("\nInterpretation:")
    if p_value < 0.05:
        print("The p-value is less than 0.05, indicating a statistically significant difference")
        print("in the proportion of patients with CEAP signs between left and right legs.")
    else:
        print("The p-value is greater than or equal to 0.05, suggesting no statistically significant")
        print("difference in the proportion of patients with CEAP signs between left and right legs.")

    print("\nNote: Cochran's Q test assesses whether there are statistically significant differences")
    print("in the proportions across multiple related groups (in this case, left and right legs).")
    
    
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not   
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
    # Exec
    stat, pval = spearmanr(indx_list_stra, colu_list_ordi)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    asso_form = "positive LE 1" if stat > 0 else "negative GE -1" if stat < 0 else "none"
    print(f"\nData : {what}\nCochran Q : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")
    write(f"\nData : {what}\nCochran Q : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")  
   
    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho EQ 0."
    Ha = f"Ha : There is a monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho NE 0."
    alpha = 0.05
    if pval < alpha:
        print(f"Cochran Q : Reject the null hypothesis:\n{Ha}")
        write(f"Cochran Q : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Cochran Q : Fail to reject the null hypothesis:\n{H0}")
        write(f"Cochran Q : Fail to reject the null hypothesis:\n{H0}")
    pass
