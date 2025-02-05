import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency

# -------------------------------
# Goodman and Kruskal's Lambda [Table] Test of Independence
# -------------------------------
    
def goodkruslamb_tabl(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True
    
    # ----
    # Exec 1
    # ----
    if False:
        data = {
            'NA': [52, 53],
            'C0': [31, 36],
            'C1': [5, 6],
            'C2': [44, 54],
            'C3': [93, 156],
            'C4': [38, 59],
            'C5': [18, 35],
            'C6': [97, 99]
        }
        df = pd.DataFrame(data, index=['M', 'F'])
        print (df)
    # Calculate row and column totals
    row_totals = df.sum(axis=1)
    # print (row_totals)
    column_totals = df.sum(axis=0)
    # print (column_totals) 
    row_modals = df.max(axis=1) # Calculate modal frequencies for rows   
    column_modals = df.max(axis=0) # Calculate modal frequencies for columns   
    row_errors = row_totals - row_modals # Calculate errors of prediction for rows  
    column_errors = column_totals - column_modals # Calculate errors of prediction for columns 
    total_errors_row = row_errors.sum()  # Total errors when predicting "ceap" from "sexe" 
    total_errors_column = column_errors.sum() # Total errors when predicting "sexe" from "ceap" 
    total_observations = row_totals.sum() # Total number of observations
    # Calculate Goodman and Kruskal's lambda
    lambda_ceap_from_sexe = total_errors_row / total_observations
    lambda_sexe_from_ceap = total_errors_column / total_observations
    
    def interpret_lambda(lambda_value):
        if lambda_value < 0.1:
            return "Negligible association"
        elif lambda_value < 0.3:
            return "Weak association"
        elif lambda_value < 0.5:
            return "Moderate association"
        else:
            return "Strong association"


    print(f"Lambda for predicting '{colu_name}' from '{indx_name}': {lambda_ceap_from_sexe:.3f}")
    print(f"Lambda for predicting '{indx_name}' from '{colu_name}': {lambda_sexe_from_ceap:.3f}")

    # ----
    # Exec 2
    # ----
    chi2, pval, dof, expected = chi2_contingency(df)

    # ----
    # Resu
    # ----
    ceap_from_sexe_form = f"{lambda_ceap_from_sexe:.3e}" if lambda_ceap_from_sexe < 0.001 else f"{lambda_ceap_from_sexe:.3f}"
    sexe_from_ceap_form = f"{lambda_sexe_from_ceap:.3e}" if lambda_sexe_from_ceap < 0.001 else f"{lambda_sexe_from_ceap:.3f}"
    stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nGoodman and Kruskal's Lambda [Tabl] predicting '{colu_name}' from '{indx_name}' : Stat:{ceap_from_sexe_form} ({interpret_lambda(lambda_ceap_from_sexe)}) Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\nGoodman and Kruskal's Lambda [Tabl] predicting '{colu_name}' from '{indx_name}' : Stat:{ceap_from_sexe_form} ({interpret_lambda(lambda_ceap_from_sexe)}) Pval:{pval_form} Dof:{dof}")
    print(f"\nData : {what}\nGoodman and Kruskal's Lambda [Tabl] predicting '{indx_name}' from '{colu_name}' : Stat:{sexe_from_ceap_form} ({interpret_lambda(lambda_sexe_from_ceap)}) Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\nGoodman and Kruskal's Lambda [Tabl] predicting '{indx_name}' from '{colu_name}' : Stat:{sexe_from_ceap_form} ({interpret_lambda(lambda_sexe_from_ceap)}) Pval:{pval_form} Dof:{dof}")

    # Intp
    H0 = f"H0 : there is no predictability between the categorical '{colu_name}' and the group '{indx_name}' variables"
    H0 += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : there is a predictability between the categorical '{colu_name}' and the group '{indx_name}' variables"
    Ha += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Goodman and Kruskal's Lambda [Tabl] : Reject the null hypothesis:\n{Ha}")
        write(f"Goodman and Kruskal's Lambda [Tabl] : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Goodman and Kruskal's Lambda [Tabl] : Fail to reject the null hypothesis:\n{H0}")
        write(f"Goodman and Kruskal's Lambda [Tabl] : Fail to reject the null hypothesis:\n{H0}")
    pass

'''
2025_01_12 Mistral

'''