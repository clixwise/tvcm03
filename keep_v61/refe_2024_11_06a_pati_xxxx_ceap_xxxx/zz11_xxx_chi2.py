import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency

# -------------------------------
# Chi-Square Test of Independence
# tuto : https://datatab.net/tutorial/chi-square-test
# -------------------------------
def chi2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    chi2, pval, dof, expected = chi2_contingency(df)

    # Resu
    stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nChi-Square : Stat:{stat_form} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\nChi-Square : Stat:{stat_form} Pval:{pval_form} Dof:{dof}")  

    # Intp
    H0 = f"H0 : there is no association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    H0 += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    #H0 += f"\nThe distribution of one variable is independent of the distribution of the other"
    #H0 += "\nThere is no difference in the distribution of venous insufficiency between the legs"
    #H0 += f"\nThe relationship between the two variables is symmetric"
    Ha = f"Ha : there is a association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    Ha += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    #Ha += f"\nThe distribution of one variable is dependent on the distribution of the other"
    #Ha += f"\nThe relationship between the two variables is asymmetric"
    alpha = 0.05
    if pval < alpha:
        print(f"Chi-Square : Reject the null hypothesis:\n{Ha}")
        write(f"Chi-Square : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Chi-Square : Fail to reject the null hypothesis:\n{H0}")
        write(f"Chi-Square : Fail to reject the null hypothesis:\n{H0}")
    pass
    
'''
Chi-Square Test: Focuses on the association or relationship between two categorical variables (age bins and sex categories). 
It checks whether the distribution of one variable is dependent on the other.
'''