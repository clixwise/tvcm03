
import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency

# Cramer's V calculation
# Degrees of Freedom Correction for Rows (rcorr) and Columns (kcorr):
# rcorr = r - ((r - 1) ** 2) / (n - 1)
# kcorr = k - ((k - 1) ** 2) / (n - 1)
# These corrections adjust the effective number of rows and columns in the contingency table. 
# The terms ((r - 1) ** 2) / (n - 1) and ((k - 1) ** 2) / (n - 1) account for 
# the variance in the row and column totals, respectively, 
# and help to provide a more accurate estimate of the degrees of freedom.
def cramers_v(df):

    # Exec
    chi2, p_value, dof, expected = chi2_contingency(df)
    n = df.sum().sum()
    phi2 = chi2 / n
    r, k = df.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    stat = cramer_v = effect_size =  np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    
    # Exit
    return stat

def cramers_v_intp(stat):
    
    interpretation = ""
    if stat == 0:
        interpretation = "No association between the variables."
    elif 0 < stat <= 0.1:
        interpretation = "Negligible association between the variables."
    elif 0.1 < stat <= 0.3:
        interpretation = "Weak association between the variables."
    elif 0.3 < stat <= 0.5:
        interpretation = "Moderate association between the variables."
    else:
        interpretation = "Strong association between the variables."

    return interpretation

# -------------------------------
# Cramer V
# -------------------------------
def cram_mist(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    stat = effect_size = cramers_v(df)
    cram_intp = cramers_v_intp(stat)
    # Calculate p-value for Cramer's V
    chi2, pval, dof, expected = chi2_contingency(df)
 
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\n(mist) Cramer V : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\n(mist) Cramer V : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")

    # Intp
    alpha = 0.05
    H0 = f"(mist) H0 : There is no association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(mist) Ha : There is an association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(mist) Cramer V : Reject the null hypothesis:\n{Ha}")
        write(f"(mist) Cramer V : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(mist) Cramer V : Fail to reject the null hypothesis:\n{H0}")
        write(f"(mist) Cramer V : Fail to reject the null hypothesis:\n{H0}")
    pass
    