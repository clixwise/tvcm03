
import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import contingency, chi2_contingency

def tschuprow_intp(stat):
    
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
# Tschuprow
# -------------------------------
def tsch_perp(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
        
    # Exec
    observed = df.values
    # Calculate association measures
    stat = tschuprow_t = contingency.association(observed, method="tschuprow")
    pear_intp = tschuprow_intp(stat)  
    # Calculate chi-square statistic and p-value
    chi2, pval, dof, expected = chi2_contingency(observed)

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"

    com1_form = "For large sample sizes, even small associations can be statistically significant."
    com2_form = "Consider both the strength of association (between 0 and 1) and statistical significance (p-value) in your interpretation."

    print(f"\nData : {what}\n(perp) Tschuprow : Stat:{stat_form} Intp: asso(effect size):{pear_intp} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\n(perp) Tschuprow : Stat:{stat_form} Intp: asso(effect size):{pear_intp} Pval:{pval_form} Dof:{dof}")
    print(f"(perp) Tschuprow : {com1_form}")
    write(f"(perp) Tschuprow : {com1_form}")
    print(f"(perp) Tschuprow : {com2_form}")
    write(f"(perp) Tschuprow : {com1_form}")

    # Intp
    alpha = 0.05
    H0 = f"(perp) H0 : There is no association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(perp) Ha : There is an association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(perp) Tschuprow : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Tschuprow : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Tschuprow : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Tschuprow : Fail to reject the null hypothesis:\n{H0}")
    pass