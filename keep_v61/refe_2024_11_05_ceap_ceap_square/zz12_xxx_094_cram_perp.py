
import numpy as np
import pandas as pd
from util_file_mngr import write      
from scipy.stats import contingency, chi2_contingency

# Cramer's V calculation
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

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
def cram_per1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    # Assuming df2 is your contingency table
    confusion_matrix = df.values
    stat = cramers_v(confusion_matrix)
    chi2, pval, dof, expected = chi2_contingency(confusion_matrix)  
    cram_intp = cramers_v_intp(stat)  

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"

    com1_form = "For large sample sizes, even small associations can be statistically significant."
    com2_form = "Consider both the strength of association (between 0 and 1) and statistical significance (p-value) in your interpretation."

    print(f"\nData : {what}\n(perp) Cramer V (1) : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\n(perp) Cramer V (1) : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    print(f"(perp) Cramer V (1) : {com1_form}")
    write(f"(perp) Cramer V (1) : {com1_form}")
    print(f"(perp) Cramer V (1) : {com2_form}")
    write(f"(perp) Cramer V (1) : {com1_form}")

    # Intp
    alpha = 0.05
    H0 = f"(perp) H0 : There is no association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(perp) Ha : There is an association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(perp) Cramer V (1) : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Cramer V (1) : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Cramer V (1) : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Cramer V (1) : Fail to reject the null hypothesis:\n{H0}")
    pass

def cram_per2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    observed = df.values
    # Calculate association measures
    stat = cramer_v = contingency.association(observed, method="cramer")
    cram_intp = cramers_v_intp(stat)  
    # Calculate chi-square statistic and p-value
    chi2, pval, dof, expected = chi2_contingency(observed)

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"

    com1_form = "For large sample sizes, even small associations can be statistically significant."
    com2_form = "Consider both the strength of association (between 0 and 1) and statistical significance (p-value) in your interpretation."

    print(f"\nData : {what}\n(perp) Cramer V (2) : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\n(perp) Cramer V (2) : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    print(f"(perp) Cramer V (2) : {com1_form}")
    write(f"(perp) Cramer V (2) : {com1_form}")
    print(f"(perp) Cramer V (2) : {com2_form}")
    write(f"(perp) Cramer V (2) : {com1_form}")

    # Intp
    alpha = 0.05
    H0 = f"(perp) H0 : There is no association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(perp) Ha : There is an association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(perp) Cramer V (2) : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Cramer V (2) : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Cramer V (2) : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Cramer V (2) : Fail to reject the null hypothesis:\n{H0}")
    pass

def cram_perp(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    cram_per1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    cram_per2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)