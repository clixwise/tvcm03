
import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency

# Cramer's V calculation
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def interpret_cramers_v(v, df):
    """
    Interpret Cramer's V value based on degrees of freedom.
    
    :param v: Cramer's V value
    :param df: Degrees of freedom (min(rows-1, columns-1))
    :return: String interpretation
    """
    # General interpretation thresholds
    general_thresholds = {
        (0.00, 0.10): "negligible",
        (0.10, 0.30): "small",
        (0.30, 0.50): "medium",
        (0.50, 0.70): "large",
        (0.70, 1.01): "very large"
    }
    
    # Cohen's thresholds based on degrees of freedom
    cohen_thresholds = {
        1: [(0.10, "small"), (0.30, "medium"), (0.50, "large")],
        2: [(0.07, "small"), (0.21, "medium"), (0.35, "large")],
        3: [(0.06, "small"), (0.17, "medium"), (0.29, "large")],
        4: [(0.05, "small"), (0.15, "medium"), (0.25, "large")],
        5: [(0.04, "small"), (0.13, "medium"), (0.22, "large")]
    }
    
    # Use Cohen's thresholds if df is 5 or less, otherwise use general interpretation
    dof_cram = min(df.shape) - 1  # degrees of freedom for Cramer are different from dof for chi2
    if dof_cram <= 5:
        thresholds = cohen_thresholds[dof_cram]
        for threshold, interpretation in thresholds:
            if v < threshold:
                return interpretation
        return "Very large"
    else:
        for (lower, upper), interpretation in general_thresholds.items():
            if lower <= v < upper:
                return interpretation
    
    return "Invalid V value"

# -------------------------------
# Cramer V
# -------------------------------
def cram_perp(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    print (df)
    stat = effect_size = cramers_v(df)
    cram_intp = interpret_cramers_v(stat, df)
    # Calculate p-value for Cramer's V
    chi2, pval, dof, expected = chi2_contingency(df)
 
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\n(perp) Cramer V : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\n(perp) Cramer V : Stat:{stat_form} Intp: asso(effect size):{cram_intp} Pval:{pval_form} Dof:{dof}")

    # Intp
    alpha = 0.05
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"(perp) H0 : There is no association between the '{colu_name}' and the counts for '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(perp) Ha : There is an association between the '{colu_name}' and the counts for '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(perp) Cramer V : Reject the null hypothesis:\n{Ha}")
        write(f"(perp) Cramer V : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(perp) Cramer V : Fail to reject the null hypothesis:\n{H0}")
        write(f"(perp) Cramer V : Fail to reject the null hypothesis:\n{H0}")
    pass
    