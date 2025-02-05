import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

def cochran_armitage_test(table):
    """
    Perform Cochran-Armitage test for trend
    table: 2xk contingency table where rows are groups and columns are ordinal categories
    """
    n_rows, n_cols = table.shape
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    grand_total = table.sum()
    
    scores = np.arange(n_cols)
    expected = (row_totals[:, np.newaxis] * col_totals) / grand_total
    
    numerator = np.sum(scores * (table[1] - expected[1]))
    denominator = np.sqrt(np.sum(scores**2 * (col_totals / grand_total) * (1 - col_totals / grand_total)) * np.prod(row_totals) / grand_total)
    
    z_statistic = numerator / denominator
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    return z_statistic, p_value

def cocharmi(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    # Exec
    print (df.values)
    stat, pval = cochran_armitage_test(df.values)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nCochran-Armitage : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nCochran-Armitage : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    # Mistral
    # H0 = "H0 : there is no trend in the binary outcomes across the different categories for the left and right sides."
    # Ha = "Ha : there is a trend in the binary outcomes across the different categories for the left and right sides."
    
    #H0 = "There is no significant evidence of a linear trend in the proportion of pet owners across age groups between genders."
    H0 = f"H0 : There is no linear trend in the proportion of observed values across '{colu_name}' between '{indx_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    #Ha = "There is a significant linear trend in the proportion of pet owners across age groups between genders"
    Ha = f"Ha : There is a linear trend in the proportion of observed values across '{colu_name}' between '{indx_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Cochran-Armitage : Reject the null hypothesis:\n{Ha}")
        write(f"Cochran-Armitage : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Cochran-Armitage : Fail to reject the null hypothesis:\n{H0}")
        write(f"Cochran-Armitage : Fail to reject the null hypothesis:\n{H0}")
    pass