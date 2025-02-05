import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import chi2_contingency
    
def mcnemar_test_returns_two_values(table):
    b = table.sum(axis=1) - np.diag(table)
    c = table.sum(axis=0) - np.diag(table)
    statistic = (b - c)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value
def mcnemar_test(table):
    b = table[0, 1]
    c = table[1, 0]
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = chi2_contingency([[b, c], [c, b]])[1]
    return statistic, p_value

# -------------------------------
# Mc Nemar Test of Independence
# -------------------------------
def nema(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):

    # Exec
    print(f"\nData : {what}\nMcNemar : Iterations per {indx_name}")
    write(f"\nData : {what}\nMcNemar : Iterations per {indx_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : There is no significant difference in the presence/absence of the CEAP classification between {indx_name} and {colu_name} classifications."
    Ha = f"Ha : There is a significant difference in the presence/absence of the CEAP classification between {indx_name} and {colu_name} classifications."
    Hx = f"The statistic represents the strength of the disagreement, with higher values indicating stronger disagreement between {indx_name} and {colu_name} classifications."
    for ceap in df.index:
        # Create 2x2 contingency table for each CEAP classification
        table = np.array([[df.loc[ceap, ceap], df.loc[ceap].sum() - df.loc[ceap, ceap]],
                          [df[ceap].sum() - df.loc[ceap, ceap], df.sum().sum() - df.loc[ceap].sum() - df[ceap].sum() + df.loc[ceap, ceap]]])   
        stat, pval = mcnemar_test(table)
        if pval < alpha:
            HV = "Ha"
        else:
            HV = "H0"
        resu_dict[ceap] = {
            'CEAP': ceap,
            'stat': stat,
            'pval': pval,
            'H': HV
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)   
    print(f"\nData : {what}\nMcNemar :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nMcNemar :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"McNemar :{H0}")
    write(f"McNemar :{H0}")
    print(f"McNemar :{Ha}")
    write(f"McNemar :{Ha}")
    print(f"McNemar :{Hx}")
    write(f"McNemar :{Hx}") 
    pass

