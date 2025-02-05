
import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats


# -------------------------------
# Fisher Exact odd's ratio
# -------------------------------
def fish_mist(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    
    # Exec
    oddsratio, pval = stats.fisher_exact(df)
    stat = oddsratio
    if oddsratio > 1:
        odds_intp = f"The odds of being in a particular '{colu_name}' group are higher for '{indx_cate_nam2}' compared to '{indx_cate_nam1}'."
    elif oddsratio < 1:
        odds_intp = f"The odds of being in a particular '{colu_name}' group are higher for '{indx_cate_nam1}' compared to '{indx_cate_nam2}'."
    else:
        odds_intp = f"The odds of being in a particular '{colu_name}' group are the same for '{indx_cate_nam1}' and '{indx_cate_nam2}'."

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\n(mist) Fisher Exact odd's ratio : Stat:{stat_form} Intp: Pval:{pval_form}\nOdds intp:{odds_intp}")
    write(f"\nData : {what}\n(mist) Fisher Exact odd's ratio : Stat:{stat_form} Intp: Pval:{pval_form}\nOdds intp:{odds_intp}")

    # Intp
    alpha = 0.05
    H0 = f"(mist) H0 : There is no association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(mist) Ha : There is an association between the '{colu_name}' categories for the '{indx_name}' categories.\n({colu_cate_list}) vs ({indx_cate_list})"
    if pval < alpha:
        print(f"(mist) Fisher Exact odd's ratio : Reject the null hypothesis:\n{Ha}")
        write(f"(mist) Fisher Exact odd's ratio : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(mist) Fisher Exact odd's ratio : Fail to reject the null hypothesis:\n{H0}")
        write(f"(mist) Fisher Exact odd's ratio : Fail to reject the null hypothesis:\n{H0}")
    pass
    