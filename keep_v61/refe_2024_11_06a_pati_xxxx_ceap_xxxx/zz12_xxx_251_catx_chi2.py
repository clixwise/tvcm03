import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

# -------------------------------
# Chi2
# -------------------------------
def catx_chi2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Test 1
    # ----
    # Perform Z-test for proportions in each age bin
    print(f"\nData : {what}\nChi2 : Iterations per {colu_name}")
    write(f"\nData : {what}\nChi2 : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} approach a 50-50 split for the given {colu_name}"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are distant from a 50-50 split for the given {colu_name}"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        
        # Observed counts for males and females in this age bin
        observed_males = df.loc[indx_cate_nam1, age_bin]
        observed_females = df.loc[indx_cate_nam2, age_bin]
        expected_half = (observed_males + observed_females) / 2
        count = [observed_males, observed_females]
        nobs = [expected_half , expected_half]
        chi2, pval = stats.chisquare(count, nobs)
        stat = chi2
        
        # Intp
        if pval < alpha:
            print(f"Chi2: Reject the null hypothesis:\n{Ha}")
            write(f"Chi2: Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            print(f"Chi2: Fail to reject the null hypothesis:\n{H0}")
            write(f"Chi2: Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Store the result
        resu_dict[age_bin] = {
            colu_name: age_bin,
            f'{indx_cate_nam1}': observed_males,
            f'{indx_cate_nam2}': observed_females,
            f'tot{indx_cate_nam1}': observed_males,
            f'tot{indx_cate_nam2}': observed_females,
            f'tot50%': expected_half,
            'stat': stat,
            'pval': pval,
            'H': HV
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    df_resu['tot50%'] = df_resu['tot50%'].apply(frmt)
        
    print(f"\nData : {what}\nChi2 :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nChi2 :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
 
