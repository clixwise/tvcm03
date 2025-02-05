import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import fisher_exact, norm
from scipy import stats

# -------------------------------
# Fisher Exact odds ratio Test of Independence
# -------------------------------
def catx_fish_odds(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
         
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]   
    
    # ----
    # Test 
    # ----
    # Iterate over each age bin and perform Fisher's Exact Test
    print(f"\nData : {what}\nFisher Exact odds ratio : Iterations per {colu_name}")
    write(f"\nData : {what}\nFisher Exact odds ratio : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : There is no association between the '{colu_name}' and the counts for '{indx_name}' groups\nThis suggests that the odds ratio is not significantly different from 1."
    Ha = f"Ha : There is an association between the '{colu_name}' and the counts for '{indx_name}' groups\nThis suggests that the odds ratio is significantly different from 1."
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        # Construct the 2x2 table for Fisher's Test for the current age_bin
        table = [[df.loc[indx_cate_nam1, age_bin], sum(df.loc[indx_cate_nam1]) - df.loc[indx_cate_nam1, age_bin]],
                 [df.loc[indx_cate_nam2, age_bin], sum(df.loc[indx_cate_nam2]) - df.loc[indx_cate_nam2, age_bin]]]     
        # Perform Fisher's Exact Test
        odds_ratio, pval = fisher_exact(table)
        stat = odds_ratio
        #
        log_odds_ratio = np.log(odds_ratio)
        np_table = np.array(table)
        se = np.sqrt(sum(1 / np_table.flatten())) # Calculate standard error
        alpha = 0.05  # or any other significance level you prefer
        ci_lower = np.exp(log_odds_ratio - norm.ppf(1 - alpha / 2) * se)
        ci_upper = np.exp(log_odds_ratio + norm.ppf(1 - alpha / 2) * se)
        
        # Intp
        if pval < alpha:
            print(f"Fisher Exact odds ratio : Reject the null hypothesis:\n{Ha}")
            write(f"Fisher Exact odds ratio : Reject the null hypothesis:\n{Ha}")
            HT = 'Ha'
            HV = Ha
        else:
            print(f"Fisher Exact odds ratio : Fail to reject the null hypothesis:\n{H0}")
            write(f"Fisher Exact odds ratio : Fail to reject the null hypothesis:\n{H0}")
            HT = 'H0'
            HV = H0
            
        # Store the result
        resu_dict[age_bin] = {
            'odds_ratio': stat,
            'pval': pval,
            'ci_lower':  ci_lower,
            'ci_upper': ci_upper,
            'H': HT
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    
    print(f"\nData : {what}\nFisher Exact odds ratio :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nFisher Exact odds ratio :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass