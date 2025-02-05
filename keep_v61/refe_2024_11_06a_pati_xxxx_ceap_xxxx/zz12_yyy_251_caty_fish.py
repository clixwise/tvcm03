import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import fisher_exact, norm

# -------------------------------
# Fisher Exact odds ratio
# -------------------------------
'''
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with the sum of the other M,F values
example :
NA :    This That
sexe
M       52   326
F       53   445 sum:876
note : same approach as 'chi2'
'''
def caty_fish(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # An example
    if False:
        data = {
            'NA': [52, 53],
            'C0': [31, 36],
            'C1': [5, 6],
            'C2': [44, 54],
            'C3': [93, 156],
            'C4': [38, 59],
            'C5': [18, 35],
            'C6': [97, 99]
        }
        df = pd.DataFrame(data, index=['Male', 'Female'])
    
    # ----
    # Prec
    # ----
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nFisher Exact odds ratio 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nFisher Exact odds ratio 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are similar for column {colu_name}"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are not similar for column {colu_name}"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for ceap_class in df.columns:
        
        # Exec
        # ----
        observed_males = df.loc[indx_cate_nam1, ceap_class]
        observed_females = df.loc[indx_cate_nam2, ceap_class]
        # Create a DataFrame for Current and Other
        this_ = df[ceap_class]
        # Calculate Other counts by summing all classes except the current one
        that_ = df.sum(axis=1) - this_
        # Create a new DataFrame for Current and Other
        df_ceap = pd.DataFrame({
            'This': this_,
            'That': that_
        })    
        if loop_jrnl:
            print(f"{ceap_class} : {df_ceap} sum:{df_ceap.sum().sum()}")
        this_dict = {index: value for index, value in df_ceap['This'].items()}
        that_dict = {index: value for index, value in df_ceap['That'].items()}
        
        # Perform Fisher's Exact Test
        odds_ratio, pval = fisher_exact(df_ceap)
        stat = odds_ratio
        #
        log_odds_ratio = np.log(odds_ratio)
        np_table = np.array(df_ceap)
        se = np.sqrt(sum(1 / np_table.flatten())) # Calculate standard error
        alpha = 0.05  # or any other significance level you prefer
        ci_lower = np.exp(log_odds_ratio - norm.ppf(1 - alpha / 2) * se)
        ci_upper = np.exp(log_odds_ratio + norm.ppf(1 - alpha / 2) * se)
        
        # Intp
        # ----
        if pval < alpha:
            if loop_jrnl:
                print(f"Fisher Exact odds ratio 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                write(f"Fisher Exact odds ratio 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Fisher Exact odds ratio 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Fisher Exact odds ratio 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Resu
        # ----
        resu_dict[ceap_class] = {
            colu_name: ceap_class,
            f'{indx_cate_nam1}': observed_males,
            f'{indx_cate_nam2}': observed_females,
            f'{indx_cate_nam1}_thi': this_dict[indx_cate_nam1],
            f'{indx_cate_nam1}_tha': that_dict[indx_cate_nam1],
            f'{indx_cate_nam1}_sum': this_dict[indx_cate_nam1]+that_dict[indx_cate_nam1],
            f'{indx_cate_nam2}_thi': this_dict[indx_cate_nam2],
            f'{indx_cate_nam2}_tha': that_dict[indx_cate_nam2],
            f'{indx_cate_nam2}_sum': this_dict[indx_cate_nam2]+that_dict[indx_cate_nam2],
            f'_sum': this_dict[indx_cate_nam1]+that_dict[indx_cate_nam1]+this_dict[indx_cate_nam2]+that_dict[indx_cate_nam2],
            'stat (odds_ratio)': stat,
            'pval': pval,
            'ci_lower':  ci_lower,
            'ci_upper': ci_upper,
            'H': HV
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat (odds_ratio)'] = df_resu['stat (odds_ratio)'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
        
    print(f"\n---\nData : {what}\nFisher Exact odds ratio 2024_12_15 [2025_01_17] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nFisher Exact odds ratio 2024_12_15 [2025_01_17] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
 
