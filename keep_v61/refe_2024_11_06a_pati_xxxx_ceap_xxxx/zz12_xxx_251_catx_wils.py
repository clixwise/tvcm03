import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

# -------------------------------
# Wilson score interval
# -------------------------------

def wilson_score_interval(count, nobs, alpha=0.05):
    """
    Calculate Wilson score interval for a proportion.
    
    Parameters:
    count (int): Number of successes
    nobs (int): Total number of observations
    alpha (float): Significance level (default 0.05 for 95% CI)
    
    Returns:
    tuple: (lower bound, upper bound) of the confidence interval
    """
    n = nobs
    p = count / n
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    return (lower_bound, upper_bound)

def catx_wils(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    if False:
        data = {
            '10-19': [3, 2],
            '20-29': [6, 7],
            '30-39': [10, 23],
            '40-49': [26, 32],
            '50-59': [41, 46],
            '60-69': [35, 46],
            '70-79': [29, 38],
            '80-89': [6, 11],
            '90-99': [0, 1]
        }
        df = pd.DataFrame(data, index=['M', 'F'])

    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Binomial Test for each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nWilson score interval 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nWilson score interval 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
    resu_dict = {}
    for age_bin in df.columns:
        
        male_count = df.loc[indx_cate_nam1, age_bin]
        female_count = df.loc[indx_cate_nam2, age_bin]
        total_count = male_count + female_count
        # Calculate proportion and confidence interval
        proportion = male_count / total_count
        ci_lower, ci_upper = wilson_score_interval(male_count, total_count)
        #print(f"Wilson score interval : Proportion of unilateral CVI: {proportion:.3f} 95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
        # You can also calculate for bilateral cases
        bi_proportion = female_count / total_count
        bi_ci_lower, bi_ci_upper = wilson_score_interval(female_count, total_count)
        #print(f"Wilson score interval : Proportion of bilateral CVI: {bi_proportion:.3f} 95% Confidence Interval: ({bi_ci_lower:.3f}, {bi_ci_upper:.3f})")
        
        # Overlap : yes,no
        overl = "Ha" if ci_upper < bi_ci_lower else "H0"
        
        resu_dict[age_bin] = {
        colu_name: age_bin,
        indx_cate_nam1: male_count,
        indx_cate_nam2: female_count,
        f'{indx_cate_nam1}_proport': proportion,
        f'{indx_cate_nam1}_ci_lowr': ci_lower,
        f'{indx_cate_nam1}_ci_uppr': ci_upper,
        f'{indx_cate_nam2}_proport': bi_proportion,
        f'{indx_cate_nam2}_ci_lowr': bi_ci_lower,
        f'{indx_cate_nam2}_ci_uppr': bi_ci_upper,
        f'overlap': overl
    }
    
    # Create DataFrame from results
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu[f'{indx_cate_nam1}_proport'] = df_resu[f'{indx_cate_nam1}_proport'].apply(frmt)
    df_resu[f'{indx_cate_nam1}_ci_lowr'] = df_resu[f'{indx_cate_nam1}_ci_lowr'].apply(frmt)
    df_resu[f'{indx_cate_nam1}_ci_uppr'] = df_resu[f'{indx_cate_nam1}_ci_uppr'].apply(frmt)
    df_resu[f'{indx_cate_nam2}_proport'] = df_resu[f'{indx_cate_nam2}_proport'].apply(frmt)
    df_resu[f'{indx_cate_nam2}_ci_lowr'] = df_resu[f'{indx_cate_nam2}_ci_lowr'].apply(frmt)
    df_resu[f'{indx_cate_nam2}_ci_uppr'] = df_resu[f'{indx_cate_nam2}_ci_uppr'].apply(frmt)
    print(f"\n---\nData : {what}\nWilson score interval 2024_12_15 [2025_01_17]:\n---")
    write(f"\n---\nData : {what}\nWilson score interval 2024_12_15 [2025_01_17]:\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    pass