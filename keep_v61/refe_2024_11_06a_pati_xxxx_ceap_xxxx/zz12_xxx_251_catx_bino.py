import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import binomtest

# -------------------------------
# Binomial Test of Independence
# -------------------------------
def bin1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

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
    # Test 1
    # ----
    # Perform Binomial Test for each age bin
    print(f"\nData : {what}\nBinomial (1) : Iterations per {colu_name}")
    write(f"\nData : {what}\nBinomial (1) : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : Observed '{indx_cate_nam1}' and '{indx_cate_nam2}' counts for '{colu_name}' do not differ from the expected count"
    Ha = f"Ha : Observed '{indx_cate_nam1}' and '{indx_cate_nam2}' counts for '{colu_name}' differ from the expected count"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        male_count = df.loc[indx_cate_nam1, age_bin]
        female_count = df.loc[indx_cate_nam2, age_bin]
        total_count = male_count + female_count
        
        # Perform Binomial Test assuming p = 0.5
        result = binomtest(male_count, total_count, p=0.5, alternative='two-sided')
        stat = result.statistic
        pval = result.pvalue
        
        # Determine H
        if pval < alpha:
            HV = 'Ha'
            HT = f"Observed {indx_cate_nam1} count in {age_bin} '{colu_name}' differs from the expected count"
        else:
            HV = 'H0'
            HT = f"Observed {indx_cate_nam1} count in {age_bin} '{colu_name}' does not differ from the expected count"
        
        resu_dict[age_bin] = {
        colu_name: age_bin,
        indx_cate_nam1: male_count,
        indx_cate_nam2: female_count,
        'tota': total_count,
        'stat': stat,
        'pval': pval,
        'H' : HV
    }
    
    # Create DataFrame from results
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    
    print(f"\nData : {what}\nBinomial (1) :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nBinomial (1) :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def bin2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Test 2
    # ----
    # Perform Binomial Test in each age bin
    print(f"\nData : {what}\nBinomial (2) : Iterations per {colu_name} and {indx_name}")
    write(f"\nData : {what}\nBinomial (2) : Iterations per {colu_name} and {indx_name}")
    F_sum = df.loc[indx_cate_nam2].sum()
    T_sum = df.sum().sum()
    F_prop = F_sum / T_sum
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : Observed '{indx_cate_nam1}' or '{indx_cate_nam2}' count for '{colu_name}' does not differ from the expected count"
    Ha = f"Ha : Observed '{indx_cate_nam1}' or '{indx_cate_nam2}' count for '{colu_name}' differs from the expected count"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        for gender in df.index:
            
            n_gender = df.loc[gender, age_bin]
            n_total = df[age_bin].sum()        
            # Use overall gender proportion as expected proportion
            expe_prop = F_prop if gender == indx_cate_nam2 else 1 - F_prop
            obsv_prop = n_gender / n_total         
            result = binomtest(n_gender, n_total, expe_prop) 
            # Calculate the confidence interval
            ci_lowr, ci_uppr = result.proportion_ci()         
            # Determine H
            stat = result.statistic
            pval = result.pvalue
            if pval < 0.05:
                hypothesis = 'Ha'
            else:
                hypothesis = 'H0'
                    
            resu_dict[f'{age_bin}_{gender}'] = {
                colu_name: age_bin,
                indx_name: gender,
                'count': n_gender,
                'tota': n_total,
                'stat': stat,
                'pval': pval,
                'H' : hypothesis,
                'obse_prop': obsv_prop,
                'expe_prop': expe_prop,
                'ci_lowr' : ci_lowr,
                'ci_uppr' : ci_uppr
            }

    # Create DataFrame from results
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['obse_prop'] = df_resu['obse_prop'].apply(frmt)
    df_resu['expe_prop'] = df_resu['expe_prop'].apply(frmt)
    df_resu['ci_lowr'] = df_resu['ci_lowr'].apply(frmt)
    df_resu['ci_uppr'] = df_resu['ci_uppr'].apply(frmt)
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    
    print(f"\nData : {what}\nBinomial (2) :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nBinomial (2) :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"{df_resu}")
        write(f"{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def catx_bino(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    bin1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    bin2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pass