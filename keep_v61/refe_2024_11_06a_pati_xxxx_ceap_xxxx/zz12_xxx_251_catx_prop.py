import numpy as np
import pandas as pd
from util_file_mngr import write
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------
# Proportion test of Independence
# -------------------------------
def pro1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

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

        df2 = pd.DataFrame(data, index=['M', 'F'])

    # Total counts of males and females across all bins
    total_males = df.loc[indx_cate_nam1].sum()
    total_females = df.loc[indx_cate_nam2].sum()

    # ----
    # Test 1
    # ----
    # Perform Z-test for proportions in each age bin
    print(f"\nData : {what}\nProportional (1) : Iterations per {colu_name}")
    write(f"\nData : {what}\nProportional (1) : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are equal for the given {colu_name} (Two-tailed test)"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are unequal for the given {colu_name} (Two-tailed test)"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        
        # Observed counts for males and females in this age bin
        observed_males = df.loc[indx_cate_nam1, age_bin]
        observed_females = df.loc[indx_cate_nam2, age_bin]
        count = [observed_males, observed_females]
        # Total counts for males and females across all bins
        nobs = [total_males, total_females]
        # Perform Proportion Z-test
        z_statistic, pval = proportions_ztest(count, nobs)
        stat = z_statistic
        
        # Intp
        if pval < alpha:
            print(f"Proportion (1): Reject the null hypothesis:\n{Ha}")
            write(f"Proportion (1): Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            print(f"Proportion (1): Fail to reject the null hypothesis:\n{H0}")
            write(f"Proportion (1): Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Store the result
        resu_dict[age_bin] = {
            colu_name: age_bin,
            f'{indx_cate_nam1}': observed_males,
            f'{indx_cate_nam2}': observed_females,
            f'tot{indx_cate_nam1}': total_males,
            f'tot{indx_cate_nam2}': total_females,
            f'pro{indx_cate_nam1}': observed_males/total_males,
            f'pro{indx_cate_nam2}': observed_females/total_females,
            'stat': stat,
            'pval': pval,
            'H': HV
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    df_resu[f'pro{indx_cate_nam1}'] = df_resu[f'pro{indx_cate_nam1}'].apply(frmt)
    df_resu[f'pro{indx_cate_nam2}'] = df_resu[f'pro{indx_cate_nam2}'].apply(frmt)
        
    print(f"\nData : {what}\nProportional (1) :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nProportional (1) :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
    
def pro2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):    
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    
    # ----
    # Test 2
    # ----
    # Perform Z-test for proportions in each age bin
    print(f"\nData : {what}\nProportional (2) : Iterations per {colu_name} and {indx_name}")
    write(f"\nData : {what}\nProportional (2) : Iterations per {colu_name} and {indx_name}")
    F_sum = df.loc[indx_cate_nam2].sum()
    T_sum = df.sum().sum()
    F_prop = F_sum / T_sum
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are equal for the given {colu_name} (Two-tailed test)"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are unequal for the given {colu_name} (Two-tailed test)"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        for gender in df.index:
            
            n_gender = df.loc[gender, age_bin]
            n_total = df[age_bin].sum()        
            # Use overall gender proportion as expected proportion
            expe_prop = F_prop if gender == indx_cate_nam2 else 1 - F_prop
            obsv_prop = n_gender / n_total         
            z_statistic, pval = proportions_ztest(n_gender, n_total, value=expe_prop)           
            stat = z_statistic
        
            # Intp
            if pval < alpha:
                print(f"Proportion (2): Reject the null hypothesis:\n{Ha}")
                write(f"Proportion (2): Reject the null hypothesis:\n{Ha}")
                HV = "Ha"
                HT = Ha
            else:
                print(f"Proportion (2): Fail to reject the null hypothesis:\n{H0}")
                write(f"Proportion (2): Fail to reject the null hypothesis:\n{H0}")
                HV = "H0"
                HT = H0
            
            # Store the result
            resu_dict[f'{age_bin}_{gender}'] = {
                colu_name: age_bin,
                indx_name: gender,
                'coun': n_gender,
                'tota': n_total,
                'obs_prop': obsv_prop,
                'exp_prop': expe_prop,
                'stat': stat,
                'pval': pval,
                'H': HV
            }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)

    print(f"\nData : {what}\nProportional (2) :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nProportional (2) :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def catx_prop(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    pro1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pro2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pass
