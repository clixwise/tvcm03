from datetime import datetime
import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from util_file_inpu_ceap import inp1
from util_file_mngr import set_file_objc, write

import numpy as np
from scipy.stats import permutation_test

def inpu(file_path):
        
    # ----
    # Exec
    # ----
    df1, df2, df3 = inp1(file_path)

    # ----
    # Exit
    # ----
    return df1, df2, df3

def spea_mann(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    trac = True

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df = df_line  
    # Remove 'NA' if it's the first element
    if False:
        if parm_ceap_list and parm_ceap_list[0] == 'NA':
            parm_ceap_list = parm_ceap_list[1:]   
    # Create numeric equivalent
    parm_ceap_dict = {value: index for index, value in enumerate(parm_ceap_list)}
    df['ceap_nume'] = df['ceap'].map(parm_ceap_dict)
    df['age_rank'] = df['age'].rank() # not used but could be
    print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # Calculate Spearman's rank correlation for all three tests
    loop_jrnl = False
    if loop_jrnl:
        print(f"\nData : {what}\Spearman [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
        write(f"\nData : {what}\Spearman [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : rhos == 0 : no monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Ha = f"Ha : rhos <> 0 :  a monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Hx = f"({parm_ceap}) vs ({parm_agee} for {parm_sexe})"

    def test_func(df, sexe):
        test_list = ['two-sided', 'greater', 'less']
        test_expl = ['two-side', 'positive', 'negative']
        for indx, test in enumerate(test_list):
            
            # Exec
            rho, pval = stats.spearmanr(df['age'], df['ceap_nume'], alternative=test)
            stat = rho
            
            # Intp rho
            def rho_intp(rho): 
                rho = max(-1, min(1, rho)) # Ensure rho is within -1 to 1 range             
                scaled_rho = abs(rho) * 5 # Convert to 0-5 scale        
                rounded_rho = round(scaled_rho) # Round to nearest integer             
                sign = '-' if rho < 0 else '' # Determine sign
                return f"{sign}{rounded_rho}/5"
            intp = rho_intp(stat)

            # Intp pval
            if pval < alpha:
                if loop_jrnl:
                    print(f"Spearman [2025_01_19] : test '{test}': Reject the null hypothesis:\n{Ha}")
                    write(f"Spearman [2025_01_19] : test '{test}': Reject the null hypothesis:\n{Ha}")
                HV = "Ha"
                HT = Ha
            else:
                if loop_jrnl:
                    print(f"Spearman [2025_01_19] : test '{test}': Fail to reject the null hypothesis:\n{H0}")
                    write(f"Spearman [2025_01_19] : test '{test}': Fail to reject the null hypothesis:\n{H0}")
                HV = "H0"
                HT = H0
            
            # Done
            resu_dict[f'{sexe} {test_expl[indx]} ({test})'] = {
                'stat': stat,
                'intp': intp,
                'pval': pval,
                'H': HV
            }
            
        # Exit
        return resu_dict
    
    def test_main(df):
        df2 = df.copy()
        resu_dict_A = test_func(df2, 'A')
        #
        df2 = df[df[parm_sexe] == sexM]
        resu_dict_M = test_func(df2, sexM)
        print (df2)
        #
        df2 = df[df[parm_sexe] == sexF]
        resu_dict_F = test_func(df2, sexF)
        print (df2)
        #
        df_resu = pd.DataFrame.from_dict({**resu_dict_A, **resu_dict_M, **resu_dict_F}, orient='index')
        frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
        df_resu['stat'] = df_resu['stat'].apply(frmt)
        df_resu['pval'] = df_resu['pval'].apply(frmt)
        return df_resu
    
    # Create an empty dictionary to store the results
    result_dict = {}
    # Iterate through age thresholds from 10 to 90 in steps of 10
    for threshold in range(10, 80, 10):
        # Filter the DataFrame for ages greater than the current threshold
        df_filt = df[df['age'] >= threshold]
        df_resu = test_main(df_filt)
        # Add the filtered result to the dictionary with the threshold as the key
        result_dict[f'age_{threshold}+ ceap count:{len(df_filt)}'] = df_resu

    print(f"\n---\nData : {what}\nSpearman [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nSpearman [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        for key in result_dict:
            print(f"\n'{key}'\n{result_dict[key]}")
            write(f"\n'{key}'\n{result_dict[key]}")
            xlsx = False
            if xlsx: 
                script_name = os.path.basename(__file__)
                file_name = f'spearmann_.xlsx'
                result_dict[key].to_excel(file_name, index=False)
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def kend_tau_(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    trac = True

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df = df_line  
    # Remove 'NA' if it's the first element
    if False:
        if parm_ceap_list and parm_ceap_list[0] == 'NA':
            parm_ceap_list = parm_ceap_list[1:]   
    # Create numeric equivalent
    parm_ceap_dict = {value: index for index, value in enumerate(parm_ceap_list)}
    df['ceap_nume'] = df['ceap'].map(parm_ceap_dict)
    df['age_rank'] = df['age'].rank()
    print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # Calculate Kendall's rank correlation for all three tests
    loop_jrnl = False
    if loop_jrnl:
        print(f"\nData : {what}\Kendall [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
        write(f"\nData : {what}\Kendall [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : tau == 0 : no monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Ha = f"Ha : tau <> 0 :  a monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Hx = f"({parm_ceap}) vs ({parm_agee} for {parm_sexe})"

    def test_func(df, sexe):
        test_list = ['two-sided', 'greater', 'less']
        test_expl = ['two-side', 'positive', 'negative']
        for indx, test in enumerate(test_list):
            
            # Exec
            tau, pval = stats.kendalltau(df['age_rank'], df['ceap_nume'], alternative=test)
            stat = tau
            
            # Intp tau
            def tau_intp(tau): 
                tau = max(-1, min(1, tau)) # Ensure tau is within -1 to 1 range             
                scaled_tau = abs(tau) * 5 # Convert to 0-5 scale        
                rounded_tau = round(scaled_tau) # Round to nearest integer             
                sign = '-' if tau < 0 else '' # Determine sign
                return f"{sign}{rounded_tau}/5"
            intp = tau_intp(stat)

            # Intp pval
            if pval < alpha:
                if loop_jrnl:
                    print(f"Kendall [2025_01_19] : test '{test}': Reject the null hypothesis:\n{Ha}")
                    write(f"Kendall [2025_01_19] : test '{test}': Reject the null hypothesis:\n{Ha}")
                HV = "Ha"
                HT = Ha
            else:
                if loop_jrnl:
                    print(f"Kendall [2025_01_19] : test '{test}': Fail to reject the null hypothesis:\n{H0}")
                    write(f"Kendall [2025_01_19] : test '{test}': Fail to reject the null hypothesis:\n{H0}")
                HV = "H0"
                HT = H0
            
            # Done
            resu_dict[f'{sexe} {test_expl[indx]} ({test})'] = {
                'stat': stat,
                'intp': intp,
                'pval': pval,
                'H': HV
            }
            
        # Exit
        return resu_dict
    
    def test_main(df):
        df2 = df.copy()
        resu_dict_A = test_func(df2, 'A')
        df2 = df[df[parm_sexe] == sexM]
        resu_dict_M = test_func(df2, sexM)
        df2 = df[df[parm_sexe] == sexF]
        resu_dict_F = test_func(df2, sexF)
        df_resu = pd.DataFrame.from_dict({**resu_dict_A, **resu_dict_M, **resu_dict_F}, orient='index')
        frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
        df_resu['stat'] = df_resu['stat'].apply(frmt)
        df_resu['pval'] = df_resu['pval'].apply(frmt)
        return df_resu
    
    # Create an empty dictionary to store the results
    result_dict = {}
    # Iterate through age thresholds from 10 to 90 in steps of 10
    for threshold in range(10, 80, 10):
        # Filter the DataFrame for ages greater than the current threshold
        df_filt = df[df['age'] >= threshold]
        df_resu = test_main(df_filt)
        # Add the filtered result to the dictionary with the threshold as the key
        result_dict[f'age_{threshold}+ ceap count:{len(df_filt)}'] = df_resu

    print(f"\n---\nData : {what}\nKendall [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nKendall [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        for key in result_dict:
            print(f"\n'{key}'\n{result_dict[key]}")
            write(f"\n'{key}'\n{result_dict[key]}")
            xlsx = False
            if xlsx: 
                script_name = os.path.basename(__file__)
                file_name = f'spearmann_.xlsx'
                result_dict[key].to_excel(file_name, index=False)
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def good_krus_gamm(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    if False:
        data = {
            'age': [25, 30, 22, 30, 28]
        }
        df = pd.DataFrame(data)
        df['age_rank'] = df['age'].rank(method='first').astype(int)
        print(df)
    
    trac = True

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df = df_line  
    # Remove 'NA' if it's the first element
    if False:
        if parm_ceap_list and parm_ceap_list[0] == 'NA':
            parm_ceap_list = parm_ceap_list[1:]   
    # Create numeric equivalent
    parm_ceap_dict = {value: index for index, value in enumerate(parm_ceap_list)}
    df['ceap_nume'] = df['ceap'].map(parm_ceap_dict)
    df['age_rank'] = df['age'].rank(method='first').astype(int)
    print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # Calculate Goodman Kruskal Gamma's rank correlation for all three tests
    loop_jrnl = False
    if loop_jrnl:
        print(f"\nData : {what}\Goodman Kruskal Gamma [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
        write(f"\nData : {what}\Goodman Kruskal Gamma [2025_01_19] : Iterations per {parm_ceap}, {parm_agee}, {parm_sexe}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : gamma == 0 : no monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Ha = f"Ha : gamma <> 0 :  a monotonic relationship exists between {parm_ceap}, {parm_agee} for {parm_sexe}"
    Hx = f"({parm_ceap}) vs ({parm_agee} for {parm_sexe})"

    def test_func(df, sexe):

        # Function to calculate Goodman and Kruskal's Gamma
        def goodman_kruskal_gamma(ranking1, ranking2):
            contingency_table = np.zeros((max(ranking1), max(ranking2)))
            for r1, r2 in zip(ranking1, ranking2):
                contingency_table[r1-1, r2-1] += 1

            n_concordant = 0
            n_discordant = 0
            n_ties_x = 0
            n_ties_y = 0

            for i in range(len(contingency_table)):
                for j in range(len(contingency_table[i])):
                    for k in range(i+1, len(contingency_table)):
                        for l in range(j+1, len(contingency_table[k])):
                            n_concordant += contingency_table[i, j] * contingency_table[k, l]
                            n_discordant += contingency_table[i, l] * contingency_table[k, j]
                    n_ties_x += np.sum(contingency_table[i, :]) * (np.sum(contingency_table[i, :]) - 1) / 2
                    n_ties_y += np.sum(contingency_table[:, j]) * (np.sum(contingency_table[:, j]) - 1) / 2

            n_total = n_concordant + n_discordant + n_ties_x + n_ties_y
            gamma = (n_concordant - n_discordant) / (n_concordant + n_discordant)
            return gamma

        # Exec
        stat = goodman_kruskal_gamma(df['age_rank'], df['ceap_nume'])
        #
        too_long = True
        if not too_long:
            def statistic(ranking1, ranking2):
                return goodman_kruskal_gamma(ranking1, ranking2)
            res = permutation_test((df['age_rank'], df['ceap_nume']), statistic, n_resamples=1000, vectorized=False, alternative='two-sided')
            pval = res.pvalue
        else:
            pval = 9999
        
        # Intp tau
        def tau_intp(tau): 
            tau = max(-1, min(1, tau)) # Ensure tau is within -1 to 1 range             
            scaled_tau = abs(tau) * 5 # Convert to 0-5 scale        
            rounded_tau = round(scaled_tau) # Round to nearest integer             
            sign = '-' if tau < 0 else '' # Determine sign
            return f"{sign}{rounded_tau}/5"
        intp = tau_intp(stat)

        # Intp pval
        if pval < alpha:
            if loop_jrnl:
                print(f"Goodman Kruskal Gamma [2025_01_19] : Reject the null hypothesis:\n{Ha}")
                write(f"Goodman Kruskal Gamma [2025_01_19] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Goodman Kruskal Gamma [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Goodman Kruskal Gamma [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Done
        resu_dict[sexe] = {
            'stat': stat,
            'intp': intp,
            'pval': pval,
            'H': HV
        }
            
        # Exit
        return resu_dict
    
    def test_main(df):
        df2 = df.copy()
        resu_dict_A = test_func(df2, 'A')
        df2 = df[df[parm_sexe] == sexM]
        resu_dict_M = test_func(df2, sexM)
        df2 = df[df[parm_sexe] == sexF]
        resu_dict_F = test_func(df2, sexF)
        df_resu = pd.DataFrame.from_dict({**resu_dict_A, **resu_dict_M, **resu_dict_F}, orient='index')
        frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
        df_resu['stat'] = df_resu['stat'].apply(frmt)
        df_resu['pval'] = df_resu['pval'].apply(frmt)
        return df_resu
    
    # Create an empty dictionary to store the results
    result_dict = {}
    # Iterate through age thresholds from 10 to 90 in steps of 10
    for threshold in range(10, 20, 10):
        # Filter the DataFrame for ages greater than the current threshold
        df_filt = df[df['age'] >= threshold]
        df_resu = test_main(df_filt)
        # Add the filtered result to the dictionary with the threshold as the key
        result_dict[f'age_{threshold}+ ceap count:{len(df_filt)}'] = df_resu

    print(f"\n---\nData : {what}\nGoodman Kruskal Gamma [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nGoodman Kruskal Gamma [2025_01_19] : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        for key in result_dict:
            print(f"\n'{key}'\n{result_dict[key]}")
            write(f"\n'{key}'\n{result_dict[key]}")
            xlsx = False
            if xlsx: 
                script_name = os.path.basename(__file__)
                file_name = f'spearmann_.xlsx'
                result_dict[key].to_excel(file_name, index=False)
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def mann_whit(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    trac = False

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    def ope1(df_line, ages_c_labl, ages_t_labl):
        df_agg = df_line #.groupby(['doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()
        resu_dict = {}
        # print (ages_t)
        for ceap in parm_ceap_list:
            if ages_c_labl == 'A' and ages_t_labl == 'A':
                ages_c = df_agg[(df_agg[parm_ceap] == ceap)][parm_agee]
                ages_t = df_agg[parm_agee]
            elif ages_c_labl == sexM and ages_t_labl == sexF:
                ages_c = df_agg[(df_agg['ceap'] == ceap) & (df_agg[parm_sexe] == ages_c_labl)]['age']
                ages_t = df_agg[(df_agg['ceap'] == ceap) & (df_agg[parm_sexe] == ages_t_labl)]['age']
            else:
                raise Exception()
            mean_ages_c = ages_c.mean()
            mean_ages_t = ages_t.mean()    
            if trac:
                print(f"{ceap} ages_c_labl : {ages_c_labl} : ages_c.size={len(ages_c)} ; ages_t_labl : {ages_t_labl} : ages_t.size={len(ages_t)}")
                write(f"{ceap} ages_c_labl : {ages_c_labl} : ages_c.size={len(ages_c)} ; ages_t_labl : {ages_t_labl} : ages_t.size={len(ages_t)}")
            if len(ages_c) > 0 and len(ages_t) > 0:  # Only run the test if both groups have data
                stat, pval = mannwhitneyu(ages_c, ages_t, alternative='two-sided')
                resu_dict[ceap] = {'iden': f'{ages_c_labl}{ages_t_labl}', 'ages_c.mean': mean_ages_c, 'ages_t.mean': mean_ages_t, 'ages_c.size': len(ages_c), 'ages_t.size': len(ages_t), 'stat': stat, 'CEAP': ceap, 'pval': pval}
            else:
                resu_dict[ceap] = {'iden': f'{ages_c_labl}{ages_t_labl}', 'ages_c.mean': None, 'ages_t.mean': None, 'ages_c.size': len(ages_c) , 'ages_t.size': len(ages_t), 'stat': None, 'CEAP': ceap, 'pval': None}
        df_resu = pd.DataFrame.from_dict(resu_dict, orient='index').dropna()
        return df_resu
    
    df_resu_AA = ope1(df_line, 'A', 'A')
    df_resu_MF = ope1(df_line, sexM, sexF)
    
    def ope2(df_resu):
        # ----
        # Corr
        # ----
        pvals = df_resu['pval'].values
        print (pvals)
        # FDR correction (Benjamini-Hochberg)
        try:
            _, p_corr_fdr, _, _ = multipletests(pvals, method='fdr_bh')
        except Exception as e:
            p_corr_fdr = None
        # Bonferroni correction
        try:
            _, p_corr__bonferroni, _, _ = multipletests(pvals, method='bonferroni')
        except Exception as e:
            p_corr__bonferroni = None
        # Add corrected p-values to the results
        df_resu['pval_corr_fdr'] = p_corr_fdr
        df_resu['pval_corr__bonferroni'] = p_corr__bonferroni
        # Step 4: Display results
        df_resu['Hx'] = df_resu['pval'].apply(lambda x: None if not x else ('H0' if x > 0.05 else 'Ha'))
        df_resu['Hx_corr_fdr'] = df_resu['pval_corr_fdr'].apply(lambda x: None if not x else ('H0' if x > 0.05 else 'Ha'))
        df_resu['Hx_corr__bonferroni'] = df_resu['pval_corr__bonferroni'].apply(lambda x: None if not x else ('H0' if x > 0.05 else 'Ha'))
    
        # ----
        # Intp
        # ----
        Hx = f"(open) Mann-Whitney U : assesses whether, for each CEAP, the distribution of ages differs significantly between the C(EAP) Cx: C(EAP)C0..C6, without assuming normality."
        H0 = f"H0 : (open) Mann-Whitney U : the distribution of ages does not differ significantly between the C(EAP) Cx: C(EAP)C0..C6, without assuming normality."
        Ha = f"Ha : (open) Mann-Whitney U : the distribution of ages differs significantly between the C(EAP) Cx: C(EAP)C0..C6, without assuming normality."
           
        frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
        df_resu['ages_c.mean'] = df_resu['ages_c.mean'].apply(frmt)
        df_resu['ages_t.mean'] = df_resu['ages_t.mean'].apply(frmt)
        df_resu['stat'] = df_resu['stat'].apply(frmt)
        df_resu['pval'] = df_resu['pval'].apply(frmt)
        df_resu['pval_corr_fdr'] = df_resu['pval_corr_fdr'].apply(frmt)
        df_resu['pval_corr__bonferroni'] = df_resu['pval_corr__bonferroni'].apply(frmt)
        
        results_df_sorted = df_resu.sort_values(by='pval', ascending=True)
        print(f"\n---\nData : {what}\n(open) Mann-Whitney U 2024_12_15 : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
        write(f"\n---\nData : {what}\n(open) Mann-Whitney U 2024_12_15 : {parm_ceap}, {parm_agee}, {parm_sexe}, {parm_ceap_list}\n{H0}\n{Ha}\n---")
        with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
            print(f"\n{results_df_sorted}")
            write(f"\n{results_df_sorted}")
        print(f"{Hx}")
        write(f"{Hx}")
    
    ope2(df_resu_AA)
    ope2(df_resu_MF)
    pass

def stat_oper_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    mann_whit(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
def stat_oper_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    spea_mann(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    kend_tau_(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    good_krus_gamm(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    
def stat_exec_NA_YES(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    stat_oper_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'seve'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0..C2', 'C3..C6'] 
    parm_sexe_list = ['M','F']
    stat_oper_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    stat_oper_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)

def stat_exec_NA_NOT(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
if __name__ == "__main__":

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
    
    # Step 2
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f"{script_name}jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
        
        # Inpu
        df1, df2, df3 = inpu(file_path)
        
        # Exec NA_YES
        exec_NA_YES = True
        if exec_NA_YES:
            write ("")
            write (">>> >>> >>>")
            write (f'{date_form} NA_YES')
            write (">>> >>> >>>")
            df_line = df1
            trac = True
            if trac:
                print(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                write(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            stat_exec_NA_YES(df_line)
        
        # Exec NA_NOT
        exec_NA_NOT = True
        if exec_NA_NOT:
            write ("")
            write (">>> >>> >>>")
            write (f'{date_form} NA_NOT')
            write (">>> >>> >>>")
            df_line = df2
            trac = True
            if trac:
                print(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                write(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            stat_exec_NA_NOT(df_line)
        pass 