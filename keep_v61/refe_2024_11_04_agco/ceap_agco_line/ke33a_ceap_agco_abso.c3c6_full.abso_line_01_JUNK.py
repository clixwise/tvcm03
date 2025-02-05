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
        
    print(f"\n---\nData : {what}\nSpearman [2025_01_19] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nSpearman [2025_01_19] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
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
    df_agg = df_line #.groupby(['doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()
    resu_dict = {}
    ages_t = df_agg[parm_agee]
    for ceap in parm_ceap_list:
        # ages_c_ = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'M')]['age']
        # ages_f_ = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'F')]['age']
        ages_c = df_agg[(df_agg[parm_ceap] == ceap)][parm_agee]
        if trac:
            print(f"ages_c : {ages_c} : ages_c.size={len(ages_c)} ; ages_t : {ages_t} : ages_t.size={len(ages_t)}")
            write(f"ages_c : {ages_c} : ages_c.size={len(ages_c)} ; ages_t : {ages_t} : ages_t.size={len(ages_t)}")
        if len(ages_c) > 0 and len(ages_t) > 0:  # Only run the test if both groups have data
            stat, pval = mannwhitneyu(ages_c, ages_t, alternative='two-sided')
            resu_dict[ceap] = {'CEAP': ceap, 'stat': stat, 'pval': pval, 'ages_c.size': len(ages_c) , 'ages_t.size': len(ages_t)}
        else:
            resu_dict[ceap] = {'CEAP': ceap, 'pval': None, 'ages_c.size': len(ages_c) , 'ages_t.size': len(ages_t)}
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index').dropna()
    
    # ----
    # Corr
    # ----
    pvals = df_resu['pval'].values
    print (pvals)
    # FDR correction (Benjamini-Hochberg)
    _, p_corr_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    # Bonferroni correction
    _, p_corr__bonferroni, _, _ = multipletests(pvals, method='bonferroni')
    # Add corrected p-values to the results
    df_resu['pval_corr_fdr'] = p_corr_fdr
    df_resu['pval_corr__bonferroni'] = p_corr__bonferroni
    results_df_sorted = df_resu.sort_values(by='pval', ascending=True)
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
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    df_resu['pval_corr_fdr'] = df_resu['pval_corr_fdr'].apply(frmt)
    df_resu['pval_corr__bonferroni'] = df_resu['pval_corr__bonferroni'].apply(frmt)
        
    print(f"\n---\nData : {what}\n(open) Mann-Whitney U 2024_12_15 :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\n(open) Mann-Whitney U 2024_12_15 :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def stat_oper_NA_YES(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    mann_whit(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
def stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    spea_mann(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    
def stat_exec_NA_YES(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_YES(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'seve'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0..C2', 'C3..C6'] 
    parm_sexe_list = ['M','F']
    stat_oper_NA_YES(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_YES(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)

def stat_exec_NA_NOT(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
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