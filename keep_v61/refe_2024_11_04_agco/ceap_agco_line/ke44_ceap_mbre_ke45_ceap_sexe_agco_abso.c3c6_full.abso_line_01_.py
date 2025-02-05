from datetime import datetime
import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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
    return df1

def stat_exec(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df_agg = df_line #.groupby(['doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()
    resu_dict = {}
    for ceap in parm_ceap_list:
        # ages_m_ = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'M')]['age']
        # ages_f_ = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'F')]['age']
        ages_m = df_agg[(df_agg[parm_ceap] == ceap) & (df_agg[parm_sexe] == sexM)][parm_agee]
        ages_f = df_agg[(df_agg[parm_ceap] == ceap) & (df_agg[parm_sexe] == sexF)][parm_agee]
        print(f"ages_m : {ages_m} : ages_m.size={len(ages_m)} ages_m.size={len(ages_m)}")
        write(f"ages_m : {ages_m} : ages_m.size={len(ages_m)} ages_m.size={len(ages_m)}")
        print(f"ages_f : {ages_f} : ages_f.size={len(ages_f)} ages_f.size={len(ages_f)}")
        write(f"ages_f : {ages_f} : ages_f.size={len(ages_f)} ages_f.size={len(ages_f)}")
        if len(ages_m) > 0 and len(ages_f) > 0:  # Only run the test if both groups have data
            stat, pval = mannwhitneyu(ages_m, ages_f, alternative='two-sided')
            resu_dict[ceap] = {'CEAP': ceap, 'stat': stat, 'pval': pval}
        else:
            resu_dict[ceap] = {'CEAP': ceap, 'pval': None}
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
    Hx = f"(open) Mann-Whitney U : assesses whether, for each CEAP, the distribution of ages differs significantly between {parm_sexe}:{parm_sexe_list}, without assuming normality."
    H0 = f"H0 : (open) Mann-Whitney U : the distribution of ages does not differ significantly between {parm_sexe}:{parm_sexe_list}, without assuming normality."
    Ha = f"Ha : (open) Mann-Whitney U : the distribution of ages differs significantly between {parm_sexe}:{parm_sexe_list}, without assuming normality."
    
    
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    df_resu['pval_corr_fdr'] = df_resu['pval_corr_fdr'].apply(frmt)
    df_resu['pval_corr__bonferroni'] = df_resu['pval_corr__bonferroni'].apply(frmt)
        
    print(f"\nData : {what}\n(open) Mann-Whitney U 2024_12_15 :\n{H0}\n{Ha}")
    write(f"\nData : {what}\n(open) Mann-Whitney U 2024_12_15 :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def stat(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_exec(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'seve'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0..C2', 'C3..C6'] 
    parm_sexe_list = ['M','F']
    stat_exec(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_exec(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
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
        df_line = inpu(file_path)
        trac = True
        if trac:
            print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        # Stat
        stat(df_line)
        pass 