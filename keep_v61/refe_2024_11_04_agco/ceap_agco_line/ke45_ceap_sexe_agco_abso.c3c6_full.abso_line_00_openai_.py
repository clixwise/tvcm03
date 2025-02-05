import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import numpy as np

from util_file_mngr import set_file_objc, write

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    write(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        write(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")

    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def stat(df_line):

    # Step 1: Aggregate ages per patient (doss), sexe, and CEAP class
    df_agg = df_line #.groupby(['doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    # Step 2: Perform Mann-Whitney U Test for each CEAP class
    results = []
    ceap_classes = df_agg['ceap'].unique()
    print (ceap_classes)
    write (str(ceap_classes))
    
    for ceap in ceap_classes:
        ages_m = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'M')]['age']
        ages_f = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'F')]['age']
        
        if len(ages_m) > 0 and len(ages_f) > 0:  # Only run the test if both groups have data
            stat, p_value = mannwhitneyu(ages_m, ages_f, alternative='two-sided')
            results.append({'CEAP': ceap, 'p_value': p_value})
        else:
            results.append({'CEAP': ceap, 'p_value': None})

    # Step 3: Correct for multiple comparisons (Bonferroni and FDR)
    results_df = pd.DataFrame(results).dropna()
    p_values = results_df['p_value'].values

    # Bonferroni correction
    _, p_corrected_bonferroni, _, _ = multipletests(p_values, method='bonferroni')
    # FDR correction (Benjamini-Hochberg)
    _, p_corrected_fdr, _, _ = multipletests(p_values, method='fdr_bh')
    # Add corrected p-values to the results
    results_df['p_value_corrected_bonferroni'] = p_corrected_bonferroni
    results_df['p_value_corrected_fdr'] = p_corrected_fdr

    # Step 4: Display results
    results_df_sorted = results_df.sort_values(by='p_value', ascending=True)
    print(results_df)
    write(f"Results\n{results_df.to_string()}")
    print(results_df_sorted)
    write ("")
    write(f"Results sorted\n{results_df_sorted.to_string()}")
    
    plot = False
    if plot:
        # Extract sorted data
        ceap_classes = results_df_sorted['CEAP']
        raw_p_values = results_df_sorted['p_value']
        bonferroni_p_values = results_df_sorted['p_value_corrected_bonferroni']
        fdr_p_values = results_df_sorted['p_value_corrected_fdr']
        x = np.arange(len(ceap_classes))  # X-axis positions
        # Plotting
        plt.figure(figsize=(10, 6))
        bar_width = 0.25
        # Bars for each type of p-value
        plt.bar(x - bar_width, raw_p_values, bar_width, label='Raw p-values', color='skyblue')
        plt.bar(x, bonferroni_p_values, bar_width, label='Bonferroni-corrected', color='orange')
        plt.bar(x + bar_width, fdr_p_values, bar_width, label='FDR-corrected', color='green')
        # Significance line
        plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Significance threshold (α=0.05)')
        # Adding labels and legend
        plt.xticks(x, ceap_classes, rotation=45)
        plt.xlabel('CEAP Classes (Ordered by Raw p-values)')
        plt.ylabel('p-values')
        plt.title('Raw and Corrected p-values by CEAP Class (Sorted)')
        plt.legend()
        plt.tight_layout()
        plt.show()

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
        
        # Step 21
        filt_valu = None
        filt_name = 'sexe'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        stat(df11)
        pass