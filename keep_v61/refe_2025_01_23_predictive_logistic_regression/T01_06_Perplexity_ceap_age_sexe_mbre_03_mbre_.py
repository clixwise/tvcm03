import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"mbre : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'mbre', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")

    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def stat(df):


    # Assuming your dataframe is named 'df' with columns 'sex', 'ceap', and 'age'

    # 1. Overall prevalence by sex
    overall_prevalence = df['mbre'].value_counts(normalize=True) * 100
    print("Overall prevalence by sex:")
    print(overall_prevalence)

    # 2. CEAP stage distribution by sex
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ceap', hue='mbre', data=df)
    plt.title('CEAP Stage Distribution by Sex')
    plt.show()

    # 3. Prevalence of C3 and C6 by sex
    for stage in ['C3', 'C6']:
        stage_prevalence = df[df['ceap'] == stage]['mbre'].value_counts(normalize=True) * 100
        print(f"\nPrevalence of {stage} by sex:")
        print(stage_prevalence)

    # 4. Chi-square test for association between sex and CEAP stages
    contingency_table = pd.crosstab(df['mbre'], df['ceap'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square test for association between sex and CEAP stages:")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")

    # 5. Odds ratio for severe stages (C4-C6) by sex
    df['severe_stage'] = df['ceap'].isin(['C4', 'C5', 'C6']).astype(int)
    contingency_table_severe = pd.crosstab(df['mbre'], df['severe_stage'])
    odds_ratio, p_value = stats.fisher_exact(contingency_table_severe)
    print(f"\nOdds ratio for severe stages (C4-C6) by sex:")
    print(f"Odds ratio: {odds_ratio:.4f}")
    print(f"p-value: {p_value:.4f}")

    # 6. Average age comparison by sex for C3 and C6
    for stage in ['C3', 'C6']:
        stage_data = df[df['ceap'] == stage]
        t_stat, p_value = stats.ttest_ind(stage_data[stage_data['mbre'] == 'G']['age'], 
                                        stage_data[stage_data['mbre'] == 'D']['age'])
        print(f"\nAverage age comparison for {stage} by sex:")
        print(f"Male average age: {stage_data[stage_data['mbre'] == 'G']['age'].mean():.2f}")
        print(f"Female average age: {stage_data[stage_data['mbre'] == 'D']['age'].mean():.2f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")


    
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
        
        # set_file_objc(file)
        
        # Step 21
        filt_valu = None
        filt_name = 'mbre'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        stat(df12)
        

        pass
    
'''
Thank you for providing this detailed analysis of the 'leg' factor. Let's interpret these results:

1. Overall prevalence by leg:
   - Left leg: 50.58%
   - Right leg: 49.42%
   This shows a nearly equal distribution between left and right legs overall.

2. Prevalence of C3 by leg:
   - Left leg: 50.60%
   - Right leg: 49.40%
   C3 stage shows almost equal distribution between left and right legs.

3. Prevalence of C6 by leg:
   - Left leg: 57.65%
   - Right leg: 42.35%
   C6 stage shows a notable preference for the left leg.

4. Chi-square test for association between leg and CEAP stages:
   - Chi-square statistic: 13.5928
   - p-value: 0.0345
   This p-value < 0.05 indicates a statistically significant association between the affected leg and CEAP stages.

5. Odds ratio for severe stages (C4-C6) by leg:
   - Odds ratio: 1.4912
   - p-value: 0.0073
   This suggests significantly higher odds of severe stages (C4-C6) in the left leg, and it's statistically significant (p < 0.05).

6. Average age comparison:
   For C3:
   - Left leg: 56.78 years
   - Right leg: 56.85 years
   - p-value: 0.9690 (not statistically significant)
   
   For C6:
   - Left leg: 54.22 years
   - Right leg: 60.63 years
   - p-value: 0.0038 (statistically significant)

Key Insights:
1. The left leg shows a higher prevalence of severe stages (C4-C6), particularly C6.
2. There's a statistically significant association between the affected leg and CEAP stages overall.
3. For C6, patients with right leg involvement are significantly older on average.
4. The leg factor seems more influential for severe stages, especially C6, than for C3.

These findings help explain why the 'leg' variable emerged as the most important feature in your logistic regression model. The significant association with CEAP stages and the higher odds of severe stages in the left leg are particularly noteworthy. This could have important clinical implications and warrants further investigation into potential anatomical, lifestyle, or other factors that might explain this left-leg preference in severe cases.
'''
