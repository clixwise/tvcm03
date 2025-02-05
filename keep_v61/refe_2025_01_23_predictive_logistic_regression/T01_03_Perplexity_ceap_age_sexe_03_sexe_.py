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
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    
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
    overall_prevalence = df['sexe'].value_counts(normalize=True) * 100
    print("Overall prevalence by sex:")
    print(overall_prevalence)

    # 2. CEAP stage distribution by sex
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ceap', hue='sexe', data=df)
    plt.title('CEAP Stage Distribution by Sex')
    plt.show()

    # 3. Prevalence of C3 and C6 by sex
    for stage in ['C3', 'C6']:
        stage_prevalence = df[df['ceap'] == stage]['sexe'].value_counts(normalize=True) * 100
        print(f"\nPrevalence of {stage} by sex:")
        print(stage_prevalence)

    # 4. Chi-square test for association between sex and CEAP stages
    contingency_table = pd.crosstab(df['sexe'], df['ceap'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square test for association between sex and CEAP stages:")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")

    # 5. Odds ratio for severe stages (C4-C6) by sex
    df['severe_stage'] = df['ceap'].isin(['C4', 'C5', 'C6']).astype(int)
    contingency_table_severe = pd.crosstab(df['sexe'], df['severe_stage'])
    odds_ratio, p_value = stats.fisher_exact(contingency_table_severe)
    print(f"\nOdds ratio for severe stages (C4-C6) by sex:")
    print(f"Odds ratio: {odds_ratio:.4f}")
    print(f"p-value: {p_value:.4f}")

    # 6. Average age comparison by sex for C3 and C6
    for stage in ['C3', 'C6']:
        stage_data = df[df['ceap'] == stage]
        t_stat, p_value = stats.ttest_ind(stage_data[stage_data['sexe'] == 'M']['age'], 
                                        stage_data[stage_data['sexe'] == 'F']['age'])
        print(f"\nAverage age comparison for {stage} by sex:")
        print(f"Male average age: {stage_data[stage_data['sexe'] == 'M']['age'].mean():.2f}")
        print(f"Female average age: {stage_data[stage_data['sexe'] == 'F']['age'].mean():.2f}")
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
        filt_name = 'sexe'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        stat(df12)
        

        pass
    
'''
Thank you for providing the results. Let's analyze these findings:

1. Overall prevalence by sex:
   - Females: 57.72%
   - Males: 42.28%
   This aligns with the general trend that CVD is more prevalent in women.

2. Prevalence of C3 by sex:
   - Females: 62.65%
   - Males: 37.35%
   C3 is notably more prevalent in females, which is consistent with general findings.

3. Prevalence of C6 by sex:
   - Females: 50.51%
   - Males: 49.49%
   C6 is almost equally distributed between sexes, which differs from some studies suggesting higher prevalence in males.

4. Chi-square test for association between sex and CEAP stages:
   - Chi-square statistic: 9.2983
   - p-value: 0.1575
   The p-value > 0.05 suggests no statistically significant association between sex and CEAP stages overall.

5. Odds ratio for severe stages (C4-C6) by sex:
   - Odds ratio: 1.1548
   - p-value: 0.3412
   This indicates slightly higher odds of severe stages in males, but it's not statistically significant (p > 0.05).

6. Average age comparison:
   For C3:
   - Males: 60.30 years
   - Females: 54.74 years
   - p-value: 0.0055 (statistically significant)
   
   For C6:
   - Males: 54.69 years
   - Females: 59.13 years
   - p-value: 0.0435 (statistically significant)

Key Insights:
1. The overall prevalence of CVD is higher in females, consistent with general findings.
2. C3 is more prevalent in females, aligning with typical observations.
3. C6 shows nearly equal prevalence between sexes, differing from some studies that find higher prevalence in males.
4. There's no strong overall association between sex and CEAP stages, but there are significant age differences within C3 and C6 stages.
5. For C3, males are significantly older on average, while for C6, females are significantly older.

These results partially align with general findings but also show some unique patterns. The lack of significant association between sex and overall CEAP stages suggests that in your population, sex might not be as strong a determinant of disease severity as sometimes observed. However, the age differences within C3 and C6 stages indicate complex interactions between age and sex that warrant further investigation.
'''
