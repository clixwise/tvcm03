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


    # Assuming your dataframe is named 'df' with columns 'age' and 'ceap'

    # 1. Age distribution for C3 and C6
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='ceap', y='age', data=df[df['ceap'].isin(['C3', 'C6'])])
    plt.title('Age Distribution for C3 and C6 Stages')
    plt.show()

    # 2. Age group analysis
    def age_group(age):
        if age < 35:
            return '18-34'
        elif age < 51:
            return '35-50'
        elif age < 65:
            return '51-64'
        else:
            return '65+'

    df['age_group'] = df['age'].apply(age_group)

    # Calculate prevalence for C3 and C6 in each age group
    for stage in ['C3', 'C6']:
        print(f"\nPrevalence of {stage} by age group:")
        prevalence = df[df['ceap'] == stage].groupby('age_group').size() / df.groupby('age_group').size() * 100
        print(prevalence)

    # 3. Statistical tests
    c3_ages = df[df['ceap'] == 'C3']['age']
    c6_ages = df[df['ceap'] == 'C6']['age']
    other_ages = df[~df['ceap'].isin(['C3', 'C6'])]['age']

    # T-test for C3 vs others
    t_stat_c3, p_value_c3 = stats.ttest_ind(c3_ages, other_ages)
    print(f"\nT-test for C3 vs others: t-statistic = {t_stat_c3:.4f}, p-value = {p_value_c3:.4f}")

    # T-test for C6 vs others
    t_stat_c6, p_value_c6 = stats.ttest_ind(c6_ages, other_ages)
    print(f"T-test for C6 vs others: t-statistic = {t_stat_c6:.4f}, p-value = {p_value_c6:.4f}")

    # 4. Age correlation with CEAP severity
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    correlation, p_value = stats.spearmanr(df['age'], df['ceap_numeric'])
    print(f"\nSpearman correlation between age and CEAP severity: correlation = {correlation:.4f}, p-value = {p_value:.4f}")

    
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
Thank you for providing the outcome. Let's analyze these results:

1. Prevalence of C3 by age group:
   - The prevalence of C3 is highest in the 51-64 age group (34.32%), followed closely by the 35-50 age group (33.92%).
   - There's a slight decrease in prevalence for the 65+ age group (30.71%).
   - The lowest prevalence is in the 18-34 age group (26.67%).

2. Prevalence of C6 by age group:
   - The highest prevalence of C6 is in the 35-50 age group (28.65%).
   - The 65+ age group has the second-highest prevalence (25.59%).
   - The 18-34 and 51-64 age groups have similar prevalences (24.00% and 23.62% respectively).

3. T-tests:
   - For C3 vs others: t-statistic = -0.2010, p-value = 0.8408
   - For C6 vs others: t-statistic = -0.1013, p-value = 0.9193
   - Both p-values are much higher than the conventional significance level of 0.05, indicating no statistically significant difference in age between C3 or C6 patients and others.

4. Spearman correlation:
   - Correlation between age and CEAP severity: 0.0306
   - P-value: 0.3968
   - This indicates a very weak positive correlation that is not statistically significant.

Interpretation:
1. Unlike the general findings we discussed earlier, your dataset doesn't show a strong increase in C3 or C6 prevalence with age. The prevalence is relatively consistent across age groups, with some fluctuations.
2. The lack of statistical significance in the t-tests suggests that age alone is not a strong differentiating factor for C3 or C6 in your dataset.
3. The weak and non-significant Spearman correlation indicates that there's no strong linear relationship between age and CEAP severity in your data.

These results differ from the general findings we discussed earlier, which showed a clearer increase in prevalence with age. Your dataset suggests that age might not be as strong a factor in determining CEAP stages as typically observed. This could be due to various factors such as sample characteristics, regional differences, or other underlying variables not captured in this analysis.

Given these results, it would be worthwhile to:
1. Investigate other factors that might be influencing CEAP stages in your population.
2. Consider if there are any unique characteristics of your sample that might explain these differences from typical findings.
3. Explore potential interactions between age and other variables (like sex or comorbidities) that might reveal more complex relationships with CEAP stages.
'''
