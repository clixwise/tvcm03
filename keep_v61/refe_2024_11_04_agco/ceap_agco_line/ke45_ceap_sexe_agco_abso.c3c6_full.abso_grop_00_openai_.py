import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from util_file_mngr import set_file_objc, write

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def stat(df_line):

    # Sample data
    # Assuming df_line contains the original data
    # Create severity groups
    severity_mapping = {
        'NA': 'Absent',
        'C0': 'Mild', 'C1': 'Mild', 'C2': 'Mild',
        'C3': 'Severe', 'C4': 'Severe', 'C5': 'Severe', 'C6': 'Severe'
    }
    df_line['severity'] = df_line['ceap'].map(severity_mapping)

    # Group by severity and sexe
    results = []
    for severity, group in df_line.groupby('severity'):
        male_ages = group[group['sexe'] == 'M']['age']
        female_ages = group[group['sexe'] == 'F']['age']
        
        # Perform Mann-Whitney U test
        if len(male_ages) > 0 and len(female_ages) > 0:
            u_stat, p_value = mannwhitneyu(male_ages, female_ages, alternative='two-sided')
        else:
            u_stat, p_value = None, None  # Not enough data for this severity level

        results.append({'severity': severity, 'p_value': p_value})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Apply multiple testing correction
    results_df['p_value_corrected_bonferroni'] = multipletests( results_df['p_value'], method='bonferroni')[1]
    results_df['p_value_corrected_fdr'] = multipletests( results_df['p_value'], method='fdr_bh')[1]

    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
            print(results_df)
            write(results_df.to_string(index=False))
    pass

def plot(df_line):
    male_color = 'blue'
    female_color = 'orange'

    # 1. Box Plot: Age Distribution by Severity and Gender
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_line, x='severity', y='age', hue='sexe', palette={'M': male_color, 'F': female_color})
    plt.title('Age Distribution by Gender and Severity Group')
    plt.xlabel('Severity Group')
    plt.ylabel('Age')
    plt.legend(title='Gender')

    # Add mean values with color-specific text
    grouped_means = df_line.groupby(['severity', 'sexe'])['age'].mean().reset_index()
    for i, severity in enumerate(df_line['severity'].unique()):
        for j, gender in enumerate(['M', 'F']):
            mean_value = grouped_means[(grouped_means['severity'] == severity) & (grouped_means['sexe'] == gender)]['age']
            if not mean_value.empty:
                text_color = 'white' if gender == 'M' else 'black'
                plt.text(x=i - 0.2 + j * 0.4, 
                        y=mean_value.values[0] + 1, 
                        s=f'{mean_value.values[0]:.1f}', 
                        ha='center', 
                        color=text_color, 
                        fontsize=10, 
                        fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 2. Grouped Bar Plot: Mean Ages by Severity and Gender
    plt.figure(figsize=(10, 6))
    severity_means = df_line.groupby(['severity', 'sexe'])['age'].mean().unstack()
    severity_std = df_line.groupby(['severity', 'sexe'])['age'].std().unstack()

    x = np.arange(len(severity_means))
    bar_width = 0.35

    plt.bar(x - bar_width / 2, severity_means['M'], bar_width, yerr=severity_std['M'], label='Male', color=male_color, capsize=5)
    plt.bar(x + bar_width / 2, severity_means['F'], bar_width, yerr=severity_std['F'], label='Female', color=female_color, capsize=5)

    plt.xticks(x, severity_means.index)
    plt.xlabel('Severity Group')
    plt.ylabel('Mean Age')
    plt.title('Mean Age by Gender and Severity Group')
    plt.legend(title='Gender')

    # Add gridlines
    plt.grid(True, linestyle='--', color='darkgray', alpha=0.7)

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
        #plot(df11)
        pass
'''
### Interpretation of Results:

#### 1. **Absent (NA)**:
   - **Raw p-value**: \( 0.233 \)
     - Suggests no strong evidence of a difference in age distribution between males and females.
   - **Bonferroni-corrected p-value**: \( 0.699 \)
   - **FDR-corrected p-value**: \( 0.514 \)
     - After correction, this remains non-significant (\( p > 0.05 \)).

#### 2. **Mild (C0, C1, C2)**:
   - **Raw p-value**: \( 0.912 \)
     - Indicates no difference in age between genders for the mild group.
   - **Bonferroni-corrected p-value**: \( 1.000 \)
   - **FDR-corrected p-value**: \( 0.912 \)
     - No significance after correction.

#### 3. **Severe (C3, C4, C5, C6)**:
   - **Raw p-value**: \( 0.343 \)
     - Suggests no evidence of a difference in age distribution between genders for the severe group.
   - **Bonferroni-corrected p-value**: \( 1.000 \)
   - **FDR-corrected p-value**: \( 0.514 \)
     - Again, non-significant after correction.

---

### Conclusion:
Across the **three severity groups**:
- There are **no significant differences in age distributions** between males and females, even before correcting for multiple tests.
- After corrections (Bonferroni and FDR), the results remain **non-significant**.

---

### Next Steps:
1. **Visualization**:
   - A bar or box plot grouped by severity and gender could help confirm and illustrate these findings visually.
   - Would you like assistance creating such a plot?

2. **Subgroup Analysis**:
   - If you’re still curious, we could explore differences within smaller subgroups (e.g., by age bins, individual CEAP classes within severity groups, etc.).

Let me know how you'd like to proceed! 🚀
'''