import numpy as np
import pandas as pd
from util_file_mngr import write
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------
# Proportion test of Independence
# -------------------------------
''' 'pro1'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with their sum [vertical approach]
example :
NA :
count: [52, 53]
nobs : [105, 105]
'''
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

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nProportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nProportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
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
        # Total counts for males and females across column
        total = observed_males + observed_females
        nobs = [total, total]
        # Perform Proportion Z-test
        if loop_jrnl:
            print (f'ceap: {age_bin} zzz proportion_test: count: {count} nobs: {nobs}')
        z_statistic, pval = proportions_ztest(count, nobs)
        stat = z_statistic
        
        # Intp
        if pval < alpha:
            if loop_jrnl:
                print(f"Proportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                write(f"Proportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Proportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Proportion (1) [ceap/colu] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Store the result
        resu_dict[age_bin] = {
            colu_name: age_bin,
            f'{indx_cate_nam1}': observed_males,
            f'{indx_cate_nam2}': observed_females,
            f'tot{indx_cate_nam1}': total,
            f'tot{indx_cate_nam2}': total,
            f'pro{indx_cate_nam1}': observed_males/total,
            f'pro{indx_cate_nam2}': observed_females/total,
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
        
    print(f"\n---\nData : {what}\nProportion (1) [ceap/colu] 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nProportion (1) [ceap/colu] 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
''' 'pro2'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with the sum of repective 'sexe' [horizontal approach]
example :
NA :
count: [52, 53]
nobs : [378, 498]
'''
def pro2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # Total counts of males and females across all bins
    total_males = df.loc[indx_cate_nam1].sum()
    total_females = df.loc[indx_cate_nam2].sum()

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nProportion (2) [ceap/line]: Iterations per {colu_name}")
        write(f"\nData : {what}\nProportion (2) [ceap/line]: Iterations per {colu_name}")
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
        if loop_jrnl:
            print (f'ceap: {age_bin} zzz proportion_test: count: {count} nobs: {nobs}')
        z_statistic, pval = proportions_ztest(count, nobs)
        stat = z_statistic
        
        # Intp
        if pval < alpha:
            if loop_jrnl:
                print(f"Proportion (2) [ceap/line] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                write(f"Proportion (2) [ceap/line] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Proportion (2) [ceap/line] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Proportion (2) [ceap/line] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
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
        
    print(f"\n---\nData : {what}\nProportion (2) [ceap/colu] 2024_12_15 [2025_01_17] [ACCEPT THIS OPTION] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nProportion (2) [ceap/colu] 2024_12_15 [2025_01_17] [ACCEPT THIS OPTION] ::\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
''' 'pro3'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with the expected ratio of its repective 'sexe' [weighted horizontal approach]
example :
NA :
count: observed:[52, 53]
nobs : expected:[..., ...] : e.g. : (observed_females + observed_males) * (total_females / (total_males + total_females))
'''
def pro3(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # Total counts of males and females across all bins
    total_males = df.loc[indx_cate_nam1].sum()
    total_females = df.loc[indx_cate_nam2].sum()

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nProportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nProportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are equal for the given {colu_name} (Two-tailed test)"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are unequal for the given {colu_name} (Two-tailed test)"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        
        # Observed counts for males and females in this age bin
        observed_males = df.loc[indx_cate_nam1, age_bin]
        observed_females = df.loc[indx_cate_nam2, age_bin]
        # Calculate expected counts based on overall proportions
        expected_female = (observed_females + observed_males) * (total_females / (total_males + total_females))
        expected_male =   (observed_females + observed_males) * (total_males / (total_males + total_females)) 
        # Perform tests using actual and expected counts
        # For example, using the proportions test
        count = np.array([observed_females, observed_males])
        nobs = np.array([expected_female, expected_male])

        # Perform Proportion Z-test
        if loop_jrnl:
            print (f'ceap: {age_bin} zzz proportion_test: count: {count} nobs: {nobs}')
        z_statistic, pval = proportions_ztest(count, nobs)
        stat = z_statistic
        
        # Intp
        if pval < alpha:
            if loop_jrnl:
                print(f"Proportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                write(f"Proportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Proportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Proportion (3) [weighted analysis] 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
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
        
    print(f"\n---\nData : {what}\nProportion (3) [ceap/colu] 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nProportion (3) [ceap/colu] 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
''' 'pro4'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with the expected ratio of its repective 'sexe' [weighted horizontal approach]
example :
NA :
test: [52, 105, ...] : ... is the overall expected proportion 
'''
def pro4(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):    
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    
    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nProportion (4) 2024_12_15 [2025_01_17] : Iterations per {colu_name} and {indx_name}")
        write(f"\nData : {what}\nProportion (4) 2024_12_15 [2025_01_17] : Iterations per {colu_name} and {indx_name}")
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
                if loop_jrnl:
                    print(f"Proportion (4) 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                    write(f"Proportion (4) 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                HV = "Ha"
                HT = Ha
            else:
                if loop_jrnl:
                    print(f"Proportion (4) 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                    write(f"Proportion (4) 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
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

    print(f"\n---\nData : {what}\nProportion (4) 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nProportion (4) 2024_12_15 [2025_01_17] [REJECT THIS OPTION] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def caty_prop(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    pro1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pro2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pro3(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pro4(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pass

'''
Comment about 'pro4'
The alternative approach you've provided for analyzing proportions involves calculating the observed and expected proportions of males and females for each age bin in your dataset. Let's break down the code and analyze its methodology, purpose, and implications.

### Breakdown of the Code

1. **Overall Female Proportion Calculation**:
   ```python
   F_sum = df.loc['Female'].sum()
   T_sum = df.sum().sum()
   F_prop = F_sum / T_sum
   ```
   - Here, you calculate the total count of females (`F_sum`) and the total count of all individuals (`T_sum`).
   - The overall female proportion (`F_prop`) is then calculated as the ratio of females to the total population.

2. **Hypotheses Definition**:
   ```python
   H0 = f"H0 : The proportions in {'Male'} and {'Female'} are equal for the given {colu_name} (Two-tailed test)"
   Ha = f"Ha : The proportions in {'Male'} and {'Female'} are unequal for the given {colu_name} (Two-tailed test)"
   ```
   - Null Hypothesis (H0): The proportions of males and females are equal for the given age bin.
   - Alternative Hypothesis (Ha): The proportions of males and females are not equal for the given age bin.

3. **Looping Through Age Bins and Genders**:
   ```python
   for age_bin in df.columns:
       for gender in df.index:
           n_gender = df.loc[gender, age_bin]
           n_total = df[age_bin].sum()
           ...
   ```
   - This nested loop iterates through each age bin (column) and each gender (index).
   - For each combination, it retrieves the count of that gender in the specific age bin (`n_gender`) and calculates the total count for that age bin (`n_total`).

4. **Expected Proportion Calculation**:
   ```python
   expe_prop = F_prop if gender == 'Female' else 1 - F_prop
   ```
   - The expected proportion (`expe_prop`) is set to the overall female proportion if the current gender is female; otherwise, it is set to `1 - F_prop` (the proportion of males).

5. **Observed Proportion Calculation and Z-Test**:
   ```python
   obsv_prop = n_gender / n_total         
   z_statistic, pval = proportions_ztest(n_gender, n_total, value=expe_prop)
   stat = z_statistic
   ```
   - The observed proportion (`obsv_prop`) is calculated as the count of that gender divided by the total count for that age bin.
   - A proportions z-test is conducted using `proportions_ztest`, comparing the observed count against the expected proportion.

### Analysis

#### Strengths

1. **Direct Comparison to Overall Proportions**: By using the overall female proportion as a benchmark, this approach allows you to assess whether observed counts within each age bin significantly deviate from what would be expected based on overall gender distribution.

2. **Clear Hypothesis Testing Framework**: The code clearly defines null and alternative hypotheses, providing a structured approach to statistical testing.

3. **Flexibility**: This method can be applied across multiple demographic bins (e.g., age bins) while maintaining a consistent comparison basis.

4. **Statistical Rigor**: Using a z-test provides a formal statistical framework to evaluate differences in proportions, enhancing the robustness of your conclusions.

#### Limitations

1. **Assumption of Independence**: The method assumes that counts in different age bins are independent, which may not always be true if there are underlying correlations between demographics.

2. **Multiple Comparisons**: If you test many age bins simultaneously, you increase the risk of Type I error (false positives). Consider applying corrections (e.g., Bonferroni correction) to account for multiple comparisons.

3. **Interpretation of Results**: While statistical significance indicates a difference from expected proportions, it does not provide context regarding effect size or practical significance.

4. **Dependence on Overall Proportions**: This approach relies on overall proportions being representative of each subgroup. If there are significant variations in gender distributions across different age bins, this could lead to misleading conclusions.

### Conclusion

This approach effectively combines overall population statistics with individual demographic analyses using a rigorous statistical framework. By employing a proportions z-test against an expected proportion based on overall gender distribution, you can draw meaningful conclusions about gender differences within specific age bins.

If you have any further questions or need assistance with specific aspects of this analysis or its implementation, feel free to ask!
'''
'''
Yes, that's correct! The way you compute the 'total' references for the proportions test can significantly influence the results and interpretations of the analysis. Let's break down the different variations you mentioned:

### 1. Column-Wise Totals

**Definition**: This approach involves calculating proportions based on the total counts within each column (e.g., for each CEAP class or age bin).

**Implementation**:
- For each CEAP class, you would compare the counts of males and females in that class against the total count for that class.
- This method is useful for understanding how gender proportions vary within each specific category.

**Example**:
```python
for ceap_class in df_table.columns:
    female_count = df_table.loc['Female', ceap_class]
    male_count = df_table.loc['Male', ceap_class]
    total_count = female_count + male_count
    
    # Perform proportions test
    count = np.array([female_count, male_count])
    nobs = np.array([total_count, total_count])  # Total for that CEAP class
```

### 2. Row-Wise Totals

**Definition**: This approach focuses on calculating proportions based on the total counts for each gender across all categories (rows).

**Implementation**:
- You would compare the counts of males and females in a specific category against their overall totals.
- This method provides insight into how a specific category's gender distribution compares to the overall gender distribution.

**Example**:
```python
total_males = df_table.loc['Male'].sum()
total_females = df_table.loc['Female'].sum()

for ceap_class in df_table.columns:
    female_count = df_table.loc['Female', ceap_class]
    male_count = df_table.loc['Male', ceap_class]
    
    # Perform proportions test using overall totals
    count = np.array([female_count, male_count])
    nobs = np.array([total_females, total_males])  # Total for males and females overall
```

### 3. Weighted-Wise Totals

**Definition**: This approach adjusts counts based on overall proportions, often accounting for population imbalances when analyzing specific categories.

**Implementation**:
- You would calculate expected counts based on overall gender proportions and use these expected values in your tests.
- This method is particularly useful when there are significant differences in sample sizes or distributions across categories.

**Example**:
```python
overall_female_prop = total_females / (total_females + total_males)

for ceap_class in df_table.columns:
    female_count = df_table.loc['Female', ceap_class]
    male_count = df_table.loc['Male', ceap_class]
    
    # Calculate expected counts based on overall proportions
    expected_female = overall_female_prop * (female_count + male_count)
    expected_male = (1 - overall_female_prop) * (female_count + male_count)

    # Perform proportions test using observed and expected counts
    count = np.array([female_count, male_count])
    nobs = np.array([expected_female, expected_male])
```

### Implications of Different Approaches

- **Statistical Significance**: Different methods may yield different p-values and conclusions regarding significance. For instance, if you have a large imbalance in gender distribution, using row-wise totals might show significant differences that column-wise totals do not.
  
- **Interpretation of Results**: The context provided by each method can lead to different interpretations. Column-wise analysis focuses on specific categories, while row-wise analysis provides a broader view of gender distribution.

- **Multiple Comparisons**: If you're testing multiple categories (e.g., multiple CEAP classes), consider adjustments for multiple comparisons regardless of the method used.

### Conclusion

Understanding these variations allows you to choose an appropriate method based on your research questions and data characteristics. Each approach has its strengths and limitations, and selecting the right one can help ensure that your findings are meaningful and robust.

If you have any further questions or need assistance with specific analyses or implementations, feel free to ask!
'''
