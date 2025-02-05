import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import binomtest

# -------------------------------
# Binomial Test of Independence
# -------------------------------
''' 'bin1'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with their sum [vertical approach]
example :
NA :
male count (52) vs total count (105)
'''
def bin1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Binomial Test for each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nBinomial (1) 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nBinomial (1) 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : Observed '{indx_cate_nam1}' and '{indx_cate_nam2}' counts for '{colu_name}' do not differ from the expected count"
    Ha = f"Ha : Observed '{indx_cate_nam1}' and '{indx_cate_nam2}' counts for '{colu_name}' differ from the expected count"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        
        male_count = df.loc[indx_cate_nam1, age_bin]
        female_count = df.loc[indx_cate_nam2, age_bin]
        total_count = male_count + female_count
        
        # Perform Binomial Test assuming p = 0.5
        result = binomtest(male_count, total_count, p=0.5, alternative='two-sided')
        stat = result.statistic
        pval = result.pvalue
        
        # Determine H
        if pval < alpha:
            HV = 'Ha'
            HT = f"Observed {indx_cate_nam1} count in {age_bin} '{colu_name}' differs from the expected count"
        else:
            HV = 'H0'
            HT = f"Observed {indx_cate_nam1} count in {age_bin} '{colu_name}' does not differ from the expected count"
        
        resu_dict[age_bin] = {
        colu_name: age_bin,
        indx_cate_nam1: male_count,
        indx_cate_nam2: female_count,
        'tota': total_count,
        'stat': stat,
        'pval': pval,
        'H' : HV
    }
    
    # Create DataFrame from results
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    
    print(f"\n---\nData : {what}\nBinomial (1) 2024_12_15 [2025_01_17] [PREFERRED NOT : ratio to column] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nBinomial (1) 2024_12_15 [2025_01_17] [PREFERRED NOT : ratio to column] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

''' 'bin2'
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with their sum [vertical approach]
example :
NA :
same as Proportion (4) but 'binomtest' i.o. 'proportions_ztest'
'''
def bin2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Binomial Test in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nBinomial (2) 2024_12_15 [2025_01_17] : Iterations per {colu_name} and {indx_name}")
        write(f"\nData : {what}\nBinomial (2) 2024_12_15 [2025_01_17] : Iterations per {colu_name} and {indx_name}")
    F_sum = df.loc[indx_cate_nam2].sum()
    T_sum = df.sum().sum()
    F_prop = F_sum / T_sum
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : Observed '{indx_cate_nam1}' or '{indx_cate_nam2}' count for '{colu_name}' does not differ from the expected count"
    Ha = f"Ha : Observed '{indx_cate_nam1}' or '{indx_cate_nam2}' count for '{colu_name}' differs from the expected count"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for age_bin in df.columns:
        for gender in df.index:
            
            n_gender = df.loc[gender, age_bin]
            n_total = df[age_bin].sum()        
            # Use overall gender proportion as expected proportion
            expe_prop = F_prop if gender == indx_cate_nam2 else 1 - F_prop
            obsv_prop = n_gender / n_total         
            result = binomtest(n_gender, n_total, expe_prop) 
            # Calculate the confidence interval
            ci_lowr, ci_uppr = result.proportion_ci()         
            # Determine H
            stat = result.statistic
            pval = result.pvalue
            if pval < 0.05:
                hypothesis = 'Ha'
            else:
                hypothesis = 'H0'
                    
            resu_dict[f'{age_bin}_{gender}'] = {
                colu_name: age_bin,
                indx_name: gender,
                'count': n_gender,
                'tota': n_total,
                'stat': stat,
                'pval': pval,
                'H' : hypothesis,
                'obse_prop': obsv_prop,
                'expe_prop': expe_prop,
                'ci_lowr' : ci_lowr,
                'ci_uppr' : ci_uppr
            }

    # Create DataFrame from results
    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['obse_prop'] = df_resu['obse_prop'].apply(frmt)
    df_resu['expe_prop'] = df_resu['expe_prop'].apply(frmt)
    df_resu['ci_lowr'] = df_resu['ci_lowr'].apply(frmt)
    df_resu['ci_uppr'] = df_resu['ci_uppr'].apply(frmt)
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
    
    print(f"\n---\nData : {what}\nBinomial (2) 2024_12_15 [2025_01_17] [PREFERRED YES : ratio to row total] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nBinomial (2) 2024_12_15 [2025_01_17] [PREFERRED YES : ratio to row total]  :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"{df_resu}")
        write(f"{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass

def caty_bino(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    bin1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    bin2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    pass
'''
The question you posed about whether the different approaches to calculating totals apply to the Binomial Test is an important one, especially in the context of how you interpret results. Let's analyze the implications of using different total references—column-wise, row-wise, and weighted-wise—specifically for the Binomial Test.

### Understanding the Binomial Test

The **Binomial Test** is typically used to determine if the proportion of successes in a binary outcome differs from a hypothesized proportion. It is particularly useful when you have a single binary variable (e.g., male vs. female) and want to test whether the observed proportion deviates from what is expected under the null hypothesis.

### Total References in Binomial Testing

1. **Column-Wise Totals**:
   - **Application**: In this approach, you would calculate the total number of observations (successes and failures) for each specific category (e.g., each CEAP class).
   - **Use Case**: This is appropriate when you want to test whether the proportion of successes (e.g., females) in that specific category differs from a hypothesized value (like 0.5).
   - **Example**: If testing CEAP class C0, you would look at how many females were in that class compared to the total number of individuals in that class.

2. **Row-Wise Totals**:
   - **Application**: Here, you would calculate totals based on all observations for each gender across all categories.
   - **Use Case**: This could be less common for a Binomial Test since it typically focuses on a single category at a time. However, it can be used if you're interested in testing whether the overall gender proportions differ from expected values across multiple categories.
   - **Example**: If you wanted to see if the overall proportion of females in your dataset deviates from an expected value, this would be suitable.

3. **Weighted-Wise Totals**:
   - **Application**: This involves adjusting counts based on overall proportions or expected distributions, which may be more complex.
   - **Use Case**: This approach is generally more relevant for tests that compare multiple groups or when there are significant imbalances in group sizes. It could theoretically be applied to a Binomial Test if you're adjusting for population proportions.
   - **Example**: If you wanted to see if females are underrepresented in a specific CEAP class relative to their overall representation in the population.

### Conclusion

For conducting a **Binomial Test**, it is most appropriate to use **column-wise totals**, as this aligns with the test's purpose of evaluating proportions within specific categories. The binomial test is designed to assess whether the observed proportion of successes in that category significantly deviates from a hypothesized proportion (e.g., 0.5).

Using row-wise or weighted totals may provide insights into broader trends but could lead to misleading conclusions if not interpreted carefully, especially since the Binomial Test focuses on specific binary outcomes within defined categories.

If you're analyzing gender differences within CEAP classes, stick with column-wise totals for your Binomial Tests while considering row-wise or weighted approaches for broader analyses or additional context.

If you have further questions or need clarification on any specific aspect of this analysis, feel free to ask!

Citations:
[1] https://peterstatistics.com/Terms/Tests/binomial-one-sample.html
[2] https://en.wikipedia.org/wiki/Binomial_test
[3] https://stats.stackexchange.com/questions/547315/binomial-distribution-for-gender-discrimination
[4] https://whitlockschluter3e.zoology.ubc.ca/RLabs/R_tutorial_Frequency_data.html
[5] https://sites.utexas.edu/sos/guided/inferential/categorical/univariate/binomial/
[6] https://stackoverflow.com/questions/64883060/apply-binom-test-to-every-row-using-columns-as-argument-in-pandas
[7] https://www.spss-tutorials.com/spss-binomial-test/
[8] https://www.crumplab.com/rstatsforpsych/binomial-test.html
'''
'''
The **binomial test** and the **proportions z-test** (or z-test for proportions) are both used to evaluate proportions, but they differ in how they calculate the test statistic, assumptions, and use cases.

---

### **1. Binomial Test**
- **Purpose**: Test whether the observed proportion in a sample differs from an expected proportion (e.g., \( p = 0.5 \)).
- **Distribution**: Uses the exact binomial distribution for small samples, making it more accurate when the sample size is small or when the proportion is near the extremes (close to 0 or 1).
- **Key Features**:
  - Works well with small sample sizes.
  - Does not approximate the sampling distribution — instead, it calculates exact probabilities based on the binomial distribution.
- **Use Case**:
  - When the sample size is small.
  - When you want an exact \( p \)-value for a hypothesis test.
- **Test Statistic**:
  - Not explicitly expressed in terms of a z-value; relies directly on the binomial probability.

---

### **2. Proportions Z-Test**
- **Purpose**: Test whether the observed proportion in a sample differs from an expected proportion or whether two proportions differ from each other.
- **Distribution**: Uses the **normal approximation** of the binomial distribution.
- **Key Features**:
  - Assumes large sample sizes (typically \( np \geq 5 \) and \( n(1-p) \geq 5 \)).
  - Approximates the test statistic using a z-score formula.
- **Use Case**:
  - When the sample size is large enough for the normal approximation to the binomial distribution to hold.
  - For comparing two proportions (e.g., proportion of males vs. females across two groups).
- **Test Statistic**:
  - \( z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1 - p_0)}{n}}} \)
    - \( \hat{p} \): Observed proportion.
    - \( p_0 \): Expected proportion under the null hypothesis.
    - \( n \): Sample size.

---

### **Key Differences**

| Feature                   | **Binomial Test**                                     | **Proportions Z-Test**                                  |
|---------------------------|------------------------------------------------------|-------------------------------------------------------|
| **Sample Size**           | Works for small or large samples.                    | Assumes large samples (\( np \geq 5 \), \( n(1-p) \geq 5 \)). |
| **Accuracy**              | Exact test based on the binomial distribution.       | Approximation using the normal distribution.          |
| **Test Statistic**        | Based on binomial probabilities.                     | Calculates a z-score.                                 |
| **Flexibility**           | Limited to single-sample proportions.                | Can compare two proportions (e.g., male vs. female).  |
| **Output**                | Exact \( p \)-value (preferred for small samples).   | Approximate \( p \)-value (better for large samples). |
| **Example Scenario**      | Small sample testing if male ratio deviates from 50%.| Large sample testing if male vs. female proportions differ. |

---

### **Example**: Testing Male-Female Proportions in CEAP Class

Suppose we are testing whether the male proportion in a CEAP class differs from 50% (\( p_0 = 0.5 \)):

#### **Using the Binomial Test**:
- Exact test based on \( P(X \geq k | n, p_0) \), where \( X \) is the number of males and \( k \) is the observed count of males.
- Suitable for small \( n \), especially if \( n \) or \( p_0 \) is close to the extremes.

#### **Using the Proportions Z-Test**:
- \( z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1 - p_0)}{n}}} \), where \( \hat{p} = \frac{\text{male count}}{\text{total count}} \).
- Approximates the test using the normal distribution.
- Accurate when \( n \) is large.

---

### When to Use Which?
- **Binomial Test**: Use when the sample size is small or when you want an exact test (no approximations).
- **Proportions Z-Test**: Use when the sample size is large and the assumptions for normal approximation hold, especially for comparisons between two groups or proportions.
'''