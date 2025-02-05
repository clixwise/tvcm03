import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

# -------------------------------
# Kolmogorov-Smirnov
# -------------------------------
def kolmsmir(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    # Trac
    trac = True

    # 'df_tabl' only 
    if df1 is None:
        print(f"\nData : {what}\n(df_table) : Kolmogorov-Smirnov : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Kolmogorov-Smirnov : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        return
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    colu_list_ord1 = df1[df1[indx_name] == indx_cate_nam1][colu_name_ordi] # colu_list_ord1 ordinal [0...n] : ceap series for M ages
    colu_list_ord2 = df1[df1[indx_name] == indx_cate_nam2][colu_name_ordi] # colu_list_ord2 ordinal [0...n] : ceap series for F ages
    if trac:
        print(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        print(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")
        write(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        write(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")

    # Exec
    stat, pval = stats.ks_2samp(colu_list_ord1, colu_list_ord2)

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nKolmogorov-Smirnov : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nKolmogorov-Smirnov : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    # Mistral
    # H0 : the distributions of the severity scores for the left and right sides are the same.
    # Ha : the distributions of the severity scores for the left and right sides are different
    H0 = f"H0 : There is no difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Kolmogorov-Smirnov : Reject the null hypothesis:\n{Ha}")
        write(f"Kolmogorov-Smirnov : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Kolmogorov-Smirnov : Fail to reject the null hypothesis:\n{H0}")
        write(f"Kolmogorov-Smirnov : Fail to reject the null hypothesis:\n{H0}")
    pass

'''
2025_01_13 MISTRAL
The Kolmogorov-Smirnov (K-S) test is a non-parametric test used to compare the distributions of two samples or to compare a sample with a reference probability distribution. It is particularly useful for comparing the shapes of distributions rather than just their central tendencies.

In your case, where you have ordinal data representing the severity of a disease for two groups (Males and Females), the K-S test can be used to determine if the distributions of disease severity are significantly different between the two groups.

### Example Data

Let's assume you have the following data:

- **Group A (Males):** 2, 3, 4, 5, 6
- **Group B (Females):** 1, 2, 3, 4, 5

### Step-by-Step Procedure

1. **State the Hypotheses:**
   - **Null Hypothesis (H0):** The distributions of disease severity are the same for males and females.
   - **Alternative Hypothesis (H1):** The distributions of disease severity are different for males and females.

2. **Perform the Kolmogorov-Smirnov Test:**
   - The K-S test compares the empirical distribution functions (ECDFs) of the two samples.

### Using Python for Kolmogorov-Smirnov Test

You can perform the Kolmogorov-Smirnov test using Python's `scipy` library:

```python
from scipy.stats import ks_2samp

# Severity scores for Group A (Males) and Group B (Females)
group_a = [2, 3, 4, 5, 6]
group_b = [1, 2, 3, 4, 5]

# Perform the Kolmogorov-Smirnov test
stat, p_value = ks_2samp(group_a, group_b)

print(f"Kolmogorov-Smirnov Statistic: {stat}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in the distribution of disease severity between males and females.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the distribution of disease severity between males and females.")
```

### Interpretation

- **Kolmogorov-Smirnov Statistic (D):** This is the maximum distance between the ECDFs of the two samples.
- **P-value:** This indicates the probability of observing the test statistic under the null hypothesis.

If the p-value is less than the chosen significance level (e.g., 0.05), you reject the null hypothesis and conclude that there is a significant difference in the distributions of disease severity between males and females. If the p-value is greater than or equal to the significance level, you fail to reject the null hypothesis.

### Important Considerations

1. **Sample Size:** The K-S test is more sensitive to differences in distribution shape when the sample sizes are large. For small sample sizes, the test may not have enough power to detect differences.
2. **Ties:** The K-S test does not handle ties well, so if your data has many tied values, the results may be less reliable.
3. **Continuous Data:** The K-S test is typically used for continuous data. While it can be applied to ordinal data, it may not be as powerful as other non-parametric tests specifically designed for ordinal data, such as the Mann-Whitney U test or the Kruskal-Wallis H test.

In summary, the Kolmogorov-Smirnov test can be used to compare the distributions of disease severity between males and females, but it is important to consider the limitations and assumptions of the test.
'''