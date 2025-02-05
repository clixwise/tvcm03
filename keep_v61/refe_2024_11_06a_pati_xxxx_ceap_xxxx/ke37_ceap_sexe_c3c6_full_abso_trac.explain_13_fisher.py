'''
Yes, we can use other tests besides Chi-square for analyzing individual CEAP classes. Here are some alternatives:

1. **Fisher's Exact Test**: This is particularly useful for small sample sizes or when the expected frequencies are low (< 5 in any cell)[1][8]. It's more accurate than Chi-square for 2x2 contingency tables with small samples.

2. **McNemar's Test**: This is appropriate for paired nominal data, which could be useful if you're comparing the same patients' CEAP classifications at different time points[1][7].

3. **Cochran's Q Test**: This can be used when comparing three or more related groups with dichotomous outcomes[7]. It could be applied if you're looking at multiple CEAP classes simultaneously within the same group of patients.

4. **Logistic Regression**: This can be used for analyzing dichotomous outcomes while considering multiple factors simultaneously[7]. It could help in understanding how various factors (including gender) influence the likelihood of a patient being in a specific CEAP class.

5. **Nonparametric Tests**: For ordinal data or when normality assumptions are not met, you might consider tests like the Wilcoxon rank-sum test (for two independent samples) or the Kruskal-Wallis test (for more than two groups)[2].

When choosing an alternative test, consider:
- Sample size
- Nature of the data (paired vs. unpaired, ordinal vs. nominal)
- Number of groups being compared
- Expected frequencies in each cell

Remember that the choice of test should align with your research question and the structure of your data[10].

Citations:
[1] https://ksumsc.com/download_center/Archive/3rd/437/Teamwork/5-Research/1-Theoratical/21-Categorical%20variables%20(tests).pdf
[2] https://www.grandacademicportal.education/assets/images/documents/20180623112719.pdf
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC4966396/
[4] https://bookdown.org/anshul302/HE802-MGHIHP-Spring2020/Chi2.html
[5] https://study.com/skill/learn/how-to-compare-two-or-more-sets-of-categorical-data-explanation.html
[6] https://www.youtube.com/watch?v=ruhV3sDZ-00
[7] https://www.brookes.ac.uk/students/academic-development/maths-and-stats/statistics/inferential-statistics/categorical-data
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC3096219/
[9] https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/
[10] https://www.scribbr.com/statistics/statistical-tests/
'''
'''
Certainly! Let's implement these tests one by one using Python. We'll use the same data structure as before, focusing on individual CEAP classes.

1. Fisher's Exact Test

```python
from scipy.stats import fisher_exact

def fisher_test(table, class_name):
    class_data = table[[class_name]]
    class_data['Other'] = table.sum(axis=1) - class_data[class_name]
    _, p_value = fisher_exact(class_data)
    return p_value

print("Fisher's Exact Test Results:")
for ceap_class in contingency_table.columns:
    p_value = fisher_test(contingency_table, ceap_class)
    print(f"{ceap_class} - p-value: {p_value:.4f}")
```

2. McNemar's Test
(Note: This test is for paired data, which doesn't directly apply to your current data structure. Here's how you'd use it if you had paired data:)

```python
from statsmodels.stats.contingency_tables import mcnemar

# Assuming you have paired data in this format:
# paired_data = [[before_treatment_class, after_treatment_class], ...]

def mcnemar_test(paired_data):
    table = [[sum(1 for x in paired_data if x == [0, 0]),
              sum(1 for x in paired_data if x == [0, 1])],
             [sum(1 for x in paired_data if x == [1, 0]),
              sum(1 for x in paired_data if x == [1, 1])]]
    result = mcnemar(table, exact=False)
    return result.pvalue

# Example usage (with hypothetical data):
# paired_data = [[0, 1], [1, 1], [0, 0], [1, 0], ...]
# p_value = mcnemar_test(paired_data)
# print(f"McNemar's Test p-value: {p_value:.4f}")
```

3. Logistic Regression

```python
import statsmodels.api as sm

def logistic_regression(table, class_name):
    y = table[class_name]
    X = sm.add_constant(table.index.map({'Male': 0, 'Female': 1}))
    model = sm.Logit(y, X).fit()
    return model.pvalues['x1']  # p-value for the gender coefficient

print("\nLogistic Regression Results:")
for ceap_class in contingency_table.columns:
    p_value = logistic_regression(contingency_table, ceap_class)
    print(f"{ceap_class} - p-value: {p_value:.4f}")
```

4. Wilcoxon Rank-Sum Test (Mann-Whitney U Test)

```python
from scipy.stats import mannwhitneyu

def wilcoxon_test(table, class_name):
    male_data = [class_name] * table.loc['Male', class_name] + ['Other'] * (table.loc['Male'].sum() - table.loc['Male', class_name])
    female_data = [class_name] * table.loc['Female', class_name] + ['Other'] * (table.loc['Female'].sum() - table.loc['Female', class_name])
    statistic, p_value = mannwhitneyu(male_data, female_data, alternative='two-sided')
    return p_value

print("\nWilcoxon Rank-Sum Test Results:")
for ceap_class in contingency_table.columns:
    p_value = wilcoxon_test(contingency_table, ceap_class)
    print(f"{ceap_class} - p-value: {p_value:.4f}")
```

These tests provide different perspectives on the data:

- Fisher's Exact Test is particularly useful for small sample sizes.
- Logistic Regression can help understand the effect of gender on the odds of being in a specific CEAP class.
- The Wilcoxon Rank-Sum Test is non-parametric and doesn't assume normality.

Remember to interpret the results in the context of your research question and the nature of your data. Each test has its own assumptions and limitations.

'''
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Create the contingency table
data = {
    'NA': [52, 53],
    'C0': [31, 36],
    'C1': [5, 6],
    'C2': [44, 54],
    'C3': [93, 156],
    'C4': [38, 59],
    'C5': [18, 35],
    'C6': [97, 99]
}

contingency_table = pd.DataFrame(data, index=['Male', 'Female'])

# Overall Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Overall Chi-Square statistic: {chi2:.4f}")
print(f"Overall p-value: {p_value:.4f}")

for ceap_class in contingency_table.columns:
    # Create a boolean mask for the current class
    mask = contingency_table[ceap_class] == contingency_table[ceap_class]
    
    # Use .loc to set values for the current class and 'Other'
    contingency_table.loc[mask, 'Current'] = contingency_table.loc[mask, ceap_class]
    contingency_table.loc[mask, 'Other'] = contingency_table.loc[mask, contingency_table.columns != ceap_class].sum(axis=1)
    
    # Perform chi-square test
    class_data = contingency_table.loc[mask, ['Current', 'Other']]
    print (f'{ceap_class} : {class_data}')
    chi2, p_value, dof, expected = chi2_contingency(class_data)
    
    print(f"{ceap_class} - Chi-Square: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Reset 'Current' and 'Other' columns for the next iteration
    contingency_table.drop(['Current', 'Other'], axis=1, inplace=True)
    
from scipy.stats import fisher_exact

from scipy.stats import fisher_exact
import pandas as pd
import numpy as np

def fisher_test(table, class_name):
    # Create a new DataFrame for the 2x2 contingency table
    contingency = pd.DataFrame(index=['Male', 'Female'], columns=['Current', 'Other'])
    
    # Fill the 'Current' column
    contingency['Current'] = table[class_name]
    
    # Calculate and fill the 'Other' column
    contingency['Other'] = table.sum(axis=1) - contingency['Current']
    
    # Perform Fisher's Exact Test
    stat, p_value = fisher_exact(contingency)
    
    return stat, p_value

# Assuming contingency_table is your original DataFrame
print("Fisher's Exact Test Results:")
for ceap_class in contingency_table.columns:
    stat, p_value = fisher_test(contingency_table, ceap_class)
    print(f"\n{ceap_class}:")
    print(f"{ceap_class} - Fisher's Exact: {stat:.4f}, p-value: {p_value:.4f}")
'''
Thank you for providing the results of the Fisher's Exact Test for each CEAP class. Let's analyze these findings:

1. **NA (No disease)**
   - p-value: 0.1727
   - Not statistically significant (p > 0.05)

2. **C0 (No visible signs)**
   - p-value: 0.6095
   - Not statistically significant

3. **C1 (Telangiectasias/reticular veins)**
   - p-value: 1.0000
   - Not statistically significant

4. **C2 (Varicose veins)**
   - p-value: 0.7459
   - Not statistically significant

5. **C3 (Edema)**
   - p-value: 0.0341
   - Statistically significant (p < 0.05)

6. **C4 (Skin changes)**
   - p-value: 0.4472
   - Not statistically significant

7. **C5 (Healed ulcer)**
   - p-value: 0.1979
   - Not statistically significant

8. **C6 (Active ulcer)**
   - p-value: 0.0493
   - Statistically significant (p < 0.05), but very close to the threshold

Key Findings:

1. C3 (Edema) shows a statistically significant difference between males and females (p = 0.0341).
2. C6 (Active ulcer) is just barely significant (p = 0.0493), which warrants careful interpretation.
3. All other CEAP classes do not show statistically significant differences between genders.

These results provide a more nuanced view compared to the Chi-square test, particularly for C3 and C6. The Fisher's Exact Test is more appropriate for smaller sample sizes or when expected frequencies are low, which might explain the differences in results.
'''