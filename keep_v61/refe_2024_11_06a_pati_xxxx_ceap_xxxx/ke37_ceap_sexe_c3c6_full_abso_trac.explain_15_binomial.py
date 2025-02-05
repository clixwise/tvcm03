'''
Yes, binomial and proportional tests can also be applied for class-by-class comparisons of CEAP classifications between genders. Here's how they apply:

1. Binomial Test: This test is suitable for comparing the proportion of one gender in a specific CEAP class to an expected proportion[3]. For example, you could test whether the proportion of females in CEAP class C3 differs significantly from 0.5 (equal distribution between genders).

2. Test of Proportions: This test compares the proportions of two independent groups[5]. In your case, it would compare the proportion of males to females for each CEAP class separately.

These tests are particularly useful when:

- You're focusing on one CEAP class at a time
- You want to compare the gender distribution within each class to an expected distribution (binomial test)
- You want to directly compare the proportion of males to females in each class (test of proportions)

Both tests assume independent observations and are suitable for categorical data, making them appropriate for your CEAP classification data[1][5].

Citations:
[1] https://www.enago.com/academy/right-statistical-test/
[2] https://blog.addgene.org/deep-dive-statistical-tests-comparisons
[3] https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/
[4] https://www.statstutor.ac.uk/resources/uploaded/tutorsquickguidetostatistics.pdf
[5] https://www.scribbr.com/statistics/statistical-tests/
[6] https://www.researchgate.net/post/What-statistical-test-to-use-in-comparing-class-grades-A-B-C-How-to-do-so-in-Graphpad-PRISM
[7] https://www.researchgate.net/post/Which_statistical_test_to_use_for_comparison_of_different_groups_in_a_categorization
'''
'''
Certainly! Let's implement both the Binomial Test and the Test of Proportions for each CEAP class using Python. We'll use the scipy library for these tests.

```python
'''
import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'df' is your DataFrame with 876 rows of individual patient data
df_line = None
if df_line: # TODO ; see df-table implementation below
    def binomial_test(data):
        results = {}
        total_patients = len(data)
        expected_proportion = 0.5  # Assuming equal distribution as null hypothesis

        for ceap_class in data['ceap'].unique():
            class_data = data[data['ceap'] == ceap_class]
            female_count = sum(class_data['sexe'] == 'F')
            total_count = len(class_data)
            
            p_value = stats.binomtest(female_count, total_count, expected_proportion).pvalue
            results[ceap_class] = p_value

        return results

    def proportion_test(data):
        results = {}
        for ceap_class in data['ceap'].unique():
            class_data = data[data['ceap'] == ceap_class]
            female_count = sum(class_data['sexe'] == 'F')
            male_count = sum(class_data['sexe'] == 'M')
            total_count = len(class_data)
            
            female_prop = female_count / total_count
            male_prop = male_count / total_count
            
            z_statistic, p_value = stats.proportions_ztest([female_count, male_count], 
                                                        [total_count, total_count])
            results[ceap_class] = p_value

        return results

    # Run the tests
    binomial_results = binomial_test(df_line)
    proportion_results = proportion_test(df_line)

    # Print results
    print("Binomial Test Results:")
    for ceap_class, p_value in binomial_results.items():
        print(f"{ceap_class} - p-value: {p_value:.4f}")

    print("\nTest of Proportions Results:")
    for ceap_class, p_value in proportion_results.items():
        print(f"{ceap_class} - p-value: {p_value:.4f}")
'''

This code does the following:

1. The `binomial_test` function:
   - Tests whether the proportion of females in each CEAP class differs significantly from 0.5.
   - You can adjust the `expected_proportion` if you have a different null hypothesis.

2. The `proportion_test` function:
   - Compares the proportion of males to females in each CEAP class.
   - Uses a two-sided z-test for proportions.

Both functions return a dictionary with CEAP classes as keys and p-values as values.

Remember to interpret these results in the context of your study:
- A low p-value (typically < 0.05) suggests a significant difference from the expected proportion (binomial test) or between male and female proportions (test of proportions).
- Consider adjusting for multiple comparisons if you're testing all CEAP classes simultaneously.

These tests provide additional perspectives on gender differences within each CEAP class, complementing your earlier analyses with Chi-square and Fisher's Exact tests.
'''
'''
Certainly! I'll recode both the Binomial Test and the Test of Proportions using the contingency table (df_table) as the starting point. Here's the implementation:

'''
import pandas as pd
import numpy as np
from scipy import stats

# Assuming df_table is your contingency table with rows for Male and Female, and columns for CEAP classes
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

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

df_table = pd.DataFrame(data, index=['Male', 'Female'])
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def binomial_test_from_table(table):
    results = {}
    expected_proportion = 0.5  # Assuming equal distribution as null hypothesis

    for ceap_class in table.columns:
        female_count = table.loc['Female', ceap_class]
        male_count = table.loc['Male', ceap_class]
        total_count = table[ceap_class].sum()
        
        result = stats.binomtest(female_count, total_count, expected_proportion)
        result = stats.binomtest(male_count, total_count, expected_proportion)
        results[ceap_class] = (result.statistic, result.pvalue)

    return results

def proportion_test_from_table(table):
    results = {}
    for ceap_class in table.columns:
        female_count = table.loc['Female', ceap_class]
        male_count = table.loc['Male', ceap_class]
        total_count = female_count + male_count
        
        count = np.array([female_count, male_count])
        nobs = np.array([total_count, total_count])
        print (f'ceap: {ceap_class} proportion_test: count: {count}')
        print (f'ceap: {ceap_class} proportion_test: nobs: {nobs}')
        
        stat, p_value = proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False)
        results[ceap_class] = (stat, p_value)

    return results

# Run the tests
binomial_results = binomial_test_from_table(df_table)
proportion_results = proportion_test_from_table(df_table)

# Print results
print("Binomial Test Results:")
for ceap_class, (statistic, p_value) in binomial_results.items():
    print(f"{ceap_class} - statistic: {statistic:.4f}, p-value: {p_value:.4f}")

print("\nTest of Proportions Results:")
for ceap_class, (statistic, p_value) in proportion_results.items():
    print(f"{ceap_class} - z-statistic: {statistic:.4f}, p-value: {p_value:.4f}")
'''
Both tests utilize totals based on counts from the same CEAP class, but they serve different purposes:
!!! The Binomial Test focuses on whether one gender's proportion deviates from an expected distribution.
!!! The Proportions Test directly compares the proportions of two groups within that same class.
This code does the following:

1. The `binomial_test_from_table` function:
   - Uses the contingency table to extract counts for each CEAP class.
   - Tests whether the proportion of females in each CEAP class differs significantly from 0.5.

2. The `proportion_test_from_table` function:
   - Extracts male and female counts for each CEAP class from the contingency table.
   - Compares the proportion of males to females in each CEAP class using a two-sided z-test.

Both functions return dictionaries with CEAP classes as keys and p-values as values.

These implementations should provide the same results as the previous versions that used the individual patient data, assuming the contingency table accurately represents the original data. The main difference is in how we extract the counts from the table structure rather than counting from individual records.

Remember to interpret these results in the context of your study, considering the significance level and potentially adjusting for multiple comparisons if necessary.

'''
'''
Thank you for providing the results. Let's analyze these findings for both the Binomial Test and the Test of Proportions:

Binomial Test Results:
This test compares the proportion of females in each CEAP class to an expected 50% (0.5).

1. Significant differences (p < 0.05):
   - C3: 62.65% female (p = 0.0001)
   - C4: 60.82% female (p = 0.0417)
   - C5: 66.04% female (p = 0.0270)

2. No significant differences:
   - NA, C0, C1, C2, C6

Test of Proportions Results:
This test compares the proportions of males and females in each CEAP class.

1. Significant differences (p < 0.05):
   - C3: z = 5.6462, p < 0.0001
   - C4: z = 3.0154, p = 0.0026
   - C5: z = 3.3024, p = 0.0010

2. No significant differences:
   - NA, C0, C1, C2, C6

Key Findings:
1. Both tests agree on significant gender differences in C3, C4, and C5 classes.
2. C3 (Edema) shows the most significant gender difference in both tests.
3. Classes NA, C0, C1, C2, and C6 show no significant gender differences in either test.
4. The proportion of females is consistently higher in the significant classes (C3, C4, C5).

Interpretation:
- There's strong evidence of gender disparity in CEAP classes C3, C4, and C5, with a higher proportion of females in these classes.
- The most pronounced difference is in C3 (Edema), suggesting this condition is significantly more common in females.
- Early stages (C0, C1, C2) and the most severe stage (C6) don't show significant gender differences.

These results provide valuable insights into the gender distribution across different CEAP classes, highlighting potential areas for further research or targeted interventions in venous disease management.

'''