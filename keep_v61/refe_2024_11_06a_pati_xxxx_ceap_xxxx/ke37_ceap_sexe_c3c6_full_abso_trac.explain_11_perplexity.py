'''
To ascertain whether the differences in CEAP class percentages between males and females are statistically significant, we can use the Chi-Square Test of Independence. This test will help us examine each CEAP class separately from a gender viewpoint. Here's how we can do this using Python:

## Chi-Square Test Implementation

First, we need to import the necessary libraries and create our contingency table:
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
# Now, let's perform the Chi-Square test for the entire table:
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Overall Chi-Square statistic: {chi2}")
print(f"Overall p-value: {p_value}")
#To examine each CEAP class separately, we can create a function to perform the test for individual classes:
def test_individual_class(table, class_name):
    class_data = table[[class_name]]
    chi2, p_value, _, _ = chi2_contingency(class_data)
    return chi2, p_value
# Test each class
for ceap_class in contingency_table.columns:
    chi2, p_value = test_individual_class(contingency_table, ceap_class)
    print(f"{ceap_class} - Chi-Square: {chi2:.4f}, p-value: {p_value:.4f}")
'''
## Interpretation

The overall Chi-Square test will tell us if there's a significant association between gender and CEAP classification in general[1]. If the p-value is less than 0.05, we can conclude that there's a statistically significant difference in the distribution of CEAP classes between males and females[3].

For individual classes, a p-value less than 0.05 indicates a significant difference between genders for that specific CEAP class[4]. For example, if the p-value for C3 is less than 0.05, we can conclude that the difference between C3 in males (25%) and females (31%) is statistically significant.

## Additional Considerations

1. **Effect Size**: While the Chi-Square test tells us if there's a significant difference, it doesn't tell us the magnitude of that difference. Consider calculating Cramer's V for effect size[1].

2. **Post-hoc Analysis**: If the overall test is significant, you might want to perform post-hoc tests to identify which specific cells in the contingency table contribute most to the chi-square statistic[2].

3. **Sample Size**: Ensure your sample size is large enough for the Chi-Square test to be valid. Generally, each cell in the contingency table should have an expected frequency of at least 5[4].

4. **Visualization**: Consider creating a stacked bar chart to visually represent the proportions of each CEAP class by gender, which can complement your statistical analysis.

By following this approach, you can systematically examine the gender differences in each CEAP class and determine which differences are statistically significant[5].

Citations:
[1] https://www.stratascratch.com/blog/chi-square-test-in-python-a-technical-guide/
[2] https://help.imsl.com/python/current/docs/stat/discrete/contingencyTable.html
[3] https://towardsdatascience.com/chi-squared-test-revealing-hidden-patterns-in-your-data-d939df2dda71
[4] https://www.jonathanbossio.com/post/two-sample-chi-square-test-with-python
[5] https://www.statsmodels.org/stable/contingency_tables.html
[6] https://stackoverflow.com/questions/66633313/cochran-armitage-test-for-ordinal-nominal-contingency-table-in-python
[7] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
[8] https://stackoverflow.com/questions/24767161/can-we-generate-contingency-table-for-chisquare-test-using-python
'''