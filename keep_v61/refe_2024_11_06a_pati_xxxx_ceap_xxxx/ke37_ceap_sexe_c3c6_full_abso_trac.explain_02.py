import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Observed counts from your table
observed_data = {
    'CEAP': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'Males': [31, 5, 44, 93, 38, 18, 97],
    'Females': [36, 6, 54, 156, 59, 35, 99]
}

df = pd.DataFrame(observed_data)

total_males = 378   # Total males in the dataset
total_females = 498 # Total females in the dataset
total_population = total_males + total_females

# Global sex ratios
male_ratio = total_males / total_population  # ~43%
female_ratio = total_females / total_population  # ~57%

# Step 1: Compute expected counts based on global proportions
df['Expected_Males'] = (df['Males'] + df['Females']) * male_ratio
df['Expected_Females'] = (df['Males'] + df['Females']) * female_ratio

# Step 2: Compare observed vs expected counts
df['Male_Deviation'] = df['Males'] - df['Expected_Males']
df['Female_Deviation'] = df['Females'] - df['Expected_Females']

# Step 3: Perform a Chi-square goodness-of-fit test for each CEAP class
def chi_square_test(row):
    observed = [row['Males'], row['Females']]
    expected = [row['Expected_Males'], row['Expected_Females']]
    chi2, pval = chi2_contingency([observed, expected])[:2]
    return pd.Series({'Chi2': chi2, 'P-value': pval})

tests = df.apply(chi_square_test, axis=1)
df = pd.concat([df, tests], axis=1)

# Display results
print(df[['CEAP', 'Males', 'Females', 'Expected_Males', 'Expected_Females', 'Male_Deviation', 'Female_Deviation', 'Chi2', 'P-value']])
'''
  CEAP  Males  Females  Expected_Males  Expected_Females  Male_Deviation  Female_Deviation      Chi2   P-value
0   C0     31       36       28.910959         38.089041        2.089041         -2.089041  0.035804  0.849921
1   C1      5        6        4.746575          6.253425        0.253425         -0.253425  0.000000  1.000000
2   C2     44       54       42.287671         55.712329        1.712329         -1.712329  0.010505  0.918363
3   C3     93      156      107.445205        141.554795      -14.445205         14.445205  1.509390  0.219232
4   C4     38       59       41.856164         55.143836       -3.856164          3.856164  0.173623  0.676912
5   C5     18       35       22.869863         30.130137       -4.869863          4.869863  0.596365  0.439968
6   C6     97       99       84.575342        111.424658       12.424658        -12.424658  1.339118  0.247190
### Interpretation of Results

Here is a breakdown of the output:

| CEAP | Chi2   | P-value  | Interpretation                       |
|------|--------|----------|--------------------------------------|
| C0   | 0.0358 | 0.8499   | No significant deviation.            |
| C1   | 0.0000 | 1.0000   | Observed matches expected perfectly. |
| C2   | 0.0105 | 0.9184   | No significant deviation.            |
| C3   | 1.5094 | 0.2192   | No significant deviation.            |
| C4   | 0.1736 | 0.6769   | No significant deviation.            |
| C5   | 0.5964 | 0.4400   | No significant deviation.            |
| C6   | 1.3391 | 0.2472   | No significant deviation.            |

### Key Points:
1. **P-values > 0.05**: None of the CEAP classes show a statistically significant deviation between observed and expected counts after adjusting for the global M/F ratio.
2. **C3 and C6**: These classes had notable male/female differences in the raw data, but once standardized to the global population proportions, the differences are no longer significant. This suggests that the observed imbalance might have been influenced by the global sex ratio rather than a clinical or behavioral factor.
3. **C1**: Perfect alignment between observed and expected counts is likely due to small counts.

---

### Next Steps:
If you are looking for further refinements:
- **Bootstrapping**: Use resampling methods to validate the robustness of the observed deviations.
- **Logistic Regression**: Test the association of CEAP classes with gender while adjusting for covariates (e.g., age or clinic visit timing).

Let me know if you'd like me to explore one of these!
'''
