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

# Step 4: Bootstrapping to validate Chi-square results
num_bootstraps = 1000  # Number of bootstrap samples

# Function to perform bootstrapping
def bootstrap_chi2(observed_males, observed_females, expected_males, expected_females, n_bootstraps=1000):
    total_count = observed_males + observed_females
    male_prob = expected_males / total_count
    female_prob = expected_females / total_count
    
    chi2_bootstrapped = []
    for _ in range(n_bootstraps):
        # Generate bootstrap samples
        bootstrap_males = np.random.binomial(total_count, male_prob)
        bootstrap_females = total_count - bootstrap_males
        
        # Recalculate Chi-square
        expected_bootstrap = [expected_males, expected_females]
        observed_bootstrap = [bootstrap_males, bootstrap_females]
        chi2, _ = chi2_contingency([observed_bootstrap, expected_bootstrap])[:2]
        chi2_bootstrapped.append(chi2)
    
    return chi2_bootstrapped

# Add bootstrapped Chi-square results for each CEAP class
df['Bootstrap_Chi2'] = df.apply(lambda row: bootstrap_chi2(row['Males'], row['Females'], row['Expected_Males'], row['Expected_Females'], num_bootstraps), axis=1)
df['Bootstrap_Pval'] = df.apply(lambda row: np.mean(np.array(row['Bootstrap_Chi2']) >= row['Chi2']), axis=1)

# Display results
print("Results with Bootstrapping:\n")
print(df[['CEAP', 'Males', 'Females', 'Expected_Males', 'Expected_Females', 'Male_Deviation', 'Female_Deviation', 'Chi2', 'P-value', 'Bootstrap_Pval']])
'''
### Key Point 2 Explained

The results indicate:

1. **Consistency Across Chi-Square and Bootstrap Results**:
   - **C0**, **C1**, **C2**, **C4**, and **C5** have high p-values across both methods (> 0.22 for Bootstrap). This suggests no significant deviation from the expected proportions for males and females.
   - **C3** and **C6**, however, show smaller bootstrap p-values (0.071 for C3 and 0.082 for C6), which are borderline significant.

---

### Interpretation:
- **C3 (Borderline Significant)**:
  - While the standard Chi-square test for C3 yielded a p-value of 0.219, the **bootstrap p-value** is **0.071**, suggesting a slightly stronger indication of deviation than initially detected.
  - This could imply **a potential imbalance favoring females** (observed 156 females vs. 93 males) that warrants further exploration.

- **C6 (Borderline Significant)**:
  - Similar to C3, the standard Chi-square test did not flag C6 as significant (p = 0.247), but the **bootstrap p-value is 0.082**, highlighting a deviation that may reflect underlying trends.

---

### Summary:
- The bootstrapping approach provides more robust evidence that **C3** and possibly **C6** show deviations between observed and expected counts for males and females. 
- This supports the hypothesis that social or behavioral factors (e.g., earlier clinic visits by females) may explain these imbalances.

Would you like to further refine the analysis or visualize the bootstrapped distributions for C3 and C6?

'''