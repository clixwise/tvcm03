'''
To **refine the analysis and visualize bootstrapped distributions** for **C3** and **C6**, we can add the following improvements:

1. Extract **C3** and **C6** bootstrapped Chi-square values.
2. Plot histograms of the bootstrapped Chi-square statistics.
3. Overlay the observed Chi-square value for comparison.

Here's the **updated code**:

### Key Changes:
1. **Visualization for C3 and C6**: Histograms of bootstrapped Chi-square values are generated for C3 and C6.
2. **Overlay Observed Chi-square**: A red dashed line represents the observed Chi-square value for easy comparison.

Run this updated code, and you will see visual insights alongside your refined statistical results. Let me know if further refinements are needed!
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Step 5: Visualization for C3 and C6
for ceap_class in ['C3', 'C6']:
    row = df[df['CEAP'] == ceap_class].iloc[0]
    print (row)
    plt.figure(figsize=(8, 5))
    plt.hist(row['Bootstrap_Chi2'], bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Bootstrapped Chi2')
    plt.axvline(row['Chi2'], color='red', linestyle='--', linewidth=2, label=f'Observed Chi2 = {row.Chi2:.3f}')
    plt.title(f'Bootstrapped Chi-square Distribution for {ceap_class}')
    plt.xlabel('Chi-square Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Display results
print("Results with Bootstrapping:\n")
print(df[['CEAP', 'Males', 'Females', 'Expected_Males', 'Expected_Females', 'Male_Deviation', 'Female_Deviation', 'Chi2', 'P-value', 'Bootstrap_Pval']])
