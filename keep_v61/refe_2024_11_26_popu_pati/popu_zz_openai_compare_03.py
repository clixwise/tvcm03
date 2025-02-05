import pandas as pd
import numpy as np
from scipy.stats import chisquare

if False:
    dfTA_ = np.array([67, 98, 249, 97, 53, 196])
    dfTB_ = np.array([294, 439, 412, 88, 19, 3])

    # Step 4: Perform Chi-Square Goodness-of-Fit Test
    chi2_stat, p_value = chisquare(f_obs=dfTA_, f_exp=dfTB_)

dataTA = {
    'M': [31, 44, 93, 38, 18, 97],
    'F': [36, 54, 156, 59, 35, 99],
    'T': [67, 98, 249, 97, 53, 196]
}
dfTA = pd.DataFrame(dataTA, index=['C0', 'C2', 'C3', 'C4', 'C5', 'C6'])
dfTA.index.name = 'ceap'
dataTB = {
    'M': [184, 167, 156, 42, 8, 2],
    'F': [110, 272, 256, 46, 11, 1],
    'T': [294, 439, 412, 88, 19, 3]
}
dfTB = pd.DataFrame(dataTB, index=['C0', 'C2', 'C3', 'C4', 'C5', 'C6'])
dfTB.index.name = 'ceap'

dfTA_ = dfTA['T'].values
dfMA_ = dfTA['M'].values
dfFA_ = dfTA['F'].values

dfTB_ = dfTB['T'].values
dfMB_ = dfTB['M'].values
dfFB_ = dfTB['F'].values

print (dfTA_)
print (dfTB_)

# Calculate the sums of the observed and expected frequencies
sum_observed = np.sum(dfTA_)
sum_expected = np.sum(dfTB_)

# Step 4: Perform Chi-Square Goodness-of-Fit Test
chi2_stat, p_value = chisquare(f_obs=dfTA_, f_exp=dfTB_)

# Normalize the frequencies so that their sums are equal
dfTA_normalized = dfTA_ * (sum_expected / sum_observed)
dfTB_normalized = dfTB_

# Print the normalized arrays
print("Normalized dfTA_:", dfTA_normalized)
print("Normalized dfTB_:", dfTB_normalized)

# Step 4: Perform Chi-Square Goodness-of-Fit Test
chi2_stat, p_value = chisquare(f_obs=dfTA_normalized, f_exp=dfTB_normalized)

# Step 5: Output results
print(f"\nChi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.4f}")

if p_value <= 0.05:
    print("Reject H0: Your sample distribution significantly differs from the reference distribution.")
else:
    print("Fail to Reject H0: Your sample distribution matches the reference distribution.")
# Step 5: Compute the standardized residuals
standardized_residuals = (dfTA_normalized - dfTB_normalized) / np.sqrt(dfTB_normalized)

# Print the standardized residuals
print("Standardized Residuals:", standardized_residuals)