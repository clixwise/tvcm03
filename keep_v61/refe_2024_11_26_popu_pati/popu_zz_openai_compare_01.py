import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate synthetic reference data for 30,000 limbs
np.random.seed(42)  # For reproducibility
reference_sample_size = 30000

# Define proportions for each CEAP level in the reference sample
reference_proportions = {
    "C0": 0.2,
    "C1": 0.05,
    "C2": 0.15,
    "C3": 0.3,
    "C4": 0.1,
    "C5": 0.1,
    "C6": 0.1
}

# Create the reference dataset
ceap_levels = list(reference_proportions.keys())
reference_counts = (np.array(list(reference_proportions.values())) * reference_sample_size).astype(int)
reference_df = pd.DataFrame({
    "ceap": np.repeat(ceap_levels, reference_counts)
})

# Verify reference distribution
reference_distribution = reference_df["ceap"].value_counts(normalize=True)
print("Reference Distribution (Proportions):\n", reference_distribution)

# Step 2: Your sample data (724 limbs)
sample_counts = np.array([67, 11, 98, 249, 97, 53, 196])  # Counts from your sample
sample_size = sample_counts.sum()
sample_df = pd.DataFrame({
    "ceap": np.repeat(ceap_levels, sample_counts)
})

# Step 3: Calculate expected counts based on reference proportions
expected_counts = np.array(list(reference_proportions.values())) * sample_size
print("\nExpected Counts (from reference proportions):", expected_counts)

# Step 4: Perform Chi-Square Goodness-of-Fit Test
chi2_stat, p_value = chisquare(f_obs=sample_counts, f_exp=expected_counts)

# Step 5: Output results
print("\nObserved Counts:", sample_counts)
print("Expected Counts:", expected_counts)
print(f"\nChi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.4f}")

if p_value <= 0.05:
    print("Reject H0: Your sample distribution significantly differs from the reference distribution.")
else:
    print("Fail to Reject H0: Your sample distribution matches the reference distribution.")

# Step 6: Visualization
# Bar plot comparing observed and expected counts
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(ceap_levels))

# Plot bars
ax.bar(x - bar_width/2, sample_counts, bar_width, label='Observed', color='skyblue')
ax.bar(x + bar_width/2, expected_counts, bar_width, label='Expected', color='orange')

# Add labels and title
ax.set_xlabel('CEAP Levels', fontsize=12)
ax.set_ylabel('Counts', fontsize=12)
ax.set_title('Comparison of Observed and Expected Counts', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(ceap_levels)
ax.legend()

# Display bar plot
plt.tight_layout()
plt.show()

# Heatmap of proportions
observed_proportions = sample_counts / sample_size
heatmap_data = pd.DataFrame({
    "Observed": observed_proportions,
    "Expected": reference_proportions.values()
}, index=ceap_levels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax)
ax.set_title("Proportions Heatmap: Observed vs Expected", fontsize=14)
plt.show()
