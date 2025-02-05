import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Create the DataFrame
data = [
    [0, 0, 0, 2, 18, 5, 0, 22],
    [0, 0, 0, 1, 8, 2, 2, 14],
    [0, 0, 1, 0, 3, 2, 1, 0],
    [3, 1, 1, 10, 14, 3, 3, 18],
    [20, 9, 0, 15, 80, 16, 3, 14],
    [9, 2, 1, 3, 17, 23, 7, 5],
    [7, 5, 1, 5, 4, 2, 6, 8],
    [39, 29, 3, 33, 9, 5, 7, 21]
]

index = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='Lcea')
columns = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='Rcea')

df_tabl = pd.DataFrame(data, index=index, columns=columns)

# Print DataFrame info
print(f"df_tabl.size: {df_tabl.size}")
print(f"df_tabl.type: {type(df_tabl)}")
print(df_tabl)
print(f"Index: {df_tabl.index}")
print(f"Columns: {df_tabl.columns}")

# Calculate chi-square test
chi2, p, dof, expected = chi2_contingency(df_tabl)

# Calculate standardized residuals
observed = df_tabl.values
standardized_residuals = (observed - expected) / np.sqrt(expected)

# Calculate adjusted residuals
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
total = observed.sum()

row_props = row_totals / total
col_props = col_totals / total

adjusted_residuals = (observed - expected) / np.sqrt(
    expected * (1 - row_props[:, np.newaxis]) * (1 - col_props)
)

# Create DataFrames for residuals
df_standardized = pd.DataFrame(standardized_residuals, index=index, columns=columns)
df_adjusted = pd.DataFrame(adjusted_residuals, index=index, columns=columns)

# Function to rank residuals
def rank_residuals(residuals):
    return pd.DataFrame(residuals).melt().sort_values('value', key=abs, ascending=False)

# Rank residuals
ranked_standardized = rank_residuals(standardized_residuals)
ranked_adjusted = rank_residuals(adjusted_residuals)

# Add cell labels and residual type
ranked_standardized['cell'] = [f'{i}{j}' for i in index for j in columns]
ranked_standardized['type'] = 'Standardized'
ranked_adjusted['cell'] = [f'{i}{j}' for i in index for j in columns]
ranked_adjusted['type'] = 'Adjusted'

# Print top 10 ranked residuals for each type
print("\nTop 10 Standardized Residuals:")
print(ranked_standardized[['cell', 'value']].head(10).to_string(index=False))
print("\nTop 10 Adjusted Residuals:")
print(ranked_adjusted[['cell', 'value']].head(10).to_string(index=False))

# Calculate Spearman correlation between rankings
spearman_corr = ranked_standardized['value'].abs().corr(ranked_adjusted['value'].abs(), method='spearman')

# Check for NaN and interpret
if np.isnan(spearman_corr):
    print("\nPerfect correlation between standardized and adjusted residual rankings (identical rankings)")
else:
    print(f"\nSpearman correlation between standardized and adjusted residual rankings: {spearman_corr:.4f}")

# Identify cells with consistently high deviations
high_deviation_cells = pd.concat([ranked_standardized, ranked_adjusted]).groupby('cell').filter(lambda x: all(abs(x['value']) > 1.96))

if not high_deviation_cells.empty:
    print("\nCells with consistently high deviations (|residual| > 1.96):")
    print(high_deviation_cells[['cell', 'type', 'value']])
else:
    print("\nNo cells show consistently high deviations across both residual types.")
