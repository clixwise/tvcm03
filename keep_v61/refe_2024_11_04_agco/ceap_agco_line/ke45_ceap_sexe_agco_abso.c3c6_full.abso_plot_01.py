import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame setup (replace this with your actual df_line)
data = {
    'name': ['YENGE MARCELA JOAO', 'YENGE MARCELA JOAO', 'YENGE MARCELA JOAO',
             'ROTT  MARC    JII', 'TSHIBASU KAGI DON'],
    'doss': ['D9972', 'D9972', 'D9972', 'D9844', 'D9921'],
    'age': [54, 54, 54, 50, 58],
    'sexe': ['F', 'F', 'F', 'M', 'M'],
    'limb': ['L', 'L', 'R', 'R', 'G'],
    'ceap': ['C2', 'C6', 'C2', 'C6', 'C2']
}

df_line = pd.DataFrame(data)

# Step 1: Aggregate ages per patient (doss), sexe, and CEAP class
df_agg = df_line.groupby(['doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

# Step 2: Perform Mann-Whitney U Test for each CEAP class
results = []
ceap_classes = df_agg['ceap'].unique()
print (ceap_classes)

for ceap in ceap_classes:
    ages_m = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'M')]['age']
    ages_f = df_agg[(df_agg['ceap'] == ceap) & (df_agg['sexe'] == 'F')]['age']
    
    if len(ages_m) > 0 and len(ages_f) > 0:  # Only run the test if both groups have data
        stat, p_value = mannwhitneyu(ages_m, ages_f, alternative='two-sided')
        results.append({'CEAP': ceap, 'p_value': p_value})
    else:
        results.append({'CEAP': ceap, 'p_value': None})

if True:
    # Step 3: Correct for multiple comparisons
    results_df = pd.DataFrame(results).dropna()
    p_values = results_df['p_value'].values
    _, p_corrected, _, _ = multipletests(p_values, method='bonferroni')  # Bonferroni correction
    results_df['p_value_corrected'] = p_corrected
    # Step 4: Display results
    print(results_df)

# ***
from statsmodels.stats.multitest import multipletests

# Step 3: Correct for multiple comparisons (Bonferroni and FDR)
results_df = pd.DataFrame(results).dropna()
p_values = results_df['p_value'].values

# Bonferroni correction
_, p_corrected_bonferroni, _, _ = multipletests(p_values, method='bonferroni')

# FDR correction (Benjamini-Hochberg)
_, p_corrected_fdr, _, _ = multipletests(p_values, method='fdr_bh')

# Add corrected p-values to the results
results_df['p_value_corrected_bonferroni'] = p_corrected_bonferroni
results_df['p_value_corrected_fdr'] = p_corrected_fdr

# Step 4: Display results
print(results_df)



# Example DataFrame for CEAP groups
data = {
    'ages': [
        np.random.randint(20, 60, size=30).tolist(),  # Absent
        np.random.randint(25, 70, size=30).tolist(),  # Mild
        np.random.randint(30, 90, size=30).tolist(),  # Severe
    ],
    'mean': [
        np.mean(np.random.randint(20, 60, size=30)),  # Mean for Absent
        np.mean(np.random.randint(25, 70, size=30)),  # Mean for Mild
        np.mean(np.random.randint(30, 90, size=30)),  # Mean for Severe
    ]
}

# Create DataFrame and set severity group as index
df = pd.DataFrame(data, index=['Absent', 'Mild', 'Severe'])

print("Example DataFrame for Boxplots:")
print(df)

"""
Generate a boxplot for CEAP groups showing age distributions.

Parameters:
- ceap_grop: Ceap_grop object with data
- pati_info: Pati_info object for patient details
- parm_data: Parm_data object for plot parameters
"""
# Extract the relevant dataframe
#df = ceap_grop.ceap_agco_abso_disp
print(f"pl33_ceap_agco: {type(df)}\n{df}\n:{df.index}\n:{df.columns}")

# Extract categories and data for plotting
ceap_categories = df.index
ages = df['ages']  # List of age data
means = df['mean']  # Mean age for each category

# Create boxplot
fig, ax = plt.subplots(figsize=(10, 6))
boxprops = dict(facecolor='orange', color='orange')  # Orange box properties
boxplot = ax.boxplot(ages, patch_artist=True, labels=ceap_categories, boxprops=boxprops)

# Overlay mean values as blue dots and annotate them
for i, mean in enumerate(means):
    ax.plot(i + 1, mean, 'bo')  # Blue dot marker at mean value
    ax.text(i + 1, mean + 1, f'{mean:.0f}', ha='center', va='bottom', color='blue', fontsize=10)

# Add titles and axis labels
'''
ax.set_title(
    f"Insuffisance veineuse des membres inférieurs [{parm_data.plot_titl}]\n"
    f"Age du patient\n[{pati_info.pati_coun} patients - {pati_info.mbre_coun} membres - {pati_info.ceap_coun} CEAP] "
    f"[période {pati_info.mini_date} à {pati_info.maxi_date}]"
)
'''
ax.set_xlabel('CEAP')
ax.set_ylabel('Age')

# Add gridlines
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', axis='y')


# Adjust layout and display the plot
plt.tight_layout()
plt.show()



# Data
ceap_classes = results_df['CEAP']
raw_p_values = results_df['p_value']
bonferroni_p_values = results_df['p_value_corrected_bonferroni']
fdr_p_values = results_df['p_value_corrected_fdr']

x = np.arange(len(ceap_classes))  # X-axis positions

# Plotting
plt.figure(figsize=(10, 6))
bar_width = 0.25

# Bars for each type of p-value
plt.bar(x - bar_width, raw_p_values, bar_width, label='Raw p-values', color='skyblue')
plt.bar(x, bonferroni_p_values, bar_width, label='Bonferroni-corrected', color='orange')
plt.bar(x + bar_width, fdr_p_values, bar_width, label='FDR-corrected', color='green')

# Significance line
plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Significance threshold (α=0.05)')

# Adding labels and legend
plt.xticks(x, ceap_classes, rotation=45)
plt.xlabel('CEAP Classes')
plt.ylabel('p-values')
plt.title('Raw and Corrected p-values by CEAP Class')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

