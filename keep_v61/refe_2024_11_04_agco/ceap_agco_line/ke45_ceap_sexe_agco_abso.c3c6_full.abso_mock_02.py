import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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
