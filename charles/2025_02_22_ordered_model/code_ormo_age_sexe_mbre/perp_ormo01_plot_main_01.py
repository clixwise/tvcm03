import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------
# VERSION AUTONOME DES PLOTS
# ces plots sont également intégrés dans 'perp_ormo01_.py'
# -------------

'''
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sexe          -0.0380      0.122     -0.312      0.755      -0.277       0.201
mbre           0.5186      0.121      4.299      0.000       0.282       0.755
age            0.0056      0.004      1.439      0.150      -0.002       0.013
0/1           -1.4662      0.248     -5.902      0.000      -1.953      -0.979
1/2           -0.5248      0.119     -4.414      0.000      -0.758      -0.292
2/3           -2.5382      0.300     -8.472      0.000      -3.125      -1.951
3/4           -0.5236      0.096     -5.429      0.000      -0.713      -0.335
4/5            0.1825      0.057      3.209      0.001       0.071       0.294
5/6           -0.6851      0.097     -7.083      0.000      -0.875      -0.496
6/7           -1.1281      0.134     -8.439      0.000      -1.390      -0.866
==============================================================================
'''
results = pd.DataFrame({
    'coef': [-0.038, 0.519, 0.006, -1.466, -0.525, -2.538, -0.524, 0.183, -0.685, -1.128],
    'std err': [0.122, 0.121, 0.004, 0.248, 0.119, 0.300, 0.096, 0.057, 0.097, 0.134],
    'odds_ratio': [0.963, 1.680, 1.006, 0.231, 0.592, 0.079, 0.592, 1.200, 0.504, 0.324],
    'ci_lower': [0.758, 1.326, 0.998, 0.142, 0.469, 0.044, 0.490, 1.074, 0.417, 0.249],
    'ci_upper': [1.222, 2.128, 1.013, 0.376, 0.747, 0.142, 0.716, 1.342, 0.609, 0.421]
}, index=['sexe', 'mbre', 'age', '0/1', '1/2', '2/3', '3/4', '4/5', '5/6', '6/7'])

# Extract only CEAP transitions
transitions = ['sexe', 'mbre', 'age', '0/1', '1/2', '2/3', '3/4', '4/5', '5/6', '6/7']
df_transitions = results.loc[transitions]

# PLOT 1
# ------
plt.figure(figsize=(8, 5))
plt.errorbar(transitions, df_transitions['odds_ratio'], 
             yerr=[df_transitions['odds_ratio'] - df_transitions['ci_lower'], 
                   df_transitions['ci_upper'] - df_transitions['odds_ratio']], 
             fmt='o', capsize=5, label="Odds Ratio")

# Labels & Styling
plt.axhline(y=1, color='gray', linestyle='--', label="No Effect (OR=1)")
plt.ylabel("Odds Ratio (log scale)")
plt.yscale("log")  # Log scale for better visualization
plt.xlabel("CEAP Transitions")
plt.title("Odds Ratios for CEAP Transitions (with 95% CI)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Annotate the plot
for i, transition in enumerate(transitions):
    or_value = df_transitions.loc[transition, 'odds_ratio']
    if or_value > 1:
        plt.text(i, or_value, f'Increased\nOR={or_value:.3f}', ha='center', va='bottom', color='green')
    elif or_value < 1:
        plt.text(i, or_value, f'Decreased\nOR={or_value:.3f}', ha='center', va='top', color='red')
    else:
        plt.text(i, or_value, f'No Effect\nOR={or_value:.3f}', ha='center', va='bottom', color='gray')

plt.show()
pass

# PLOT 2
# ------
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot coefficients and confidence intervals
results['coef'].plot(kind='bar', yerr=results['std err'], ax=ax1, capsize=5)
ax1.set_title('Coefficients with 95% Confidence Intervals')
ax1.set_ylabel('Coefficient Value')
ax1.axhline(y=0, color='r', linestyle='--')

# Plot odds ratios and confidence intervals
results['odds_ratio'].plot(kind='bar', yerr=[results['odds_ratio'] - results['ci_lower'], 
                                             results['ci_upper'] - results['odds_ratio']], 
                           ax=ax2, capsize=5, log=True)
ax2.set_title('Odds Ratios with 95% Confidence Intervals')
ax2.set_ylabel('Odds Ratio (log scale)')
ax2.axhline(y=1, color='r', linestyle='--')

# Rotate x-axis labels for both plots
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Adjust layout and display
plt.tight_layout()
plt.show()
pass