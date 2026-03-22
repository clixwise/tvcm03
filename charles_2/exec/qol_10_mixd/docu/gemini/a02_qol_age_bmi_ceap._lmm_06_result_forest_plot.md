```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from Run 1 provided by the user
data = {
    'Variable': ['CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'Month', 'BMI'],
    'Coef': [-7.414, -11.752, -13.889, -14.531, -23.058, 0.461, -0.574],
    'StdErr': [2.831, 2.379, 2.644, 2.988, 2.488, 0.037, 0.172]
}

df_results = pd.DataFrame(data)

# Calculate 95% Confidence Intervals
df_results['CI_lower'] = df_results['Coef'] - 1.96 * df_results['StdErr']
df_results['CI_upper'] = df_results['Coef'] + 1.96 * df_results['StdErr']

# Sort for better visualization (Optional, but here we keep logical order)
df_results = df_results.iloc[::-1] # Reverse to have CEAP at top in horizontal plot

# Create the Forest Plot
plt.figure(figsize=(10, 6))

# Error bars for CI and markers for coefficients
for i, row in df_results.iterrows():
    color = 'red' if (row['CI_lower'] < 0 < row['CI_upper']) == False else 'gray'
    plt.errorbar(row['Coef'], i, xerr=[[row['Coef'] - row['CI_lower']], [row['CI_upper'] - row['Coef']]], 
                 fmt='o', color=color, ecolor='black', capsize=5, markersize=8)

# Vertical line at zero
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Labels and styling
plt.yticks(range(len(df_results)), df_results['Variable'])
plt.xlabel('Coefficient Estimate (95% CI)', fontsize=12)
plt.title('Forest Plot of LMM Results (VEINES-QOL Predictors)', fontsize=14)
plt.grid(axis='x', linestyle=':', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('lmm_forest_plot.png')
# Also save the summary table to CSV for the user
df_results.to_csv('lmm_summary_results.csv', index=False)



```

```text
Traceback (most recent call last):
  File "<string>", line 45, in <module>
  File "categorical.py", line 2239, in boxplot
    plotter.plot(ax, kwargs)
  File "categorical.py", line 888, in plot
    self.draw_boxplot(ax, boxplot_kws)
  File "categorical.py", line 849, in draw_boxplot
    artist_dict = ax.boxplot(box_data,
  File "__init__.py", line 1423, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
TypeError: Axes.boxplot() got an unexpected keyword argument 'legend'


```

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from User's Run 1 (excluding intercept for scale clarity)
results = {
    'Variable': ['CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'Month', 'BMI'],
    'Coef': [-7.414, -11.752, -13.889, -14.531, -23.058, 0.461, -0.574],
    'StdErr': [2.831, 2.379, 2.644, 2.988, 2.488, 0.037, 0.172]
}

df_results = pd.DataFrame(results)

# Calculate 95% Confidence Intervals
df_results['CI_lower'] = df_results['Coef'] - 1.96 * df_results['StdErr']
df_results['CI_upper'] = df_results['Coef'] + 1.96 * df_results['StdErr']

# Reorder for plotting (optional, but often looks better)
df_results = df_results.iloc[::-1] # Reverse to have CEAP at top

# Create Forest Plot
plt.figure(figsize=(10, 6))

# Error bars for CI
plt.errorbar(df_results['Coef'], range(len(df_results)), 
             xerr=[df_results['Coef'] - df_results['CI_lower'], df_results['CI_upper'] - df_results['Coef']],
             fmt='o', color='black', capsize=5, markersize=8, label='Coefficient (95% CI)')

# Vertical line at zero
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

# Formatting
plt.yticks(range(len(df_results)), df_results['Variable'], fontsize=12)
plt.xticks(fontsize=10)
plt.xlabel('Effect Size (Coefficient)', fontsize=12)
plt.title('Forest Plot of LMM Fixed Effects (Run 1 Results)', fontsize=14)
plt.grid(axis='x', linestyle=':', alpha=0.6)

# Save image
plt.tight_layout()
plt.savefig('lmm_forest_plot.png')
plt.close()

# Save results table to CSV
df_results.to_csv('lmm_regression_summary.csv', index=False)

print("Forest plot and summary CSV generated.")


```

```text
Forest plot and summary CSV generated.


```

To visualize the results of your Linear Mixed Model (LMM) runs, a **Forest Plot** is the professional standard. It allows you to see the "Effect Size" and its precision (Confidence Interval) at a glance.

The code below extracts the data from your "Run 1" and generates the plot. Note that I have omitted the **Intercept** (value ~54) from the plot to allow the scale of the x-axis to clearly show the impact of the predictors.

### 1. Forest Plot of LMM Results

In this plot:

* **The Dot:** Represents the Estimated Coefficient ().
* **The Error Bars:** Represent the 95% Confidence Interval.
* **The Red Line (0):** Any variable whose error bar crosses the red line is **not** statistically significant.

---

### 2. Python Code for the Forest Plot

You can use this snippet to generate the plot for any future runs.

```python
import matplotlib.pyplot as plt
import pandas as pd

# 1. Define your LMM results from Run 1
data = {
    'Variable': ['CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'Month', 'BMI'],
    'Coef': [-7.414, -11.752, -13.889, -14.531, -23.058, 0.461, -0.574],
    'StdErr': [2.831, 2.379, 2.644, 2.988, 2.488, 0.037, 0.172]
}

df = pd.DataFrame(data)

# 2. Calculate 95% CI (1.96 * StdErr)
df['low'] = df['Coef'] - (1.96 * df['StdErr'])
df['high'] = df['Coef'] + (1.96 * df['StdErr'])

# 3. Create the Plot
plt.figure(figsize=(10, 6))
plt.errorbar(df['Coef'], range(len(df)), 
             xerr=[df['Coef']-df['low'], df['high']-df['Coef']], 
             fmt='o', color='navy', capsize=5, elinewidth=2)

# 4. Add formatting and the "Zero line"
plt.axvline(x=0, color='crimson', linestyle='--')
plt.yticks(range(len(df)), df['Variable'])
plt.xlabel('Coefficient Value (Effect on VEINES-QOL)')
plt.title('Forest Plot of Predictors (LMM Run 1)')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('lmm_forest_plot.png')

```

---

### 3. Key Observations from your Specific Data

* **Clinical Degradation:** The coefficient for **CEAP C6** is the most negative (), showing that an active ulcer has a vastly greater impact on QoL than any other stage.
* **Recovery Rate:** The **Month** coefficient is positive (). This means that for every month of follow-up, patients gain nearly half a point in QoL, independent of their starting point.
* **Precision:** Notice how thin the error bar is for **Month** (StdErr 0.037). This indicates a very high degree of confidence in the rate of improvement across the population.

The final summary of these calculations is available in `lmm_regression_summary.csv`.

Would you like to move on to the **Tier 3 — Secondary Sensitivity Analysis** (e.g., handling non-linear trends or outlier influence)?