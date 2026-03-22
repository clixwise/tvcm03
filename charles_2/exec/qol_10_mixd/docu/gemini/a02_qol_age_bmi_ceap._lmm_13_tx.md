```python
import pandas as pd
import numpy as np

# Data from User's Run 1 and Run 2
# Note: Intercepts are high (50-66), we include them for completeness
run1_data = {
    'Variable': ['Intercept', 'Visit[T.T1_3Mo]', 'Visit[T.T2_6Mo]', 'Visit[T.T3_12Mo]', 
                 'CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'BMI'],
    'Coef': [50.083, 1.912, 3.426, 6.466, -5.197, -12.850, -14.520, -16.658, -24.744, -0.384],
    'StdErr': [5.295, 0.474, 0.492, 0.513, 2.187, 2.347, 2.688, 2.817, 2.598, 0.177],
    'P': [0.000, 0.000, 0.000, 0.000, 0.017, 0.000, 0.000, 0.000, 0.000, 0.030]
}

run2_data = {
    'Variable': ['Intercept', 'Visit[T.T1_3Mo]', 'Visit[T.T2_6Mo]', 'Visit[T.T3_12Mo]', 
                 'CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'BMI'],
    'Coef': [66.552, 0.849, 2.355, 5.280, -3.882, -11.278, -14.475, -17.444, -21.424, -1.035],
    'StdErr': [5.388, 0.429, 0.447, 0.463, 2.202, 2.900, 2.388, 2.245, 2.368, 0.167],
    'P': [0.000, 0.048, 0.000, 0.000, 0.078, 0.000, 0.000, 0.000, 0.000, 0.000]
}

# 1. Generate df_detail (Long format with all stats)
df1 = pd.DataFrame(run1_data)
df1['Run'] = 'Run 1'
df2 = pd.DataFrame(run2_data)
df2['Run'] = 'Run 2'

df_detail = pd.concat([df1, df2], ignore_index=True)

# 2. Generate df_global (Side-by-side comparison for publication)
df_global = df1[['Variable', 'Coef', 'P']].copy()
df_global.columns = ['Variable', 'Run1_Coef', 'Run1_P']
df_global['Run2_Coef'] = df2['Coef']
df_global['Run2_P'] = df2['P']

# Formatting for a "Global" look
df_global['Run1_Sig'] = df_global['Run1_P'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')
df_global['Run2_Sig'] = df_global['Run2_P'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')

print("df_detail sample:")
print(df_detail.head())
print("\ndf_global sample:")
print(df_global[['Variable', 'Run1_Coef', 'Run1_Sig', 'Run2_Coef', 'Run2_Sig']])

# Export
df_detail.to_csv('lmm_results_detail.csv', index=False)
df_global.to_csv('lmm_results_global.csv', index=False)


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
import pandas as pd
import numpy as np

# Data from User's Runs
# Run 1
run1_data = {
    'Variable': ['Intercept', 'Visit[T.T1_3Mo]', 'Visit[T.T2_6Mo]', 'Visit[T.T3_12Mo]', 
                 'CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'BMI'],
    'Coef_R1': [50.083, 1.912, 3.426, 6.466, -5.197, -12.850, -14.520, -16.658, -24.744, -0.384],
    'P_R1': [0.000, 0.000, 0.000, 0.000, 0.017, 0.000, 0.000, 0.000, 0.000, 0.030]
}

# Run 2
run2_data = {
    'Variable': ['Intercept', 'Visit[T.T1_3Mo]', 'Visit[T.T2_6Mo]', 'Visit[T.T3_12Mo]', 
                 'CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'BMI'],
    'Coef_R2': [66.552, 0.849, 2.355, 5.280, -3.882, -11.278, -14.475, -17.444, -21.424, -1.035],
    'P_R2': [0.000, 0.048, 0.000, 0.000, 0.078, 0.000, 0.000, 0.000, 0.000, 0.000]
}

df1 = pd.DataFrame(run1_data)
df2 = pd.DataFrame(run2_data)

# Create df_global: A high-level comparison of the two cohorts
# Focusing on the key dimensions: Baseline, Time Effect, CEAP Impact, and BMI Sensitivity
global_metrics = {
    'Metric': ['Model Baseline', 'Mean Annual Recovery', 'Max CEAP Impact (C6)', 'BMI Sensitivity'],
    'Run_1': [50.08, 6.47, -24.74, -0.38],
    'Run_2': [66.55, 5.28, -21.42, -1.04]
}
df_global = pd.DataFrame(global_metrics)

# Create df_detail: The full merge of both LMM runs
df_detail = pd.merge(df1, df2, on='Variable')

# Formatting for clean output
df_detail['Significant_Both'] = (df_detail['P_R1'] < 0.05) & (df_detail['P_R2'] < 0.05)

print("df_global:")
print(df_global)
print("\ndf_detail (first 5 rows):")
print(df_detail.head())

# Save to CSV for the user
df_global.to_csv('lmm_global_summary.csv', index=False)
df_detail.to_csv('lmm_detailed_results.csv', index=False)



```

```text
df_global:
                 Metric  Run_1  Run_2
0        Model Baseline  50.08  66.55
1  Mean Annual Recovery   6.47   5.28
2  Max CEAP Impact (C6) -24.74 -21.42
3       BMI Sensitivity  -0.38  -1.04

df_detail (first 5 rows):
           Variable  Coef_R1   P_R1  Coef_R2   P_R2  Significant_Both
0         Intercept   50.083  0.000   66.552  0.000              True
1   Visit[T.T1_3Mo]    1.912  0.000    0.849  0.048              True
2   Visit[T.T2_6Mo]    3.426  0.000    2.355  0.000              True
3  Visit[T.T3_12Mo]    6.466  0.000    5.280  0.000              True
4        CEAP[T.C2]   -5.197  0.017   -3.882  0.078             False


```

To properly organize your Mixed Linear Model results, we can structure the data into two distinct DataFrames: **df_global** (for high-level cohort comparison) and **df_detail** (for specific coefficient analysis).

---

### 1. High-Level Summary (`df_global`)

This table summarizes the "character" of your two mock runs. It’s ideal for a "Results at a Glance" section in a presentation.

| Metric | Run 1 (N=182) | Run 2 (N=177) | Clinical Meaning |
| --- | --- | --- | --- |
| **Model Baseline** | 50.08 | 66.55 | Starting QoL for a C1 patient. |
| **Mean Annual Recovery** | +6.47 | +5.28 | Total QoL points gained at 12 months. |
| **Max CEAP Impact (C6)** | -24.74 | -21.42 | Severity of QoL loss in ulcer patients. |
| **BMI Sensitivity** | -0.38 | -1.04 | Strength of BMI as a negative predictor. |

---

### 2. Full Statistical Breakdown (`df_detail`)

This DataFrame merges your two runs to allow for a direct comparison of significance and effect sizes across all variables.

| Variable | Coef (Run 1) | P-value (R1) | Coef (Run 2) | P-value (R2) | Sig. Both? |
| --- | --- | --- | --- | --- | --- |
| **Intercept** | 50.08 | <.001 | 66.55 | <.001 | Yes |
| **Visit [T1_3Mo]** | 1.91 | <.001 | 0.85 | 0.048 | **Yes** |
| **Visit [T2_6Mo]** | 3.43 | <.001 | 2.36 | <.001 | Yes |
| **Visit [T3_12Mo]** | 6.47 | <.001 | 5.28 | <.001 | Yes |
| **CEAP [C2]** | -5.20 | 0.017 | -3.88 | 0.078 | No |
| **CEAP [C6]** | -24.74 | <.001 | -21.42 | <.001 | Yes |
| **BMI** | -0.38 | 0.030 | -1.04 | <.001 | Yes |

---

### 3. Python Implementation

You can use the following code to generate these tables from your raw results:

```python
import pandas as pd

# Creating df_detail from your results
data = {
    'Variable': ['Intercept', 'Visit[T.T1_3Mo]', 'Visit[T.T2_6Mo]', 'Visit[T.T3_12Mo]', 'CEAP[C2]', 'CEAP[C6]', 'BMI'],
    'Coef_R1': [50.08, 1.91, 3.43, 6.47, -5.20, -24.74, -0.38],
    'P_R1': [0.000, 0.000, 0.000, 0.000, 0.017, 0.000, 0.030],
    'Coef_R2': [66.55, 0.85, 2.36, 5.28, -3.88, -21.42, -1.04],
    'P_R2': [0.000, 0.048, 0.000, 0.000, 0.078, 0.000, 0.000]
}
df_detail = pd.DataFrame(data)

# Calculating global trends
df_global = pd.DataFrame({
    'Metric': ['Baseline', '12mo Gain', 'C6 Impact', 'BMI Impact'],
    'Run_1': [50.08, 6.47, -24.74, -0.38],
    'Run_2': [66.55, 5.28, -21.42, -1.04]
})

```

These DataFrames provide the "meat" of your statistical reporting. **df_global** tells the story, while **df_detail** provides the rigorous evidence.

Would you like me to help you create a **summary visualization** (like a grouped bar chart) comparing the CEAP impacts between these two runs?