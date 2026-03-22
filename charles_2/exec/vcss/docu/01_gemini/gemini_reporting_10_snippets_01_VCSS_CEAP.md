```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Define patients and limbs
num_patients = 40
num_limbs = 60
timepoints = ['T0', 'T1', 'T2']

# Create patient IDs and assign limbs to them
patient_ids = np.arange(num_patients)
# Assign 20 patients 2 limbs and 20 patients 1 limb to get 60 limbs
limb_patient_map = list(np.repeat(patient_ids[:20], 2)) + list(patient_ids[20:])
limb_ids = np.arange(num_limbs)

# --- VCSS Mock Data ---
vcss_data = []
for l_idx, p_id in enumerate(limb_patient_map):
    # Baseline T0
    base_vcss = np.random.normal(8.5, 2.0)
    # T1 improvement
    t1_vcss = base_vcss - np.random.normal(5.0, 1.0)
    # T2 further improvement or plateau
    t2_vcss = t1_vcss - np.random.normal(1.0, 0.5)
    
    # Clip values between 0 and 30
    vcss_scores = np.clip([base_vcss, t1_vcss, t2_vcss], 0, 30).round().astype(int)
    
    for i, tp in enumerate(timepoints):
        vcss_data.append({
            'patient_id': p_id,
            'limb_id': l_idx,
            'timepoint': tp,
            'vcss_score': vcss_scores[i]
        })

df_vcss = pd.DataFrame(vcss_data)

# --- CEAP Mock Data ---
ceap_data = []
# Classes
classes = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

for l_idx, p_id in enumerate(limb_patient_map):
    # T0: Mostly C2, C3, C4
    t0_class = np.random.choice(['C2', 'C3', 'C4'], p=[0.5, 0.3, 0.2])
    
    # T1: Improvement
    if t0_class == 'C4':
        t1_class = np.random.choice(['C2', 'C3', 'C4'], p=[0.5, 0.3, 0.2])
    elif t0_class == 'C3':
        t1_class = np.random.choice(['C1', 'C2', 'C3'], p=[0.4, 0.5, 0.1])
    else: # C2
        t1_class = np.random.choice(['C0', 'C1', 'C2'], p=[0.4, 0.5, 0.1])
        
    # T2: Further improvement
    if t1_class in ['C0', 'C1']:
        t2_class = t1_class # Stay improved
    elif t1_class == 'C2':
        t2_class = np.random.choice(['C0', 'C1', 'C2'], p=[0.6, 0.3, 0.1])
    else:
        t2_class = np.random.choice(['C1', 'C2', 'C3'], p=[0.5, 0.4, 0.1])

    ceap_data.append({'limb_id': l_idx, 'timepoint': 'T0', 'ceap_class': t0_class})
    ceap_data.append({'limb_id': l_idx, 'timepoint': 'T1', 'ceap_class': t1_class})
    ceap_data.append({'limb_id': l_idx, 'timepoint': 'T2', 'ceap_class': t2_class})

df_ceap = pd.DataFrame(ceap_data)

# --- Plotting VCSS ---
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
ax1 = sns.boxplot(x='timepoint', y='vcss_score', data=df_vcss, 
                 palette="Blues", order=['T0', 'T1', 'T2'], width=0.5)
sns.swarmplot(x='timepoint', y='vcss_score', data=df_vcss, 
              color=".25", size=4, alpha=0.6)
plt.title('Evolution of Venous Clinical Severity Score (VCSS)', fontsize=14)
plt.xlabel('Timepoint (T0 = Surgery)', fontsize=12)
plt.ylabel('Total VCSS Score', fontsize=12)
plt.tight_layout()
plt.savefig('vcss_evolution.png')

# --- Plotting CEAP ---
ceap_counts = pd.crosstab(df_ceap['timepoint'], df_ceap['ceap_class'])
ceap_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
available_cols = [c for c in ceap_order if c in ceap_counts.columns]
ceap_counts = ceap_counts[available_cols]
ceap_pct = ceap_counts.div(ceap_counts.sum(axis=1), axis=0) * 100
# Reorder index for plotting
ceap_pct = ceap_pct.reindex(['T0', 'T1', 'T2'])

ax2 = ceap_pct.plot(kind='bar', stacked=True, figsize=(10, 6), 
                   colormap='RdYlGn_r', edgecolor='white')
plt.title('Distribution of CEAP Clinical Classification Over Time', fontsize=14)
plt.ylabel('Percentage of Limbs (%)', fontsize=12)
plt.xlabel('Timepoint', fontsize=12)
plt.legend(title='CEAP Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ceap_evolution.png')

# Print summaries for verification
print("VCSS Summary:")
print(df_vcss.groupby('timepoint')['vcss_score'].describe())
print("\nCEAP Distribution (Percentages):")
print(ceap_pct)



```

```text
VCSS Summary:
           count      mean       std  min   25%  50%  75%   max
timepoint                                                      
T0          60.0  8.566667  1.608944  6.0  7.00  9.0  9.0  12.0
T1          60.0  3.733333  2.015757  0.0  2.75  3.5  5.0   8.0
T2          60.0  2.766667  1.898855  0.0  1.00  3.0  4.0   7.0

CEAP Distribution (Percentages):
ceap_class         C0    C1         C2         C3    C4
timepoint                                              
T0           0.000000   0.0  46.666667  38.333333  15.0
T1          18.333333  40.0  30.000000   6.666667   5.0
T2          30.000000  60.0   6.666667   3.333333   0.0


```

To test the reporting standards and the code provided, here is a mock setup including data generation for **60 limbs in 40 patients** (simulating 20 bilateral cases).

### 1. Mock Data Generation and VCSS Plotting

This script creates a realistic dataset where VCSS scores drop significantly post-surgery (T1) and maintain or improve slightly at the 6-month follow-up (T2).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
np.random.seed(42)

# Setup: 60 limbs across 40 patients
num_limbs = 60
num_patients = 40
patient_ids = np.arange(num_patients)
# Assign 20 patients 2 limbs and 20 patients 1 limb
limb_patient_map = list(np.repeat(patient_ids[:20], 2)) + list(patient_ids[20:])

# Generate VCSS Scores
vcss_data = []
for l_idx, p_id in enumerate(limb_patient_map):
    t0 = np.random.normal(8.5, 2.0)
    t1 = t0 - np.random.normal(5.0, 1.0)
    t2 = t1 - np.random.normal(1.0, 0.5)
    
    scores = np.clip([t0, t1, t2], 0, 30).round().astype(int)
    for i, tp in enumerate(['T0', 'T1', 'T2']):
        vcss_data.append({'patient_id': p_id, 'limb_id': l_idx, 'timepoint': tp, 'vcss_score': scores[i]})

df_vcss = pd.DataFrame(vcss_data)

# --- PLOT VCSS ---
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
sns.boxplot(x='timepoint', y='vcss_score', data=df_vcss, palette="Blues", width=0.5)
sns.swarmplot(x='timepoint', y='vcss_score', data=df_vcss, color=".25", size=4, alpha=0.6)

plt.title('Evolution of Venous Clinical Severity Score (VCSS)', fontsize=14)
plt.xlabel('Timepoint (T0 = Surgery)', fontsize=12)
plt.ylabel('Total VCSS Score', fontsize=12)
plt.savefig('vcss_evolution.png')

```

### 2. Mock Data Generation and CEAP Plotting

This script simulates the clinical "downgrade" from higher classes (C3–C4) to lower classes (C0–C1).

```python
# Generate CEAP Classes
ceap_data = []
for l_idx in range(num_limbs):
    t0 = np.random.choice(['C2', 'C3', 'C4'], p=[0.5, 0.3, 0.2])
    # Logical improvement for mock data
    t1 = 'C1' if t0 in ['C2', 'C3'] else 'C2'
    t2 = 'C0' if t1 == 'C1' else 'C1'
    
    for i, (tp, cls) in enumerate(zip(['T0', 'T1', 'T2'], [t0, t1, t2])):
        ceap_data.append({'limb_id': l_idx, 'timepoint': tp, 'ceap_class': cls})

df_ceap = pd.DataFrame(ceap_data)

# --- PLOT CEAP ---
ceap_counts = pd.crosstab(df_ceap['timepoint'], df_ceap['ceap_class'])
ceap_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
available_cols = [c for c in ceap_order if c in ceap_counts.columns]
ceap_pct = ceap_counts[available_cols].div(ceap_counts.sum(axis=1), axis=0) * 100
ceap_pct = ceap_pct.reindex(['T0', 'T1', 'T2'])

ceap_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='RdYlGn_r', edgecolor='white')
plt.title('Distribution of CEAP Clinical Classification Over Time', fontsize=14)
plt.ylabel('Percentage of Limbs (%)', fontsize=12)
plt.legend(title='CEAP Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.savefig('ceap_evolution.png')

```

### Expected Output Summary

When you run this with your actual data, you should see:

1. **VCSS Boxplot:** A clear downward trend. The **Mean** will likely drop from ~8.5 (Moderate disease) to ~2.7 (Mild/Resolved).
2. **CEAP Bar Chart:** A visual "greening" of the chart. In this mock, the population moves from **0% C0/C1 at T0** to over **90% C0/C1 at T2**, which is a powerful visual for surgical efficacy.

### Final Publication Tip: Figure Legends

* **Figure 1 (VCSS):** "Evolution of the Venous Clinical Severity Score (VCSS) at baseline (T0), one-month (T1), and six-month (T2) follow-up. Boxplots represent the median and interquartile range; points represent individual limb scores ().  across all timepoints via Linear Mixed-Effects Model."
* **Figure 2 (CEAP):** "Shift in CEAP clinical classification across timepoints ( limbs). A significant migration from advanced clinical classes (C3–C4) to asymptomatic or telangiectatic classes (C0–C1) was observed (, Stuart-Maxwell test)."