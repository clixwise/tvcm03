
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
plt.show()

# Print summaries for verification
print("\nCEAP Distribution (Percentages):")
print(ceap_pct)