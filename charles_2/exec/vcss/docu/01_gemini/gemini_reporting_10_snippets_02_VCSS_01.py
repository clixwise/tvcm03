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
print (df_vcss)


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
plt.show()
