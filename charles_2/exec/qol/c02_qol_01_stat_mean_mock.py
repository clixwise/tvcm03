# ****
#
# ****
import sys
import os
# This adds the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import numpy as np
from util.stat_help import get_comparison_stats, summarize_continuous_edit

def plot_veines_qol_sophisticated(df, outcome_col="VEINES_QOL_t", time_col="timepoint"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    timepoints = ["T0", "T1", "T2"]
    x = np.arange(len(timepoints))
    
    # 1. Calculate Aggregates
    means = []
    cis = []
    for tp in timepoints:
        data = df[df[time_col] == tp][outcome_col].dropna()
        n = len(data)
        avg = data.mean()
        se = data.std() / np.sqrt(n)
        ci = stats.t.interval(0.95, n-1, loc=avg, scale=se)
        means.append(avg)
        cis.append(ci)
    
    means = np.array(means)
    cis = np.array(cis) # [lower, upper]

    # 2. Plot Individual Patient Trajectories (Spaghetti Plot)
    # Assumes df has a 'patient_id'
    if 'patient_id' in df.columns:
        for pid in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == pid].set_index(time_col).reindex(timepoints)[outcome_col]
            ax.plot(x, patient_data, color='gray', alpha=0.15, lw=1, zorder=1)

    # 3. Plot the Group Ribbon (CI) and Mean Line
    ax.fill_between(x, cis[:, 0], cis[:, 1], color='#3498db', alpha=0.2, label='95% CI', zorder=2)
    ax.plot(x, means, marker='o', color='#2980b9', lw=3, ms=8, label='Group Mean', zorder=3)

    # 4. Significance Annotations (The "Brackets")
    # Example: T0 vs T2
    def add_sig_bracket(x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y], color='black', lw=1)
        ax.text((x1+x2)/2, y+1.5, text, ha='center', va='bottom', fontsize=10)

    # Assuming p < 0.001 for T0-T2 based on previous discussion
    add_sig_bracket(0, 2, max(cis[:, 1]) + 2, "p < 0.001")

    # 5. MCID Reference Line (Improvement of 3 points from T0)
    mcid_level = means[0] + 3
    ax.axhline(mcid_level, color='red', linestyle='--', alpha=0.5, lw=1)
    ax.text(2.4, mcid_level, 'MCID (+3)', color='red', va='center', fontweight='bold')

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline (T0)", "6 Months (T1)", "12 Months (T2)"])
    ax.set_ylabel("VEINES-QOL Score (T-score)")
    ax.set_title("Longitudinal Evolution of Patient Quality of Life", pad=20, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')

    plt.tight_layout()
    return fig

# Usage
# fig = plot_veines_qol_sophisticated(df_fram)
# plt.show()

# ****
#
# ****


def generate_mock_veines(n_patients=30):
    np.random.seed(42) # For reproducibility
    
    data = []
    # Patient IDs
    p_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
    
    # Baselines (T0) around 50
    t0_scores = np.random.normal(50, 4, n_patients)
    
    # Improvements (T1 ≈ +3, T2 ≈ +6) with some individual variance
    t1_improvements = np.random.normal(3, 2, n_patients)
    t2_improvements = np.random.normal(6, 3, n_patients)
    
    for i, pid in enumerate(p_ids):
        # T0
        data.append([pid, "T0", t0_scores[i]])
        # T1
        data.append([pid, "T1", t0_scores[i] + t1_improvements[i]])
        # T2
        data.append([pid, "T2", t0_scores[i] + t2_improvements[i]])
        
    df = pd.DataFrame(data, columns=["patient_id", "timepoint", "VEINES_QOL_t"])
    return df

# Create the test dataframe
# df_fram = generate_mock_veines()

# ****
#
# ****
# --- 1. Generate Data ---
df_test = generate_mock_veines(30)
publ_list = []
trac = True

# --- 2. Run Analytics Loop ---
for tipo in ["T0", "T1", "T2"]:
    df_tipo = df_test[df_test["timepoint"] == tipo]
    
    # Descriptive (Mean ± SD [CI])
    val = summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "jrnl_qual")
    
    # Comparative (Change & P-value)
    # Note: Use the function we defined earlier that handles Paired T-tests
    diff, p_val = get_comparison_stats(df_test, tipo, "T0")
    
    publ_list.append(["StatQOL", "VEINES-QOL", tipo, val, diff, p_val])

# --- 3. Display Table ---
cols = ['ID', 'Metric', 'Timepoint', 'Value', 'Diff', 'P-value']
df_publ = pd.DataFrame(publ_list, columns=cols).set_index(['ID', 'Metric'])
print(df_publ)

# --- 4. Generate Plot ---
fig = plot_veines_qol_sophisticated(df_test)
plt.show()
