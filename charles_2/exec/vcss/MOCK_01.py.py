import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.data_31_figu import FiguTran1x1, FiguTran2x1
# https://gemini.google.com/app/5716240478c0e5f3 

# ----
#
# ----

# 1. Setup Mock Data
data = {
    'patient_id': [f'PT_{i}' for i in range(12)],
    'timepoint': ['T0']*4 + ['T1']*4 + ['T2']*4,
    'Age': [59, 45, 51, 30, 60, 46, 52, 31, 61, 47, 53, 32],
    'Sexe': ['F', 'F', 'F', 'M'] * 3,
    'VCSS_R': np.random.randint(5, 11, 12),
    'VCSS_L': np.random.randint(5, 11, 12),
    'VCSS_P': np.random.randint(5, 11, 12)
}
df_raw = pd.DataFrame(data)

def get_stats(df_subset):
    # Melt VCSS columns to transform them into row identifiers
    melted = df_subset.melt(
        id_vars=['timepoint', 'Age'], 
        value_vars=['VCSS_R', 'VCSS_L', 'VCSS_P'],
        var_name='VCSS_Type', value_name='Value'
    )
    
    # Group by time and VCSS type to create the 9 rows
    grouped = melted.groupby(['timepoint', 'VCSS_Type'])
    
    return pd.DataFrame({
        'median': grouped['Value'].median(),
        'mean': grouped['Value'].mean(),
        # Store as sorted lists of integers
        'age': grouped['Age'].apply(lambda x: sorted(list(x)))
    })

# 2. Generate the three segments
df_a = get_stats(df_raw)
df_m = get_stats(df_raw[df_raw['Sexe'] == 'M'])
df_f = get_stats(df_raw[df_raw['Sexe'] == 'F'])

# 3. Concatenate with MultiIndex columns
df_final = pd.concat({'A': df_a, 'M': df_m, 'F': df_f}, axis=1)

# 4. Display results using your preferred options
with pd.option_context(
    'display.max_columns', None,
    'display.width', 1000,
    'display.precision', 2,
    'display.colheader_justify', 'left'
):
    print("--- Multi-Indexed Row and Column DataFrame ---")
    print(df_final)

print("\n--- Detailed Dtype Scan ---")
for col in df_final.columns:
    actual = type(df_final[col].dropna().iloc[0]).__name__
    print(f"{col}: {df_final[col].dtype} (Underlying: {actual})")
    
# ----
#
# ----
exe1 = False
if exe1:
    # Define the groups we want to plot
    groups = ['A', 'M', 'F']
    vcss_types = ['VCSS_R', 'VCSS_L', 'VCSS_P']

    # Example: Extracting Mean values for 'All Patients' (A)
    # This creates a table where rows are Timepoints and columns are VCSS types
    plot_data_a = df_final['A']['mean'].unstack(level=1)

    # Assuming your FiguTran1x1 is defined in your environment
    for group in groups:
        # 1. Initialize your custom figure class
        view = FiguTran1x1()
        view.titl = f"VCSS Trends: Group {group}"
        view.size = (8, 5)
        view.upda() # Creates the fig and ax1
        
        # 2. Extract data for this group (Mean values)
        # .unstack(level=1) turns the 'VCSS_Type' index into columns
        data_to_plot = df_final[group]['mean'].unstack(level=1)
        
        # 3. Plotting on the custom axis
        data_to_plot.plot(ax=view.ax1, marker='o', linewidth=2)
        
        # 4. Styling
        view.ax1.set_ylabel("Mean VCSS Score")
        view.ax1.set_xlabel("Timepoint")
        view.ax1.grid(True, linestyle='--', alpha=0.7)
        view.ax1.legend(title="VCSS Type")
        # Get n-count from the first available row in the age column
        n_count = len(df_final[group]['age'].iloc[0])
        view.titl = f"Group {group} (n={n_count})"
        plt.show()
        pass

# ----
#
# ----
# Assuming FiguTran2x1 and df_final are already defined

exe2 = False
if exe2:
    groups = ['A', 'M', 'F']

    for group in groups:
        # 1. Initialize the 2x1 Figure
        view = FiguTran2x1()
        view.titl = f"VCSS Statistical Comparison: Group {group}"
        view.size = (10, 8)
        view.hspa = 0.4  # Add some height space between plots
        view.upda()
        
        # 2. Extract Data
        # Rows = Timepoint, Columns = VCSS_Type
        df_mean = df_final[group]['mean'].unstack(level=1)
        df_median = df_final[group]['median'].unstack(level=1)
        
        # 3. Plot Mean (Top Axis)
        df_mean.plot(ax=view.ax1, marker='o', linestyle='-')
        view.ax1.set_title("Mean VCSS Scores")
        view.ax1.set_ylabel("Score")
        view.ax1.grid(True, alpha=0.3)
        
        # 4. Plot Median (Bottom Axis)
        df_median.plot(ax=view.ax2, marker='s', linestyle='--')
        view.ax2.set_title("Median VCSS Scores")
        view.ax2.set_ylabel("Score")
        view.ax2.set_xlabel("Timepoint")
        view.ax2.grid(True, alpha=0.3)
        
        # Optional: Place legend outside or simplify
        view.ax1.legend(loc='upper right', fontsize='small')
        view.ax2.legend(loc='upper right', fontsize='small')
        # Get n-count from the first available row in the age column
        n_count = len(df_final[group]['age'].iloc[0])
        view.titl = f"Group {group} (n={n_count})"
        plt.show()
    
# ----
#
# ----
exe3 = False
if exe3:
    groups = ['A', 'M', 'F']
    for group in groups:
        view = FiguTran2x1()
        
        # 1. Calculate total n for the group title
        # We look at the first row (T0, VCSS_L) to get the patient count
        sample_age_list = df_final[(group, 'age')].iloc[0]
        total_n = len(sample_age_list)
        
        view.titl = f"VCSS Analysis: {group} (Total n={total_n})"
        view.size = (10, 8)
        view.hspa = 0.5
        view.upda()
        
        # 2. Extract Data
        df_mean = df_final[group]['mean'].unstack(level=1)
        df_median = df_final[group]['median'].unstack(level=1)
        
        # 3. Create Dynamic X-Labels (e.g., "T0 (n=4)")
        # We extract the age list for each timepoint (level 0 of the index)
        age_series = df_final[group]['age'].unstack(level=1).iloc[:, 0]
        new_labels = [f"{tp}\n(n={len(ages)})" for tp, ages in age_series.items()]
        
        # 4. Plot Mean
        df_mean.plot(ax=view.ax1, marker='o')
        view.ax1.set_title("Mean Scores")
        view.ax1.set_xticks(range(len(new_labels)))
        view.ax1.set_xticklabels(new_labels)
        
        # 5. Plot Median
        df_median.plot(ax=view.ax2, marker='s', linestyle='--')
        view.ax2.set_title("Median Scores")
        view.ax2.set_xticks(range(len(new_labels)))
        view.ax2.set_xticklabels(new_labels)
        
        # Global styling
        for ax in [view.ax1, view.ax2]:
            ax.set_ylabel("VCSS Value")
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
# ----
#
# ----
exe4 = False
if exe4:
    
    def get_stats(df_subset):
        melted = df_subset.melt(
            id_vars=['timepoint', 'Age'], 
            value_vars=['VCSS_R', 'VCSS_L', 'VCSS_P'],
            var_name='VCSS_Type', value_name='Value'
        )
        
        grouped = melted.groupby(['timepoint', 'VCSS_Type'])
        
        return pd.DataFrame({
            'median': grouped['Value'].median(),
            'mean': grouped['Value'].mean(),
            'std': grouped['Value'].std().fillna(0), # Standard Deviation for error bars
            'age': grouped['Age'].apply(lambda x: sorted(list(x)))
        })

    # Re-run the segments
    df_a = get_stats(df_raw)
    df_m = get_stats(df_raw[df_raw['Sexe'] == 'M'])
    df_f = get_stats(df_raw[df_raw['Sexe'] == 'F'])
    df_final = pd.concat({'A': df_a, 'M': df_m, 'F': df_f}, axis=1)

    # ----
    #
    # ----
    '''
    Spread of Data: The error bars show how much your patients' VCSS scores vary. 
    Large bars mean the patients are very different from each other; small bars mean they are recovering at a similar rate.
    Overlap Check: If the error bars for VCSS_R and VCSS_L overlap significantly, 
    the difference between the right and left leg is likely not statistically significant.
    '''
    groups = ['A', 'M', 'F']
    for group in groups:
        view = FiguTran2x1()
        view.size = (10, 8)
        view.upda()
        
        # Extract Data
        means = df_final[group]['mean'].unstack(level=1)
        stds = df_final[group]['std'].unstack(level=1)
        medians = df_final[group]['median'].unstack(level=1)
        
        # X-axis setup
        x_indices = np.arange(len(means.index))
        vcss_cols = means.columns # ['VCSS_R', 'VCSS_L', 'VCSS_P']
        
        # --- TOP PLOT: MEAN + ERROR BARS ---
        for col in vcss_cols:
            view.ax1.errorbar(
                x_indices, means[col], yerr=stds[col], 
                label=col, marker='o', capsize=5, elinewidth=1
            )
        
        view.ax1.set_title(f"Mean VCSS Score (±SD) - Group {group}")
        view.ax1.set_xticks(x_indices)
        view.ax1.set_xticklabels([f"{tp}\n(n={len(df_final.loc[(tp, 'VCSS_R'), (group, 'age')])})" for tp in means.index])

        # --- BOTTOM PLOT: MEDIAN ---
        medians.plot(ax=view.ax2, marker='s', linestyle='--')
        view.ax2.set_title("Median VCSS Score")
        view.ax2.set_xticks(x_indices)
        view.ax2.set_xticklabels(view.ax1.get_xticklabels()) # Keep labels consistent

        # Global Styling
        view.ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        view.ax2.get_legend().remove() # Avoid redundancy
        
        plt.tight_layout()
        plt.show()
    
    
# ----
#
# ----
exe5 = False
if exe5:
    
    def get_stats(df_subset):
        melted = df_subset.melt(
            id_vars=['timepoint', 'Age'], 
            value_vars=['VCSS_R', 'VCSS_L', 'VCSS_P'],
            var_name='VCSS_Type', value_name='Value'
        )
        
        grouped = melted.groupby(['timepoint', 'VCSS_Type'])
        
        return pd.DataFrame({
            'median': grouped['Value'].median(),
            'mean': grouped['Value'].mean(),
            'std': grouped['Value'].std().fillna(0), # Standard Deviation for error bars
            'age': grouped['Age'].apply(lambda x: sorted(list(x)))
        })

    # Re-run the segments
    df_a = get_stats(df_raw)
    df_m = get_stats(df_raw[df_raw['Sexe'] == 'M'])
    df_f = get_stats(df_raw[df_raw['Sexe'] == 'F'])
    df_final = pd.concat({'A': df_a, 'M': df_m, 'F': df_f}, axis=1)

    # ----
    #
    # ----
    def get_sig_star(p_value):
        if p_value < 0.001: return '***'
        if p_value < 0.01:  return '**'
        if p_value < 0.05:  return '*'
        return 'ns'
    '''
    Spread of Data: The error bars show how much your patients' VCSS scores vary. 
    Large bars mean the patients are very different from each other; small bars mean they are recovering at a similar rate.
    Overlap Check: If the error bars for VCSS_R and VCSS_L overlap significantly, 
    the difference between the right and left leg is likely not statistically significant.
    '''
    groups = ['A', 'M', 'F']    
    for group in groups:
        view = FiguTran2x1()
        view.size = (10, 8)
        view.upda()
        
        # Data extraction
        means = df_final[group]['mean'].unstack(level=1)
        stds = df_final[group]['std'].unstack(level=1)
        
        # 3. Plot Mean + Error Bars
        x = np.arange(len(means.index))
        for i, col in enumerate(means.columns):
            # Plot the line and error bars
            line = view.ax1.errorbar(x, means[col], yerr=stds[col], 
                                    label=col, marker='o', capsize=5)
            
            # --- Significance Calculation (T0 vs T2) ---
            # We need the raw data lists stored in our 'age' column 
            # (Assuming VCSS values were stored or accessible)
            # For this mock, we will simulate a p-value calculation
            p_val = 0.002  # Simulated: in real case, run stats.ttest_rel()
            star = get_sig_star(p_val)
            
            # Annotate the star above the T2 point (index 2)
            y_pos = means[col].iloc[2] + stds[col].iloc[2] + 0.5
            view.ax1.text(2, y_pos, star, ha='center', fontweight='bold', 
                        color=line[0].get_color())

        # --- Formatting ---
        view.ax1.set_title(f"Clinical Progress (Mean ± SD) - Group {group}")
        view.ax2.set_title("Median Distribution")
        
        # Add n-count labels
        tp_labels = [f"{tp}\n(n={len(df_final.loc[(tp, 'VCSS_R'), (group, 'age')])})" 
                    for tp in means.index]
        
        for ax in [view.ax1, view.ax2]:
            ax.set_xticks(x)
            ax.set_xticklabels(tp_labels)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

# ----
# Data
# ----
'''
Group,Category,T0_Mean,T2_Mean,P-Value,Significance
A,VCSS_R,8.50,5.20,0.0024,**
A,VCSS_L,8.40,5.15,0.0011,**
M,VCSS_R,9.00,8.80,0.4500,ns
'''

def get_significance_summary(df_raw, group_name):
    """
    Performs a paired T-test for T0 vs T2 for each VCSS category.
    group_name: 'A', 'M', or 'F'
    """
    # Filter raw data for the specific group (if not 'A')
    if group_name != 'A':
        df_group = df_raw[df_raw['Sexe'] == group_name]
    else:
        df_group = df_raw

    vcss_cols = ['VCSS_R', 'VCSS_L', 'VCSS_P']
    summary_data = []

    for col in vcss_cols:
        # Get values for T0 and T2 for the same patients
        # We pivot to ensure patient_id alignment
        pivot = df_group.pivot(index='patient_id', columns='timepoint', values=col).dropna()
        
        if not pivot.empty and 'T0' in pivot.columns and 'T2' in pivot.columns:
            # Perform Paired T-Test
            t_stat, p_val = stats.ttest_rel(pivot['T0'], pivot['T2'])
            
            # Map p-value to stars
            if p_val < 0.001: star = '***'
            elif p_val < 0.01: star = '**'
            elif p_val < 0.05: star = '*'
            else: star = 'ns'
            
            summary_data.append({
                'Group': group_name,
                'Category': col,
                'T0_Mean': pivot['T0'].mean(),
                'T2_Mean': pivot['T2'].mean(),
                'P-Value': p_val,
                'Significance': star
            })

    return pd.DataFrame(summary_data)

# Usage
summary_all = get_significance_summary(df_raw, 'A')
summary_male = get_significance_summary(df_raw, 'M')
summary_female = get_significance_summary(df_raw, 'F')

# Combine all into one clean table
final_stats_table = pd.concat([summary_all, summary_male, summary_female], ignore_index=True)

# ----
# Frame : Integration
# ----
def get_stats(df_subset):
    # 1. Standard Aggregations
    melted = df_subset.melt(
        id_vars=['patient_id', 'timepoint', 'Age'], 
        value_vars=['VCSS_R', 'VCSS_L', 'VCSS_P'],
        var_name='VCSS_Type', value_name='Value'
    )
    
    grouped = melted.groupby(['timepoint', 'VCSS_Type'])
    
    # Basic Stats
    df_res = pd.DataFrame({
        'median': grouped['Value'].median(),
        'mean': grouped['Value'].mean(),
        'std': grouped['Value'].std().fillna(0),
        'count': grouped['Value'].count(),  # Added Count
        'age': grouped['Age'].apply(lambda x: sorted(list(x)))
    })

    # 2. Calculate P-Values (relative to T0)
    # Pivot to align same patient_id across timepoints
    pivot = melted.pivot(index=['patient_id', 'VCSS_Type'], columns='timepoint', values='Value')
    
    p_vals = []
    for (tp, v_type) in df_res.index:
        if tp == 'T0':
            p_vals.append(np.nan) # No change from T0 to T0
        else:
            # Pair patients who have both T0 and the current TP
            pairs = pivot[['T0', tp]].dropna()
            if len(pairs) > 1:
                _, p = stats.ttest_rel(pairs['T0'], pairs[tp])
                p_vals.append(p)
            else:
                p_vals.append(np.nan)
                
    df_res['p_val_vs_T0'] = p_vals
    return df_res

exe6 = True
if exe6:
    

    # 1. Prepare the raw data (Assuming df_raw exists from our previous steps)
    # ---------------------------------------------------------------------

    # 2. Call get_stats for each cohort
    # --------------------------------
    # Cohort A: All Patients
    df_a = get_stats(df_raw)

    # Cohort M: Males Only
    df_m = get_stats(df_raw[df_raw['Sexe'] == 'M'])

    # Cohort F: Females Only
    df_f = get_stats(df_raw[df_raw['Sexe'] == 'F'])

    # 3. Integrate into the MultiIndex DataFrame
    # ------------------------------------------
    cohorts = {'A': df_raw}

    # Only add M/F if they actually exist in the data
    for gender in ['M', 'F']:
        subset = df_raw[df_raw['Sexe'] == gender]
        if not subset.empty:
            cohorts[gender] = subset

    # Dictionary comprehension to run get_stats on all valid cohorts
    df_final = pd.concat({k: get_stats(v) for k, v in cohorts.items()}, axis=1)
    # This creates the (Group, Metric) column hierarchy
    '''
    df_final = pd.concat({
        'A': df_a, 
        'M': df_m, 
        'F': df_f
    }, axis=1)
    '''

    # 4. (Optional) Trace/Validation
    # ------------------------------
    print("Structure of df_final columns:")
    # pprint(df_final.columns.tolist())

# Now the plotting loop will work perfectly!
    def get_star(p):
        if pd.isna(p) or p > 0.05: return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        return "*"

    for group in ['A', 'M', 'F']:
        view = FiguTran2x1()
        view.size = (10, 8)
        view.hspa = 0.5
        view.upda()
        
        # Extract data for the group
        data = df_final[group]
        vcss_types = data.index.get_level_values('VCSS_Type').unique()
        
        # 1. Plot Mean + Error Bars + Stars
        for col_name in vcss_types:
            subset = data.xs(col_name, level='VCSS_Type')
            x = np.arange(len(subset.index))
            
            # Plot lines
            line = view.ax1.errorbar(x, subset['mean'], yerr=subset['std'], 
                                    label=col_name, marker='o', capsize=4)
            
            # Add Significance Stars for T1 and T2
            for i, (tp, p_val) in enumerate(subset['p_val_vs_T0'].items()):
                star = get_star(p_val)
                if star:
                    # Position star slightly above the error bar
                    y_star = subset['mean'].iloc[i] + subset['std'].iloc[i] + 0.2
                    view.ax1.text(i, y_star, star, ha='center', color=line[0].get_color(), fontweight='bold')

        # 2. Plot Median
        for col_name in vcss_types:
            subset = data.xs(col_name, level='VCSS_Type')
            view.ax2.plot(subset.index, subset['median'], marker='s', linestyle='--', label=col_name)

        # 3. Dynamic X-Axis Labels with Counts
        # We pull the 'count' specifically for the first VCSS type as a representative
        counts = data.xs(vcss_types[0], level='VCSS_Type')['count']
        new_labels = [f"{tp}\n(n={int(n)})" for tp, n in counts.items()]
        
        for ax in [view.ax1, view.ax2]:
            ax.set_xticks(range(len(new_labels)))
            ax.set_xticklabels(new_labels)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        view.ax1.set_title(f"Group {group}: Mean Progress with Significance vs T0")
        view.ax2.set_title(f"Group {group}: Median Distribution")
        
        plt.tight_layout()
        plt.show()
        pass
    
    
        import pickle

        # --- SAVE ---
        # This saves the entire MultiIndex df_final to a binary file
        file_path = "vcss_analysis_results.pkl"

        with open(file_path, 'wb') as f:
            pickle.dump(df_final, f)
        print(f"Data saved successfully to {file_path}")

        # --- LOAD ---
        # When you want to plot later (perhaps in a different script)
        with open(file_path, 'rb') as f:
            df_loaded = pickle.load(f)

        print("Data loaded. Ready for plotting.")
        
        # Save
        df_final.to_pickle("vcss_results.pkl")

        # Load
        df_final = pd.read_pickle("vcss_results.pkl")