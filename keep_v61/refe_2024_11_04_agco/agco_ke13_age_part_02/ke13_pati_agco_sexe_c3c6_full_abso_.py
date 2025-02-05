from datetime import datetime
import os
import sys
import pandas as pd
from util_file_inpu_pati import inp1
from util_file_mngr import set_file_objc, write
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from matplotlib.lines import Line2D

# -----------------
def inpu(df, mode):
# -----------------
        if mode == "T":
            # agco_abso_sexe = df[~df['ceap'].isin(['NA'])] # eliminate 'NA'
            df_age = df[['doss', 'age_bin', 'age', 'sexe']].drop_duplicates()
            # print_write(f"agco_abso_sexe_T:{type(df_age)}\n{df_age}\n:{df_age.index}")
        else:
            df_age = df[['doss', 'age_bin', 'age', 'sexe']].drop_duplicates()
            df_age = df_age[df_age['sexe'] == mode]
            df_age = df_age[['doss', 'age_bin', 'age', 'sexe']].drop_duplicates()
        #
        #print_write(f"df_age:{type(df_age)}\n{df_age}\n:{df_age.index}")
        print_write(f"INPUT for {mode} patients : df_age:{type(df_age)} Number of {mode} patients {len(df_age)}")
        #
        return df_age

# ----------------------   
def analyze_peaks_valleys_global(df, mode, grain=200):
# ----------------------
    ages_array = df['age'].values
    
    # Compute KDE
    kde = gaussian_kde(ages_array)
    # Create fine-grained x points for evaluation
    x_eval = np.linspace(ages_array.min(), ages_array.max(), grain)
    # Get density values
    density = kde.evaluate(x_eval)
    # Find peaks and valleys
    peaks, _ = find_peaks(density)
    valleys, _ = find_peaks(-density)
    
    peaks_M = x_eval[peaks]
    valleys_M = x_eval[valleys]

    print_write (f"---------------------------------------")
    print_write (f"analyze_peaks_valleys_global for {mode}")
    print_write (f"---------------------------------------")
    print_write (f"Peaks {mode}: {peaks_M}")
    print_write (f"Valleys {mode}: {valleys_M}")
    #print_write (f"Density {mode}:\n{density_M}")

# -----------------------
def analyze_peaks_valleys_refine(df, mode, grain=300, bandwidth=None): # Use smaller bandwidth for finer detail
# -----------------------
    ages_array = df['age'].values
    
    kde = gaussian_kde(ages_array, bw_method=bandwidth)
    x_eval = np.linspace(ages_array.min(), ages_array.max(), grain)  # More points
    density = kde.evaluate(x_eval)
    peaks, peak_props = find_peaks(density, prominence=0.001)  # Add prominence threshold
    valleys, valley_props = find_peaks(-density, prominence=0.001)
    
    print_write (f"---------------------------------------")
    print_write (f"analyze_peaks_valleys_refine for {mode}")
    print_write (f"---------------------------------------")
    print_write (f"bandwidth {mode}: {bandwidth}")
    print_write (f"peaks {mode}: {peaks}")
    print_write (f"peak_props {mode}: {peak_props}")
    print_write (f"valleys {mode}: {valleys}")
    print_write (f"valley_props {mode}: {valley_props}")
    return x_eval[peaks], x_eval[valleys], density

# --------------------
def analyze_age_groups_global(df, mode):
# --------------------
    ages_array = df['age'].values
    # Create age bins (e.g., 5-year intervals)
    bins = np.arange(ages_array.min(), ages_array.max()+5, 5)
    # Get histogram
    hist, bin_edges = np.histogram(ages_array, bins=bins)
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    maxima = argrelextrema(hist, np.greater)[0]
    minima = argrelextrema(hist, np.less)[0]
    
    print_write (f"------------------------------------")
    print_write (f"analyze_age_groups_global for {mode}")
    print_write (f"------------------------------------")
    print_write (f"Maxima {mode}: {maxima}")
    print_write (f"Minima {mode}: {minima}")

    return maxima, minima, hist, bin_edges

# ----------------------------
def analyze_age_groups_refine(df, mode):
# ----------------------------
    ages_array = df['age'].values
    # Use smaller bin size
    bins = np.arange(ages_array.min(), ages_array.max()+2, 2)  # 2-year bins
    hist, bin_edges = np.histogram(ages_array, bins=bins)
    # Add smoothing before finding extrema
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(hist, sigma=1)
    from scipy.signal import argrelextrema
    maxima = argrelextrema(smoothed, np.greater)[0]
    minima = argrelextrema(smoothed, np.less)[0]
    
    print_write (f"------------------------------------")
    print_write (f"analyze_age_groups_refine for {mode}")
    print_write (f"------------------------------------")
    print_write (f"Maxima {mode}: {maxima}")
    print_write (f"Minima {mode}: {minima}")
    print_write (f"Hist {mode}: {hist}")
    print_write (f"Bin_edges {mode}: {bin_edges}")
    return maxima, minima, hist, bin_edges
# ----------------------------
def analyze_age_groups_plot(df, mode):
# ----------------------------

    # Sample age data (Replace this with df['age'].values in actual implementation)
    ages_array = df['age'].values  # Extract age values as a NumPy array
    # Create bins with a 2-year interval
    bins = np.arange(ages_array.min(), ages_array.max() + 2, 2)
    hist, bin_edges = np.histogram(ages_array, bins=bins)
    # Apply Gaussian smoothing
    smoothed = gaussian_filter1d(hist, sigma=1)
    # Find local maxima (peaks) and minima (troughs)
    maxima = argrelextrema(smoothed, np.greater)[0]
    minima = argrelextrema(smoothed, np.less)[0]
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))
    # Histogram
    ax.bar(bin_edges[:-1], hist, width=2, color='gray', alpha=0.5, label='Age Distribution (Histogram)')
    # Smoothed curve (aligned to bin centers)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, smoothed, color='black', label='Smoothed Curve', linewidth=1)
    # Mark extrema
    ax.scatter(bin_centers[maxima], smoothed[maxima], color='red', label='Maxima (Peaks)', zorder=3, s=25)
    ax.scatter(bin_centers[minima], smoothed[minima], color='blue', label='Minima (Troughs)', zorder=3, s=25)
    # Labels and title
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Patient Count")
    ax.set_title("Refined Age Distribution with Peaks and Troughs")
    # X-axis tick improvements
    ax.set_xticks(bin_edges[::2])  # Show every second bin for better spacing
    plt.xticks(rotation=30, ha='right')
    # Grid and legend
    ax.grid(True, linestyle="--", alpha=0.5)
    # Get existing handles and labels
    handles, labels = ax.get_legend_handles_labels()
    custom_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=0)
    ax.legend([custom_handle] + handles, ['Gender: M'] + labels, loc='upper left', fontsize=12)
    # Show the plot
    plt.tight_layout()
    plt.show()

'''
### **Refined Age Group Analysis in Male Patients**

In order to better understand the age distribution of male patients, we conducted a refined analysis using a **2-year bin size**. This finer granularity allows us to detect subtle variations in the age distribution that may be overlooked in broader age categories. Additionally, we applied a **Gaussian smoothing filter** to reduce noise and identify significant peaks (**maxima**) and troughs (**minima**) in the histogram.

---

### **Key Findings**
#### **1. Histogram of Age Distribution (2-Year Bins)**
The histogram represents the frequency of patients in each 2-year age interval, with the following notable characteristics:

- The highest counts occur in the middle-aged and older adult groups.
- The frequency varies, showing alternating periods of higher and lower representation.

#### **2. Detected Maxima (Local Peaks)**
- **Maxima at bin index 17**, corresponding to **age 51**.
  - This suggests that there is a notable concentration of male patients around this age.

#### **3. Detected Minima (Local Troughs)**
- **Minima at bin indices 24 and 27**, corresponding to **ages 65 and 71**.
  - These dips suggest relatively fewer male patients in these age intervals.

---

### **Interpretation and Implications**
1. **Peak at Age 51**:
   - This age group may be a key demographic affected by the condition under study.
   - It is crucial to explore whether this concentration is linked to occupational factors, genetic predispositions, or lifestyle influences.

2. **Dips at Ages 65 and 71**:
   - These age gaps may indicate a lower prevalence or underrepresentation of cases in these intervals.
   - Possible reasons include cohort effects, health-related survival differences, or diagnostic patterns.

3. **Overall Trend**:
   - The general shape of the histogram suggests an increasing prevalence with age, consistent with chronic progressive conditions.
   - The observed oscillations highlight the importance of using a fine-scale binning method for detecting age-specific variations.

---

### **Conclusion**
This refined analysis reveals that the age distribution of male patients is not uniform but characterized by distinct peaks and troughs. The **notable concentration at age 51** and the **relative dips at 65 and 71** warrant further investigation into potential biological, environmental, or healthcare access factors contributing to these trends.

Would you like any additional statistical tests or visualization adjustments to support these findings?
'''

# -----------------
def bootstrap_peaks(df, mode, n_bootstrap=1000): # 4. Bootstrap Analysis for Peak Significance:
# -----------------
    ages_array = df['age'].values
    peaks_locations = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(ages_array, size=len(ages_array))
        kde = gaussian_kde(sample)
        x_eval = np.linspace(ages_array.min(), ages_array.max(), 200)
        density = kde.evaluate(x_eval)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density)
        peaks_locations.append(x_eval[peaks])
    print_write (f"ANALYZE BOOTSTRAP PEAKS for {mode}")
    print_write (f"Peaks_locations {mode}: {peaks_locations}")
    return peaks_locations

# ----
# Util
# ----
def print_write(line):
    print (line)
    write (line)

def mai1(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path):
        
    # Selector
    # --------
    df1, df2, df3 = inp1(file_path) 

    def exe1(df, mode):
        print_write ("") 
        print_write (">>>")
        print_write (f"Patients {mode}")
        print_write (">>>")
        #
        df_age = inpu(df, mode)
        print (df_age)
        #
        analyze_peaks_valleys_global(df_age, mode)
        analyze_peaks_valleys_refine(df_age, mode)
        #
        analyze_age_groups_global(df_age, mode)
        analyze_age_groups_refine(df_age, mode)
        analyze_age_groups_plot(df_age, mode)
        #
        # bootstrap_peaks(df_age, mode, n_bootstrap=1000)

    # Exec   
    df = df1 
    print (df)
    mode = "M"
    exe1(df, mode)
    mode = "F"
    exe1(df, mode)
    pass

def mai2():

    # Create data for the DataFrame
    data = [
        # Age periods with low frequency
        ["14-30", "F", "Low", "scattered", 1, 2, "Low frequency period"],
        ["17-35", "M", "Low", "valley+rise", 1, 3, "Low frequency then slight rise"],
        
        # Early peaks and transitions
        ["30-40", "F", "Peak", "rise+valley", 1, 9, "First rise with peak 34-36"],
        ["35-45", "M", "Rising", "rise", 3, 7, "Rising period"],
        
        # Middle age periods
        ["40-54", "F", "Rising", "rise", 4, 10, "Gradual increase"],
        ["45-55", "M", "Peak", "major", 8, 11, "First major peak"],
        ["54-70", "F", "High", "plateau", 9, 11, "High stable plateau"],
        ["55-65", "M", "High", "plateau", 7, 10, "Plateau period"],
        
        # Late middle age
        ["65-73", "M", "Valley", "major", 2, 3, "Major valley period"],
        ["70-76", "F", "Valley", "minor", 7, 8, "Minor valley"],
        
        # Late age peaks and decline
        ["73-77", "M", "Peak", "major", 9, 9, "Second major peak"],
        ["76-78", "F", "Peak", "major", 12, 12, "Late peak"],
        ["77-85", "M", "Decline", "gradual", 1, 5, "Decline period"],
        ["78-90", "F", "Decline", "gradual", 1, 7, "Gradual decline"]
    ]

    # Create DataFrame
    df_periods = pd.DataFrame(data, columns=[
        'age_period', 
        'gender', 
        'pattern_type',
        'pattern_subtype', 
        'min_patients',
        'max_patients',
        'description'
    ])

    # Display the DataFrame
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_periods}")
        write(f"\n{df_periods}")
    xlsx = True
    if xlsx: 
        file_name = 'df_periods.xlsx'
        df_periods.to_excel(file_name, index=False)
 
# ----------
# TO CORRECT
# ----------
def mai3(file_path):

    if False:
        data = {
            'doss': ['D9921', 'D9825', 'D9730', 'D9674', 'D9574', 'D10315', 'D10291', 'D10213', 'D10129', 'D10103'],
            'age_bin': ['50-59', '70-79', '70-79', '40-49', '60-69', '50-59', '70-79', '60-69', '70-79', '60-69'],
            'age': [54, 70, 73, 47, 60, 57, 75, 67, 75, 68],
            'sexe': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M']
        }
        df = pd.DataFrame(data)

    # Selector
    # --------
    df1, df2, df3 = inp1(file_path) 
    df = df1
    df1 = df1[df1['sexe'] == 'M']
    print (df1)

    if False:
        data = {'patient_id': ['D9921', 'D9825', 'D9730', 'D9674', 'D9574', 'D10315', 'D10291', 'D10213', 'D10129', 'D10103'],
                'age_bin': ['50-59', '70-79', '70-79', '40-49', '60-69', '50-59', '70-79', '60-69', '70-79', '60-69'],
                'age': [54, 70, 73, 47, 60, 57, 75, 67, 75, 68],
                'gender': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M']}
        df1 = pd.DataFrame(data)



    # Sample DataFrame structure
    # df_input = pd.read_csv("your_data.csv")  # Assuming you load it from a CSV file

    # Function to dynamically segment age bins
    def dynamic_age_bins(df, n_clusters=5):
        age_values = df['age'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        df['age_cluster'] = kmeans.fit_predict(age_values)
        df['age_bin'] = df.groupby('age_cluster')['age'].transform(lambda x: f"{x.min()}-{x.max()}")
        return df

    # Function to identify peaks, valleys, and plateaus
    def identify_patterns(counts):
        patterns = []
        peaks, _ = scipy.signal.find_peaks(counts)
        valleys, _ = scipy.signal.find_peaks(-counts)
        plateaus = np.where(np.diff(counts) == 0)[0]  # Identify plateaus
        
        for i in range(len(counts) - 1):
            if i in peaks:
                pattern = "Peak"
                subtype = "major" if counts[i] > np.percentile(counts, 90) else "minor"
                description = "Major peak" if subtype == "major" else "Minor peak"
            elif i in valleys:
                pattern = "Valley"
                subtype = "major" if counts[i] < np.percentile(counts, 10) else "minor"
                description = "Major valley" if subtype == "major" else "Minor valley"
            elif i in plateaus:
                pattern = "Plateau"
                subtype = "stable"
                description = "Stable period"
            else:
                pattern = "Trend"
                subtype = "rising" if counts[i + 1] > counts[i] else "declining"
                description = "Increasing" if subtype == "rising" else "Decreasing"
            
            patterns.append((i, i + 1, pattern, subtype, description))
        
        return patterns

    # Sample DataFrame with patient age & gender
    df_input = pd.DataFrame({
        'patient_id': [f'D{10000 + i}' for i in range(50)],
        'age': np.random.randint(20, 80, 50),
        'gender': np.random.choice(['M', 'F'], 50)
    })
    df_input = df1

    # Apply dynamic binning
    df_segmented = dynamic_age_bins(df_input, n_clusters=6)

    # Group by age_bin and count patients
    age_bin_counts = df_segmented.groupby('age_bin').size().reset_index(name='counts')

    # Identify patterns
    patterns = identify_patterns(age_bin_counts['counts'])

    # Create output DataFrame
    df_output = pd.DataFrame(columns=['age_period', 'gender', 'pattern_type', 'pattern_subtype', 'min_patients', 'max_patients', 'description'])

    for start_idx, end_idx, pattern_type, pattern_subtype, description in patterns:
        start_age = age_bin_counts.loc[start_idx, 'age_bin']
        end_age = age_bin_counts.loc[end_idx, 'age_bin']
        min_patients = age_bin_counts.loc[start_idx:end_idx, 'counts'].min()
        max_patients = age_bin_counts.loc[start_idx:end_idx, 'counts'].max()
        new_row = pd.DataFrame([{
            'age_period': f'{start_age}-{end_age}',
            'gender': 'M',  # Change dynamically if needed
            'pattern_type': pattern_type,
            'pattern_subtype': pattern_subtype,
            'min_patients': min_patients,
            'max_patients': max_patients,
            'description': description
        }])
        df_output = pd.concat([df_output, new_row], ignore_index=True)

    # Display the results
    print(df_output)


    # Display the output DataFrame
    print(df_output)
    pass


def main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
    
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
        mai1(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path)
        mai2() 
        mai3(file_path)
        pass

def ke13_pati_agco_sexe_c3c6_full_abso():

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
    #
    ceap_mono = False
    indx_name = 'age_bin'  
    indx_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    filt_name = 'sexe'
    filt_valu = None # 'G' 'D'
    #    
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}jrnl.txt' if filt_valu is not None else f'{script_name}jrnl.txt')
    main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    ke13_pati_agco_sexe_c3c6_full_abso()
    
'''
Thank you for the corrected data! Now I can analyze the genuine gender differences:

Male Distribution Analysis:
1. Age Groups (2-year bins):
- Single prominent maximum at index 17 (around 51-53 years)
- Notable frequencies:
  * Highest peak: 11 patients (51-53 years)
  * Secondary concentrations: 9-10 patients in several bins around 43-45, 73-75 years
- Clear valleys at indices 24 and 27 (approximately 65-67 and 71-73 years)
- Age range: 17-85 years
- More irregular distribution pattern

Female Distribution Analysis:
1. Age Groups (2-year bins):
- Two maxima at indices 3 and 10 (approximately 20-22 and 34-36 years)
- Notable frequencies:
  * Highest concentration: 12 patients (76-78 years)
  * Consistent frequencies of 10-11 patients across 50-70 years
- Valleys at indices 4, 12, and 29 (approximately 22-24, 38-40, and 72-74 years)
- Age range: 14-90 years
- More uniform distribution in middle ages

Key Differences:
1. Distribution Shape:
- Males show more pronounced clustering around specific ages
- Females show more uniform distribution with smaller variations

2. Peak Characteristics:
- Male peak is more concentrated around early 50s
- Female distribution has multiple smaller peaks with more even distribution

3. Age Range:
- Females have slightly wider age range (14-90 vs 17-85)
- Females show more consistent representation across age groups

4. Pattern Stability:
- Male distribution shows more abrupt changes between adjacent age groups
- Female distribution shows smoother transitions between age groups

These results better align with our visual observations and provide quantitative support for the gender-specific patterns we noticed in the plots. 
Would you like me to explore any specific aspect of these differences in more detail?
'''
'''
I'll create a synthesis of significant age periods, identifying peaks and valleys for each gender. I'll organize them chronologically and include patient counts.

For Males (M):
1. 17-35: Low frequency period
   - Valley at 17-25: ~1-2 patients per 2-year bin
   - Slight increase 25-35: ~2-3 patients per 2-year bin
   
2. 35-45: Rising period
   - Growing peak: 3-7 patients per 2-year bin
   - More consistent numbers

3. 45-55: First major peak
   - Peak at 51-53: 11 patients
   - Surrounding bins: 8-9 patients

4. 55-65: Plateau period
   - Stable numbers: 7-10 patients per bin
   
5. 65-73: Valley period
   - Valley at 65-67: 3 patients
   - Valley at 71-73: 2 patients

6. 73-77: Second major peak
   - Peak around 73-75: 9 patients
   
7. 77-85: Decline period
   - Decreasing from 5 to 1 patient per bin

For Females (F):
1. 14-30: Low frequency period
   - Scattered 1-2 patients per bin
   - Small peak at 20-22

2. 30-40: First rise
   - Peak at 34-36: 9 patients
   - Valley at 38-40: 1 patient

3. 40-54: Gradual increase
   - Growing from 4 to 10 patients per bin
   - More uniform distribution

4. 54-70: High plateau
   - Consistent 9-11 patients per bin
   - Most stable period

5. 70-76: Minor valley
   - Slight dip to 7-8 patients per bin

6. 76-78: Late peak
   - Peak: 12 patients

7. 78-90: Gradual decline
   - Decreasing from 7 to 1-2 patients per bin

Key Observations:
- Males show more pronounced peaks and valleys
- Females show longer stable periods
- Males have stronger age clustering
- Females have wider age range
- Both show late-age peaks but at different ages
- Female distribution is generally smoother

Would you like me to focus on any particular time period or provide more detailed patient counts for specific ranges?
'''