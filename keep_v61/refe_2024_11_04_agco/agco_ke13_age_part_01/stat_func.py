from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from util_file_mngr import set_file_objc, write
from scipy.stats import norm, gaussian_kde, skew, median_abs_deviation, mode, kurtosis
from scipy import stats

# ----
# Desc
# ----
def desc(what, df, colu_cate_list, indx_name, colu_name):
    
    # Initialize lists for ageM and ageF
    col1_name = colu_cate_list[0]
    col1_nump = df[col1_name].to_numpy()
    
    # Normal distribution
    # -------------------
    mu, sigma = norm.fit(col1_nump)
    da01 = {
    'Mu': [mu],
    'Sigma': [sigma]
    }
    
    # Quartiles and Outliers
    # ----------------------
    def quar_outl(ages_):
        q1_, q3_ = np.percentile(ages_, [25, 75])
        below_q1_ = len(ages_[ages_ < q1_])
        above_q3_ = len(ages_[ages_ > q3_])
        within_iqr_ = len(ages_) - below_q1_ - above_q3_
        pati_belo_ = below_q1_ / len(ages_) * 100
        pati_with_ = within_iqr_ / len(ages_) * 100
        pati_abov_ = above_q3_ / len(ages_) * 100
        return q1_, q3_, below_q1_, above_q3_, within_iqr_, pati_belo_, pati_with_, pati_abov_
    q1, q3, below_q1, above_q3, within_iqr, pati_belo, pati_with, pati_abov = quar_outl(col1_nump)
    percentiles = np.percentile(col1_nump, [5, 10, 90, 95])
    da02 = {
    'Q1': [q1],
    'Q3': [q3],
    'Below_Q1': [below_q1],
    'Above_Q3': [above_q3],
    'Within_IQR': [within_iqr],
    'Patients_Below_Q1': [pati_belo],
    'Patients_Within_IQR': [pati_with],
    'Patients_Above_Q3': [pati_abov],
    '5th_perc': percentiles[0],
    '10th_perc': percentiles[1],
    '90th_perc': percentiles[2],
    '95th_perc': percentiles[3]
    }
    
    # Median
    # ------
    median_age = np.median(col1_nump)
    da03 = {
    'Median': [median_age]
    }
    
    # MAD range
    # ---------
    mad = stats.median_abs_deviation(col1_nump)
    da04 = {
    'Mad': [mad]
    }
    
    # Mean
    # ----
    mean = col1_nump.mean()
    da05 = {
    'Mean': [mean]
    }
    
    # Std
    # ----
    std_ = col1_nump.std()
    da06 = {
    'Std': [std_]
    }
    
    # Mode
    # ----
    mode_ = stats.mode(col1_nump) ; mode_ = mode_.mode
    if np.isscalar(mode_): 
        mode_ = float(mode_) 
    else: 
        mode_= mode_[0]
    da07 = {
    'Mode': [mode_]
    }
    
    # Skeness
    # -------
    skew_ = skew(col1_nump)
    da08 = {
    'Skewness': [skew_]
    }
    
    # Kurtosis
    # --------
    kurt_ = kurtosis(col1_nump)
    da09 = {
    'Kurtosis': [kurt_]
    }
    
    # Resu
    data = {**da01, **da02, **da03, **da04, **da05, **da06, **da07, **da08, **da09}
    return data

# -------------------------------
# Shapiro-Wilk Test for Normality
# -------------------------------
def shap(what, df, colu_cate_list, indx_name, colu_name, H0, HA):

    # Initialize lists for ageM and ageF
    col1_name = colu_cate_list[0]
    col1_nump = df[col1_name].to_numpy()
    
    # Test for male ages
    stat, pval = shapiro(col1_nump)

    # Resu
    sta1_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pva1_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nShapiro-Wilk test for Normality {col1_name}: Stat:{sta1_form} Pval:{pva1_form}") 
    write(f"\nData : {what}\nShapiro-Wilk test for Normality {col1_name}: Stat:{sta1_form} Pval:{pva1_form}")        
        
    # Intp
    alpha = 0.05
    if pval < alpha:
        print(f"Shapiro-Wilk test for Normality : Reject the null hypothesis:\n{HA}")
        write(f"Shapiro-Wilk test for Normality : Reject the null hypothesis:\n{HA}")
    else:
        print(f"Shapiro-Wilk test for Normality : Fail to reject the null hypothesis:\n{H0}")
        write(f"Shapiro-Wilk test for Normality : Fail to reject the null hypothesis:\n{H0}") 
    pass

    # Exit
    da01 = {
        'Shapiro-Wilk Stat': [stat],
        'Shapiro-Wilk Pval': [pval]
        }
    return da01

def stat_resu(what, data, colu_cate_list, indx_name, colu_name):
    
    # Initialize lists for ageM and ageF
    col1_name = colu_cate_list[0]
    
    df_resu = pd.DataFrame(data)
    print(f"\nStep 1 : df_resu.size:{len(df_resu)} df_resu.type:{type(df_resu)}\n{df_resu}\n:{df_resu.index}\n:{df_resu.columns}")

    # Function to format the values
    def format_value(value):
        if isinstance(value, (int, float)):
            if value < 0.001:
                return f"{value:.3e}"
            else:
                return f"{value:.3f}"
        return value
    df_resu = df_resu.applymap(format_value)

    # Display the DataFrame
    print(f"\nData : {what}\nDesc {col1_name}:")
    write(f"\nData : {what}\nDesc {col1_name}:")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
        
    # Exit
    return df_resu

def stat_exec(what, df, colu_cate_list, indx_name, colu_name):
    
    da01 = desc(what, df, colu_cate_list, indx_name, colu_name)
    
    H0 = f"The data appears to be normally distributed."
    HA = f"The data does not appear to be normally distributed"
    da02 = shap(what, df, colu_cate_list, indx_name, colu_name, H0, HA)
    
    data = {**da01, **da02}
    df_resu = stat_resu(what, data, colu_cate_list, indx_name, colu_name)
    
    return df_resu