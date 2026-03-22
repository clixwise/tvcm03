import numpy as np
from scipy import stats
import pandas as pd

# ****
# Help
# ****
 
# ----
# Help 1
# ----

# Continuous variables (Veins/QOL, ...)
def summarize_continuous_stru(series, method):
    
    # Exec
    # ----
    match method:
        case "mean_sd":
            return series.mean(), series.std() # f"{series.mean():.1f}", f"{series.std():.1f}"
        case "median_iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            return series.median(), q1, q3 # f"{series.median():.1f},[{q1:.1f}–{q3:.1f}]"
        case _:
            raise Exception()

# Categorical variables (VCSS, ...)
def summarize_categorical_stru(series):
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    return {
        k: f"{v} ({100*v/total:.1f}%)"
        for k, v in counts.items()
    }
    
# ----
# Help 2
# ---- 
   
# Continuous variables (Veins/QOL, ...)
def summarize_continuous_edit(series, method):
    
    # Prec
    # ----
    series = series.dropna()
    n = len(series)
    if n == 0:
        return "N/A"
    # print (series)
    
    # Note
    # ----
    # Optional: Quick check for your own knowledge (won't change output if forced)
    force_normality = True
    if not force_normality and n > 3:
        stat, p = stats.shapiro(series)
        if p < 0.05:
            method = "median_iqr" # Switch if data is truly messy
    
    # Exec
    # ----
    match method:
        case "sum":
            decimals = 0
            return f"{series.sum():.{decimals}f}"
        case "mean_sd":
            return f"{series.mean():.1f} ± {series.std():.1f}"
        case "median_iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            return f"{series.median():.1f} ({q1:.1f}–{q3:.1f})"
        case "jrnl_qual":
                # Mean ± SD [95% CI lower – upper]
                avg = series.mean()
                sd = series.std()
                se = sd / np.sqrt(n)
                # Use t-distribution for clinical accuracy with smaller samples (n=30)
                ci_low, ci_high = stats.t.interval(0.95, n-1, loc=avg, scale=se)
                
                return f"{avg:.1f} ± {sd:.1f} [{ci_low:.1f}–{ci_high:.1f}]"
        case _:
            raise Exception()
    
# Categorical variables (VCSS, ...)
# cate_sort=True : the caller's sorting is respected (i.e. as determined by pd.Categorical)
# cate_sort=False: the caller's sorting is NOT respected (the result is retured by decreasing value count)
def summarize_categorical_edit(series, nan_labl="-", cate_sort=True): # nan_labl=np.nan
    # 1. Fill NaNs with a readable label so they can be treated as a category
    # 2. 'value_counts()' defaults to sort=True ; which prioritizes the frequency (counts) over the categorical order.
    # 3. sort=False respects the Categorical order defined in the dataframe (else, it would be sorted on 'count')
    # 2. Safely add the nan_labl to categories if it's not already there
    if nan_labl not in series.cat.categories:
        series = series.cat.add_categories([nan_labl])
    counts = series.fillna(nan_labl).value_counts(dropna = False, sort = not cate_sort)
    #
    total = counts.sum()
    if total == 0:
        return {}
    #
    return {
        k: f"{v} ({100*v/total:.1f}%)"
        for k, v in counts.items()
        if v > 0  # Ignore counts of 0
    }
    
# ----
# Help 3 : Returns the mean change and the p-value compared to baseline.
# ----
def get_comparison_stats(df_fram, tipo_TX, tipo_T0="T0"):
    
    trac = True
        
    # Exec T0
    # ----
    if tipo_TX == tipo_T0:
        return ""
    
    df_wide = df_fram.pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
    if trac:
        print_yes(df_wide, "df_wide")

    # Prec : drop any NaN : i.e. patients that did not complete TX
    # ----
    df_pure = df_wide[[tipo_T0,tipo_TX]].dropna()
    
    # Exec
    # ----
    df_delt = df_pure[tipo_TX] - df_pure[tipo_T0]
    #
    delt_mean = df_delt.mean()
    delt_sd   = df_delt.std(ddof=1)
    delt_perc = (delt_mean / df_pure["T0"].mean()) * 100
    
    if trac:
        print("Mean change:", round(delt_mean,2))
        print("SD change  :", round(delt_sd,2))
        print("% change   :", round(delt_perc,1))
    
    # Exit
    # ----
    valu = f"{round(delt_mean,1)} ± {round(delt_sd,1)} ({round(delt_perc,1)}%)"
    return valu


def get_comparison_stats_needs_same_count_patients(df, tipo_TX, tipo_T0="T0"):
    
    # Trac
    trac = True
    data_T0 = df[df["timepoint"] == tipo_T0]
    data_TX = df[df["timepoint"] == tipo_TX]
    if trac:
        print_yes(data_T0, "df_T0")
        print_yes(data_TX, "df_TX")
    
    # Prec
    # ----
    data_T0 = df[df["timepoint"] == tipo_T0]["VEINES_QOL_t"].dropna()
    data_TX = df[df["timepoint"] == tipo_TX]["VEINES_QOL_t"].dropna()
    
    # Exec T0
    # ----
    if tipo_TX == tipo_T0:
        # Exit
        return "Ref.", ""
    
    # Exec TX 1 : Check if the datasets are identical or have zero variance in differences
    # ----
    if np.array_equal(data_TX, data_T0):
        # Exit
        return "+0.0", "p = 1.000"

    # Exec TX 2
    # ----
    t_stat, p_val = stats.ttest_rel(data_TX, data_T0)
    
    # Handle the case where ttest_rel might still return NaN 
    # (e.g., if all differences are the same constant value)
    if np.isnan(p_val):
        return f"{data_TX.mean() - data_T0.mean():.1f}", "p = 1.000"

    # P-val : format
    if p_val < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_val:.3f}"
    
    # Mean diff : format
    mean_diff = data_TX.mean() - data_T0.mean()
    sign = "+" if mean_diff > 0 else ""
    
    # Exit
    return f"{sign}{mean_diff:.1f}", p_str
    
def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass