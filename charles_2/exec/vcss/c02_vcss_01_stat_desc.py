    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranVCSS_01_desc
    
import pandas as pd
import sys
import os  
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import summarize_continuous_edit
from pprint import pprint
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from vcss.c02_vcss_01_stat_adat import StatTranVCSS_01_desc
        
# ----
# https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4
# ----
'''
Timepoint                   T0             T1
Outcome
VCSS (max limb)  5.5 [4.8–6.2]  5.5 [5.2–5.8]
VEINES-VCSS         56.0 ± 11.0     58.2 ± 4.5
VEINES-VCSS         60.8 ± 12.2     62.5 ± 5.4
'''
def exec_stat_desc(stat_tran_desc: StatTranVCSS_01_desc):
    _exec_stat_desc_gemi(stat_tran_desc)
    _exec_stat_desc_open(stat_tran_desc)
    
# --------------
# Version Openai [2026-01-19]
# --------------
def _exec_stat_desc_open(stat_tran_desc: StatTranVCSS_01_desc):
    _exec_stat_desc_open_mean(stat_tran_desc)
    _exec_stat_desc_open_delt(stat_tran_desc)

def _exec_stat_desc_open_mean(stat_tran_desc: StatTranVCSS_01_desc) -> None:
    # from vcss.c02_vcss_01_stat_ import StatTranVCSS_01_desc
    
    trac = True

    # Data
    # ---- 
    df_frax = stat_tran_desc.stat_tran.frax
    
    # Trac
    # ----
    if trac:
        print_yes(df_frax, labl="df_frax")
        
    ## 
    ## **Table 4 – VCSS descriptive (Median [IQR])**
    ## 
    def median_iqr_1(x):
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        return f"{x.median():.1f} [{q1:.1f}–{q3:.1f}]"
    def median_iqr_2(x):
        q1 = np.percentile(x,25)
        q3 = np.percentile(x,75)
        return f"{np.median(x):.1f} [{q1:.1f}–{q3:.1f}]"

    #
    df_publ = (
        df_frax
        .assign(Metric=lambda d: "VCSS_" + d["Limb"].astype(str))
        .groupby(["Metric", "timepoint"])["VCSS"]
        .apply(median_iqr_1)
        .reset_index(name="Value")
    )
    '''
       Metric timepoint           Value
    0  VCSS_L        T0  5.0 [3.0–10.8]
    1  VCSS_L        T1  5.0 [3.0–10.8]
    2  VCSS_L        T2  5.0 [3.0–10.8]
    3  VCSS_R        T0   5.0 [3.2–8.8]
    4  VCSS_R        T1   5.0 [3.2–8.8]
    5  VCSS_R        T2   5.0 [3.2–8.8]
    '''
    df_publ.insert(0, "ID", StatTranVCSS_01_desc.__name__)
    df_publ.rename(columns={"timepoint": "Timepoint"}, inplace=True)

    # Trac
    # ----
    '''
       ID                    Metric  Timepoint Value
    0  StatTranVCSS_01_desc  VCSS_L  T0        5.0 [3.0–10.8]
    1  StatTranVCSS_01_desc  VCSS_L  T1        5.0 [3.0–10.8]
    2  StatTranVCSS_01_desc  VCSS_L  T2        5.0 [3.0–10.8]
    3  StatTranVCSS_01_desc  VCSS_R  T0         5.0 [3.2–8.8]
    4  StatTranVCSS_01_desc  VCSS_R  T1         5.0 [3.2–8.8]
    5  StatTranVCSS_01_desc  VCSS_R  T2         5.0 [3.2–8.8]
    '''
    if trac:
        print_yes(df_publ, labl="df_publ")     
    pass

def _exec_stat_desc_open_delt(stat_tran_desc: StatTranVCSS_01_desc) -> None:
    from vcss.c02_vcss_01_stat_ import StatTranVCSS_01_desc
    
    trac = True

    # Data
    # ---- 
    df_frax = stat_tran_desc.stat_tran.frax
    
    # Trac
    # ----
    if trac:
        print_yes(df_frax, labl="df_frax")
        
    ## 
    ## **Table 5 – VCSS change vs baseline (mock model-style)**
    ## 
    rows = []
    limb_cate_list = df_frax['Limb'].cat.categories.tolist()
    time_cate_list = df_frax['timepoint'].cat.categories.tolist()
    time_cate_list.remove("T0")
    for limb_item in limb_cate_list:
        base = df_frax.query("Limb == @limb_item and timepoint == 'T0'")["VCSS"].mean()

        for tipo in time_cate_list:
            vals = df_frax.query("Limb == @limb_item and timepoint == @tipo")["VCSS"]
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            lo = mean - 1.96 * se
            hi = mean + 1.96 * se

            rows.append({
                "ID": StatTranVCSS_01_desc.__name__,
                "Metric": f"VCSS_{limb_item}",
                "Timepoint": tipo,
                "Mean Change vs T0 (95% CI)":
                    f"{mean - base:+.1f} [{lo - base:+.1f}–{hi - base:+.1f}]"
            })
    #
    df_publ = pd.DataFrame(rows)

    # Trac
    # ----
    '''
    ID                       Metric  Timepoint Mean Change vs T0 (95% CI)
    0  StatTranVCSS_01_desc  VCSS_R  T1        +0.0 [-1.8–+1.8]
    1  StatTranVCSS_01_desc  VCSS_R  T2        +0.0 [-1.8–+1.8]
    2  StatTranVCSS_01_desc  VCSS_L  T1        +0.0 [-1.9–+1.9]
    3  StatTranVCSS_01_desc  VCSS_L  T2        +0.0 [-1.9–+1.9]
    '''
    if trac:
        print_yes(df_publ, labl="df_publ")     
    pass

# --------------
# Version Gemini [2025-12-01]
# --------------
def _exec_stat_desc_gemi(stat_tran_desc: StatTranVCSS_01_desc) -> None:
    from vcss.c02_vcss_01_stat_ import StatTranVCSS_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram
    mark_dict = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta01_base_char
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
    
    # Exec : tech
    # ---- 
    tech_list = []
    for tipo in ["T0", "T1", "T2"]:
        
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            continue
        tech_list.append([StatTranVCSS_01_desc.__name__, "VCSS_P", tipo, summarize_continuous_edit(df_tipo["P"], "median_iqr")])
        tech_list.append([StatTranVCSS_01_desc.__name__, "VCSS_R", tipo, summarize_continuous_edit(df_tipo["R"], "median_iqr")])
        tech_list.append([StatTranVCSS_01_desc.__name__, "VCSS_L", tipo, summarize_continuous_edit(df_tipo["L"], "median_iqr")])
    #
    if trac:
        print (tech_list)
        
    # Exec : publ
    # ---- 
    publ_list = []
    for tipo in ["T0", "T1", "T2"]:
        
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            continue
        publ_list.append([StatTranVCSS_01_desc.__name__, "VCSS_P", tipo, summarize_continuous_edit(df_tipo["P"], "median_iqr")])
        publ_list.append([StatTranVCSS_01_desc.__name__, "VCSS_R", tipo, summarize_continuous_edit(df_tipo["R"], "median_iqr")])
        publ_list.append([StatTranVCSS_01_desc.__name__, "VCSS_L", tipo, summarize_continuous_edit(df_tipo["L"], "median_iqr")])
    #
    if trac:
        print (publ_list)
        
    # Exec : mark
    # ----
    df_tipo = df_fram[df_fram["timepoint"] == 'T0'] 
    vein_info = summarize_continuous_edit(df_tipo["P"], "median_iqr")  
    mark_dict['.VCSS'] = (vein_info, Path(__file__).stem) # VCSS 54.8 ± 11.6

    # Oupu
    # ----
    cols = ['ID', 'Metric', 'Timepoint', 'Value']
    #
    df_tech = pd.DataFrame(tech_list, columns=cols)
    df_tech.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_tech, labl="df_tech")
    #
    df_publ = pd.DataFrame(publ_list, columns=cols)
    df_publ.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_publ, labl="df_publ")
    #
    if trac:
        pprint(mark_dict, indent=4, sort_dicts=False)

    # Exit
    # ----
    stat_tran_desc.resu_tech = df_tech
    stat_tran_desc.resu_publ = df_publ
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$


    baseline = df_fram[df_fram["timepoint"] == "T0"]["P"]

    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(baseline)
    print("Shapiro-Wilk p-value:", shapiro_test.pvalue)

    # Histogram
    plt.figure()
    plt.hist(baseline, bins=8)
    plt.xlabel("VCSS (T0)")
    plt.ylabel("Frequency")
    plt.title("Baseline VCSS Distribution")
    #plt.show()

    # Q-Q plot
    plt.figure()
    stats.probplot(baseline, dist="norm", plot=plt)
    plt.title("Q-Q Plot Baseline VCSS")
    #plt.show()
    pass


    summary = df_fram.groupby("timepoint")["P"].agg(
        mean="mean",
        sd="std",
        n="count"
    )

    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci_lower"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci_upper"] = summary["mean"] + 1.96 * summary["se"]

    print(summary)
    
    

    timepoints = summary.index
    means = summary["mean"]
    ci_lower = summary["ci_lower"]
    ci_upper = summary["ci_upper"]

    plt.figure()

    plt.plot(timepoints, means, marker='o')
    plt.fill_between(timepoints, ci_lower, ci_upper, alpha=0.2)

    plt.xlabel("Timepoint")
    plt.ylabel("VCSS (mean ± 95% CI)")
    plt.title("VCSS Over Time")

    #plt.show()
    
    
    
    plt.figure()

    for pid in df_fram["patient_id"].unique():
        patient_data = df_fram[df_fram["patient_id"] == pid]
        plt.plot(patient_data["timepoint"], patient_data["P"], alpha=0.3)

    plt.xlabel("Timepoint")
    plt.ylabel("VCSS")
    plt.title("Individual Patient Trajectories")

    #plt.show()
    pass

# $$$$$$$$$$$$$$$$$$$$$
# COHEN T0-> T1
# $$$$$$$$$$$$$$$$$$$$$


    # Pivot to wide format
    df_wide = df_fram.pivot(index="patient_id", columns="timepoint", values="P")

    # Compute change
    change_T1 = df_wide["T1"] - df_wide["T0"]

    mean_change = np.mean(change_T1)
    sd_change = np.std(change_T1, ddof=1)

    cohens_d_T1 = mean_change / sd_change

    print("T0 → T1")
    print("Mean change:", round(mean_change, 2))
    print("SD change:", round(sd_change, 2))
    print("Cohen's d:", round(cohens_d_T1, 3))
 
# $$$$$$$$$$$$$$$$$$$$$
# COHEN T0-> T2
# $$$$$$$$$$$$$$$$$$$$$   
 
    change_T2 = df_wide["T2"] - df_wide["T0"]

    mean_change = np.mean(change_T2)
    sd_change = np.std(change_T2, ddof=1)

    cohens_d_T2 = mean_change / sd_change

    print("T0 → T2")
    print("Mean change:", round(mean_change, 2))
    print("SD change:", round(sd_change, 2))
    print("Cohen's d:", round(cohens_d_T2, 3))   
    
    '''
    | d value | Interpretation                      |
    | ------- | ----------------------------------- |
    | 0.2     | Small                               |
    | 0.5     | Moderate                            |
    | 0.8     | Large                               |
    | 1.2+    | Very large                          |
    | 2.0+    | Massive (rare in clinical medicine) |
    VCSS improved from 10.9 ± 3.1 at baseline to 6.4 ± 2.8 at 12 months (
    mean change −4.5 ± 2.9; Cohen’s d = 1.55, 95% CI 1.05–2.05), indicating a large treatment effect.

    '''


    n = len(change_T1)
    se_d = np.sqrt((1/n) + (cohens_d_T1**2 / (2*n)))

    ci_low = cohens_d_T1 - 1.96 * se_d
    ci_high = cohens_d_T1 + 1.96 * se_d

    print("95% CI:", round(ci_low,3), "to", round(ci_high,3))
    
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Standardized Response Mean (SRM)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    df_wide = df_fram.pivot(index="patient_id", columns="timepoint", values="P")

    change_T2 = df_wide["T2"] - df_wide["T0"]

    mean_change = np.mean(change_T2)
    sd_change = np.std(change_T2, ddof=1)

    srm_T2 = mean_change / sd_change

    print("Mean change:", round(mean_change,2))
    print("SD change:", round(sd_change,2))
    print("SRM:", round(srm_T2,3))
    '''
    | SRM  | Interpretation |
    | ---- | -------------- |
    | 0.20 | Small          |
    | 0.50 | Moderate       |
    | 0.80 | Large          |
    | >1.0 | Very large     |
    | >1.5 | Exceptional    |

    '''
    # $$$$$$$$$$$$$
    # Minimal Clinically Important Difference (MCID) : VCSS MCID (≥2-point improvement)
    # At 12 months, 83% of patients achieved a ≥2-point reduction in VCSS, meeting the predefined MCID threshold.
    # $$$$$$$$$$$$$
    mcid_threshold = -2  # reduction of 2 points

    responders = change_T2 <= mcid_threshold

    prop_responders = np.mean(responders)

    print("MCID responders (%):", round(prop_responders * 100,1))
    
    # VEINES-QOL MCID (≥5-point increase) NOTE : we rather use the 1 yea improvement
    
    
    
    
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