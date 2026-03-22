 
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_81_stat_ import StatTranQOL_81_mcid_copi
    
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import sem, t
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.utils import resample
from qol_80_mixd_sens.c02_qol_81_stat_adat import StatTranQOL_81_mcid_copi   


# ****
# 1. Anchor‑based (mean change)
# 2. Anchor‑based (ROC/Youden)
# 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)
# ****

def exec_stat_mcid_copi_1(stat_tran_adat: StatTranQOL_81_mcid_copi) -> None:
    # from qol_80_mixd_sens.c02_qol_81_stat_ import StatTranQOL_81_mcid_copi

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta07_mcid

    if trac:
        print_yes(df_fram, labl='df_fram')      

    # ====
    # Build the T0–T2 change dataset
    # ====
    
    # Step 1 : Pivot to wide: one row per patient, T0 and T2 columns
    # ----
    df_wide = df_fram.pivot_table(index='patient_id', columns='timepoint', values='VEINES_QOL_t').reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={'T0': 'VEINES_QOL_T0', 'T1': 'VEINES_QOL_T1', 'T2': 'VEINES_QOL_T2'})
    if trac:
        print_yes(df_wide, labl='df_wide')
    # Merge Q3 at T2 (assuming one row per patient at T2 with Q3_T2)
    q3_t2 = df_fram.loc[df_fram['timepoint'] == 'T2', ['patient_id', 'C_3']].drop_duplicates()
    df_mc = df_wide.merge(q3_t2, on='patient_id', how='inner')
    df_mc = df_mc.rename(columns={'C_3': 'Q3_T2'})
    if trac:
        print_yes(df_mc, labl='df_mc')
    # Compute change [note: ionly used in 1. and 2., not in 3.]
    df_mc['delta_T0_T2'] = df_mc['VEINES_QOL_T2'] - df_mc['VEINES_QOL_T0']
    if trac:
        print_yes(df_mc, labl='df_mc')
    pass
    '''
    patient_id         VEINES_QOL_T0  VEINES_QOL_T1  VEINES_QOL_T2  Q3_T2  delta_T0_T2
    0   PT_2024_02_00078  46.9           45.86          53.35          2.0     6.45
    1   PT_2024_02_08277  47.2           51.99          54.80          5.0     7.60
    2   PT_2024_02_10578  50.5           54.05          58.92          2.0     8.42
    3   PT_2024_02_11301  53.7           55.73          62.43          1.0     8.73
    etc
    '''
    
    # ====
    # 1. Anchor‑based (mean change)
    # 2. Anchor‑based (ROC/Youden)
    # -> 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)
    #    3. Note : Distribution‑based MCID is only meaningfully computed for the baseline 'T0'
    #       https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
    # ****
    # Compute distribution‑based MCID estimates from your own baseline data 
    # Note : DO NOT drop patients in T0 when computing T0_T1 and a T1 patient is absent ; idem for T2 
    # https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy
    # ====  
    baseline = df_mc['VEINES_QOL_T0'].dropna().values
    sd_baseline = np.std(baseline, ddof=1)

    # Bootstrap CI function
    def bootstrap_ci(values, func, n_boot=1000, alpha=0.05):
        boot_vals = []
        for _ in range(n_boot):
            sample = resample(values)
            boot_vals.append(func(sample))
        lo = np.percentile(boot_vals, 100 * alpha/2)
        hi = np.percentile(boot_vals, 100 * (1 - alpha/2))
        return lo, hi
    
    # Distribution-based MCID estimate candidates from your own baseline data    
    mcid_sd_05 = 0.5 * sd_baseline # Cohen's convention
    mcid_sd_03 = 0.3 * sd_baseline
    
    # Reliability estimate (Cronbach alpha or ICC)
    # If you have VEINES-QOL reliability from literature, plug it here.
    # For now, assume alpha = 0.90 (typical for VEINES-QOL)
    # SEM using reliability (classical test theory)
    # How much of the observed score variation is due to measurement noise?
    alpha = reliability = 0.90 ; mcid_SEM = sd_baseline * np.sqrt(1 - alpha)
    # MDC95 (Minimal Detectable Change at 95% confidence)
    mcid_MDC95 = mcid_SEM * 1.96 * np.sqrt(2)
    
    # Bootstrap CIs
    ci_sd_05 = bootstrap_ci(baseline, lambda x: 0.5 * np.std(x, ddof=1))
    ci_sd_03 = bootstrap_ci(baseline, lambda x: 0.3 * np.std(x, ddof=1))
    ci_sem  = bootstrap_ci(baseline, lambda x: np.std(x, ddof=1) * np.sqrt(1 - alpha))
    
    # These values describe measurement error and detectable change at baseline.
    df_mc_3_dist = pd.DataFrame([
    {
        'ID': "3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)",
        'Method': 'Distribution-based (0.5 SD baseline)',
        'MCID estimate': f"{mcid_sd_05:.2f}",
        '95% CI': f"[{ci_sd_05[0]:.2f}, {ci_sd_05[1]:.2f}]"
    },
    {
        'ID': "3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)",
        'Method': 'Distribution-based (0.3 SD baseline)',
        'MCID estimate': f"{mcid_sd_03:.2f}",
        '95% CI': f"[{ci_sd_03[0]:.2f}, {ci_sd_03[1]:.2f}]"
    },
    {
        'ID': "3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)",
        'Method': 'Distribution-based (SEM)',
        'MCID estimate': f"{mcid_SEM:.2f}",
        '95% CI': f"[{ci_sem[0]:.2f}, {ci_sem[1]:.2f}]"
    }
    ,
    {
        'ID': "3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)",
        'Method': 'Distribution-based (MDC95 : Minimal Detectable Change at 95% confidence)',
        'MCID estimate': f"{mcid_MDC95:.2f}",
        # '95% CI': f"[{ci_sem[0]:.2f}, {ci_sem[1]:.2f}]"
    }
    ])
    if trac:
        print_yes(df_mc_3_dist, labl="3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)")
    pass
    '''
       ID                                                Method                                                                    MCID estimate 95% CI
    0  3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)                                      Distribution-based (0.5 SD baseline)  1.76          [1.27, 2.18]
    1  3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)                                      Distribution-based (0.3 SD baseline)  1.05          [0.74, 1.32]
    2  3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)                                                  Distribution-based (SEM)  1.11          [0.79, 1.36]
    3  3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)  Distribution-based (MDC95 : Minimal Detectable Change at 95% confidence)  3.08                   NaN
        '''
 
    # ====
    # -> 1. Anchor‑based (mean change)
    # 2. Anchor‑based (ROC/Youden)
    # 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI) (T0 only)
    # ****
    # Descriptive MCID: mean change per anchor group
    # The primary anchor‑based MCID estimate was defined as the mean change [Mean ΔVEINES_QOL (T2–T0)] 
    # among patients reporting being “somewhat better”, representing the smallest category of perceived improvement
    # '6.5' is the primary anchor‑based estimate of the MCID.
    # ****
    # Purpose : to provide a a patient‑centered calibration of the QOL scale.
    # In this cohort cohort, patients who feel “somewhat better” after one year typically improve by about 6.5 VEINES‑QOL points.
    # Patients in this cohort need a moderate improvement to feel “somewhat better” : 
    # Not tiny (1–2 points), not huge (10+ points), but around 6–7 points.
    #
    # 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI) (T0 only)
    # This threshold is higher than measurement error (1–2 points) : So it reflects real change, not noise.
    # This comes directly from the '3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI) (T0 only)' calculations. Specifically:
    # ✔ SEM (Standard Error of Measurement) = 1.11 ; 0.3 SD = 1.05 ; 0.5 SD = 1.76
    # These are the variables that quantify 'measurement noise' and 'smallest detectable change':
    # Variable names that correspond to “1–2 points”:

    # | Concept    | Variable name                   | Value    | Meaning                    |
    # |------------|---------------------------------|----------|----------------------------|
    # | **SEM**    | `SEM_T0` or `df_mcid_T0["SEM"]` | **1.11** | Pure measurement error     |
    # | **0.3 SD** | `MCID_0.3SD_T0`                 | **1.05** | Small effect threshold     |
    # | **0.5 SD** | `MCID_0.5SD_T0`                 | **1.76** | Moderate effect threshold  |

    # This 6.5 threshold is higher than measurement error (1–2 points), referring to **SEM (≈1.1)** and **0.3–0.5 SD (≈1.0–1.8)**.
    # These are the **distribution‑based MCID candidates** that define the *lower bound* of meaningful change.
    # ====
    '''
    “Compared to 1 year ago, has your leg problem…”  
    1 = Greatly improved
    2 = Slightly improved
    3 = About the same
    4 = Slightly worsened
    5 = Greatly worsened
    6 = I had no problem
    '''

    # Step 2 : Binary anchor for ROC: 1 = at least somewhat better, 0 = no improvement / worse 
    # ----
    df_mc = df_mc[df_mc['Q3_T2'] != 6].copy()
    # Define anchor groups
    def anchor_group(q):
        if q == 1:
            return 'Substantially better'
        elif q == 2:
            return 'Somewhat better'
        elif q == 3:
            return 'About the same'
        elif q == 4:
            return 'Somewhat worse'
        elif q == 5:
            return 'Much worse'
        elif q == 6:
            return np.nan # Exclude Q3 = 6 (no leg problem last year)
        elif np.isnan(q):
            return np.nan # TODO
        else:
            print (f"q:{q}")
            raise Exception()
    df_mc['anchor_group'] = df_mc['Q3_T2'].apply(anchor_group)
    # Binary anchor for ROC: 
    # [1, 2] =  (1: at least somewhat better, 0:no improvement) -> 1
    # [3,4,5] = (no improvement / worse)                        -> 0
    df_mc['anchor_binary'] = np.where(df_mc['Q3_T2'].isin([1, 2]), 1, 0)
    if trac:
        print_yes(df_mc, labl='df_mc')
    pass
    '''
        patient_id        VEINES_QOL_T0  VEINES_QOL_T1  VEINES_QOL_T2  Q3_T2   delta_T0_T2         anchor_group  anchor_binary
    0   PT_2024_02_00078  46.9           45.86          53.35          2.0     6.45             Somewhat better  1
    1   PT_2024_02_08277  47.2           51.99          54.80          5.0     7.60                  Much worse  0
    2   PT_2024_02_10578  50.5           54.05          58.92          2.0     8.42             Somewhat better  1
    etc
    '''
    
    # Step 3 : Descriptive MCID: mean change per anchor group
    # ----
    def mean_ci(x, alpha=0.05):
        x = np.array(x)
        n = len(x)
        m = np.mean(x)
        s = sem(x) # standard error of the mean
        if n > 1:
            h = s * t.ppf(1 - alpha/2, n - 1)
            return m, m - h, m + h
        else:
            return m, np.nan, np.nan

    desc_rows = []
    for g, sub in df_mc.groupby('anchor_group'):
        m, lo, hi = mean_ci(sub['delta_T0_T2'])
        desc_rows.append({
            'ID': '1. Anchor‑based (mean change)',
            'Anchor group': g,
            'N': len(sub),
            'Mean ΔVEINES_QOL (T2–T0)': f"{m:.2f}",
            '95% CI': f"[{lo:.2f}, {hi:.2f}]",
            'Script': exec_stat_mcid_copi_1.__name__
        })

    df_mc_1_anch_mean_change = pd.DataFrame(desc_rows)
    # Define the custom order mapping
    order_map = {
        'Substantially better': 1,
        'Somewhat better': 2,
        'About the same': 3,
        'Somewhat worse': 4,
        'Much worse': 5
    }
    # Create a temporary column with the numeric order to sort by
    df_mc_1_anch_mean_change['sort_key'] = df_mc_1_anch_mean_change['Anchor group'].map(order_map)
    # Sort the DataFrame and drop the temporary column
    df_mc_1_anch_mean_change = df_mc_1_anch_mean_change.sort_values(by='sort_key').drop(columns=['sort_key'])
    if trac:
        print_yes(df_mc_1_anch_mean_change, labl="1. Anchor-based (mean change in 'Somewhat better')")
    pass
    '''
    ID                                        Anchor group   N   Mean ΔVEINES_QOL (T2–T0)       95% CI  Script
    4  1. Anchor‑based (mean change)  Substantially better   7   7.72                    [3.30, 12.14]  exec_stat_mcid_copi_1
    2  1. Anchor‑based (mean change)       Somewhat better  13   6.50                     [3.67, 9.32]  exec_stat_mcid_copi_1
    0  1. Anchor‑based (mean change)        About the same   7   9.15                    [5.37, 12.94]  exec_stat_mcid_copi_1
    3  1. Anchor‑based (mean change)        Somewhat worse   1  -2.14                       [nan, nan]  exec_stat_mcid_copi_1
    1  1. Anchor‑based (mean change)            Much worse   1   7.60                       [nan, nan]  exec_stat_mcid_copi_1
    # Philosophy : https://copilot.microsoft.com/chats/sPDu3ZRV7BMm15S8tz2Qo
    Hence, the anchor‑based MCID is: 6.5 points (mean change in “somewhat better”)
    Because both the change score and the global rating of change are patient‑reported, 
    anchor‑based MCID directly reflects the amount of improvement that patients perceive as meaningful.
    The 6.5 value provides calibration of the QOL score measurement
    '''

    # ====
    # 1. Anchor‑based (mean change) 
    # -> 2. Anchor‑based (ROC/Youden)
    # 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)
    # ****
    # ROC-based MCID (Youden index)
    # Your ROC-based MCID candidate is: best_threshold (ΔVEINES_QOL T2–T0).
    # ====
    # 
    if trac:
        print_yes(df_mc, labl="df_mc")
    y = df_mc['anchor_binary'].values
    x = df_mc['delta_T0_T2'].values

    fpr, tpr, thresholds = roc_curve(y, x)
    auc_val = roc_auc_score(y, x)

    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]
    #
    df_metrics = pd.DataFrame({
        'ID': "2. Anchor‑based (ROC/Youden)",
        'Metric': [
            'ROC AUC', 
            'Best threshold (Youden)', 
            'Sensitivity at threshold', 
            'Specificity at threshold'
        ],
        'Value': [
            round(auc_val, 3), 
            round(best_threshold, 2), 
            round(best_sens, 2), 
            round(best_spec, 2)
        ],
        'Script': exec_stat_mcid_copi_1.__name__
    })
    if trac:
        print_yes(df_metrics, labl= "2.a. Anchor‑based (ROC/Youden)")
    pass
    '''
       ID                            Metric                     Value Script
    0  2. Anchor‑based (ROC/Youden)                   ROC AUC   0.43  exec_stat_mcid_copi_1
    1  2. Anchor‑based (ROC/Youden)   Best threshold (Youden)  11.79  exec_stat_mcid_copi_1
    2  2. Anchor‑based (ROC/Youden)  Sensitivity at threshold   0.15  exec_stat_mcid_copi_1
    3  2. Anchor‑based (ROC/Youden)  Specificity at threshold   0.90  exec_stat_mcid_copi_1
    '''

    # Mean-change MCID from "Somewhat better"
    # ----
    sub_somewhat = df_mc[df_mc['anchor_group'] == 'Somewhat better']['delta_T0_T2']
    mcid_mean, mcid_lo, mcid_hi = mean_ci(sub_somewhat)
    df_mc_2_anch_roc = pd.DataFrame([
        {
            'ID': "32. Anchor‑based (ROC/Youden)",
            'Method': "Mean change in 'Somewhat better'",
            'MCID estimate': f"{mcid_mean:.2f}",
            '95% CI': f"[{mcid_lo:.2f}, {mcid_hi:.2f}]",
            'Script': exec_stat_mcid_copi_1.__name__
        },
        {
            'ID': "32. Anchor‑based (ROC/Youden)",
            'Method': 'Best threshold',
            'MCID estimate': f"{best_threshold:.2f}",
            '95% CI': 'N/A (threshold)',
            'Script': exec_stat_mcid_copi_1.__name__
        }
    ])
    if trac:
        print_yes(df_mc_2_anch_roc, labl='2.b. Anchor‑based (ROC/Youden)')
    pass
    '''
       ID                             Method                            MCID estimate 95% CI           Script
    0  32. Anchor‑based (ROC/Youden)  Mean change in 'Somewhat better'   6.50            [3.67, 9.32]  exec_stat_mcid_copi_1
    1  32. Anchor‑based (ROC/Youden)                    Best threshold  11.79         N/A (threshold)  exec_stat_mcid_copi_1
    '''
  
    # Plot
    # ----
    # Create df_with_data for the plot
    # This includes the curve points and the specific 'best' point
    df_plot_roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })

    # Store metadata (AUC and Youden point) as attributes or in a separate dict
    plot_roc_meta = {
        'auc': auc_val,
        'best_fpr': 1 - best_spec,
        'best_tpr': best_sens,
        'threshold': best_threshold
    }
  
    # ====
    # 1. Anchor‑based (mean change) 
    # 2. Anchor‑based (ROC/Youden)
    # 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)
    # -> 4. Variability indices (T0, T1, T2) : an 'extention' af (3.)
    # These values do not define MCID. They are descriptive information in providing the 'evolution' of MCID["T0"] towards MCID["T1"] and MCID["T2"]
    # For each timepoint (T0, T1, T2), they describe:
    # 1. SD    : Spread (SD) : how spread changes over time
    # 2. SEM   : Standard Error of Measurement : how measurement noise (SEM) scales with SD
    # 3. MDC95 : Minimal Detectable Change (95%) beyond measurement error : how detectable change (MDC95) evolves
    # how spread (SD) changes over time
    # how measurement noise (SEM) scales with SD
    # how detectable change (MDC95) evolves
    # how variability increases as patients diverge in outcomes
    # old : https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy
    # new : https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
    # ====
    if trac:
        print_yes(df_wide, labl="df_wide")
    '''
        patient_id        VEINES_QOL_T0  VEINES_QOL_T1  VEINES_QOL_T2
    0   PT_2024_02_00078  46.9           45.86          53.35
    1   PT_2024_02_08277  47.2           51.99          54.80
    2   PT_2024_02_10578  50.5           54.05          58.92
    '''
    
    # Compute mean, SD, n for each timepoint
    df_var = (
        df_wide[["VEINES_QOL_T0", "VEINES_QOL_T1", "VEINES_QOL_T2"]]
        .agg(["mean", "std", "count"])
        .T
        .rename(columns={"count": "n"})
    )

    # Add SEM and MDC95 for descriptive purposes only
    df_var["SEM"] = df_var["std"] * np.sqrt(1 - reliability)
    df_var["MDC95"] = df_var["SEM"] * 1.96 * np.sqrt(2)

    # Add SD-based descriptive thresholds
    df_var["0.3SD"] = 0.30 * df_var["std"]
    df_var["0.5SD"] = 0.50 * df_var["std"]
    df_var["1.0SD"] = 1.00 * df_var["std"]

    if trac:
        print_yes(df_var, labl="df_variablity indices")
        '''
                    mean   std   n     SEM   MDC95  0.3SD  0.5SD  1.0SD
        VEINES_QOL_T0  50.02  3.51  30.0  1.11  3.08   1.05   1.76   3.51
        VEINES_QOL_T1  54.00  5.17  24.0  1.64  4.54   1.55   2.59   5.17
        VEINES_QOL_T2  57.29  6.17  30.0  1.95  5.41   1.85   3.08   6.17
        '''
        
    pass 
        
        
    '''
        timepoint        patient_id    T0         T1         T2   delta_T1   delta_T2  mcid_T1  mcid_T2
    0          PT_2024_02_00078  46.9  45.860276  53.349384  -1.039724   6.449384    False     True
    1          PT_2024_02_08277  47.2  51.987495  54.795861   4.787495   7.595861    False     True
    2          PT_2024_02_10578  50.5  54.055014  58.917011   3.555014   8.417011    False     True
    '''
    
    # Plot
    # ----
    df_anchor = pd.DataFrame({
    "Meaning": ["T0 Anchor-based data"],
    "Metric": ["Anchor-based"],
    "Value": [mcid_mean],
    "CI_low": [mcid_lo],
    "CI_high": [mcid_hi],
    "Label": ["Anchor-based\n(Somewhat better)"],
    "Script": exec_stat_mcid_copi_1.__name__
    })

    df_dist = pd.DataFrame({
        "Meaning": [
            "T0 Distribution Small effect threshold",
            "T0 Distribution Moderate effect threshold",
            "T0 Distribution Pure measurement error [SEM]"
        ],
        "Metric": ["0.3 SD", "0.5 SD", "SEM"],
        "Value": [mcid_sd_03, mcid_sd_05, mcid_SEM],
        "Label": ["0.3 SD", "0.5 SD", "SEM"],
        "Script": exec_stat_mcid_copi_1.__name__
        
    })

    df_plot_anch = pd.concat([df_anchor, df_dist], ignore_index=True)

    if trac:
        print_yes(df_plot_anch, labl="df_plot_anch")

    # Synt [stat]
    # ----
    if trac:
        print_yes(df_wide, labl="df_wide")
        print_yes(df_mc_1_anch_mean_change, labl="df_mc_1_anch_mean_change")
        print_yes(df_mc_2_anch_roc, labl="df_mc_2_anch_roc")
        print_yes(df_mc_3_dist,  labl="df_mc_3_dist")
    
    # Mark
    # ----

    # ============================================================
    # Build MCID synthesis table from:
    # df_mc_1_anch_mean_change
    # df_mc_2_anch_roc
    # df_mc_3_dist
    # ============================================================

    # ------------------------------------------------------------
    # 1. Anchor-based (mean change) — retain only "Somewhat better"
    # ------------------------------------------------------------
    df_anchor_mean = (
        df_mc_1_anch_mean_change
        .loc[df_mc_1_anch_mean_change["Anchor group"] == "Somewhat better",
            ["Mean ΔVEINES_QOL (T2–T0)", "95% CI", "N"]]
        .rename(columns={
            "Mean ΔVEINES_QOL (T2–T0)": "MCID estimate",
            "95% CI": "95% CI"
        })
    )
    df_anchor_mean.insert(0, "Method", "Anchor-based (mean change)")

    # ------------------------------------------------------------
    # 2. Anchor-based (ROC/Youden)
    # ------------------------------------------------------------
    df_anchor_roc = df_mc_2_anch_roc.copy()
    df_anchor_roc = df_anchor_roc[["Method", "MCID estimate", "95% CI"]]

    # ------------------------------------------------------------
    # 3. Distribution-based (T0 only)
    # ------------------------------------------------------------
    df_dist = df_mc_3_dist.copy()
    df_dist = df_dist[["Method", "MCID estimate", "95% CI"]]

    # ------------------------------------------------------------
    # 4. Combine all into a single synthesis table
    # ------------------------------------------------------------
    df_mcid_synthesis = pd.concat(
        [df_anchor_mean, df_anchor_roc, df_dist],
        ignore_index=True
    )
    if trac: 
            print_yes(df_mcid_synthesis)

    # ------------------------------------------------------------
    # 5. Clean formatting
    # ------------------------------------------------------------


    # Function to merge Estimate and N
    def merge_estimate_n(row):
        est = f"{row['MCID estimate']}"
        # Only add (N=...) if N is not NaN
        if pd.notna(row['N']):
            return f"{est} (N={int(row['N'])})"
        return est

    # 1. Update the MCID estimate column
    df_mcid_synthesis['MCID estimate'] = df_mcid_synthesis.apply(merge_estimate_n, axis=1)

    # 2. Now create the final "Value" column by adding the 95% CI
    # We handle "N/A" or "nan" strings in the CI column to keep it clean
    df_mcid_synthesis['Value'] = df_mcid_synthesis.apply(
        lambda x: f"{x['MCID estimate']} {x['95% CI']}" 
        if pd.notna(x['95% CI']) and str(x['95% CI']).lower() != 'nan' 
        else x['MCID estimate'], 
        axis=1
    )

    # Show the simplified report
    df_report = df_mcid_synthesis[['Method', 'Value']]
    print(df_report)
    if trac: 
            print_yes(df_report)
 
    df_report.set_index('Method', inplace=True)
    mark_dict['Anchor-based (mean change)'] = (df_report.loc["Anchor-based (mean change)", "Value"], Path(__file__).stem)
    mark_dict["Mean change in 'Somewhat better'"] = (df_report.loc["Mean change in 'Somewhat better'", "Value"], Path(__file__).stem)
    mark_dict['Best threshold'] = (df_report.loc["Best threshold", "Value"], Path(__file__).stem)
    mark_dict['Distribution-based (0.5 SD baseline)'] = (df_report.loc['Distribution-based (0.5 SD baseline)', "Value"], Path(__file__).stem)
    mark_dict['Distribution-based (0.3 SD baseline)'] = (df_report.loc["Distribution-based (0.3 SD baseline)", "Value"], Path(__file__).stem)
    mark_dict['Distribution-based (SEM)'] = (df_report.loc["Distribution-based (SEM)", "Value"], Path(__file__).stem)
    mark_dict['Distribution-based (MDC95 : Minimal Detectable Change at 95% confidence)'] = (df_report.loc["Distribution-based (MDC95 : Minimal Detectable Change at 95% confidence)"], Path(__file__).stem)
    df_report = df_report.reset_index(drop=False)
    if trac: 
        print_yes(df_report)

    # Exit
    # ----
    stat_tran_adat.resu_wide = df_wide
    stat_tran_adat.resu_1_anch_mean_change = df_mc_1_anch_mean_change
    stat_tran_adat.resu_2_anch_roc = df_mc_2_anch_roc
    stat_tran_adat.resu_3_dist = df_mc_3_dist
    stat_tran_adat.resu_4_variability = df_var
    stat_tran_adat.resu_synt = df_mcid_synthesis
    
    stat_tran_adat.plot_anch     = df_plot_anch
    stat_tran_adat.plot_roc_data = df_plot_roc_data
    stat_tran_adat.plot_roc_meta = plot_roc_meta
    
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