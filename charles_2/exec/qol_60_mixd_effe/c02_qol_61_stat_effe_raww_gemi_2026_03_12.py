    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_61_stat_ import StatTranQOL_61_cohe
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re
from pathlib import Path
import matplotlib.pyplot as plt
from qol_60_mixd_effe.c02_qol_61_stat_adat import StatTranQOL_61_cohe
from itertools import combinations
from pathlib import Path
from pprint import pprint

# ----
# Publ : https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4
# ----

# https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy

# https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy
# https://gemini.google.com/app/3d83302995de6337 : VALIDATION ON 2026-03-12 <- REFERENCE pour concept
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4 : VALIDATION ON 2026-03-13 <- REFERENCE pour code

def exec_stat_effe_raww(stat_tran_adat: StatTranQOL_61_cohe) -> None:
    # from qol_60_mixd_effe.c02_qol_61_stat_ import StatTranQOL_61_cohe
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta06_endp_effe_size
       
    '''
    df_fram
    -------
    patient_id        timepoint VEINES_QOL_t  pati_isok  pati_50pc  C_3
    0  PT_2024_02_00078  T0        46.90         True       True       2.0
    1  PT_2024_02_00078  T1        45.86         True       True       2.0
    2  PT_2024_02_00078  T2        53.35         True       True       2.0
    df_wide
    -------
    timepoint  patient_id        T0    T1     T2   
    0          PT_2024_02_00078  46.9  45.86  53.35
    1          PT_2024_02_08277  47.2  51.99  54.80
    2          PT_2024_02_10578  50.5  54.05  58.92
    3          PT_2024_02_11301  53.7  55.73  62.43
    '''
    df_wide = (
        df_fram
        .pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
        .reset_index()
    )
    if trac:
        print_yes(df_fram, labl='df_fram')
        print_yes(df_wide, labl='df_wide')

    # Util
    # ----
    # Paired_effect_sizes
    # https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4 
    def paired_effect_sizes(a, b, baseline_a):
        
        # Exec
        # ----
        delta = b - a
        mean_change = delta.mean()
        sd_change = delta.std()
        sd_baseline = baseline_a.std() # Use T0 spread for Cohen's d

        # Paired Cohen's d: Effect relative to population baseline spread
        d = mean_change / sd_baseline if sd_baseline != 0 else np.nan
        
        # SRM: Responsiveness relative to change noise
        srm = mean_change / sd_change if sd_change != 0 else np.nan

        # Interpretation (using Cohen's d)
        if abs(d) < 0.20: interp_d = "negligible"
        elif abs(d) < 0.50: interp_d = "small"
        elif abs(d) < 0.80: interp_d = "moderate"
        else: interp_d = "large"

        # Interpretation for SRM (Responsiveness - Kazis et al. standards)
        if abs(srm) < 0.20: interp_srm = "trivial"
        elif abs(srm) < 0.50: interp_srm = "small"
        elif abs(srm) < 0.80: interp_srm = "moderate"
        else: interp_srm = "large"
        
        # Exit
        # ----
        return mean_change, sd_change, d, srm, interp_d, interp_srm
    

    # Bootstrap 95% CI for paired Cohen's d.
    def bootstrap_cohens_d(a, b, baseline_a, n_boot=5000, seed=42):
        """
        Bootstrap 95% CI for paired Cohen's d (baseline-anchored).
        """
        a = pd.Series(a)
        b = pd.Series(b)
        baseline_a = pd.Series(baseline_a)

        mask = a.notna() & b.notna() & baseline_a.notna()
        a = a[mask].to_numpy()
        b = b[mask].to_numpy()
        baseline = baseline_a[mask].to_numpy()

        if len(a) == 0:
            return np.nan, np.nan

        rng = np.random.default_rng(seed)
        n = len(a)
        sd_baseline = baseline.std()

        boot_ds = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            delta = b[idx] - a[idx]
            mean_change = delta.mean()
            d = mean_change / sd_baseline if sd_baseline != 0 else np.nan
            boot_ds.append(d)

        return (
            np.nanpercentile(boot_ds, 2.5),
            np.nanpercentile(boot_ds, 97.5)
        )

    def bootstrap_srm(a, b, n_boot=5000, seed=42):
        """
        Bootstrap 95% CI for SRM (Standardized Response Mean).
        """
        a = pd.Series(a)
        b = pd.Series(b)

        mask = a.notna() & b.notna()
        a = a[mask].to_numpy()
        b = b[mask].to_numpy()

        if len(a) == 0:
            return np.nan, np.nan

        rng = np.random.default_rng(seed)
        n = len(a)

        boot_srms = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            delta = b[idx] - a[idx]
            mean_change = delta.mean()
            sd_change = delta.std()
            srm = mean_change / sd_change if sd_change != 0 else np.nan
            boot_srms.append(srm)

        return (
            np.nanpercentile(boot_srms, 2.5),
            np.nanpercentile(boot_srms, 97.5)
        )

    def paired_effect_sizes_with_missing_and_bootstrap(a, b, baseline_a, 
                                                   n_boot=5000, seed=42):
        """
        Returns:
        - df_results: effect sizes + bootstrap CIs
        - df_missing: missingness diagnostics
        """

        # Convert to pandas Series for easy masking
        a = pd.Series(a)
        b = pd.Series(b)
        baseline_a = pd.Series(baseline_a)

        # Identify complete cases
        mask_complete = a.notna() & b.notna() & baseline_a.notna()
        n_complete = mask_complete.sum()
        n_missing = len(a) - n_complete
        missing_indices = mask_complete[~mask_complete].index.tolist()

        # Filter arrays
        a_c = a[mask_complete].to_numpy()
        b_c = b[mask_complete].to_numpy()
        baseline_c = baseline_a[mask_complete].to_numpy()

        # If no complete cases, return NA structures
        if n_complete == 0:
            df_results = pd.DataFrame([{
                "mean_change": np.nan,
                "sd_change": np.nan,
                "cohen_d": np.nan,
                "cohen_d_ci_low": np.nan,
                "cohen_d_ci_high": np.nan,
                "srm": np.nan,
                "srm_ci_low": np.nan,
                "srm_ci_high": np.nan,
                "interpretation_d": "insufficient data",
                "interpretation_srm": "insufficient data"
            }])

            df_missing = pd.DataFrame([{
                "n_complete": 0,
                "n_missing": n_missing,
                "missing_indices": missing_indices
            }])

            return df_results, df_missing

        # Compute effect sizes using your original function
        mean_change, sd_change, d, srm, interp_d, interp_srm = paired_effect_sizes(
            a_c, b_c, baseline_c
        )

        # Bootsrap mean and sd change
        # Raw-data mean change CI
        se_mean_change = sd_change / np.sqrt(n_complete)
        mean_ci_low = mean_change - 1.96 * se_mean_change
        mean_ci_high = mean_change + 1.96 * se_mean_change

        # Bootstrap CIs
        d_low, d_high = bootstrap_cohens_d(a_c, b_c, baseline_c,
                                        n_boot=n_boot, seed=seed)
        srm_low, srm_high = bootstrap_srm(a_c, b_c,
                                        n_boot=n_boot, seed=seed)

        # Build results DataFrame
        df_results = pd.DataFrame([{
            "mean_change": mean_change,
            "mean_ci_low": mean_ci_low,
            "mean_ci_high": mean_ci_high,
            "sd_change": sd_change,
            "cohen_d": d,
            "cohen_d_ci_low": d_low,
            "cohen_d_ci_high": d_high,
            "srm": srm,
            "srm_ci_low": srm_low,
            "srm_ci_high": srm_high,
            "interpretation_d": interp_d,
            "interpretation_srm": interp_srm
        }])

        # Build missingness DataFrame
        df_missing = pd.DataFrame([{
            "n_complete": n_complete,
            "n_missing": n_missing,
            "missing_indices": missing_indices
        }])

        return df_results, df_missing


    # Exec : Compute effect sizes for each contrast ir between timepoints
    # ----
    def veines_effect_sizes(df_fram, df_wide):
        df_wide = df_fram.pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
        tps = sorted(df_fram["timepoint"].unique())
        
        # Define contrasts
        contrasts = {
            "T1 - T0": ("T0", "T1"),
            "T2 - T1": ("T1", "T2"),
            "T2 - T0": ("T0", "T2"),
        }
        contrast_lookup = {v: k for k, v in contrasts.items()}

        results_list = []
        missing_list = []

        for t0, t1 in combinations(tps, 2):
            a = df_wide[t0]
            b = df_wide[t1]

            df_results, df_missing = paired_effect_sizes_with_missing_and_bootstrap(
                a, b, baseline_a=df_wide[tps[0]]  # always T0 as baseline
            )

            # Annotate with transition
            df_results["start"] = t0
            df_results["end"] = t1

            df_missing["start"] = t0
            df_missing["end"] = t1
            
            # Add contrast name
            df_results["contrast"] = contrast_lookup.get((t0, t1), None)
            df_missing["contrast"] = contrast_lookup.get((t0, t1), None)

            results_list.append(df_results)
            missing_list.append(df_missing)

        # Concatenate all transitions
        df_results_all = pd.concat(results_list, ignore_index=True)
        df_missing_all = pd.concat(missing_list, ignore_index=True)

        return df_results_all, df_missing_all
    
    df_cohe_srmm_resu, df_cohe_srmm_miss = veines_effect_sizes(df_fram, df_wide)
    df_cohe_srmm_resu = df_cohe_srmm_resu[["contrast", "mean_change", "mean_ci_low", "mean_ci_high", "sd_change", "cohen_d", "cohen_d_ci_low", "cohen_d_ci_high", "srm", "srm_ci_low", "srm_ci_high", "interpretation_d", "interpretation_srm"]]
    
    # Edit
    # ----
    def summarize_effect_sizes(df_tech):
        """
        Convert the technical effect-size dataframe into a clean,
        human-readable summary table including CIs.
        """

        df = df_tech.copy()
        print (df)

        # Format Cohen's d with CI
        df["cohen_d_fmt"] = (
            df["cohen_d"].round(2).astype(str)
            + " (" + df["interpretation_d"].str.capitalize() + ") "
            + "[" 
            + df["cohen_d_ci_low"].round(2).astype(str)
            + "–"
            + df["cohen_d_ci_high"].round(2).astype(str)
            + "]"
        )

        # Format SRM with CI
        df["srm_fmt"] = (
            df["srm"].round(2).astype(str)
            + " (" + df["interpretation_srm"].str.capitalize() + ") "
            + "[" 
            + df["srm_ci_low"].round(2).astype(str)
            + "–"
            + df["srm_ci_high"].round(2).astype(str)
            + "]"
        )

        # Mean change
        df["mean_change_fmt"] = (
            df["mean_change"].round(2).astype(str)
            + "[" 
            + df["mean_ci_low"].round(2).astype(str)
            + "–"
            + df["mean_ci_high"].round(2).astype(str)
            + "]"
        )

        # Final summary table
        df_summary = df[[
            "contrast",
            "mean_change_fmt",
            "cohen_d_fmt",
            "srm_fmt"
        ]].rename(columns={
            "contrast": "Contrast",
            "mean_change_fmt": "Mean Change",
            "cohen_d_fmt": "Cohen's d (Interp + CI)",
            "srm_fmt": "SRM (Interp + CI)"
        })

        return df_summary
    
    def summarize_missingness(df_missing):
        """
        Produce a clean, human-readable summary of missingness
        from the technical missingness dataframe.
        """

        df = df_missing.copy()

        # Format missing indices as comma-separated strings
        df["missing_fmt"] = df["missing_indices"].apply(
            lambda lst: ", ".join(map(str, lst)) if isinstance(lst, list) else ""
        )

        # Select and rename columns
        df_summary = df[[
            "contrast",
            "n_complete",
            "n_missing",
            "missing_fmt"
        ]].rename(columns={
            "contrast": "Contrast",
            "n_complete": "N Complete",
            "n_missing": "N Missing",
            "missing_fmt": "Missing Patients"
        })

        return df_summary

    df_cohe_srmm_resu_edit = summarize_effect_sizes(df_cohe_srmm_resu)
    df_cohe_srmm_miss_edit = summarize_missingness(df_cohe_srmm_miss)
    
    '''
    ----
    Fram labl : df_cohe_srmm_resu_edit
    ----
    df:3 type:<class 'pandas.core.frame.DataFrame'>
    Contrast  Mean Change Cohen's d (Interp + CI)   SRM (Interp + CI)
    0  T1 - T0  4.30         1.23 (Large) [0.87–1.62]  1.31 (Large) [0.92–2.09]
    1  T2 - T0  7.27          2.1 (Large) [1.64–2.58]  1.59 (Large) [1.13–2.48]
    2  T2 - T1  3.09         0.89 (Large) [0.05–1.68]  0.44 (Small) [0.02–1.12]
    '''
    #
    if trac:
        print_yes(df_wide, labl="df_wide")
        print_yes(df_cohe_srmm_resu_edit, labl="df_cohe_srmm_resu_edit")
        print_yes(df_cohe_srmm_miss_edit, labl="df_cohe_srmm_miss_edit")
        df_cohe_srmm_resu_edit = df_cohe_srmm_resu_edit.set_index("Contrast")
        if trac:
            print_yes(df_cohe_srmm_resu_edit, labl="df_cohe_edit_3")
        
    # Mark
    # ----
    mean_info = df_cohe_srmm_resu_edit.loc["T2 - T0", "Mean Change"]
    cohe_info = df_cohe_srmm_resu_edit.loc["T2 - T0", "Cohen's d (Interp + CI)"]
    srmm_info = df_cohe_srmm_resu_edit.loc["T2 - T0", "SRM (Interp + CI)"]
    mark_dict["Mean change (T0–T2)"]              = (mean_info, Path(__file__).stem)
    mark_dict["Cohen’s d"]                        = (cohe_info, Path(__file__).stem)
    mark_dict["Standardized response mean (SRM)"] = (srmm_info, Path(__file__).stem)
    if trac:
        pprint(mark_dict)
    
    # Exit
    # ----
    stat_tran_adat.resu_cohe_raww_stat = df_cohe_srmm_resu_edit
    stat_tran_adat.resu_cohe_raww_plot = df_cohe_srmm_resu
    
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