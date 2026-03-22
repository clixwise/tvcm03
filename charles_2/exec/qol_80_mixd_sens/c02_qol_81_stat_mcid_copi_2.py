 
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
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem, t
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.utils import resample
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ----
# Timepoint outcomes
# ----

# https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R -> OLD
# MCID : BEST : distribution-base, anchor-based, roc-based
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4 -> NEW [2026-03-16]

# ****
# MCID ATTAINMENT ANALYSIS based on preceding : 
# 1. Anchor‑based (mean change)
# 2. Anchor‑based (ROC/Youden)
# 3. Distribution‑based (0.5 SD, 0.3 SD, SEM ± CI)
# ****

def exec_stat_mcid_copi_2(stat_tran_adat: StatTranQOL_81_mcid_copi) -> None:
    # from qol_80_mixd_sens.c02_qol_81_stat_ import StatTranQOL_81_mcid_copi

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, "df_fram")

    # Data : df_fram
    # ----
    df_modl = df_fram.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
   
    # Modl : df_modl [Fit the linear mixed-effects model]
    # ---- 
    modl = smf.mixedlm("VEINES_QOL_t ~ C(timepoint)", df_modl, groups=df_modl["patient_id"])
    result = modl.fit(reml=True, method="powell")
    if trac:
        print(result.summary())

    if trac:
        print_yes(df_fram, labl='df_fram')      

    # ------------------------------------------------------------
    # 1. Fit the mixed model (statistical significance layer)
    # ------------------------------------------------------------
    modl = smf.mixedlm(
        "VEINES_QOL_t ~ C(timepoint)",
        df_modl,
        groups=df_modl["patient_id"]
    )
    result = modl.fit(reml=True, method="powell")

    # Extract fixed-effect coefficients for timepoints
    coef_series = result.params[result.params.index.str.startswith("C(timepoint)")]
    lmm_coef = (
        pd.DataFrame({
            "param": coef_series.index,
            "LMM_coef": coef_series.values,
        })
    )
    lmm_coef["timepoint"] = lmm_coef["param"].str.extract(r"\[T\.(.+)\]")

    # ------------------------------------------------------------
    # 2. Prepare wide-format data for patient-level change scores
    # ------------------------------------------------------------
    df_wide = (
        df_modl
        .pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
        .reset_index()
    )

    # ------------------------------------------------------------
    # 3. Compute patient-level change and MCID attainment
    # ------------------------------------------------------------
    MCID = 5.0   # chosen clinical threshold (pragmatic, not estimated)

    timepoints = [tp for tp in df_wide.columns if tp not in ["patient_id", "T0"]]
    for tp in timepoints:
        df_wide[f"delta_{tp}"] = df_wide[tp] - df_wide["T0"]
        df_wide[f"mcid_{tp}"]  = df_wide[f"delta_{tp}"] >= MCID

    # ------------------------------------------------------------
    # 4. Group-level summary: mean change + % achieving MCID
    # ------------------------------------------------------------
    mcid_summary = {}
    for tp in timepoints:
        mcid_summary[tp] = {
            "mean_delta": df_wide[f"delta_{tp}"].mean(),
            "pct_achieved_MCID": df_wide[f"mcid_{tp}"].mean() * 100,
        }

    mcid_df = (
        pd.DataFrame(mcid_summary)
        .T
        .reset_index()
        .rename(columns={"index": "timepoint"})
    )

    # ------------------------------------------------------------
    # 5. Synthesis table: statistical + clinical significance
    # ------------------------------------------------------------
    df_mcid = lmm_coef.merge(mcid_df, on="timepoint", how="left")
    df_mcid["MCID"] = MCID
    df_mcid["exceeds_MCID"] = df_mcid["mean_delta"] >= MCID

    df_mcid = df_mcid[[
        "timepoint",
        "LMM_coef",            # model-based estimate
        "mean_delta",          # raw observed change
        "MCID",                # chosen threshold
        "exceeds_MCID",        # does mean change exceed MCID?
        "pct_achieved_MCID",   # % of patients exceeding MCID
    ]].round(2)

    print_yes(df_mcid, labl="MCID attainment analysis")

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