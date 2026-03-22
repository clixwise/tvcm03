    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_71_assu_ import AssuTranQOL_71_dist
    
import pandas as pd
import numpy as np
import sys
import os  
import scipy.stats as stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
# ----
# Main: https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R 
# ----

def exec_assu_dist(assu_tran_dist: AssuTranQOL_71_dist) -> None:
    from qol_70_mixd_mcid.c02_qol_71_assu_ import AssuTranQOL_71_dist
    
    trac = True

    # Data
    # ---- 
    df_fram = assu_tran_dist.assu_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")

    # Exec
    # ----  
    '''
               patient_id        T0    T1         T2 
    0          PT_2024_02_00078  46.9  45.860276  53.349384
    1          PT_2024_02_08277  47.2  51.987495  54.795861
    2          PT_2024_02_10578  50.5  54.055014  58.917011
    '''
    df_wide = (
        df_fram
        .pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
        .reset_index()
    )  
    #
    time_list = ["T0", "T1", "T2"]
    rows = []

    for tipo in time_list:
        VEINES_QOL_t = df_wide[tipo]
        n = VEINES_QOL_t.shape[0]

        # Descriptives
        desc      = VEINES_QOL_t.describe()
        mean      = desc["mean"]
        std       = desc["std"]
        cv        = std / mean * 100
        skewness  = VEINES_QOL_t.skew()
        kurtosis  = VEINES_QOL_t.kurt()
        q1, q3    = VEINES_QOL_t.quantile([0.25, 0.75])
        iqr       = q3 - q1
        range_min = VEINES_QOL_t.min()
        range_max = VEINES_QOL_t.max()

        # 95% CI for the mean
        se = std / np.sqrt(n)
        ci_low  = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        # Shapiro–Wilk
        stat, p_shapiro = stats.shapiro(VEINES_QOL_t)
        normality_flag = "Normal" if p_shapiro > 0.05 else "Non-normal"
        normality_interpretation =  "'Normal' if p_shapiro > 0.05 else 'Non-normal' ie MCID SEM method less reliable"

        row = {
            "timepoint": tipo,
            "mean": mean,
            "std": std,
            "min": desc["min"],
            "25%": desc["25%"],
            "50%": desc["50%"],
            "75%": desc["75%"],
            "max": desc["max"],
            "cv": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "iqr": iqr,
            "range_min": range_min,
            "range_max": range_max,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "shapiro_W": stat,
            "shapiro_p": p_shapiro,
            "normality": normality_flag,
            "normality_interpretation": normality_interpretation,
        }

        rows.append(row)

    # Oupu
    # ----   
    df_dist = pd.DataFrame(rows).set_index("timepoint")    
    #
    if trac:
        print_yes(df_wide, labl="df_wide")
        print_yes(df_dist, labl="df_dist")
    
    # Exit
    # ----
    assu_tran_dist.resu_wide = df_wide
    assu_tran_dist.resu_dist = df_dist
    
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