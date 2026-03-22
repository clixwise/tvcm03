    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_01_assu_ import AssuTranQOL_01_synt, AssuTranQOL_01_isok, AssuTranQOL_01_summ, AssuTranQOL_01_ceil
    
import pandas as pd
import sys
import os  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
# ----
# Assumption : histogram normality thru qq-plot
# https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
# https://gemini.google.com/app/0f865d976b405499
# A1 — completeness (pati_isok)
# A2 — score distribution (histogram, Q–Q, skew/kurt)
# C1 — ceiling/floor effects
# ----

def exec_assu_synt(assu_tran_synt: AssuTranQOL_01_synt, assu_tran_isok:AssuTranQOL_01_isok, assu_tran_summ:AssuTranQOL_01_summ, assu_tran_ceil:AssuTranQOL_01_ceil) -> None:
    from qol.c02_qol_01_assu_ import AssuTranQOL_01_synt, AssuTranQOL_01_isok, AssuTranQOL_01_summ, AssuTranQOL_01_ceil
    
    trac = True

    # Data
    # ---- 
    df_fram = assu_tran_synt.assu_tran.fram
    df_resu_isok = assu_tran_isok.resu_tech
    df_resu_summ = assu_tran_summ.resu_tech
    df_resu_ceil = assu_tran_ceil.resu_tech
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        print_yes(df_resu_isok, labl="df_resu_isok")
        print_yes(df_resu_summ, labl="df_resu_summ")
        print_yes(df_resu_ceil, labl="df_resu_ceil")
    
    # ----
    # Metric,Source Column,Flag Condition
    # Completeness,pct_valid,< 85%
    # Floor effect,pct_min,> 15%
    # Ceiling effect,pct_max,> 15%
    # Skewness,`,skew
    # Kurtosis,`,kurt
    
    # Exec : Completeness
    # ----
    # Define threshold
    min_valid_pct = 85
    # 1. Group by timepoint and calculate the percentage of True values in pati_isok
    df_vali = df_fram.groupby("timepoint")["pati_50pc"].mean() * 100
    # 2. Convert to DataFrame for the report
    df_resu_comp_ = df_vali.to_frame(name="COMPLETE_vali_prct")
    # 3. Create the flag: True if validity is LESS THAN 85%
    df_resu_comp_["COMPLETE_vali_flag"] = df_resu_comp_["COMPLETE_vali_prct"] < min_valid_pct
    # Display result
    if trac:
        print_yes(df_resu_comp_, labl="df_comp")
        
    # Exec : Ceil, Floor
    # ----     
    # Define your threshold
    ceiling_floor_pct = 15
    #
    df_resu_ceil_ = df_resu_ceil.copy()
    df_resu_ceil_.rename(columns={'pct_max': 'CEIL_FLOO_ceil_prct'}, inplace=True)
    df_resu_ceil_.rename(columns={'pct_min': 'CEIL_FLOO_floo_prct'}, inplace=True)
    #
    df_resu_ceil_["CEIL_FLOO_ceil_flag"] = df_resu_ceil_["CEIL_FLOO_ceil_prct"] >= ceiling_floor_pct
    df_resu_ceil_["CEIL_FLOO_floo_flag"] = df_resu_ceil_["CEIL_FLOO_floo_prct"] >= ceiling_floor_pct
    # Display the result
    if trac:
        print_yes(df_resu_ceil_, labl="df_resu_ceil")
    
    # Exec : Normality
    # ---- 
    # Define limits
    skew_limit = 1
    kurt_limit = 3
    #
    df_resu_summ_ = df_resu_summ.copy()
    df_resu_summ_.rename(columns={'skew': 'NORM_skew'}, inplace=True)
    df_resu_summ_.rename(columns={'kurt': 'NORM_kurt'}, inplace=True)
    #
    df_resu_summ_["NORM_skew_flag"] = df_resu_summ_["NORM_skew"].abs() > skew_limit
    df_resu_summ_["NORM_kurt_flag"] = df_resu_summ_["NORM_kurt"].abs() > kurt_limit
    # Display the result
    if trac:
        print_yes(df_resu_summ_, labl="df_resu_summ_")

    # Exec : Synthesis
    # ----
    # 1. Concatenate all relevant columns horizontally
    df_synt = pd.concat([
        df_resu_comp_[["COMPLETE_vali_prct", "COMPLETE_vali_flag"]],
        df_resu_ceil_[["CEIL_FLOO_floo_prct", "CEIL_FLOO_floo_flag", "CEIL_FLOO_ceil_prct", "CEIL_FLOO_ceil_flag"]],
        df_resu_summ_[["NORM_skew", "NORM_skew_flag", "NORM_kurt", "NORM_kurt_flag"]]
    ], axis=1)
    # 2. Optional: Create a "Global Pass" column
    # It's True only if ALL flags are False
    flag_cols = [c for c in df_synt.columns if "flag" in c]
    df_synt["all_passed"] = ~df_synt[flag_cols].any(axis=1)
    #
    df_warn = df_synt[flag_cols]
    print_yes(df_warn, labl="Quality Warnings")
    # 3. Trac
    if trac:
        print_yes(df_synt, labl="df_synt")
    
    # Trac
    # ----
    if trac:
        print_yes(df_synt, labl="df_synt")
        print_yes(df_warn, labl="df_warn")
    
    # Exit
    # ----
    assu_tran_synt.resu_synt = df_synt
    assu_tran_synt.resu_warn = df_warn
    
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