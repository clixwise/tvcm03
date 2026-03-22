    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_stat_ import StatTranQOL_01_desc
    
import pandas as pd
import sys
import os  
import numpy as np
from pprint import pprint
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.stat_help import get_comparison_stats, summarize_continuous_edit
from qol.c02_qol_01_stat_adat import StatTranQOL_01_desc
        
# ----
# Timepoint outcomes
# ----
'''
Timepoint                   T0             T1
Outcome
VCSS (max limb)  5.5 [4.8–6.2]  5.5 [5.2–5.8]
VEINES-QOL         56.0 ± 11.0     58.2 ± 4.5
VEINES-QOL         60.8 ± 12.2     62.5 ± 5.4
'''
def exec_stat_desc(stat_tran_desc: StatTranQOL_01_desc):
    _exec_stat_desc_gemi(stat_tran_desc)
    _exec_stat_desc_open_mean_sd(stat_tran_desc)
    _exec_stat_desc_open_mean_ci(stat_tran_desc)
    
# --------------
# Version Openai [2026-01-19]
# --------------
def _exec_stat_desc_open_mean_sd(stat_tran_desc: StatTranQOL_01_desc) -> None:
    # from qol.c02_qol_01_stat_ import StatTranQOL_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
       
    ## 
    ## **Table 2 – VEINES descriptive (Mean ± SD)**
    ## 
    def mean_ci(x):
        mean = x.mean()
        se = x.std(ddof=1) / np.sqrt(len(x))
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        return mean, ci_low, ci_high
    #
    rows = []
    time_cate_list = df_fram['timepoint'].cat.categories.tolist()
    for metric in ["VEINES_QOL_t"]:
        baseline_mean, _, _ = mean_ci(
            df_fram.loc[df_fram.timepoint == "T0", metric]
        )
        #
        for tipo in time_cate_list:
            # Mean
            mean, lo, hi = mean_ci(
                df_fram.loc[df_fram.timepoint == tipo, metric]
            )
            # Mean change
            rows.append({
                "ID": StatTranQOL_01_desc.__name__,
                "Metric": metric,
                "Timepoint": tipo,
                "Estimated Mean (95% CI)": f"{mean:.1f} [{lo:.1f}–{hi:.1f}]",
                "Mean Change vs T0 (95% CI)":
                    "—" if tipo == "T0"
                    else f"{mean - baseline_mean:+.1f} [{lo - baseline_mean:+.1f}–{hi - baseline_mean:+.1f}]"
            })
    #
    df_publ = pd.DataFrame(rows)

    # Trac
    # ----
    '''
    ID              Metric                Timepoint Estimated Mean (95% CI) Mean Change vs T0 (95% CI)
    0  StatTranQOL_01_desc  VEINES_QOL_t  T0        50.0 [48.8–51.3]                       —    
    1  StatTranQOL_01_desc  VEINES_QOL_t  T1        50.0 [48.8–51.3]        +0.0 [-1.3–+1.3]    
    2  StatTranQOL_01_desc  VEINES_QOL_t  T2        50.0 [48.8–51.3]        +0.0 [-1.3–+1.3]  
    '''
    if trac:
        print_yes(df_publ, labl="df_publ")     
    pass

def _exec_stat_desc_open_mean_ci(stat_tran_desc: StatTranQOL_01_desc) -> None:
    from qol.c02_qol_01_stat_ import StatTranQOL_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
       
    ## 
    ## **Table 2 – VEINES descriptive (Mean ± SD)**
    ## 
    def mean_sd(x):
        return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"
    #
    df_publ = (
        df_fram
        .melt(id_vars=["workbook", "patient_id", "timepoint"],
            value_vars=["VEINES_QOL_t"],
            var_name="Metric",
            value_name="ValueNum")
        .groupby(["Metric", "timepoint"])["ValueNum"]
        .apply(mean_sd)
        .reset_index(name="Value")
    )
    '''
       Metric timepoint       Value
    0  VEINES_QOL_t        T0  50.0 ± 3.5
    1  VEINES_QOL_t        T1  50.0 ± 3.5
    2  VEINES_QOL_t        T2  50.0 ± 3.5
    '''
    df_publ.insert(0, "ID", StatTranQOL_01_desc.__name__)
    df_publ.rename(columns={"timepoint": "Timepoint"}, inplace=True)

    # Trac
    # ----
    '''
       ID                   Metric        Timepoint Value
    0  StatTranQOL_01_desc  VEINES_QOL_t  T0        50.0 ± 3.5
    1  StatTranQOL_01_desc  VEINES_QOL_t  T1        50.0 ± 3.5
    2  StatTranQOL_01_desc  VEINES_QOL_t  T2        50.0 ± 3.5
    '''
    if trac:
        print_yes(df_publ, labl="df_publ")     
    pass

# --------------
# Version Gemini [2025-12-01]
# --------------
def _exec_stat_desc_gemi(stat_tran_desc: StatTranQOL_01_desc) -> None:
    from qol.c02_qol_01_stat_ import StatTranQOL_01_desc
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_desc.stat_tran.fram
    mark_dic2 = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta01_base_char
    mark_dic3 = stat_tran_desc.stat_tran.proc_tran.orch_tran.ta03_endp_prim_raww
    
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
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_none_t", tipo, summarize_continuous_edit(df_tipo["none_t"], "mean_sd")])
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_mean_t", tipo, summarize_continuous_edit(df_tipo["mean_t"], "mean_sd")])
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_iter_t", tipo, summarize_continuous_edit(df_tipo["iter_t"], "mean_sd")])
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_gemi_t", tipo, summarize_continuous_edit(df_tipo["gemi_t"], "mean_sd")])
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_copi_t", tipo, summarize_continuous_edit(df_tipo["copi_t"], "mean_sd")])
        tech_list.append([StatTranQOL_01_desc.__name__, "QOL_VEINES_QOL_t", tipo, summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "mean_sd")])
        '''
        tech_list.append([StatTranSym_01.__name__, "SYM_none_t", tipo, summarize_continuous_edit(df_tipo["S_none_t"], "mean_sd")])
        tech_list.append([StatTranSym_01.__name__, "SYM_mean_t", tipo, summarize_continuous_edit(df_tipo["S_mean_t"], "mean_sd")])
        tech_list.append([StatTranSym_01.__name__, "SYM_VEINES_QOL_t", tipo, summarize_continuous_edit(df_tipo["S_VEINES_QOL_t"], "mean_sd")])
        tech_list.append([StatTranSym_01.__name__, "SYM_gemi_t", tipo, summarize_continuous_edit(df_tipo["S_gemi_t"], "mean_sd")])
        '''
    #
    if trac:
        print(tech_list)
        
    # Exec : publ
    # ---- 
    publ_list = []
    baseline_label = "T0"

    for tipo in ["T0", "T1", "T2"]:
        df_tipo = df_fram[df_fram["timepoint"] == tipo]
        if df_tipo.empty:
            continue
        
        # 1. Descriptive stats (Mean ± SD [CI])
        jrnl_val = summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "jrnl_qual")     
        # 2. Comparative stats (vs T0)
        # We call the function we built in the previous step
        TX_T0 = get_comparison_stats(df_fram, tipo, baseline_label)    
        # 3. Build the row
        # We include Change and P-value as separate columns for a professional table
        publ_list.append([StatTranQOL_01_desc.__name__, "VEINES-QOL", tipo,  jrnl_val, TX_T0, ""])
        
        # Exec : mark
        # ----
        if tipo == 'T0':
            qol_info = summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "mean_sd") # "median_iqr")  
            mark_dic2['.VEINES-QOL'] = (qol_info, Path(__file__).stem)
        #
        match tipo:
            case 'T0':
                jrnl_val = summarize_continuous_edit(df_tipo["VEINES_QOL_t"], "jrnl_qual")
                mark_dic3['Mean ± SD at T0'] = (jrnl_val, Path(__file__).stem)
            case 'T1':
                mark_dic3['Mean ± SD at T1'] = (jrnl_val, Path(__file__).stem)
            case 'T2':
                mark_dic3['Mean ± SD at T2'] = (jrnl_val, Path(__file__).stem)
                TX_T0 = get_comparison_stats(df_fram, tipo, baseline_label) 
                mark_dic3['T0–T2 : Absolute mean ± SD change (%)'] = (TX_T0, Path(__file__).stem)
            case _:
                raise Exception() 

    # Oupu
    # ----
    cols = ['ID', 'Metric', 'Timepoint', 'Value']
    df_tech = pd.DataFrame(tech_list, columns=cols)
    df_tech.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_tech, labl="df_tech")
    #
    cols = ['ID', 'Metric', 'Timepoint', 'Value', 'Change (vs T0)', 'p-value']
    df_publ = pd.DataFrame(publ_list, columns=cols)
    df_publ.set_index(['ID', 'Metric'], inplace=True)
    if trac:
        print_yes(df_publ, labl="df_publ")
    
    # Exit
    # ----
    stat_tran_desc.resu_tech = df_tech
    stat_tran_desc.resu_publ = df_publ
    
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