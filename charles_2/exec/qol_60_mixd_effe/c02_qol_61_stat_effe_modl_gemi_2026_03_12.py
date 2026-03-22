    
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

# ----
# Publ : https://chatgpt.com/c/69a41826-783c-8394-a3d4-737c80a8d4b4
# ----

# https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy

# https://copilot.microsoft.com/shares/zJuaaaTriQmCZrAu5vs3R
# https://copilot.microsoft.com/shares/UbUqoVA3REt31nrcUVsYy
# https://gemini.google.com/app/3d83302995de6337 : VALIDATION ON 2026-03-12 <- REFERENCE pour concept
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4 : VALIDATION ON 2026-03-13 <- REFERENCE pour code

def exec_stat_effe_modl(stat_tran_adat: StatTranQOL_61_cohe) -> None:
    #from qol_60_mixd_effe.c02_qol_61_stat_ import StatTranQOL_61_cohe
    
    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta06_endp_effe_size

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
        
    # Prec : should be bell-shaped
    # ----
    plot_exec = False
    if plot_exec:
        plt.hist(result.resid)
        plt.title("Check if residuals are normally distributed")
        plt.show()
    
    # Deta
    # ----
    params = result.params            # coefficients
    bse = result.bse                  # standard errors
    zvals = result.tvalues            # z statistics
    pvals = result.pvalues            # p-values
    conf_int = result.conf_int()      # confidence intervals (DataFrame)
    # 
    df_deta = pd.DataFrame({
        "Coef": params,
        "Std.Err": bse,
        "z": zvals,
        "P>|z|": pvals,
        "CI_lower": conf_int[0],
        "CI_upper": conf_int[1]
    })
    if trac:
        print_yes(df_deta, labl="df_deta")
        
    # ====
    # Exec 1 : not used
    # ====
    def cohe_main_1(result):
    
        # Util 1 : model-based Cohen [didactic: not used]
        # ----
        def cohe_util_1(result, time):
            
            # Exec
            # ----
            # Model coefficient
            beta_T2 = result.params[f'C(timepoint)[T.{time}]']
            # Model residual variance
            residual_var = result.scale
            residual_sd = np.sqrt(residual_var)
            # Model Cohen's d
            cohen_d = beta_T2 / residual_sd
            # Interpretation
            if cohen_d < 0.20:
                interp = "negligible effect"
            elif cohen_d < 0.50:
                interp = "small effect"
            elif cohen_d < 0.80:
                interp = "medium effect"
            # elif cohen_d < 1.20:
            #     interp = "large effect"
            # else:
            #     interp = "very large effect"
            else:
                interp = "large effect"
            #
            df_out = pd.DataFrame({
            "timepoint": time, # also called : "contrast": [time],
            "beta_T2": [beta_T2],
            "residual_var": [residual_var],
            "residual_sd": [residual_sd],
            "cohen_d": [cohen_d],
            "interpretation": [interp]
            })
            
            # Exit
            # ----
            return df_out
        
        # Exec
        # ----
        df_cohe_1_T1 = cohe_util_1(result, 'T1')
        df_cohe_1_T2 = cohe_util_1(result, 'T2')
        #
        if trac:
            print_yes(df_cohe_1_T1, labl="df_cohe_1_T1")
            print_yes(df_cohe_1_T2, labl="df_cohe_1_T2")
    
    # Main 1
    # ----
    cohe_main_1(result)
 
    # ====
    # Exec 2 : not used
    # ====
    def cohe_main_2(result):
    
        # Util 2 : model-based Cohen [production: not used]
        # ----
        def cohe_util_2(result):
            
            # Exec
            # ----
            rows = []
            # Parcourt tous les paramètres du modèle
            for name in result.params.index:
                # On ne garde que les contrastes du type C(timepoint)[T.X]
                m = re.match(r"C\(timepoint\)\[T\.(.+)\]", name)
                if m:
                    tp = m.group(1)  # ex: "T1", "T2"
                    #
                    beta = result.params[name]
                    residual_var = result.scale
                    residual_sd = np.sqrt(residual_var)
                    cohen_d = beta / residual_sd
                    #
                    if cohen_d < 0.20:
                        interp = "negligible effect"
                    elif cohen_d < 0.50:
                        interp = "small effect"
                    elif cohen_d < 0.80:
                        interp = "medium effect"
                    elif cohen_d < 1.20:
                        interp = "large effect"
                    else:
                        interp = "very large effect"
                    #
                    rows.append({
                        "contrast": tp,
                        "beta": beta,
                        "residual_var": residual_var,
                        "residual_sd": residual_sd,
                        "cohen_d": cohen_d,
                        "interpretation": interp
                    })

            # Exit
            # ----
            return pd.DataFrame(rows)
        
        # Exec
        # ----
        df_cohe_2 = cohe_util_2(result)
        if trac:
            print_yes(df_cohe_2, labl="df_cohe_2")
        
        # Merg
        # ----
        # 1. Clean df_cohe to ensure no trailing/leading spaces
        df_cohe_2['contrast'] = df_cohe_2['contrast'].astype(str).str.strip()

        # 2. Add empty columns to df_deta for the new data
        df_merg_2 = df_deta.copy()
        df_merg_2['cohen_d'] = None
        df_merg_2['interpretation'] = None

        # 3. Explicitly map based on string contains
        # This avoids regex failures by looking for the specific labels
        for idx in df_merg_2.index:
            for _, row in df_cohe_2.iterrows():
                if row['contrast'] in idx:  # e.g., if 'T1' is in 'C(timepoint)[T.T1]'
                    df_merg_2.at[idx, 'cohen_d'] = row['cohen_d']
                    df_merg_2.at[idx, 'interpretation'] = row['interpretation']

        # 4. Convert cohen_d back to numeric (it becomes 'object' during manual assignment)
        df_merg_2['cohen_d'] = pd.to_numeric(df_merg_2['cohen_d'])
        if trac:
            print_yes(df_merg_2, labl="df_merg_2")
        
        # Mark 2
        # ----
        exec_mark_2 = False
        if exec_mark_2:
            coef_info = df_merg_2.loc['C(timepoint)[T.T2]', 'Coef']
            cohe_info = df_merg_2.loc['C(timepoint)[T.T2]', 'cohen_d']
            mark_dict['Mean change (T0–T2)'] = (coef_info.round(2), Path(__file__).stem)
            mark_dict['Model-based Cohen’s d'] = (cohe_info.round(2), Path(__file__).stem)
            
    # Main 2
    # ----
    cohe_main_2(result)

    # ====
    # Exec 3: used
    # ====

    def cohe_main_3(result):
    
        # Util 3
        # ----
        def cohe_util_3(result):
            """
            Extract model-based contrasts (T1-T0, T2-T0, T2-T1)
            with estimates, SEs, CIs, and model-based Cohen's d.
            """
            # Extract coefficients
            beta_T1 = result.params["C(timepoint)[T.T1]"]
            beta_T2 = result.params["C(timepoint)[T.T2]"]

            # Extract SEs
            se_T1 = result.bse["C(timepoint)[T.T1]"]
            se_T2 = result.bse["C(timepoint)[T.T2]"]

            # Residual SD
            resid_sd = np.sqrt(result.scale)

            # Compute T2-T1 contrast
            beta_T2_T1 = beta_T2 - beta_T1
            se_T2_T1 = np.sqrt(se_T1**2 + se_T2**2)  # conservative

            # Build dataframe
            df = pd.DataFrame([
                ["T1 - T0", beta_T1, se_T1, resid_sd],
                ["T2 - T0", beta_T2, se_T2, resid_sd],
                ["T2 - T1", beta_T2_T1, se_T2_T1, resid_sd],
            ], columns=["contrast", "estimate", "se", "resid_sd"])

            # Compute model-based Cohen's d
            df["cohen_d"] = df["estimate"] / df["resid_sd"]
            
            # Delta-method SE for d
            df["se_d"] = df["se"] / df["resid_sd"]

            # CI for d
            df["cohen_d_ci_low"] = df["cohen_d"] - 1.96 * df["se_d"]
            df["cohen_d_ci_high"] = df["cohen_d"] + 1.96 * df["se_d"]

            # Interpretation
            df["interpretation"] = df["cohen_d"].apply(
                lambda d: "negligible" if abs(d) < 0.20 else
                        "small" if abs(d) < 0.50 else
                        "moderate" if abs(d) < 0.80 else
                        "large"
            )

            # 95% CI for estimate
            df["ci_low"] = df["estimate"] - 1.96 * df["se"]
            df["ci_high"] = df["estimate"] + 1.96 * df["se"]

            return df
        
        # Exec 3
        # ----
        df_cohe_3 = cohe_util_3(result)
        if trac:
            print_yes(df_cohe_3, labl="df_cohe_3")
        
        # Edit 3
        # ----  
        def cohe_util_3_edit(df_mod):
            df = df_mod.copy()
            print_yes(df_mod, labl='df_cohe_3')

            df["estimate_fmt"] = (
                df["estimate"].round(2).astype(str)
                + " [" + df["ci_low"].round(2).astype(str)
                + "–" + df["ci_high"].round(2).astype(str) + "]"
            )

            df["cohen_d_fmt"] = (
                df["cohen_d"].round(2).astype(str)
                + " (" + df["interpretation"].str.capitalize() + ") "
                + "[" + df["cohen_d_ci_low"].round(2).astype(str)
                + "–" + df["cohen_d_ci_high"].round(2).astype(str) + "]"
            )

            df_summary = df[[
                "contrast",
                "estimate_fmt",
                "cohen_d_fmt"
            ]].rename(columns={
                "contrast": "Contrast",
                "estimate_fmt": "Model Estimate (CI)",
                "cohen_d_fmt": "Model Cohen's d (Interp)"
            })

            return df_summary
        
        # Edit 3
        # ----
        df_cohe_edit_3 = cohe_util_3_edit(df_cohe_3)
        if trac:
            print_yes(df_cohe_edit_3, labl="df_cohe_edit_3")
        df_cohe_edit_3 = df_cohe_edit_3.set_index("Contrast")
        if trac:
            print_yes(df_cohe_edit_3, labl="df_cohe_edit_3")
          
        # Exit 3
        # ----
        return df_cohe_3, df_cohe_edit_3
            
                
    # Main 3
    # ----
    df_cohe_3, df_cohe_edit_3 = cohe_main_3(result)

    # Mark 3
    # ----
    exec_mark_3 = True
    if exec_mark_3:
        coef_info = df_cohe_edit_3.loc["T2 - T0", "Model Estimate (CI)"]
        cohe_info = df_cohe_edit_3.loc["T2 - T0", "Model Cohen's d (Interp)"]
        mark_dict['Model-based mean change (T0–T2)']   = (coef_info, Path(__file__).stem)
        mark_dict['Model-based Cohen’s d'] = (cohe_info, Path(__file__).stem)
        
    # Exit
    # ----
    stat_tran_adat.resu_cohe_modl_plot = df_cohe_3
    stat_tran_adat.resu_cohe_modl_stat = df_cohe_edit_3 
    
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