    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_31_stat_ import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# ----
# Timepoint outcomes
# ----
#
# https://copilot.microsoft.com/shares/Q5DZZFvNPuaU7NF4xrwn6
#
def exec_stat_mixd_mean_modl(stat_tran_adat: StatTranQOL_31_mixd, df_modl, result) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True
    
    # Publ
    # ----
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta04_endp_prim_modl

    # Exec : df_mean : modeled means
    # ----
    params = result.params
    vcov = result.cov_params()
    
    # Basic
    # -----
    def modl_mean_1():
        mean_T0 = params["Intercept"]  # 50.003
        mean_T1 = params["Intercept"] + params["C(timepoint)[T.T1]"] # 55.003
        mean_T2 = params["Intercept"] + params["C(timepoint)[T.T2]"] # 57.503
        return mean_T0, mean_T1, mean_T2
    #
    mean_T0, mean_T1, mean_T2 = modl_mean_1()
    df_mean_not_used = pd.DataFrame({
        "timepoint": ["T0", "T1", "T2"],
        "mean": [mean_T0, mean_T1, mean_T2]
    })
    if trac:
        print_yes(df_mean_not_used)
    
    # Extnd [with CIs] : open_ai (flexible TX)
    # -----
    def modl_mean_2(level):
        if level == "T0":
            mean = params["Intercept"]
            vari = vcov.loc["Intercept", "Intercept"]
        else:
            coef = f"C(timepoint)[T.{level}]"
            mean = params["Intercept"] + params[coef]
            vari = (vcov.loc["Intercept", "Intercept"] + vcov.loc[coef, coef] + 2 * vcov.loc["Intercept", coef])
        se = np.sqrt(vari) #  standard error
        ci_low = mean - 1.96 * se
        ci_up = mean + 1.96 * se
        return level, mean, se, ci_low, ci_up
    #
    timepoints = ["T0", "T1", "T2"]
    estimates = [modl_mean_2(tp) for tp in timepoints]
    df_mean = pd.DataFrame(estimates, columns=["timepoint", "mean", "se", "ci_lower", "ci_upper"])
    df_mean["timepoint"] = pd.Categorical(df_mean["timepoint"], categories=df_modl["timepoint"].cat.categories, ordered=df_modl["timepoint"].cat.ordered)
    df_mean['n'] = [df_modl[df_modl['timepoint'] == tp]['patient_id'].nunique() for tp in timepoints]
    df_mean['95% CI'] = (df_mean['ci_lower'].round(2).astype(str) + '–' + df_mean['ci_upper'].round(2).astype(str))
    #
    df_mean_not_used = df_mean[["timepoint", "mean", "n", "se", "ci_lower", "ci_upper", "95% CI"]]
    #
    if trac:
        print_yes(df_mean_not_used, "df_mean")
        
    # Extnd [with CIs] : claude (didactic TX) https://chat.mistral.ai/chat/8a997b72-eed0-4629-b55e-9d3aa089ae45
    # -----
    def modl_mean_3(result):
        
        # Coefficients
        # ------------
        intercept = result.params['Intercept']
        effect_T0 = 0
        effect_T1 = result.params['C(timepoint)[T.T1]']
        effect_T2 = result.params['C(timepoint)[T.T2]']
        #
        conf = result.conf_int()
        intercept_ci = conf.loc["Intercept"]
        effect_T0_ci = 0
        effect_T1_ci = conf.loc["C(timepoint)[T.T1]"] # "C(timepoint, Treatment(reference='T0'))[T.T1]"
        effect_T2_ci = conf.loc["C(timepoint)[T.T2]"]

        # Modeled means
        # -------------
        mean_T0 = intercept # 50.003
        mean_T1 = intercept + effect_T1 # 55.003
        mean_T2 = intercept + effect_T2  # 57.503

        # Coefficients : Standard errors
        # ------------------------------
        se_intercept = result.bse['Intercept']
        se_T1 = result.bse['C(timepoint)[T.T1]']
        se_T2 = result.bse['C(timepoint)[T.T2]']

        # Modeled means : Standard errors
        # -------------------------------
        # Need covariance between intercept and effect_TX
        cov_matrix = result.cov_params()
        # T0: SE of intercept
        se_mean_T0 = se_intercept
        # T1: SE of (intercept + effect_T1)
        cov_int_T1 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T1]']
        se_mean_T1 = np.sqrt(se_intercept**2 + se_T1**2 + 2*cov_int_T1)
        # T2: SE of (intercept + effect_T2)
        cov_int_T2 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T2]']
        se_mean_T2 = np.sqrt(se_intercept**2 + se_T2**2 + 2*cov_int_T2)

        # Modeled means : $95% CI
        # -----------------------
        ci_T0_lower = mean_T0 - 1.96 * se_mean_T0
        ci_T0_upper = mean_T0 + 1.96 * se_mean_T0
        #
        ci_T1_lower = mean_T1 - 1.96 * se_mean_T1
        ci_T1_upper = mean_T1 + 1.96 * se_mean_T1
        #
        ci_T2_lower = mean_T2 - 1.96 * se_mean_T2
        ci_T2_upper = mean_T2 + 1.96 * se_mean_T2

        # Summary dataframe
        # -----------------
        df_mean = pd.DataFrame({
            'timepoint': ['T0', 'T1', 'T2'],
            'mean': [mean_T0, mean_T1, mean_T2],
            'se': [se_mean_T0, se_mean_T1, se_mean_T2],
            'ci_lower': [ci_T0_lower, ci_T1_lower, ci_T2_lower],
            'ci_upper': [ci_T0_upper, ci_T1_upper, ci_T2_upper],
            'effect': [effect_T0, effect_T1, effect_T2],
            'effect_ci_lower': [effect_T0_ci, effect_T1_ci[0], effect_T2_ci[0]],
            'effect_ci_upper': [effect_T0_ci, effect_T1_ci[1], effect_T2_ci[1]]
        })
        df_mean["timepoint"] = pd.Categorical(df_mean["timepoint"], categories=df_modl["timepoint"].cat.categories, ordered=df_modl["timepoint"].cat.ordered)
        df_mean['n'] = [df_modl[df_modl['timepoint'] == tp]['patient_id'].nunique() for tp in timepoints]
        df_mean['95% CI'] = (df_mean['ci_lower'].round(2).astype(str) + '–' + df_mean['ci_upper'].round(2).astype(str))
        #
        df_mean = df_mean[["timepoint", "mean", "n", "se", "ci_lower", "ci_upper", "95% CI", "effect", "effect_ci_lower", "effect_ci_upper"]]
        
        # Exit
        # ----
        return df_mean
    
    df_mean = modl_mean_3(result)
    if trac:
        print_yes(df_mean, labl="df_mean")

    # Mark : At T0, the mean score was 54.16 ± 1.00 (95% CI 52.20–56.13; n = 24).
    # ----
    df_mean = df_mean.set_index("timepoint") # 'timepoint' column is now also row index
    #
    def mean_util(df, time):
        mean_info = (
        f"{df.loc[time, 'mean']:.2f} ± {df.loc[time, 'se']:.2f} "
        f"(95% CI {df.loc[time, 'ci_lower']:.2f}–{df.loc[time, 'ci_upper']:.2f}; "
        f"n = {df.loc[time, 'n']})"
        )
        return mean_info
    T0_mean = mean_util(df_mean, 'T0') ; mark_dict['Mean score at T0'] = (T0_mean, Path(__file__).stem)
    T1_mean = mean_util(df_mean, 'T1') ; mark_dict['Mean score at T1'] = (T1_mean, Path(__file__).stem)
    T2_mean = mean_util(df_mean, 'T2') ; mark_dict['Mean score at T2'] = (T2_mean, Path(__file__).stem)
    #
    effect_T1 = f"{df_mean.loc['T1', 'effect']:.2f}" ; mark_dict['Adjusted mean difference at T1'] = (effect_T1, Path(__file__).stem)
    effect_T2 = f"{df_mean.loc['T2', 'effect']:.2f}" ; mark_dict['Adjusted mean difference at T2'] = (effect_T2, Path(__file__).stem)
    #
    df_mean = df_mean.reset_index(drop=False) # 'timepoint' column is kept but is not index anymore ; index = 0,1,...

    # Mark
    # ----
    df_mark = df_mean.copy()
    df_mark['Script'] = Path(__file__).stem
    stat_tran_adat.stat_tran.proc_tran.orch_tran.df_ta05_endp_prim_modl = df_mark
    
    # Exit
    # ----    
    stat_tran_adat.mixd_mean_modl = df_mean

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