    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_11_stat_ import StatTranQOL_11_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import patsy
from pandas import DataFrame
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.stats import shapiro, normaltest

# https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c
def exec_stat_mixd_open(stat_tran_adat: StatTranQOL_11_mixd, df_lon1:DataFrame) -> None:
    from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11_mixd

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, "df_fram")

    # Data : df_fram
    # ====
    df_modl = df_fram.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
    
    # Modl : df_modl [Fit the linear mixed-effects model]
    # ==== 
    model = smf.mixedlm("VEINES_QOL_t ~ C(timepoint)", df_modl, groups=df_modl["patient_id"])
    result = model.fit(reml=True, method="powell")
    print(result.summary())

    # Plot
    # ====
    stra = '1'
    match stra:
        case '1':
            # ----
            # Extract fixed effects and covariance
            # ----
            fe_params = result.fe_params              # pandas Series
            cov_all = result.cov_params()             # full covariance matrix
            cov_fe = cov_all.loc[
                fe_params.index,
                fe_params.index
            ]

            # ----
            # Build aligned EMM design matrix
            # ----
            emm_design = pd.DataFrame(
                0.0,
                index=["T0", "T1", "T2"],
                columns=fe_params.index
            )

            # Intercept
            emm_design.loc[:, "Intercept"] = 1.0

            # Timepoint contrasts
            if "C(timepoint)[T.T1]" in emm_design.columns:
                emm_design.loc["T1", "C(timepoint)[T.T1]"] = 1.0

            if "C(timepoint)[T.T2]" in emm_design.columns:
                emm_design.loc["T2", "C(timepoint)[T.T2]"] = 1.0

            # ----
            # EMMs, SEs, CIs
            # ----
            emm_mean = emm_design @ fe_params
            emm_var = np.diag(
                emm_design @ cov_fe @ emm_design.T
            )
            emm_se = np.sqrt(emm_var)

            # Fini
            # ----
            uu = len(df_modl)
            df_emm = pd.DataFrame({
                "timepoint": emm_design.index,
                "modl_mean": emm_mean.values,
                "modl_se": emm_se,
                "modl_ci_lower": emm_mean.values - 1.96 * emm_se,
                "modl_ci_upper": emm_mean.values + 1.96 * emm_se
            })
            df_emm["n"] = (
            df_modl
            .groupby("timepoint", observed=False)["VEINES_QOL_t"]
            .count()
            .reindex(["T0", "T1", "T2"])
            .values
            )
            pass
        case '2': # [TODO bug]
            # Prediction frame (one row per timepoint)
            pred_df = pd.DataFrame({
                "timepoint": df_modl["timepoint"].cat.categories
            })

            pred_df["timepoint"] = pd.Categorical(
                pred_df["timepoint"],
                categories=df_modl["timepoint"].cat.categories,
                ordered=True
            )

            # DESIGN MATRIX — INCLUDE INTERCEPT
            X_pred = patsy.dmatrix(
                "1 + C(timepoint)",
                pred_df,
                return_type="dataframe"
            )

            # Sanity check (recommended once)
            assert X_pred.shape[1] == len(result.fe_params)

            # Estimates
            model_estimate = X_pred @ result.fe_params

            # SE via delta method
            cov_fe = result.cov_params()
            model_se = np.sqrt(np.diag(X_pred @ cov_fe @ X_pred.T))

            # 95% CI
            df_emm = pd.DataFrame({
                "timepoint": pred_df["timepoint"],
                "modl_mean": model_estimate,
                "modl_se": model_se,
                "modl_ci_lower": model_estimate - 1.96 * model_se,
                "modl_ci_upper": model_estimate + 1.96 * model_se,
            })
            df_emm["n"] = (
            df_modl
            .groupby("timepoint", observed=False)["VEINES_QOL_t"]
            .count()
            .reindex(["T0", "T1", "T2"])
            .values
            )
            pass
        #
    if trac:
        print_yes(df_emm, labl="df_emm")

    # Merg
    # ----
    if len(df_lon1) != len(df_emm):
            raise ValueError(f"DataFrames have different number of rows: {len(df_lon1)} vs {len(df_emm)}") 
    df_plot = pd.merge(df_lon1, df_emm, on=['timepoint', 'n'], how='inner')
    df_plot = df_plot[['timepoint', 'n', 'mean', 'ci_lower', 'ci_upper', 'sd', 'se', 'modl_mean', 'modl_ci_lower', 'modl_ci_upper']]
    df_plot['timepoint'] = pd.Categorical(df_plot['timepoint'], categories=df_lon1['timepoint'].cat.categories, ordered=df_lon1['timepoint'].cat.ordered)
    if trac:
        print_yes(df_plot, labl="df_plot")
        
    # ****
    # MCID : https://chatgpt.com/c/69772ac1-14dc-832b-bef7-3937abac6a77 
    # ****     

    does_work = False
    if does_work:
        # Design matrix for EMMs
        timepoints = ["T0", "T1", "T2"]

        # Get fixed effects
        fe_params = result.fe_params
        cov_fe = result.cov_params()

        # Build contrast vectors manually
        # Order must match fe_params.index
        # Typically: Intercept, C(timepoint)[T.T1], C(timepoint)[T.T2]

        contrast_vectors = {
            "T1 - T0": np.array([0, 1, 0]),
            "T2 - T0": np.array([0, 0, 1])
        }

        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha / 2)

        mcid_threshold = 3.0

        results = []

        for label, cvec in contrast_vectors.items():
            estimate = np.dot(cvec, fe_params.values)
            se = np.sqrt(np.dot(cvec.T, np.dot(cov_fe.values, cvec)))
            
            ci_lower = estimate - z_crit * se
            ci_upper = estimate + z_crit * se
            
            clinically_meaningful = ci_lower >= mcid_threshold
            
            results.append({
                "Contrast": label,
                "EMM Difference": round(estimate, 2),
                "95% CI Lower": round(ci_lower, 2),
                "95% CI Upper": round(ci_upper, 2),
                "MCID Exceeded (Lower CI ≥ 3)": clinically_meaningful
            })

        mcid_emm_df = pd.DataFrame(results)
        print(mcid_emm_df)
    pass


    # Exit
    # ----
    df_emm = df_emm.rename(columns={"modl_mean":"mean", "modl_se":"se", "modl_ci_lower":"ci_lower", "modl_ci_upper":"ci_upper"})
    cate_list = ['T0','T1','T2'] ; df_emm["timepoint"] = pd.Categorical(df_emm["timepoint"], categories=cate_list, ordered=True)
    #
    stat_tran_adat.open_resu_fram = df_fram # original Tx records / patient
    stat_tran_adat.open_resu_plot = df_plot # derived  Tx records / timepoint
    stat_tran_adat.open_resu_plot_raw = df_lon1
    stat_tran_adat.open_resu_plot_lme = df_emm

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