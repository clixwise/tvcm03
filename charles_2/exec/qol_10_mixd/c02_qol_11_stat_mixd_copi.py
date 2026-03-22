    
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

# ----
# Timepoint outcomes
# ----
#
# https://copilot.microsoft.com/shares/Q5DZZFvNPuaU7NF4xrwn6
#
def exec_stat_mixd_copi(stat_tran_adat: StatTranQOL_11_mixd, df_lon1:DataFrame) -> None:
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

    # ****
    # Modeled means
    # ****
    
    # Basic
    # -----
    params = result.params
    mean_T0 = params["Intercept"]
    mean_T1 = params["Intercept"] + params["C(timepoint)[T.T1]"]
    mean_T2 = params["Intercept"] + params["C(timepoint)[T.T2]"]
    df_mean_not_used = pd.DataFrame({
        "timepoint": ["T0", "T1", "T2"],
        "modeled_mean": [mean_T0, mean_T1, mean_T2]
    })
    
    # Extnd ; with CIs
    # -----
    params = result.params
    vcov = result.cov_params()
    #
    def modeled_mean_and_ci(level):
        if level == "T0":
            est = params["Intercept"]
            var = vcov.loc["Intercept", "Intercept"]
        else:
            coef = f"C(timepoint)[T.{level}]"
            est = params["Intercept"] + params[coef]
            var = (vcov.loc["Intercept", "Intercept"] + vcov.loc[coef, coef] + 2 * vcov.loc["Intercept", coef])
        se = np.sqrt(var) #  standard error
        ci_low = est - 1.96 * se
        ci_up = est + 1.96 * se
        return level, est, se, ci_low, ci_up
    #
    timepoints = ["T0", "T1", "T2"]
    estimates = [modeled_mean_and_ci(tp) for tp in timepoints]
    df_mean = pd.DataFrame(estimates, columns=["timepoint", "mean", "se", "ci_lower", "ci_upper"])
    df_mean["timepoint"] = pd.Categorical(df_mean["timepoint"], categories=df_modl["timepoint"].cat.categories, ordered=df_modl["timepoint"].cat.ordered)
    df_mean['n'] = [df_modl[df_modl['timepoint'] == tp]['patient_id'].nunique() for tp in timepoints]
    #
    df_mean = df_mean[["timepoint", "mean", "n", "se", "ci_lower", "ci_upper"]]
    
    # ****
    # EMM https://copilot.microsoft.com/shares/M71sUTzXVPVFDSEeZkTLq
    # ****
    def exec_emms(m_fit, df):
        """
        EMMs for timepoint in model: VEINES_QOL_t ~ C(timepoint)
        Assumes categories: T0 (ref), T1, T2
        """

        # Fixed-effect params only
        fe = m_fit.fe_params
        vcov_fe = m_fit.cov_params().loc[fe.index, fe.index]

        levels = pd.Categorical(df["timepoint"]).categories.tolist()
        ref = levels[0]  # should be 'T0'

        rows = []

        for lvl in levels:
            # Build contrast vector for fixed effects
            c = np.zeros(len(fe))
            # Intercept always 1
            c[fe.index.get_loc("Intercept")] = 1.0

            # Add dummy if not reference
            if lvl != ref:
                # param name is like C(timepoint)[T.T1] for lvl='T1'
                pname = f"C(timepoint)[T.{lvl}]"
                if pname in fe.index:
                    c[fe.index.get_loc(pname)] = 1.0

            emm = float(c @ fe)
            se = float(np.sqrt(c @ vcov_fe @ c))

            rows.append({
                "timepoint": lvl,
                "EMM": emm,
                "SE": se,
                "Lower95": emm - 1.96 * se,
                "Upper95": emm + 1.96 * se
            })

        return pd.DataFrame(rows)
    #
    df_emms = exec_emms(result, df_fram)
    if trac:
        print_yes(df_emms, "df_emms")

    def plot_emm_trajectory(emms):
        """
        Pure matplotlib trajectory plot for EMMs with 95% CI.
        Expects columns: timepoint, EMM, Lower95, Upper95
        """

        # X positions
        x = range(len(emms))

        plt.figure(figsize=(6, 4))

        # Plot the EMM line
        plt.plot(
            x,
            emms["EMM"],
            marker="o",
            color="black",
            linewidth=2
        )

        # Add vertical error bars
        for i, row in emms.iterrows():
            plt.vlines(
                x=i,
                ymin=row["Lower95"],
                ymax=row["Upper95"],
                color="black",
                linewidth=2
            )

        # Formatting
        plt.xticks(x, emms["timepoint"])
        plt.xlabel("Timepoint")
        plt.ylabel("VEINES t-score (EMM ± 95% CI)")
        plt.title("Estimated Marginal Means of VEINES t-score")
        plt.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()
        pass

    plot_emms = False
    if plot_emms:
        plot_emm_trajectory(df_emms)


    # ****
    # Paiwise contrats
    # ****

    def pairwise_contrasts(m_fit):
        fe = m_fit.fe_params
        vcov = m_fit.cov_params().loc[fe.index, fe.index]

        # Extract coefficients
        b_T1 = fe["C(timepoint)[T.T1]"]
        b_T2 = fe["C(timepoint)[T.T2]"]

        # Extract variances and covariance
        var_T1 = vcov.loc["C(timepoint)[T.T1]", "C(timepoint)[T.T1]"]
        var_T2 = vcov.loc["C(timepoint)[T.T2]", "C(timepoint)[T.T2]"]
        cov_T1_T2 = vcov.loc["C(timepoint)[T.T1]", "C(timepoint)[T.T2]"]

        # Contrasts
        contrasts = []

        # T1 - T0
        est = b_T1
        se = np.sqrt(var_T1)
        contrasts.append(["T1 - T0", est, se, est - 1.96*se, est + 1.96*se])

        # T2 - T0
        est = b_T2
        se = np.sqrt(var_T2)
        contrasts.append(["T2 - T0", est, se, est - 1.96*se, est + 1.96*se])

        # T2 - T1 = (T2 - T0) - (T1 - T0)
        est = b_T2 - b_T1
        se = np.sqrt(var_T2 + var_T1 - 2*cov_T1_T2)
        contrasts.append(["T2 - T1", est, se, est - 1.96*se, est + 1.96*se])

        return pd.DataFrame(contrasts, columns=["Contrast", "Estimate", "SE", "Lower95", "Upper95"])
    
    df_pair = pairwise_contrasts(result)
    if trac:
        print_yes(df_pair, "df_pair")


    def plot_forest(df_pair):
        """
        Forest plot for pairwise contrasts using pure matplotlib.
        Expects columns: Contrast, Estimate, SE, Lower95, Upper95
        """

        # Reverse order so the last contrast appears at the top
        df_plot = df_pair.iloc[::-1].reset_index(drop=True)

        y_positions = range(len(df_plot))

        plt.figure(figsize=(6, 4))

        # Plot the point estimates
        plt.scatter(df_plot["Estimate"], y_positions, color="black", zorder=3)

        # Plot the confidence intervals
        for i, row in df_plot.iterrows():
            plt.hlines(
                y=i,
                xmin=row["Lower95"],
                xmax=row["Upper95"],
                color="black",
                linewidth=2
            )

        # Add a vertical reference line at 0
        plt.axvline(0, color="grey", linestyle="--", linewidth=1)

        # Formatting
        plt.yticks(y_positions, df_plot["Contrast"])
        plt.xlabel("Difference in VEINES t-score")
        plt.title("Pairwise Contrasts (95% CI)")
        plt.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()
        pass
    plot_pair = False
    if plot_pair:
        plot_forest(df_pair)
    
    # ****
    # Result for Forest plot
    # ****
    if trac:
        print_yes(df_lon1, "df_lon1")
        print_yes(df_mean, "df_mean")
        print_yes(df_emms, "df_emms")
        print_yes(df_pair, "df_pair")
        
    stat_tran_adat.copi_mixd_raww_mean = df_lon1
    stat_tran_adat.copi_mixd_modl_mean = df_mean
    #
    stat_tran_adat.copi_mixd_modl_emms = df_emms
    stat_tran_adat.copi_mixd_modl_pair = df_emms
    pass

    ''''
    `var` is the **variance of the estimated mean** at each timepoint, extracted from the model's **variance-covariance matrix** (`vcov`).
    ## Breakdown by Timepoint
    **For T0 (reference):**
    var = vcov.loc["Intercept", "Intercept"]
    - Pure variance of the Intercept estimate
    - `Var(μ_T0) = Var(β_Intercept)`

    **For T1/T2 (vs T0):**
    var = (vcov.loc["Intercept", "Intercept"] + 
        vcov.loc[coef, coef] + 
        2 * vcov.loc["Intercept", coef])
    - `Var(μ_T1) = Var(β_Intercept + β_T1)`
    - **Variance addition formula**: `Var(a + b) = Var(a) + Var(b) + 2⋅Cov(a,b)`
    - `coef = "C(timepoint)[T.T1]"` or `"C(timepoint)[T.T2]"`

    ## Why This Works
    Your **linear combination** `μ_T1 = Intercept + C(timepoint)[T.T1]` has variance:
    Var(μ_T1) = Var(Intercept) + Var(β_T1) + 2 × Cov(Intercept, β_T1)
    The `vcov` matrix contains **all pairwise variances/covariances** from the mixed model fit.

    ## Then...
    `se = np.sqrt(var)` → **standard error** of the mean at each timepoint
    `ci_low = est - 1.96 * se` → **95% CI lower bound**

    **Publication note**: "Means and 95% CIs derived from model variance-covariance matrix 
    accounting for random effects and repeated measures." 
    [groups.google](https://groups.google.com/g/pystatsmodels/c/KXF3CxqYZcI)
    '''
    '''
    T0:  est=50.003, se=1.23  → 95% CI: 47.58–52.43  [precise population mean estimate]
    T0: residuals SD=4.54     → individual patients scatter ±4.54 around that mean [data spread]
    '''

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