    
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
import matplotlib.pyplot as plt
from scipy import stats

# ----
# Post-hoc contrasts (paiwise comparisons)
# ----
#
# https://copilot.microsoft.com/shares/Q5DZZFvNPuaU7NF4xrwn6
#
def exec_stat_mixd_pair_modl(stat_tran_adat: StatTranQOL_31_mixd, df_modl, result) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # ====
    # Exec 1
    # ====
    def exec_1(result):
    
        # Exec : df-pair = Paiwise contrasts
        # ----
        fe = result.fe_params
        vcov = result.cov_params().loc[fe.index, fe.index]

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

        # Fram
        df_pair = pd.DataFrame(contrasts, columns=["Contrast", "Estimate", "SE", "Lower95", "Upper95"])
        #
        if trac:
            print_yes(df_pair, "df_pair")
            
        # Exit
        # ----
        return df_pair
            
    df_pair_1 = exec_1(result)
    
    # ====
    # Exec 2
    # ====
    def exec_2(result):
        
        # Coefficients
        # ------------
        intercept = result.params['Intercept']
        effect_T1 = result.params['C(timepoint)[T.T1]']
        effect_T2 = result.params['C(timepoint)[T.T2]']
        
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
    
        # Contrasts
        # ---------
        # T1-T0
        contrast_T1_T0 = effect_T1
        se_contrast_T1_T0 = se_T1
        z_T1_T0 = contrast_T1_T0 / se_contrast_T1_T0
        p_T1_T0 = 2 * (1 - stats.norm.cdf(abs(z_T1_T0)))
        ci_T1_T0_lower = contrast_T1_T0 - 1.96 * se_contrast_T1_T0
        ci_T1_T0_upper = contrast_T1_T0 + 1.96 * se_contrast_T1_T0
        # T2-T0
        contrast_T2_T0 = effect_T2
        se_contrast_T2_T0 = se_T2
        z_T2_T0 = contrast_T2_T0 / se_contrast_T2_T0
        p_T2_T0 = 2 * (1 - stats.norm.cdf(abs(z_T2_T0)))
        ci_T2_T0_lower = contrast_T2_T0 - 1.96 * se_contrast_T2_T0
        ci_T2_T0_upper = contrast_T2_T0 + 1.96 * se_contrast_T2_T0
        # T2-T1 : This is (intercept + effect_T2) - (intercept + effect_T1) = effect_T2 - effect_T1
        contrast_T2_T1 = effect_T2 - effect_T1
        # SE needs covariance between T1 and T2 effects
        cov_T1_T2 = cov_matrix.loc['C(timepoint)[T.T1]', 'C(timepoint)[T.T2]']
        se_contrast_T2_T1 = np.sqrt(se_T1**2 + se_T2**2 - 2*cov_T1_T2)
        z_T2_T1 = contrast_T2_T1 / se_contrast_T2_T1
        p_T2_T1 = 2 * (1 - stats.norm.cdf(abs(z_T2_T1)))
        ci_T2_T1_lower = contrast_T2_T1 - 1.96 * se_contrast_T2_T1
        ci_T2_T1_upper = contrast_T2_T1 + 1.96 * se_contrast_T2_T1

        # Dataframe
        df_modl_delt = pd.DataFrame({
            'Comparison': ['T1 - T0', 'T2 - T0', 'T2 - T1'],
            'Difference': [contrast_T1_T0, contrast_T2_T0, contrast_T2_T1],
            'SE': [se_contrast_T1_T0, se_contrast_T2_T0, se_contrast_T2_T1],
            'Z': [z_T1_T0, z_T2_T0, z_T2_T1],
            'P_value': [p_T1_T0, p_T2_T0, p_T2_T1],
            'CI_Lower': [ci_T1_T0_lower, ci_T2_T0_lower, ci_T2_T1_lower],
            'CI_Upper': [ci_T1_T0_upper, ci_T2_T0_upper, ci_T2_T1_upper]
        })
        df_modl_delt['95% CI'] = (df_modl_delt['CI_Lower'].round(2).astype(str) + '–' + df_modl_delt['CI_Upper'].round(2).astype(str))

        # 3. Bonferroni correction for 3 contrast p-values (optional)
        # Note : Statistically significant pairwise comparisons 
        # indicate progressive improvement in QOL scores over time.
        # ================================================
        n_comparisons = 3
        df_modl_delt['Bonf_P_value'] = df_modl_delt['P_value'] * n_comparisons
        df_modl_delt['Bonf_P_value'] = df_modl_delt['Bonf_P_value'].clip(upper=1.0)
        df_modl_delt['Bonf_Significant'] = df_modl_delt['Bonf_P_value'] < 0.05
        #
        df_modl_delt = df_modl_delt[['Comparison', 'Difference', '95% CI', 'SE', 'Z', 'P_value', 'Bonf_P_value', 'Bonf_Significant']]
        if trac:
            print_yes(df_modl_delt, "df_modl_delt")
            
        # Exit
        # ----
        return df_modl_delt
    
    df_pair_2 = exec_2(result)
        
    # Exit
    # ----    
    stat_tran_adat.mixd_pair_modl_1 = df_pair_1
    stat_tran_adat.mixd_pair_modl_2 = df_pair_2
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TODO
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

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
        plot_forest(df_pair_1)

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