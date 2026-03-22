    
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
from scipy.stats import shapiro, normaltest, levene, skew, kurtosis
import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew, kurtosis, anderson
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm

# ----
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
# ----
#
# Random effects : checking for normality
# ============
# Grph
# ====
# see    : c02_qol_11_grph_plot_mixd_resi : residuals
# Desc : check for non-normality LMM approach [https://chat.mistral.ai/chat/8a997b72-eed0-4629-b55e-9d3aa089ae45]
#
def exec_stat_mixd_assu_resi(stat_tran_adat: StatTranQOL_31_mixd, df_modl, result) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True
    
    # Exec
    # ----
    resid = result.resid

    # Basic stats
    res_mean = np.mean(resid)
    res_std = np.std(resid)
    res_skew = skew(resid)
    res_kurt = kurtosis(resid)

    # Normality tests
    res_shapiro_stat, res_shapiro_p = shapiro(resid)
    res_anderson = anderson(resid)

    res_results = [
        {
            'Component': 'Residuals',
            'Assumption': 'Mean ≈ 0',
            'Metric': 'Mean',
            'Value': f"{res_mean:.4f}",
            'Comment': "✓ Close to zero" if abs(res_mean) < 0.1 else "⚠ Deviates from zero"
        },
        {
            'Component': 'Residuals',
            'Assumption': 'Dispersion',
            'Metric': 'Std Dev',
            'Value': f"{res_std:.4f}",
            'Comment': "✓ Reasonable"
        },
        {
            'Component': 'Residuals',
            'Assumption': 'Normality',
            'Metric': f"Shapiro-Wilk W={res_shapiro_stat:.4f}",
            'Value': f"p={res_shapiro_p:.4f}",
            'Comment': "✓ Normal" if res_shapiro_p > 0.05 else "⚠ Non-normal"
        },
        {
            'Component': 'Residuals',
            'Assumption': 'Normality (Anderson)',
            'Metric': f"A²={res_anderson.statistic:.4f}",
            'Value': f"Critical={res_anderson.critical_values[2]:.4f}",
            'Comment': "✓ Normal" if res_anderson.statistic < res_anderson.critical_values[2] else "⚠ Non-normal"
        },
        {
            'Component': 'Residuals',
            'Assumption': 'Shape',
            'Metric': 'Skewness',
            'Value': f"{res_skew:.4f}",
            'Comment': "✓ Symmetric" if abs(res_skew) < 0.5 else "⚠ Skewed"
        },
        {
            'Component': 'Residuals',
            'Assumption': 'Shape',
            'Metric': 'Kurtosis',
            'Value': f"{res_kurt:.4f}",
            'Comment': "✓ Mesokurtic" if abs(res_kurt) < 1 else "⚠ Heavy/light tails"
        }
    ]
    
    '''
    # RESIDUALS vs FITTED
    #Homoscedasticity Tests
    #We can apply Breusch–Pagan and White tests to the residuals vs. fitted relationship.
    fitted = result.fittedvalues
    resid = result.resid
    # Prepare design matrix for tests
    X = sm.add_constant(fitted)

    # Breusch–Pagan test
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, X)

    # White test
    white_stat, white_pvalue, _, _ = het_white(resid, X)

    print(f"Breusch–Pagan p-value: {bp_pvalue:.4f}")
    print(f"White test p-value: {white_pvalue:.4f}")
    '''
    '''
    # Standardized residuals
    std_resid = resid / np.sqrt(result.scale)
    # Number of fixed effects
    p = len(result.params)
    # Cook's distance approximation
    cooks_d = (std_resid**2) / (p)
    # Flag influential points
    influential = np.where(cooks_d > 4 / len(resid))[0]
    print("Potentially influential observations:", influential)
    '''
    
    df_resid = pd.DataFrame(res_results)
    if trac:
        print_yes(df_resid, labl="df_resi")  

    # Exit
    # ----    
    stat_tran_adat.mixd_assu_resi = df_resid
    stat_tran_adat.mixd_assu_resi_plot = result.resid.to_frame(name='Residuals')
    stat_tran_adat.mixd_assu_resi_resu = result

    # OLD FROM CLAUDE
    old = False
    if old:
        # Summary Statistics
        # ------------------
        # Overall first
        df_modl['fitted'] = result.fittedvalues # TODO : not used
        df_modl['residuals'] = result.resid
        overall_mean = df_modl['residuals'].mean()
        overall_sd = df_modl['residuals'].std()
        print(f"Mean:     {overall_mean:.6f} (should be ≈ 0)")

        results = []
        results.append({
            'Timepoint': 'Global', 
            'Metric': 'mean_sd', 
            'Value': f"{overall_mean:.2f} ± {overall_sd:.2f}",
            'Comment': 'should be ≈ 0'
        })
        # Per timepoint
        grouped_residuals = df_modl.groupby('timepoint')['residuals']
        for timepoint, residuals in grouped_residuals:
            mean_value = residuals.mean()
            sd_value = residuals.std()
            results.append({
                'Timepoint': timepoint, 
                'Metric': 'mean_sd', 
                'Value': f"{mean_value:.2f} ± {sd_value:.2f}",
                'Comment': 'check homoscedasticity'
            })
        df_resi_desc = pd.DataFrame(results)
        if trac:
            print_yes(df_resi_desc, labl="df_resi_desc")  
        
        # Stat
        # ====
        def calculate_skew_kurtosis(residuals):
            return pd.Series({
                'Skewness': round(skew(residuals), 4),
                'Kurtosis': round(kurtosis(residuals, fisher=True), 4)  # Fisher=True pour kurtosis excess
            })

        grouped_residuals = df_modl.groupby('timepoint')['residuals']
        normality_results = []
        #
        for timepoint, residuals in grouped_residuals:
            # Test de normalité : Shapiro-Wilk
            shapiro_stat, shapiro_p = shapiro(residuals)
            normality_results.append({
                'Timepoint': timepoint,
                'Assumption': 'Normality',
                'Metric': 'Shapiro-Wilk',
                'Value': round(shapiro_stat, 4),
                'Effect Size': 'W (closer to 1 = more normal)',  # Interprétation de W
                'P-value': round(shapiro_p, 4),
                'Assumption Met': shapiro_p > 0.05,
                'Sample Size': len(residuals),
                'Comment': "✓ Normal (p > 0.05)" if shapiro_p > 0.05 else "⚠ Non-normal (p < 0.05)"
            })
            # Test de normalité : D'Agostino-Pearson + skewness/kurtosis
            dagostino_stat, dagostino_p = normaltest(residuals)
            skew_kurt = calculate_skew_kurtosis(residuals)
            normality_results.append({
                'Timepoint': timepoint,
                'Assumption': 'Normality',
                'Metric': "D'Agostino-Pearson",
                'Value': round(dagostino_stat, 4),
                'Effect Size': f"Skewness={skew_kurt['Skewness']}, Kurtosis={skew_kurt['Kurtosis']}",
                'P-value': round(dagostino_p, 4),
                'Assumption Met': dagostino_p > 0.05,
                'Sample Size': len(residuals),
                'Comment': "✓ Normal (p > 0.05)" if dagostino_p > 0.05 else "⚠ Non-normal (p < 0.05)"
            })

        # Test d'homogénéité des variances : Levene
        groups = [group['residuals'].values for name, group in df_modl.groupby('timepoint')]
        levene_stat, levene_p = levene(*groups)
        # Calcul du ratio de variance
        variances = [np.var(group['residuals']) for name, group in df_modl.groupby('timepoint')]
        variance_ratio = round(max(variances) / min(variances), 4)
        levene_result = {
            'Timepoint': 'All',
            'Assumption': 'Homogeneity',
            'Metric': "Levene's test",
            'Value': round(levene_stat, 4),
            'Effect Size': f"Variance Ratio={variance_ratio}",
            'P-value': round(levene_p, 4),
            'Assumption Met': levene_p > 0.05,
            'Sample Size': sum([len(group['residuals']) for name, group in df_modl.groupby('timepoint')]),
            'Comment': "✓ Homogeneous (p > 0.05)" if levene_p > 0.05 else "⚠ Heteroscedastic (p < 0.05)"
        }

        df_modl_mean_bool = df_modl['residuals'].mean() > 0.1
        mean_result = {
            'Timepoint': 'All',
            'Assumption': 'Residual mean close to zero',
            'Metric': "Residual mean > 0.1",
            'Value': df_modl_mean_bool,
            'Comment': "✓ Residual mean close to zero" if not df_modl_mean_bool else "⚠ Residual mean not close to zero"
        }
        # Combinaison des résultats
        df_resi_stat_mean = pd.DataFrame([mean_result])
        df_resi_stat_norm = pd.DataFrame(normality_results)
        df_resi_stat_homo = pd.DataFrame([levene_result])
        df_resi_stat = pd.concat([df_resi_stat_mean, df_resi_stat_norm, df_resi_stat_homo], ignore_index=True)
        #
        df_resi_stat = df_resi_stat[['Timepoint', 'Assumption', 'Metric', 'Value', 'Effect Size', 'P-value', 'Assumption Met', 'Sample Size', 'Comment']]
        # !!! Variance Ratio : Un ratio > 2–3 suggère une hétéroscédasticité marquée, justifiant des modèles robustes. !!!
        if trac:
            print_yes(df_resi_stat, labl="df_resi_stat")  
        
        # Exit
        # ----    
        stat_tran_adat.mixd_mist_modl_resi_desc = df_resi_desc
        stat_tran_adat.mixd_mist_modl_resi_stat = df_resi_stat    

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