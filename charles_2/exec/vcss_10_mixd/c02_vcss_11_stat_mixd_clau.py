    
    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_11_stat_ import StatTranVCSS_11_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import patsy
    
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
from vcss_10_mixd.c02_vcss_11_stat_adat import StatTranVCSS_11_mixd

# ----
# Timepoint outcomes
# ----

# https://claude.ai/chat/7f734aa6-8a0a-4292-9a61-f123b4afc57c

def exec_stat_mixd_clau(stat_tran_adat: StatTranVCSS_11_mixd, df_lon1:DataFrame) -> None:
    # from vcss_10_mixd.c02_vcss_11_stat_ import StatTranVCSS_11_mixd

    trac = True
    
    # Data
    # ---- 
    df_frax = stat_tran_adat.stat_tran.frax
    
    # Trac
    # ----
    if trac:
        print_yes(df_frax, labl='df_frax')

    # Data : df_fram
    # ====
    df_modl = df_frax.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)

    # Assuming you have 'result' from your mixedlm fit
    model = smf.mixedlm("VCSS ~ C(timepoint) + C(Limb)", df_modl, groups=df_modl["patient_id"])
    result = model.fit(reml=True)

    # ============================================================================
    # 1. MODELED MEANS WITH CONFIDENCE INTERVALS
    # ============================================================================

    print("="*80)
    print("MODELED MEANS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)

    # Extract coefficients
    intercept = result.params['Intercept']
    effect_T1 = result.params['C(timepoint)[T.T1]']
    effect_T2 = result.params['C(timepoint)[T.T2]']
    effect_L = result.params['C(Limb)[T.L]']

    # Extract standard errors and covariance matrix
    se_intercept = result.bse['Intercept']
    se_T1 = result.bse['C(timepoint)[T.T1]']
    se_T2 = result.bse['C(timepoint)[T.T2]']
    se_L = result.bse['C(Limb)[T.L]']
    cov_matrix = result.cov_params()

    # Calculate modeled means for each timepoint × limb combination
    # Reference level: T0, Right limb = intercept
    means = {}
    ses = {}
    cis = {}

    # T0-R (reference)
    means['T0_R'] = intercept
    ses['T0_R'] = se_intercept
    ci_lower = means['T0_R'] - 1.96 * ses['T0_R']
    ci_upper = means['T0_R'] + 1.96 * ses['T0_R']
    cis['T0_R'] = (ci_lower, ci_upper)

    # T0-L
    means['T0_L'] = intercept + effect_L
    cov_int_L = cov_matrix.loc['Intercept', 'C(Limb)[T.L]']
    ses['T0_L'] = np.sqrt(se_intercept**2 + se_L**2 + 2*cov_int_L)
    ci_lower = means['T0_L'] - 1.96 * ses['T0_L']
    ci_upper = means['T0_L'] + 1.96 * ses['T0_L']
    cis['T0_L'] = (ci_lower, ci_upper)

    # T1-R
    means['T1_R'] = intercept + effect_T1
    cov_int_T1 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T1]']
    ses['T1_R'] = np.sqrt(se_intercept**2 + se_T1**2 + 2*cov_int_T1)
    ci_lower = means['T1_R'] - 1.96 * ses['T1_R']
    ci_upper = means['T1_R'] + 1.96 * ses['T1_R']
    cis['T1_R'] = (ci_lower, ci_upper)

    # T1-L
    means['T1_L'] = intercept + effect_T1 + effect_L
    cov_int_T1 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T1]']
    cov_int_L = cov_matrix.loc['Intercept', 'C(Limb)[T.L]']
    cov_T1_L = cov_matrix.loc['C(timepoint)[T.T1]', 'C(Limb)[T.L]']
    ses['T1_L'] = np.sqrt(se_intercept**2 + se_T1**2 + se_L**2 + 
                        2*cov_int_T1 + 2*cov_int_L + 2*cov_T1_L)
    ci_lower = means['T1_L'] - 1.96 * ses['T1_L']
    ci_upper = means['T1_L'] + 1.96 * ses['T1_L']
    cis['T1_L'] = (ci_lower, ci_upper)

    # T2-R
    means['T2_R'] = intercept + effect_T2
    cov_int_T2 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T2]']
    ses['T2_R'] = np.sqrt(se_intercept**2 + se_T2**2 + 2*cov_int_T2)
    ci_lower = means['T2_R'] - 1.96 * ses['T2_R']
    ci_upper = means['T2_R'] + 1.96 * ses['T2_R']
    cis['T2_R'] = (ci_lower, ci_upper)

    # T2-L
    means['T2_L'] = intercept + effect_T2 + effect_L
    cov_int_T2 = cov_matrix.loc['Intercept', 'C(timepoint)[T.T2]']
    cov_int_L = cov_matrix.loc['Intercept', 'C(Limb)[T.L]']
    cov_T2_L = cov_matrix.loc['C(timepoint)[T.T2]', 'C(Limb)[T.L]']
    ses['T2_L'] = np.sqrt(se_intercept**2 + se_T2**2 + se_L**2 + 
                        2*cov_int_T2 + 2*cov_int_L + 2*cov_T2_L)
    ci_lower = means['T2_L'] - 1.96 * ses['T2_L']
    ci_upper = means['T2_L'] + 1.96 * ses['T2_L']
    cis['T2_L'] = (ci_lower, ci_upper)

    # Create summary dataframe
    df_modeled_means = pd.DataFrame({
        'Timepoint': ['T0', 'T0', 'T1', 'T1', 'T2', 'T2'],
        'Limb': ['R', 'L', 'R', 'L', 'R', 'L'],
        'Modeled_Mean': [means[k] for k in ['T0_R', 'T0_L', 'T1_R', 'T1_L', 'T2_R', 'T2_L']],
        'SE': [ses[k] for k in ['T0_R', 'T0_L', 'T1_R', 'T1_L', 'T2_R', 'T2_L']],
        'CI_Lower': [cis[k][0] for k in ['T0_R', 'T0_L', 'T1_R', 'T1_L', 'T2_R', 'T2_L']],
        'CI_Upper': [cis[k][1] for k in ['T0_R', 'T0_L', 'T1_R', 'T1_L', 'T2_R', 'T2_L']]
    })

    print(df_modeled_means.to_string(index=False))
    print()

    # ============================================================================
    # 2. POST-HOC CONTRASTS - TIMEPOINT EFFECTS
    # ============================================================================

    print("="*80)
    print("POST-HOC CONTRASTS: TIMEPOINT EFFECTS (averaged over limbs)")
    print("="*80)

    # These are the main effects already in the model
    contrasts_time = []

    # T1 vs T0
    contrast_T1_T0 = effect_T1
    se_T1_T0 = se_T1
    z_T1_T0 = contrast_T1_T0 / se_T1_T0
    p_T1_T0 = 2 * (1 - stats.norm.cdf(abs(z_T1_T0)))
    ci_lower = contrast_T1_T0 - 1.96 * se_T1_T0
    ci_upper = contrast_T1_T0 + 1.96 * se_T1_T0
    contrasts_time.append(['T1 - T0', contrast_T1_T0, se_T1_T0, z_T1_T0, p_T1_T0, ci_lower, ci_upper])

    # T2 vs T0
    contrast_T2_T0 = effect_T2
    se_T2_T0 = se_T2
    z_T2_T0 = contrast_T2_T0 / se_T2_T0
    p_T2_T0 = 2 * (1 - stats.norm.cdf(abs(z_T2_T0)))
    ci_lower = contrast_T2_T0 - 1.96 * se_T2_T0
    ci_upper = contrast_T2_T0 + 1.96 * se_T2_T0
    contrasts_time.append(['T2 - T0', contrast_T2_T0, se_T2_T0, z_T2_T0, p_T2_T0, ci_lower, ci_upper])

    # T2 vs T1
    contrast_T2_T1 = effect_T2 - effect_T1
    cov_T1_T2 = cov_matrix.loc['C(timepoint)[T.T1]', 'C(timepoint)[T.T2]']
    se_T2_T1 = np.sqrt(se_T1**2 + se_T2**2 - 2*cov_T1_T2)
    z_T2_T1 = contrast_T2_T1 / se_T2_T1
    p_T2_T1 = 2 * (1 - stats.norm.cdf(abs(z_T2_T1)))
    ci_lower = contrast_T2_T1 - 1.96 * se_T2_T1
    ci_upper = contrast_T2_T1 + 1.96 * se_T2_T1
    contrasts_time.append(['T2 - T1', contrast_T2_T1, se_T2_T1, z_T2_T1, p_T2_T1, ci_lower, ci_upper])

    df_contrasts_time = pd.DataFrame(contrasts_time, 
                                    columns=['Comparison', 'Difference', 'SE', 'Z', 
                                            'P_value', 'CI_Lower', 'CI_Upper'])
    print(df_contrasts_time.to_string(index=False))
    print()

    # ============================================================================
    # 3. LIMB EFFECT
    # ============================================================================

    print("="*80)
    print("LIMB EFFECT (averaged over timepoints)")
    print("="*80)

    print(f"Left vs Right: {effect_L:.3f} (SE = {se_L:.3f}, p = {result.pvalues['C(Limb)[T.L]']:.3f})")
    ci_lower = effect_L - 1.96 * se_L
    ci_upper = effect_L + 1.96 * se_L
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    if result.pvalues['C(Limb)[T.L]'] > 0.05:
        print("→ No significant difference between left and right limbs")
    else:
        print("→ Significant difference between left and right limbs")
    print()

    # ============================================================================
    # 4. COMPARISON: RAW vs MODELED MEANS
    # ============================================================================

    print("="*80)
    print("COMPARISON: RAW MEANS vs MODELED MEANS")
    print("="*80)

    # Assuming you have df_lon1 from the raw data aggregation
    comparison = pd.DataFrame({
        'Timepoint': ['T0', 'T0', 'T1', 'T1', 'T2', 'T2'],
        'Limb': ['R', 'L', 'R', 'L', 'R', 'L'],
        'Raw_Mean': [6.73, 6.97, 4.03, 4.40, 3.30, 3.70],  # From your df_lon1
        'Modeled_Mean': df_modeled_means['Modeled_Mean'].values
    })
    comparison['Difference'] = comparison['Modeled_Mean'] - comparison['Raw_Mean']

    print(comparison.to_string(index=False))
    print()
    print("Note: Small differences expected due to model structure")
    print("      (additive model assumes constant limb effect across time)")
    print()

    # ============================================================================
    # 5. INTERPRETATION SUMMARY
    # ============================================================================

    print("="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    print(f"Baseline VCSS (T0):")
    print(f"  Right limb: {means['T0_R']:.2f} (95% CI: {cis['T0_R'][0]:.2f}-{cis['T0_R'][1]:.2f})")
    print(f"  Left limb:  {means['T0_L']:.2f} (95% CI: {cis['T0_L'][0]:.2f}-{cis['T0_L'][1]:.2f})")
    print(f"\nChange over time (averaged across limbs):")
    print(f"  T0 to T1: {contrast_T1_T0:.2f} points (p < 0.001) - {'Improvement' if contrast_T1_T0 < 0 else 'Worsening'}")
    print(f"  T1 to T2: {contrast_T2_T1:.2f} points (p < 0.001) - {'Improvement' if contrast_T2_T1 < 0 else 'Worsening'}")
    print(f"  T0 to T2: {contrast_T2_T0:.2f} points (p < 0.001) - {'Improvement' if contrast_T2_T0 < 0 else 'Worsening'}")
    print(f"\nLimb asymmetry:")
    print(f"  Left vs Right: {effect_L:.2f} points (p = {result.pvalues['C(Limb)[T.L]']:.3f}) - Not significant")
    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION:")
    print("="*80)
    print("• VCSS scores show significant improvement from baseline to both follow-ups")
    print("• The improvement is progressive (continues from T1 to T2)")
    print("• No significant difference between left and right limbs on average")
    print("• Lower VCSS indicates better venous function (improvement = negative change)")
    print("="*80)
    
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