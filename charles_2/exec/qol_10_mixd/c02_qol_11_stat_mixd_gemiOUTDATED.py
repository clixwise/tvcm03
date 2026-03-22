    
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
            
from statsmodels.stats.proportion import proportion_confint

# ----
# Timepoint outcomes
# ----
#
# https://gemini.google.com/app/0f865d976b405499
#
def exec_stat_mixd_gemi(stat_tran_adat: StatTranQOL_11_mixd, df_lon1:DataFrame) -> None:
    from qol_10_mixd.c02_qol_11_stat_ import StatTranQOL_11_mixd

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, "df_fram")
        print_yes(df_fram[['patient_id','timepoint','VEINES_QOL_t']], "df_fram")

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Exec -1 : Modl
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # Data : df_fram
    # ====
    df_modl = df_fram.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
   
    # Modl : df_modl [Fit the linear mixed-effects model]
    # ==== 
    model = smf.mixedlm("VEINES_QOL_t ~ C(timepoint)", df_fram, groups=df_fram["patient_id"])
    result = model.fit(reml=True, method="powell")
    print(result.summary())
    
    
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Exec 0 : Compute MCID threshold
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    # ****
    # MCID (Step 1)
    # Calculate MCID using distribution-based methods using
    # variance components from the Longitudinal Mixed Model.
    # https://gemini.google.com/app/51e2b25e86000839
    # ****
    def calculate_distribution_based_mcid(df, lmm_result):
        
        # Exec
        # ----
        
        # 1. Baseline SD (Standard deviation of T0 scores)
        baseline_scores = df[df['timepoint'] == "T0"]["VEINES_QOL_t"]
        sd_baseline = baseline_scores.std()
        
        # 2. Extract Variance Components from the model
        # var_between: Variance of the random intercept (patient-level)
        var_between = lmm_result.cov_re.iloc[0, 0]
        # var_within: Residual variance (error/unexplained)
        var_within = lmm_result.scale
        
        # 3. Calculate ICC (Intraclass Correlation Coefficient)
        # This represents the reliability/consistency of the scores in your cohort
        icc = var_between / (var_between + var_within)
        
        # 4. Calculate SEM (Standard Error of Measurement)
        # Correct formula: SEM = SD_baseline * sqrt(1 - ICC)
        sem_icc = sd_baseline * np.sqrt(1 - icc)
        
        # 5. Smallest Detectable Change at the 95% confidence level (SDC_95) using the standard PROM formula:
        sdc_95 = 1.96 * np.sqrt(2) * sem_icc
        #
        mcid_dict = {
            '0.3_SD': 0.3 * sd_baseline,   # Small effect
            '0.5_SD': 0.5 * sd_baseline,   # Moderate effect (Commonly used MCID)
            'SEM': sem_icc,                # Threshold for "True" change (Measurement certainty)
            'SDC_95': sdc_95,              # Smallest Detectable Change at the 95% confidence level
            'baseline_SD': sd_baseline,
            'ICC_Reliability': icc
        }
        df_mcid = pd.DataFrame(list(mcid_dict.items()), columns=['Metric', 'Value'])
        #
        '''
        Metric            Value
        0           0.3_SD  1.05
        1           0.5_SD  1.76
        2              SEM  2.48
        3           SDC_95  6.87
        4      baseline_SD  3.51
        5  ICC_Reliability  0.50
        '''
        print_yes(df_mcid, labl="df_mcid")    

        # Exit
        # ----
        return df_mcid
    #
    df_mcid = calculate_distribution_based_mcid(df_fram, result)

    print_yes(df_mcid, labl="df_mcid") 
    # Setup Thresholds (Using your 0.5_SD result) for Steps 2 and 3
    # Usually, the standard MCID is : mcid_threshold = 3.0 ; here = 1.76   
    mcid_thre = df_mcid.loc[df_mcid['Metric'] == '0.5_SD', 'Value'].values[0]

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Exec 2 : Group-Level Analysis (The "Mean" Approach)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    # ****
    # MCID (Step 2) : Group-Level Analysis (The "Mean" Approach)
    # **** 
    
    # Extract coefficients from your LMM result object
    lmm_params = result.params
    t1_mean_gain = lmm_params["C(timepoint)[T.T1]"]
    t2_mean_gain = lmm_params["C(timepoint)[T.T2]"]

    group_results = {
    "T1_vs_T0": {"Gain": t1_mean_gain, "Clinically_Sig": t1_mean_gain >= mcid_thre},
    "T2_vs_T0": {"Gain": t2_mean_gain, "Clinically_Sig": t2_mean_gain >= mcid_thre}
    }
    df_mcid_grop = pd.DataFrame([
        {"Comparison": comp, "Gain": res["Gain"], "Clinically_Sig": res["Clinically_Sig"]}
        for comp, res in group_results.items()
    ])
    df_mcid_grop_name = df_mcid_grop.copy().add_prefix('mcid_grop_').rename(columns={'mcid_grop_Comparison': 'Comparison'}),
    '''
    Comparison  mcid_grop_Gain  mcid_grop_Clinically_Sig
    0  T1_vs_T0   5.0             True
    1  T2_vs_T0   7.5             True
    '''
    print_yes(df_mcid_grop, labl="df_mcid_grop")

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Exec 3 : Patient-Level Analysis (The "Responder" Approach)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
      
    # ****
    # MCID (Step 3) Patient-Level Analysis (The "Responder" Approach)
    # **** 
    # Pivot df_modl to calculate individual deltas
    df_pivot = df_modl.pivot(index='patient_id', columns='timepoint', values='VEINES_QOL_t')
    df_pivot['delta_T1'] = df_pivot['T1'] - df_pivot['T0']
    df_pivot['delta_T2'] = df_pivot['T2'] - df_pivot['T0']

    # Identify Responders (Delta >= MCID)
    df_pivot['is_responder_T1'] = df_pivot['delta_T1'] >= mcid_thre
    df_pivot['is_responder_T2'] = df_pivot['delta_T2'] >= mcid_thre

    responder_rate_t1 = df_pivot['is_responder_T1'].mean() * 100
    responder_rate_t2 = df_pivot['is_responder_T2'].mean() * 100

    df_mcid_pat1 = pd.DataFrame({
        "Comparison": ["T1_vs_T0", "T2_vs_T0"],
        "Group Mean Gain": [t1_mean_gain, t2_mean_gain],
        "Individual Responder Rate": [f"{responder_rate_t1:.1f}%", f"{responder_rate_t2:.1f}%"]
    })
    df_mcid_pat1_name = df_mcid_pat1.copy().add_prefix('mcid_pat1_').rename(columns={'mcid_pat1_Comparison': 'Comparison'}),
    '''
     Comparison    Group Mean Gain  Individual Responder Rate
    0  T1_vs_T0        5.0              100.0%
    1  T2_vs_T0        7.5              100.0%
    '''
    print_yes(df_mcid_pat1, labl="df_mcid_pat1")
    
    # ****
    # MCID (Step 3)
    # ****
    def _wilson_ci(successes, n, alpha=0.05):
        """Calculate Wilson score confidence interval for proportions."""
        if n == 0:
            return 0, 0
        
        p = successes / n
        z = stats.norm.ppf(1 - alpha/2)
        
        denominator = 1 + z**2/n
        centre = (p + z**2/(2*n)) / denominator
        adjustment = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator
        
        return max(0, centre - adjustment), min(1, centre + adjustment)
   
    def individual_responder_analysis(df, mcid_threshold, direction):
        """
        Calculate proportion of patients achieving MCID at individual level.
        
        Parameters:
        -----------
        mcid_threshold : float
            MCID threshold value
        direction : str
            'improvement' (positive change), 'deterioration' (negative change), or 'both'
        """
        # Pivot to wide format
        wide_df = df.pivot(index='patient_id', columns='timepoint', values='VEINES_QOL_t')
        
         # Get timepoint labels
        baseline_label = 'T0'
        timepoints = ['T0','T1','T2']
        timepoints.remove(baseline_label)
        
        responder_results = []
        
        for tp in timepoints:
            change = wide_df[tp] - wide_df[baseline_label]
            
            if direction == 'incr':
                responders = (change >= mcid_threshold).sum()
            elif direction == 'decr':
                responders = (change <= -mcid_threshold).sum()
            elif direction == 'both':
                responders = (abs(change) >= mcid_threshold).sum()
            
            total = len(change.dropna())
            proportion = responders / total if total > 0 else 0
            
            # 95% CI for proportion (Wilson score interval) IGNORE FOR SMALL SAMPLES
            # ci_lower, ci_upper = _wilson_ci(responders, total)


            # Recalculate CIs using the 'beta' method (Clopper-Pearson)
            # This ensures ci_upper will NEVER exceed 1.0 (100%)
            ci_lower, ci_upper = proportion_confint(
                count=responders, 
                nobs=total, 
                alpha=0.05, 
                method='beta'  # 'beta' is the Clopper-Pearson exact interval
            )

            
            responder_results.append({
                'Comparison': f'{tp}_vs_{baseline_label}',
                'evolution': direction,
                'n_total': total,
                'mcid_threshold': mcid_threshold,
                'direction': direction,
                'n_responders': responders,
                'proportion': proportion,
                'percentage': proportion * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100
            })
        
        df_resp_anal = pd.DataFrame(responder_results)
        '''
          comparison  n_responders  n_total  proportion  percentage  ci_lower  ci_upper  mcid_threshold direction
        0  T1_vs_T0   30            30       1.0         100.0       88.65     100.0     1.76            improvement
        1  T2_vs_T0   30            30       1.0         100.0       88.65     100.0     1.76            improvement
        '''
        return df_resp_anal

    df_resp_anal_incr = individual_responder_analysis(df_modl, mcid_thre, direction='incr')
    df_resp_anal_decr = individual_responder_analysis(df_modl, mcid_thre, direction='decr')
    #
    direction='incr' ; resp_anal= f'resp_anal_{direction}'
    df_resp_anal_incr_name = df_resp_anal_incr.copy().add_prefix(f'{resp_anal}_').rename(columns={f'{resp_anal}_Comparison': 'Comparison'}),
    print_yes(df_resp_anal_incr, labl="df_resp_anal_incr")
       
    # ----
    # Tabl : https://gemini.google.com/app/51e2b25e86000839
    # ----
    # Create a copy for the publication table
    pub_table = df_resp_anal_incr.copy()
    # 1. Format the Proportion/Percentage with CI: "100.0 (88.4–100.0)"
    pub_table['Responder Rate (95% CI)'] = pub_table.apply(
        lambda x: f"{x['percentage']:.1f}% ({x['ci_lower']:.1f}–{x['ci_upper']:.1f}%)", axis=1
    )
    # 2. Format the Count: "30/30"
    pub_table['n/N'] = pub_table.apply(
        lambda x: f"{x['n_responders']}/{x['n_total']}", axis=1
    )
    # 3. Select and Rename columns for the final report
    final_table = pub_table[[
        'Comparison', 
        'mcid_threshold', 
        'n/N', 
        'Responder Rate (95% CI)'
    ]].rename(columns={
        'Comparison': 'Time Point',
        'mcid_threshold': 'MCID Threshold'
    })
    print(final_table.to_string(index=False))
    
    # ----
    # Tabl (variant)
    # ----
    '''
    2. Responder Analysis (Subplot 2)
    This reflects the individual patient success rate.
    '''
    '''
    "95% confidence intervals for responder rates were calculated using the Clopper-Pearson exact method to account for the bounded nature of proportions."
    '''
    # Formatting Table 2: Responders
    tab2 = df_resp_anal_incr.copy()
    tab2['Responders, n (%)'] = tab2.apply(lambda x: f"{x['n_responders']} ({x['percentage']:.1f}%)", axis=1)
    tab2['95% CI (%)'] = tab2.apply(lambda x: f"{x['ci_lower']:.1f}–{x['ci_upper']:.1f}", axis=1)
    tab2 = tab2[['Comparison', 'n_total', 'Responders, n (%)', '95% CI (%)']].rename(columns={'n_total': 'Total N'})
    print (tab2)
        
    # ****
    # MCID (Step 3)
    # ****     
    # 2. Extract coefficients and p-values from the LMM result
    summary_df = result.params
    p_values = result.pvalues

    # 3. Focus on the timepoint effects
    # result.params typically looks like: Intercept, C(timepoint)[T.T1], C(timepoint)[T.T2]
    t1_delta = summary_df["C(timepoint)[T.T1]"]
    t2_delta = summary_df["C(timepoint)[T.T2]"]

    # 4. Create a logic check for Clinical Significance
    analysis = []
    for label, delta, p in [("T1_vs_T0", t1_delta, p_values[1]), ("T2_vs_T0", t2_delta, p_values[2])]:
        is_significant = p < 0.05
        is_clinically_meaningful = abs(delta) >= mcid_thre
        #
        analysis.append({
            "Comparison": label,
            "Mean Change": round(delta, 3),
            "P-Value": round(p, 4),
            "Statistically Significant": is_significant,
            "Clinically Meaningful (MCID)": is_clinically_meaningful
        })

    '''
     Comparison   Mean Change  P-Value  Statistically Significant  Clinically Meaningful (MCID)
    0  T1_vs_T0   5.0          0.0      True                       True
    1  T2_vs_T0   7.5          0.0      True                       True
    '''
    df_mcid_pat2 = pd.DataFrame(analysis)
    df_mcid_pat2_name = df_mcid_pat2.copy().add_prefix('mcid_pat2_').rename(columns={'mcid_pat2_Comparison': 'Comparison'}),
    print_yes(df_mcid_pat2, labl="df_mcid_pat2")
 
    # ****
    # Population changes
    # ****
    def extract_population_changes(result):
        """
        Extract population-level mean changes from LMM.
        """
        params = result.params
        conf_int = result.conf_int()
        pvalues = result.pvalues
        
        # Get timepoint labels
        baseline_label = 'T0'
        timepoints = ['T0','T1','T2']
        timepoints.remove(baseline_label)
        
        changes = []
        for tp in timepoints:
            # 'C(timepoint)[T.T1]'
            param_name = f"C(timepoint)[T.{tp}]"
            
            if param_name in params.index:
                change_dict = {
                    'Comparison': f'{tp}_vs_{baseline_label}',
                    'mean_change': params[param_name],
                    'ci_lower': conf_int.loc[param_name, 0],
                    'ci_upper': conf_int.loc[param_name, 1],
                    'se': result.bse[param_name],
                    'p_value': pvalues[param_name]
                }
                changes.append(change_dict)
        
        '''
          comparison  mean_change  ci_lower  ci_upper        se  p_value
        0   T1_vs_T0     5.001667  4.865893  5.137440  0.069273      0.0
        1   T2_vs_T0     7.502500  7.366727  7.638273  0.069273      0.0
        '''
        df_popu_delt = pd.DataFrame(changes)
        return df_popu_delt
    
    df_popu_delt = extract_population_changes(result)
    df_popu_delt_name = df_popu_delt.copy().add_prefix('popu_delt_').rename(columns={'popu_delt_Comparison': 'Comparison'}),
    print_yes (df_popu_delt, labl="df_popu_delt")
    
    # ----
    # Tabl
    # ----
    '''
    1. Population-Level Change (Subplot 1)
    This table focuses on the magnitude of change across the whole group.
    '''
    # Formatting Table 1: Mean Changes
    tab1 = df_popu_delt[['Comparison', 'mean_change', 'ci_lower', 'ci_upper']].copy()
    tab1['Mean Change (95% CI)'] = tab1.apply(
    lambda x: f"{x['mean_change']:+.2f} ({x['ci_lower']:+.2f} to {x['ci_upper']:+.2f})", axis=1
    )
    tab1 = tab1[['Comparison', 'Mean Change (95% CI)']].rename(columns={'Comparison': 'Comparison'})
    print (tab1)
    
    # ****
    # Effect size
    # ****
    def calculate_effect_sizes(df):
        """
        Calculate Cohen's d effect sizes for each timepoint comparison.
        """
        baseline_scores = df[df['timepoint'] == "T0"]["VEINES_QOL_t"]
        baseline_sd = baseline_scores.std()
        
        # Get timepoint labels
        baseline_label = 'T0'
        timepoints = ['T0','T1','T2']
        timepoints.remove(baseline_label)
        
        effect_sizes = []
        for tp in timepoints:
            tp_scores = df[df['timepoint'] == tp]['VEINES_QOL_t']
            
            # Pooled SD
            pooled_sd = np.sqrt((baseline_sd**2 + tp_scores.std()**2) / 2)
            
            # Mean difference
            mean_diff = tp_scores.mean() - baseline_scores.mean()
            
            # Cohen's d
            cohens_d = mean_diff / pooled_sd
            
            effect_sizes.append({
                'Comparison': f'{tp}_vs_{baseline_label}',
                'cohens_d': cohens_d,
                'interpretation': _interpret_cohens_d(cohens_d)
            })
        '''
          comparison  cohens_d interpretation
        0  T1_vs_T0   1.35      large
        1  T2_vs_T0   1.98      large
        '''
        df_effe_size = pd.DataFrame(effect_sizes)
        return df_effe_size
    
    def _interpret_cohens_d(d):
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
        
    df_effe_size = calculate_effect_sizes(df_modl)
    df_effe_size_name = df_effe_size.copy().add_prefix('effe_size_').rename(columns={'effe_size_Comparison': 'Comparison'}),
    print_yes (df_effe_size, labl="df_effe_size")
    
    # ----
    # Tabl
    # ----
    # ====
    '''
    3. Effect Size Magnitude (Subplot 3)
    Journals often require standardized effect sizes to compare your results to other QoL studies.
    '''
    # Formatting Table 3: Effect Sizes
    tab3 = df_effe_size[['Comparison', 'cohens_d', 'interpretation']].copy()
    tab3.columns = ['Comparison', "Cohen's d", 'Magnitude']
    print (tab3) 

    # ****
    # Results
    # ****
    if trac:
        print_yes(df_mcid_grop, labl="df_mcid_grop")
        print_yes(df_mcid_pat1, labl="df_mcid_pat1")
        print_yes(df_mcid_pat2, labl="df_mcid_pat2")
        print_yes(df_resp_anal_incr, labl="df_resp_anal_incr")
        print_yes(df_resp_anal_decr, labl="df_resp_anal_decr")
        print_yes(df_popu_delt, labl="df_popu_delt")
        print_yes(df_effe_size, labl="df_effe_size")
        
    stat_tran_adat.gemi_mixd_mcid = df_mcid
    stat_tran_adat.gemi_mixd_mcid_grop = df_mcid_grop
    stat_tran_adat.gemi_mixd_mcid_pat1 = df_mcid_pat1
    stat_tran_adat.gemi_mixd_mcid_pat2 = df_mcid_pat2
    stat_tran_adat.gemi_mixd_mcid_anal = df_resp_anal_incr
    stat_tran_adat.gemi_mixd_mcid_popu_delt = df_popu_delt
    stat_tran_adat.gemi_mixd_mcid_effe_size = df_effe_size
    
    df_popu_delt_ = df_popu_delt.copy()
    df_effe_size_ = df_effe_size.copy()
    df_resp_anal_incr_ = df_resp_anal_incr.copy()
    df_resp_anal_incr_.rename(columns={'ci_lower': 'responder_ci_lower','ci_upper': 'responder_ci_upper'}, inplace=True),
    df_resp_anal_incr_ = df_resp_anal_incr_[['Comparison', 'proportion', 'responder_ci_lower', 'responder_ci_upper', 'n_responders', 'n_total']]
    
    df_full = df_popu_delt_.merge(df_effe_size_, on='Comparison')
    df_full = df_full.merge(df_resp_anal_incr_, on='Comparison')
    # Add interpretation columns
    df_full['statistically_significant'] = df_full['p_value'] < 0.05
    df_full['clinically_meaningful'] = df_full['mean_change'].abs() >= mcid_thre
    df_full['meets_both_criteria'] = (df_full['statistically_significant'] & df_full['clinically_meaningful'])
    print_yes(df_full, labl="df_full")
    '''
    4. Significance Summary (Subplot 4)
    This is your "Significance Matrix" in table form, useful for the Appendix or a Summary Results section.
    '''
    # Formatting Table 4: Summary Matrix
    tab4 = df_full[['Comparison', 'statistically_significant', 'clinically_meaningful', 'meets_both_criteria']].copy()
    tab4 = tab4.replace({True: 'Yes', False: 'No'})
    tab4.columns = ['Comparison', 'Stat. Sig (p<0.05)', 'Clin. Sig (≥MCID)', 'Combined Success']
    print (tab4)

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