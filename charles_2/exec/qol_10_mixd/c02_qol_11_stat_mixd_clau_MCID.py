"""
Comprehensive MCID Analysis for VEINES-QOL
Publication-ready analysis with multiple MCID approaches
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class VEINESMCIDAnalysis:
    """
    Comprehensive analysis framework for establishing and evaluating MCID
    in VEINES-QOL scores across multiple timepoints.
    """
    
    def __init__(self, df, score_col='VEINES_QOL_t', time_col='timepoint', 
                 patient_col='patient_id', baseline_label='T0'):
        """
        Initialize the analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Long-format dataframe with columns for patient_id, timepoint, and score
        score_col : str
            Name of the QOL score column
        time_col : str
            Name of the timepoint column
        patient_col : str
            Name of the patient ID column
        baseline_label : str
            Label for baseline timepoint
        """
        self.df = df.copy()
        self.score_col = score_col
        self.time_col = time_col
        self.patient_col = patient_col
        self.baseline_label = baseline_label
        self.results = {}
 
    def calculate_distribution_based_mcid(self, lmm_result):
        """
        Calculate MCID using distribution-based methods correctly using
        variance components from the longitudinal Mixed Model.
        
        Args:
            lmm_result: The fitted MixedLM result object (from T0, T1, T2 data).
        """
        # 1. Baseline SD (Standard deviation of T0 scores)
        baseline_scores = self.df[self.df[self.time_col] == self.baseline_label][self.score_col]
        sd_baseline = baseline_scores.std()
        
        # 2. Extract Variance Components from the LONGITUDINAL model
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
        
        mcid_dict = {
            '0.3_SD': 0.3 * sd_baseline,   # Small effect
            '0.5_SD': 0.5 * sd_baseline,   # Moderate effect (Commonly used MCID)
            'SEM': sem_icc,                # Threshold for "True" change (Measurement certainty)
            'baseline_SD': sd_baseline,
            'ICC_Reliability': icc
        }
        
        self.results['distribution_mcid'] = mcid_dict
        df_mcid = pd.DataFrame(list(mcid_dict.items()), columns=['Metric', 'Value'])
        #
        print_yes(df_mcid, labl="df_mcid") 
        return mcid_dict

    def fit_mixed_model(self):
        """
        Fit linear mixed model with timepoint as fixed effect and patient as random effect.
        """
        formula = f"{self.score_col} ~ C({self.time_col})"
        model = smf.mixedlm(formula, self.df, groups=self.df[self.patient_col])
        result = model.fit(reml=True, method="powell")
        
        self.results['lmm_result'] = result
        return result
    
    def extract_population_changes(self, result):
        """
        Extract population-level mean changes from LMM.
        """
        params = result.params
        conf_int = result.conf_int()
        pvalues = result.pvalues
        
        # Get timepoint labels
        timepoints = sorted(self.df[self.time_col].unique())
        timepoints.remove(self.baseline_label)
        
        changes = []
        for tp in timepoints:
            param_name = f"C({self.time_col})[T.{tp}]"
            
            if param_name in params.index:
                change_dict = {
                    'comparison': f'{self.baseline_label} to {tp}',
                    'mean_change': params[param_name],
                    'ci_lower': conf_int.loc[param_name, 0],
                    'ci_upper': conf_int.loc[param_name, 1],
                    'se': result.bse[param_name],
                    'p_value': pvalues[param_name]
                }
                changes.append(change_dict)
        
        self.results['population_changes'] = pd.DataFrame(changes)
        df_popu_changes = pd.DataFrame(changes)
        print (df_popu_changes)
        return pd.DataFrame(changes)
    
    def calculate_effect_sizes(self):
        """
        Calculate Cohen's d effect sizes for each timepoint comparison.
        """
        baseline_scores = self.df[self.df[self.time_col] == self.baseline_label][self.score_col]
        baseline_sd = baseline_scores.std()
        
        timepoints = sorted(self.df[self.time_col].unique())
        timepoints.remove(self.baseline_label)
        
        effect_sizes = []
        for tp in timepoints:
            tp_scores = self.df[self.df[self.time_col] == tp][self.score_col]
            
            # Pooled SD
            pooled_sd = np.sqrt((baseline_sd**2 + tp_scores.std()**2) / 2)
            
            # Mean difference
            mean_diff = tp_scores.mean() - baseline_scores.mean()
            
            # Cohen's d
            cohens_d = mean_diff / pooled_sd
            
            effect_sizes.append({
                'comparison': f'{self.baseline_label} to {tp}',
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            })
        
        self.results['effect_sizes'] = pd.DataFrame(effect_sizes)
        df_effect_sizes  = pd.DataFrame(effect_sizes)
        print_yes(df_effect_sizes)
        return df_effect_sizes
    
    def _interpret_cohens_d(self, d):
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
    
    def individual_responder_analysis(self, mcid_threshold, direction='improvement'):
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
        wide_df = self.df.pivot(index=self.patient_col, 
                                columns=self.time_col, 
                                values=self.score_col)
        
        timepoints = sorted(self.df[self.time_col].unique())
        timepoints.remove(self.baseline_label)
        
        responder_results = []
        
        for tp in timepoints:
            change = wide_df[tp] - wide_df[self.baseline_label]
            
            if direction == 'improvement':
                responders = (change >= mcid_threshold).sum()
            elif direction == 'deterioration':
                responders = (change <= -mcid_threshold).sum()
            elif direction == 'both':
                responders = (abs(change) >= mcid_threshold).sum()
            
            total = len(change.dropna())
            proportion = responders / total if total > 0 else 0
            
            # 95% CI for proportion (Wilson score interval)
            ci_lower, ci_upper = self._wilson_ci(responders, total)
            
            responder_results.append({
                'comparison': f'{self.baseline_label} to {tp}',
                'n_responders': responders,
                'n_total': total,
                'proportion': proportion,
                'percentage': proportion * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'mcid_threshold': mcid_threshold,
                'direction': direction
            })
        
        self.results['responder_analysis'] = pd.DataFrame(responder_results)
        print_yes (self.results['responder_analysis'], labl='responder_analysis')
        return pd.DataFrame(responder_results)
    
    def _wilson_ci(self, successes, n, alpha=0.05):
        """Calculate Wilson score confidence interval for proportions."""
        if n == 0:
            return 0, 0
        
        p = successes / n
        z = stats.norm.ppf(1 - alpha/2)
        
        denominator = 1 + z**2/n
        centre = (p + z**2/(2*n)) / denominator
        adjustment = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator
        
        return max(0, centre - adjustment), min(1, centre + adjustment)
    
    def comprehensive_mcid_table(self, mcid_threshold=None, use_half_sd=True):
        """
        Create comprehensive table combining statistical and clinical significance.
        
        Parameters:
        -----------
        mcid_threshold : float or None
            If None, uses 0.5×SD from distribution-based methods
        use_half_sd : bool
            If True and mcid_threshold is None, uses 0.5×SD as threshold
        """
        if mcid_threshold is None:
            if 'distribution_mcid' not in self.results:
                self.calculate_distribution_based_mcid()
            mcid_threshold = self.results['distribution_mcid']['0.5_SD'] if use_half_sd else self.results['distribution_mcid']['SEM']
        
        # Get population changes
        if 'population_changes' not in self.results:
            result = self.fit_mixed_model()
            self.extract_population_changes(result)
        
        # Get effect sizes
        if 'effect_sizes' not in self.results:
            self.calculate_effect_sizes()
        
        # Get responder analysis
        self.individual_responder_analysis(mcid_threshold, direction='improvement')
        
        # Merge all results
        pop_df = self.results['population_changes'].copy()
        es_df = self.results['effect_sizes'].copy()
        resp_df = self.results['responder_analysis'].copy()
        
        # Rename responder analysis CI columns to avoid conflict
        resp_df = resp_df.rename(columns={
            'ci_lower': 'responder_ci_lower',
            'ci_upper': 'responder_ci_upper'
        })
        
        comprehensive = pop_df.merge(es_df, on='comparison')
        comprehensive = comprehensive.merge(
            resp_df[['comparison', 'proportion', 'responder_ci_lower', 'responder_ci_upper', 'n_responders', 'n_total']], 
            on='comparison'
        )
        print_yes (comprehensive, labl='comprehensive')
        
        # Add interpretation columns
        comprehensive['statistically_significant'] = comprehensive['p_value'] < 0.05
        comprehensive['clinically_meaningful'] = comprehensive['mean_change'].abs() >= mcid_threshold
        comprehensive['meets_both_criteria'] = (comprehensive['statistically_significant'] & 
                                               comprehensive['clinically_meaningful'])
        
        self.results['comprehensive_table'] = comprehensive
        return comprehensive
    
    def generate_publication_summary(self, mcid_threshold=None):
        """
        Generate formatted text summary suitable for publication Methods and Results sections.
        """
        if mcid_threshold is None:
            if 'distribution_mcid' not in self.results:
                self.calculate_distribution_based_mcid()
            mcid_threshold = self.results['distribution_mcid']['0.5_SD']
        
        # Ensure all analyses are run
        self.comprehensive_mcid_table(mcid_threshold)
        
        summary = []
        summary.append("=" * 80)
        summary.append("MINIMAL CLINICALLY IMPORTANT DIFFERENCE (MCID) ANALYSIS")
        summary.append("=" * 80)
        
        # Distribution-based MCID
        summary.append("\n1. MCID DETERMINATION (Distribution-Based Methods)")
        summary.append("-" * 80)
        dist = self.results['distribution_mcid']
        summary.append(f"Baseline SD: {dist['baseline_SD']:.2f}")
        summary.append(f"Intraclass Correlation (ICC): {dist['ICC_Reliability']:.3f}")
        summary.append(f"\nProposed MCID thresholds:")
        summary.append(f"  • 0.5 × SD (primary): {dist['0.5_SD']:.2f} points")
        summary.append(f"  • 0.3 × SD (small effect): {dist['0.3_SD']:.2f} points")
        summary.append(f"  • SEM-based: {dist['SEM']:.2f} points")
        summary.append(f"\nSelected MCID threshold: {mcid_threshold:.2f} points")
        
        # Population-level changes
        summary.append("\n2. POPULATION-LEVEL CHANGES (Linear Mixed Model)")
        summary.append("-" * 80)
        comp_table = self.results['comprehensive_table']
        
        for _, row in comp_table.iterrows():
            summary.append(f"\n{row['comparison']}:")
            summary.append(f"  Mean change: {row['mean_change']:.2f} (95% CI: {row['ci_lower']:.2f} to {row['ci_upper']:.2f})")
            summary.append(f"  p-value: {row['p_value']:.4f}")
            summary.append(f"  Cohen's d: {row['cohens_d']:.2f} ({row['interpretation']})")
            summary.append(f"  Statistical significance: {'Yes' if row['statistically_significant'] else 'No'}")
            summary.append(f"  Clinical meaningfulness (≥{mcid_threshold:.2f}): {'Yes' if row['clinically_meaningful'] else 'No'}")
        
        # Individual responder analysis
        summary.append("\n3. INDIVIDUAL RESPONDER ANALYSIS")
        summary.append("-" * 80)
        for _, row in comp_table.iterrows():
            summary.append(f"\n{row['comparison']}:")
            summary.append(f"  Patients achieving MCID: {row['n_responders']}/{row['n_total']} ({row['proportion']*100:.1f}%)")
            summary.append(f"  95% CI: {row['responder_ci_lower']:.1f}% to {row['responder_ci_upper']:.1f}%")
        
        # Summary interpretation
        summary.append("\n4. CLINICAL INTERPRETATION")
        summary.append("-" * 80)
        
        for _, row in comp_table.iterrows():
            if row['meets_both_criteria']:
                interpretation = f"The change from {row['comparison']} demonstrates both statistical significance (p={row['p_value']:.4f}) and clinical meaningfulness (mean change {row['mean_change']:.2f} exceeds MCID threshold of {mcid_threshold:.2f}). Individual-level analysis shows {row['proportion']*100:.1f}% of patients achieved clinically meaningful improvement."
            elif row['statistically_significant'] and not row['clinically_meaningful']:
                interpretation = f"The change from {row['comparison']} is statistically significant (p={row['p_value']:.4f}) but does not meet the threshold for clinical meaningfulness (mean change {row['mean_change']:.2f} < MCID {mcid_threshold:.2f})."
            elif row['clinically_meaningful'] and not row['statistically_significant']:
                interpretation = f"The change from {row['comparison']} exceeds the MCID threshold ({row['mean_change']:.2f} ≥ {mcid_threshold:.2f}) but lacks statistical significance (p={row['p_value']:.4f}), possibly due to sample size limitations."
            else:
                interpretation = f"The change from {row['comparison']} demonstrates neither statistical significance (p={row['p_value']:.4f}) nor clinical meaningfulness (mean change {row['mean_change']:.2f} < MCID {mcid_threshold:.2f})."
            
            summary.append(f"\n{row['comparison']}: {interpretation}")
        
        summary.append("\n" + "=" * 80)
        
        return "\n".join(summary)
    
    def plot_mcid_analysis(self, mcid_threshold=None, figsize=(14, 10)):
        """
        Create comprehensive visualization of MCID analysis.
        """
        if mcid_threshold is None:
            if 'distribution_mcid' not in self.results:
                self.calculate_distribution_based_mcid()
            mcid_threshold = self.results['distribution_mcid']['0.5_SD']
        
        # Ensure comprehensive table exists
        if 'comprehensive_table' not in self.results:
            self.comprehensive_mcid_table(mcid_threshold)
        
        comp_table = self.results['comprehensive_table']
        print_yes (comp_table, labl="comp_table")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        verti = True
        
        if verti:
            # ====
            '''
            1. Population-Level Change (Subplot 1)
            This table focuses on the magnitude of change across the whole group.
            '''
            # Formatting Table 1: Mean Changes
            tab1 = comp_table[['comparison', 'mean_change', 'ci_lower', 'ci_upper']].copy()
            tab1['Mean Change (95% CI)'] = tab1.apply(
                lambda x: f"{x['mean_change']:+.2f} ({x['ci_lower']:+.2f} to {x['ci_upper']:+.2f})", axis=1
            )
            tab1 = tab1[['comparison', 'Mean Change (95% CI)']].rename(columns={'comparison': 'Comparison'})
            print (tab1)
            # ====
            ax1 = axes[0, 0]
            comparisons = comp_table['comparison']
            means = comp_table['mean_change']
            ci_lower = comp_table['ci_lower']
            ci_upper = comp_table['ci_upper']

            colors = ['green' if x else 'red' for x in comp_table['meets_both_criteria']]
            x_pos = np.arange(len(comparisons))

            # 1. Change barh to bar (x_pos first, then means)
            ax1.bar(x_pos, means, color=colors, alpha=0.6)

            # 2. Update errorbar (swap means and x_pos)
            # Note: xerr becomes yerr
            ax1.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means],
                        fmt='none', ecolor='black', capsize=5)

            # 3. Change axvline (vertical line) to axhline (horizontal line)
            ax1.axhline(y=mcid_threshold, color='blue', linestyle='--', linewidth=2, label=f'MCID = {mcid_threshold:.2f}')
            ax1.axhline(y=-mcid_threshold, color='blue', linestyle='--', linewidth=2)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # 4. Update ticks to the x-axis
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(comparisons)

            # 5. Swap labels
            ax1.set_ylabel('Mean Change (95% CI)', fontsize=11)
            ax1.set_xlabel('Comparisons', fontsize=11)
            ax1.set_title('Population-Level Changes vs MCID Threshold', fontsize=12, fontweight='bold')

            ax1.legend()
            ax1.grid(axis='y', alpha=0.3) # Grid on y-axis is more helpful for vertical bars
            # plt.show()
        else:    
            # Plot 1: Mean changes with confidence intervals and MCID threshold
            ax1 = axes[0, 0]
            comparisons = comp_table['comparison']
            means = comp_table['mean_change']
            ci_lower = comp_table['ci_lower']
            ci_upper = comp_table['ci_upper']
            print_yes (comp_table[['comparison', 'mean_change', 'ci_lower', 'ci_upper' ]])
            colors = ['green' if x else 'red' for x in comp_table['meets_both_criteria']]
            
            y_pos = np.arange(len(comparisons))
            ax1.barh(y_pos, means, color=colors, alpha=0.6)
            ax1.errorbar(means, y_pos, xerr=[means - ci_lower, ci_upper - means],
                        fmt='none', ecolor='black', capsize=5)
            ax1.axvline(x=mcid_threshold, color='blue', linestyle='--', linewidth=2, label=f'MCID = {mcid_threshold:.2f}')
            ax1.axvline(x=-mcid_threshold, color='blue', linestyle='--', linewidth=2)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(comparisons)
            ax1.set_xlabel('Mean Change (95% CI)', fontsize=11)
            ax1.set_title('Population-Level Changes vs MCID Threshold', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Individual responder rates
        new = True
        if new:
            # ====
            '''
            2. Responder Analysis (Subplot 2)
            This reflects the individual patient success rate.
            '''
            # Formatting Table 2: Responders
            '''
            tab2 = df_resp_anal_incr.copy()
            tab2['Responders, n (%)'] = tab2.apply(lambda x: f"{x['n_responders']} ({x['percentage']:.1f}%)", axis=1)
            tab2['95% CI (%)'] = tab2.apply(lambda x: f"{x['ci_lower']:.1f}–{x['ci_upper']:.1f}", axis=1)
            tab2 = tab2[['Comparison', 'n_total', 'Responders, n (%)', '95% CI (%)']].rename(columns={'n_total': 'Total N'})
            print (tab2)
            '''
            # ====
            ax2 = axes[0, 1]
            # Convert to percentages
            proportions = comp_table['proportion'] * 100
            ci_low_resp = comp_table['responder_ci_lower']
            ci_up_resp = comp_table['responder_ci_upper']

            # --- THE FIX: Cap bounds at logical limits ---
            ci_up_capped = ci_up_resp.clip(upper=100)
            ci_low_capped = ci_low_resp.clip(lower=0)

            # Calculate distances from the point estimate
            lower_err = (proportions - ci_low_capped).clip(lower=0)
            upper_err = (ci_up_capped - proportions).clip(lower=0)

            x_pos = np.arange(len(comp_table['comparison']))

            # 1. Bars
            ax2.bar(x_pos, proportions, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.8)

            # 2. Error Bars (Now strictly bounded at 100%)
            # Ensure distances are non-negative and capped logically
            '''
            "95% confidence intervals for responder rates were calculated using the Clopper-Pearson exact method to account for the bounded nature of proportions."
            '''
            proportions = comp_table['proportion'] * 100
            lower_dist = (proportions - comp_table['responder_ci_lower']).clip(lower=0)
            upper_dist = (comp_table['responder_ci_upper'] - proportions).clip(lower=0)

            # This will now run without the 'yerr' ValueError
            ax2.errorbar(x_pos, proportions, yerr=[lower_dist, upper_dist],
                        fmt='none', ecolor='black', capsize=5, elinewidth=1)

            # 3. Smart Labels for VEINES-QoL Responders
            for i, (v, n_resp, n_tot) in enumerate(zip(proportions, comp_table['n_responders'], comp_table['n_total'])):
                if v > 95:
                    # For 100% cases, place text INSIDE the bar to keep the top clean
                    y_loc = v - 2  # Slightly below the top
                    va = 'top'
                    color = 'white'
                else:
                    # Standard placement for lower responder rates
                    y_loc = v + (upper_err[i] if upper_err[i] > 0 else 2) + 2
                    va = 'bottom'
                    color = 'black'
                    
                ax2.text(i, y_loc, f'{v:.1f}%\n({n_resp}/{n_tot})', 
                        ha='center', va=va, fontsize=9, fontweight='bold', color=color)

            # 4. Final Formatting
            ax2.set_ylim(0, 110) # Enough room for the title and caps
            ax2.set_ylabel('Responders (%)', fontsize=11, fontweight='bold')
            ax2.set_title(f'VEINES-QoL Responders\n(MCID ≥ {mcid_threshold})', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(comp_table['comparison'], rotation=0) # T1, T2 usually fit without rotation
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
        else:
            ax2 = axes[0, 1]
            proportions = comp_table['proportion'] * 100
            ci_low_resp = comp_table['responder_ci_lower']
            ci_up_resp = comp_table['responder_ci_upper']
            print_yes (comp_table[['comparison', 'proportion', 'responder_ci_lower', 'responder_ci_upper', 'n_total', 'n_responders' ]])
            y_pos = np.arange(len(comparisons))
            ax2.bar(y_pos, proportions, color='steelblue', alpha=0.7)
            #ax2.errorbar(y_pos, proportions, yerr=[proportions - ci_low_resp, ci_up_resp - proportions],
            #            fmt='none', ecolor='black', capsize=5)
            ax2.set_xticks(y_pos)
            ax2.set_xticklabels(comparisons, rotation=45, ha='right')
            ax2.set_ylabel('Percentage of Patients (%)', fontsize=11)
            ax2.set_title(f'Individual Responders (Achieving MCID ≥ {mcid_threshold:.2f})', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 100)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add percentage labels on bars
            for i, (v, n_resp, n_tot) in enumerate(zip(proportions, comp_table['n_responders'], comp_table['n_total'])):
                ax2.text(i, v + 3, f'{v:.1f}%\n({n_resp}/{n_tot})', ha='center', va='bottom', fontsize=9)
        #plt.show()
                    
        # Plot 3: Effect sizes (Cohen's d)
        vert = True
        if vert:
            # ====
            '''
            3. Effect Size Magnitude (Subplot 3)
            Journals often require standardized effect sizes to compare your results to other QoL studies.
            '''
            # Formatting Table 3: Effect Sizes
            tab3 = comp_table[['comparison', 'cohens_d', 'interpretation']].copy()
            tab3.columns = ['Comparison', "Cohen's d", 'Magnitude']
            print (tab3)
            # ====
            # Plot 3: Effect sizes (Cohen's d) - Vertical Style
            ax3 = axes[1, 0]
            cohens_d = comp_table['cohens_d']
            comparisons = comp_table['comparison']
            x_pos = np.arange(len(comparisons))

            # Color logic: Green for Large, Orange for Medium, Red for Small
            colors_d = ['#2ca02c' if abs(d) >= 0.8 else '#ff7f0e' if abs(d) >= 0.5 else '#d62728' for d in cohens_d]

            # 1. Vertical Bars
            ax3.bar(x_pos, cohens_d, color=colors_d, alpha=0.7, edgecolor='black', linewidth=0.8)

            # 2. Horizontal Threshold Lines (Standard Cohen's benchmarks)
            ax3.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.8)
            ax3.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.8)
            ax3.axhline(y=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # 3. Text Labels for Benchmarks (Cleaner than a full legend)
            ax3.text(len(x_pos)-0.5, 0.22, 'Small', fontsize=8, color='gray', ha='right')
            ax3.text(len(x_pos)-0.5, 0.52, 'Medium', fontsize=8, color='gray', ha='right')
            ax3.text(len(x_pos)-0.5, 0.82, 'Large', fontsize=8, color='gray', ha='right')

            # 4. Data Labels on Top of Bars
            for i, d in enumerate(cohens_d):
                ax3.text(i, d + 0.05, f'{d:.2f}', ha='center', va='bottom', fontweight='bold')

            # 5. Formatting
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(comparisons, rotation=45, ha='right')
            ax3.set_ylabel("Cohen's d", fontsize=11)
            ax3.set_title('Magnitude of Effect (Cohen\'s d)', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, max(cohens_d) * 1.2) # Dynamic height
            ax3.grid(axis='y', linestyle=':', alpha=0.5)

            # Remove top/right spines
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
        else:
            ax3 = axes[1, 0]
            cohens_d = comp_table['cohens_d']
            colors_d = ['green' if abs(d) >= 0.5 else 'orange' if abs(d) >= 0.2 else 'red' for d in cohens_d]
            
            ax3.barh(y_pos, cohens_d, color=colors_d, alpha=0.6)
            ax3.axvline(x=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.2)')
            ax3.axvline(x=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.5)')
            ax3.axvline(x=0.8, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Large (0.8)')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(comparisons)
            ax3.set_xlabel("Cohen's d", fontsize=11)
            ax3.set_title('Effect Sizes', fontsize=12, fontweight='bold')
            ax3.legend(loc='lower right', fontsize=9)
            ax3.grid(axis='x', alpha=0.3)
        # plt.show()
        # Plot 4: Statistical vs Clinical Significance Matrix
        better = True
        '''
        4. Significance Summary (Subplot 4)
        This is your "Significance Matrix" in table form, useful for the Appendix or a Summary Results section.
        '''
        # Formatting Table 4: Summary Matrix
        tab4 = comp_table[['comparison', 'statistically_significant', 'clinically_meaningful', 'meets_both_criteria']].copy()
        tab4 = tab4.replace({True: 'Yes', False: 'No'})
        tab4.columns = ['Comparison', 'Stat. Sig (p<0.05)', 'Clin. Sig (≥MCID)', 'Combined Success']
        print (tab4)
        if better:
            ax4 = axes[1, 1]
            ax4.axis('off')

            matrix_data = []
            for _, row in comp_table.iterrows():
                # Using slightly more modern symbols or clean text
                stat_sig = 'Yes' if row['statistically_significant'] else 'No'
                clin_sig = 'Yes' if row['clinically_meaningful'] else 'No'
                # 'Pass' is a strong, clear word for both criteria
                both = 'PASS' if row['meets_both_criteria'] else 'FAIL'
                matrix_data.append([row['comparison'], stat_sig, clin_sig, both])

            # Create Table
            table = ax4.table(
                cellText=matrix_data,
                colLabels=['Comparison', 'Statistical\n(p < 0.05)', 'Clinical\n(≥ MCID)', 'Combined\nResult'],
                cellLoc='center',
                loc='center',
                bbox=[0.0, 0.2, 1.0, 0.6] # Adjusted to give the title breathing room
            )

            # 1. Styling the Table
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # 2. Advanced Cell Styling
            for (row, col), cell in table.get_celld().items():
                # Header row styling
                if row == 0:
                    cell.set_text_props(fontweight='bold', color='white')
                    cell.set_facecolor('#444444') # Dark gray header
                else:
                    # Style the 'Combined Result' column (column index 3)
                    if col == 3:
                        val = matrix_data[row-1][3]
                        if val == 'PASS':
                            cell.set_facecolor('#d4edda') # Soft green
                            cell.set_text_props(fontweight='bold', color='#155724')
                        else:
                            cell.set_facecolor('#f8d7da') # Soft red
                            cell.set_text_props(fontweight='bold', color='#721c24')
                    
                    # Consistent row height
                    cell.set_height(0.2)

            ax4.set_title('Significance Summary Matrix', fontsize=12, fontweight='bold', pad=10)
        else:
            print_yes (comp_table[['comparison', 'statistically_significant','clinically_meaningful', 'meets_both_criteria']], labl="comp_table")
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            matrix_data = []
            for _, row in comp_table.iterrows():
                stat_sig = '✓' if row['statistically_significant'] else '✗'
                clin_sig = '✓' if row['clinically_meaningful'] else '✗'
                both = '✓✓' if row['meets_both_criteria'] else ''
                matrix_data.append([row['comparison'], stat_sig, clin_sig, both])
            
            table = ax4.table(cellText=matrix_data,
                            colLabels=['Comparison', 'Statistical\n(p<0.05)', 'Clinical\n(≥MCID)', 'Both\nCriteria'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color code the cells
            for i in range(1, len(matrix_data) + 1):
                if matrix_data[i-1][3] == '✓✓':
                    table[(i, 3)].set_facecolor('#90EE90')
            
            ax4.set_title('Significance Summary Matrix', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def export_results_to_excel(self, filename='VEINES_MCID_Analysis.xlsx'):
        """
        Export all analysis results to formatted Excel file.
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: MCID thresholds
            mcid_df = pd.DataFrame([self.results['distribution_mcid']])
            mcid_df.to_excel(writer, sheet_name='MCID_Thresholds', index=False)
            
            # Sheet 2: Comprehensive results
            self.results['comprehensive_table'].to_excel(writer, sheet_name='Comprehensive_Results', index=False)
            
            # Sheet 3: LMM summary
            lmm_summary = self.results['lmm_result'].summary()
            # Convert summary to dataframe
            lmm_df = pd.DataFrame({'Summary': [str(lmm_summary)]})
            lmm_df.to_excel(writer, sheet_name='LMM_Summary', index=False)
            
            # Sheet 4: Raw data
            self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        print(f"Results exported to {filename}")


# Example usage
if __name__ == "__main__":
      
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
            
    # Example with sample data structure
    # Replace this with your actual data loading
    
    # Sample data generation (replace with your actual data)
    np.random.seed(42)
    n_patients = 30
    
    data = []
    for patient_id in range(1, n_patients + 1):
        t0_score = np.random.normal(50, 10)
        t1_score = t0_score + np.random.normal(3, 5)  # Mean improvement of 3
        t2_score = t0_score + np.random.normal(5, 6)  # Mean improvement of 5
        
        data.append({'patient_id': patient_id, 'timepoint': 'T0', 'VEINES_QOL_t': t0_score})
        data.append({'patient_id': patient_id, 'timepoint': 'T1', 'VEINES_QOL_t': t1_score})
        data.append({'patient_id': patient_id, 'timepoint': 'T2', 'VEINES_QOL_t': t2_score})
    
    df_fram = pd.DataFrame(data)
    '''
            patient_id timepoint  VEINES_QOL_t
    0            1        T0     54.967142
    1            1        T1     57.275820
    2            1        T2     63.853273
    '''  
    print (df_fram)
    
    file_name = "../../../../../data_qol_00_data/01_saisie/2099-01-01 2099-01-01 TX/09_QUES_FRAME_MOCK/df_pati,qol,sym,vcss,exam incr.csv" # T0, T1, T2
    # Inpu
    # ----
    inpu_file = file_name
    scri_path = os.path.abspath(__file__)
    scri_dire = os.path.dirname(scri_path)
    path_inpu = os.path.join(scri_dire, inpu_file)
    path_inpu = Path(path_inpu).resolve()
    path_inpu = os.path.normpath(path_inpu)
    print (path_inpu)
    df_full = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)
    df_full.columns = df_full.columns.str.strip()
    df_full.rename(columns={'Q_copi_t': 'VEINES_QOL_t'}, inplace=True)
    df_full =  df_full[['patient_id', 'timepoint','VEINES_QOL_t']]
    df_full['timepoint'] = pd.Categorical(df_fram['timepoint'], categories=["T0","T1","T2"], ordered=True)
    print_yes(df_full,labl="df_full")

    # Initialize analysis
    analyzer = VEINESMCIDAnalysis(df_full)
      
    print("\nFitting linear mixed model...")
    lmm_result = analyzer.fit_mixed_model()
    print (lmm_result.summary())
      
    # Run complete analysis
    print("Calculating distribution-based MCID...")
    # https://gemini.google.com/app/0f865d976b405499
    mcid_thresholds = analyzer.calculate_distribution_based_mcid(lmm_result)
    
    
    # **********************************************************************
    
    # Gemini# 
    result = lmm_result
    df_modl = df_full
    # 1. Setup Thresholds (Using your 0.5 SD result)
    mcid_threshold = 1.76  # Derived from your 0.5_SD calculation

    # 2. Group-Level Analysis (The "Mean" Approach)
    # Extract coefficients from your LMM result object
    lmm_params = result.params
    t1_mean_gain = lmm_params["C(timepoint)[T.T1]"]
    t2_mean_gain = lmm_params["C(timepoint)[T.T2]"]

    group_results = {
        "T1_vs_T0": {"Gain": t1_mean_gain, "Clinically_Sig": t1_mean_gain >= mcid_threshold},
        "T2_vs_T0": {"Gain": t2_mean_gain, "Clinically_Sig": t2_mean_gain >= mcid_threshold}
    }

    # 3. Patient-Level Analysis (The "Responder" Approach)
    # Pivot df_modl to calculate individual deltas
    df_pivot = df_modl.pivot(index='patient_id', columns='timepoint', values='VEINES_QOL_t')
    df_pivot['delta_T1'] = df_pivot['T1'] - df_pivot['T0']
    df_pivot['delta_T2'] = df_pivot['T2'] - df_pivot['T0']

    # Identify Responders (Delta >= MCID)
    df_pivot['is_responder_T1'] = df_pivot['delta_T1'] >= mcid_threshold
    df_pivot['is_responder_T2'] = df_pivot['delta_T2'] >= mcid_threshold

    responder_rate_t1 = df_pivot['is_responder_T1'].mean() * 100
    responder_rate_t2 = df_pivot['is_responder_T2'].mean() * 100

    print(f"Group Mean Gain T2: {t2_mean_gain:.2f}")
    print(f"Individual Responder Rate T2: {responder_rate_t2:.1f}%")

    # ******
    
    print("\nExtracting population changes...")
    pop_changes = analyzer.extract_population_changes(lmm_result)
    
    print("\nCalculating effect sizes...")
    effect_sizes = analyzer.calculate_effect_sizes()
    
    print("\nRunning comprehensive analysis...")
    comprehensive = analyzer.comprehensive_mcid_table()
    
    print("\n" + "="*80)
    print(analyzer.generate_publication_summary())
    
    # Generate visualization
    fig = analyzer.plot_mcid_analysis()
    plt.savefig('VEINES_MCID_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Export to Excel
    analyzer.export_results_to_excel()
    
    print("\n✓ Analysis complete!")
    print("  - Visualization saved as 'VEINES_MCID_Analysis.png'")
    print("  - Results exported to 'VEINES_MCID_Analysis.xlsx'")

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