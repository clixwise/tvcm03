import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency, norm

# -------------------------------
# Residuals
# -------------------------------
# def resi_anal(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

def residual_significance(residuals, p_threshold):
        """
        Identify significant residuals based on the p-value threshold.

        Parameters:
        residuals (pd.DataFrame): DataFrame of residuals.
        p_threshold (float): Significance level (e.g., 0.05 or 0.01).

        Returns:
        pd.DataFrame: DataFrame with significance flags.
        """
        critical_value = norm.ppf(1 - p_threshold / 2)  # Two-tailed critical value
        return residuals.applymap(lambda x: abs(x) > critical_value)

def main():
    # Input Data: Standardized and Adjusted Residuals
    standardized_residuals = pd.DataFrame(
        [
            [0.99, 0.39, 0.12, 0.26, -1.39, -0.60, -1.02, 1.35],
            [-0.87, -0.34, -0.10, -0.23, 1.21, 0.52, 0.89, -1.18],
        ],
        index=['M', 'F'],
        columns=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    )

    adjusted_residuals = pd.DataFrame(
        [
            [1.41, 0.54, 0.16, 0.37, -2.18, -0.84, -1.39, 2.03],
            [-1.41, -0.54, -0.16, -0.37, 2.18, 0.84, 1.39, -2.03],
        ],
        index=['M', 'F'],
        columns=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    )

    # Significance Analysis
    p_threshold_05 = 0.05
    p_threshold_01 = 0.01

    standardized_significance_05 = residual_significance(standardized_residuals, p_threshold_05)
    adjusted_significance_05 = residual_significance(adjusted_residuals, p_threshold_05)

    standardized_significance_01 = residual_significance(standardized_residuals, p_threshold_01)
    adjusted_significance_01 = residual_significance(adjusted_residuals, p_threshold_01)

    # Print Results
    print("Standardized Residuals - Significant at p < 0.05:\n", standardized_significance_05)
    print("Adjusted Residuals - Significant at p < 0.05:\n", adjusted_significance_05)
    print("Standardized Residuals - Significant at p < 0.01:\n", standardized_significance_01)
    print("Adjusted Residuals - Significant at p < 0.01:\n", adjusted_significance_01)

    # Stratified Analysis: C3 and C6 Residuals
    c3_c6_adjusted_residuals = adjusted_residuals[['C3', 'C6']]
    c3_c6_significance_05 = residual_significance(c3_c6_adjusted_residuals, p_threshold_05)
    c3_c6_significance_01 = residual_significance(c3_c6_adjusted_residuals, p_threshold_01)

    print("C3 and C6 Adjusted Residuals - Significant at p < 0.05:\n", c3_c6_significance_05)
    print("C3 and C6 Adjusted Residuals - Significant at p < 0.01:\n", c3_c6_significance_01)

    # Hypothesis Testing: Stratified Analysis
    from scipy.stats import chi2_contingency

    # Example Contingency Table (Counts for M and F in C3 and C6, adjust with real data)
    contingency_table = pd.DataFrame(
        {
            "C3": [50, 70],  # Observed counts for Male (M) and Female (F)
            "C6": [80, 60],
        },
        index=['M', 'F']
    )

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("Chi-square Test Results for C3 and C6:")
    print("Chi2 Statistic:", chi2)
    print("p-value:", p)
    print("Degrees of Freedom:", dof)
    print("Expected Frequencies:\n", expected)
    pass

if __name__ == "__main__":
    main()
    pass
'''
### Results Analysis

1. **Standardized Residuals**:
   - None of the residuals are significant at \( p < 0.05 \) or \( p < 0.01 \).
   - Indicates that the standardized residuals do not reveal strong deviations between observed and expected counts.

2. **Adjusted Residuals**:
   - At \( p < 0.05 \):
     - \( M(C3) \) and \( M(C6) \) are significant.
     - \( F(C3) \) and \( F(C6) \) are also significant.
   - At \( p < 0.01 \):
     - None of the adjusted residuals are significant.
   - Suggests moderate deviations for \( C3 \) and \( C6 \), particularly for both males and females, but these deviations are not extreme.

3. **C3 and C6 Stratified Significance**:
   - \( C3 \) and \( C6 \) are both significant at \( p < 0.05 \) for males and females, as indicated by adjusted residuals.

4. **Chi-Square Test**:
   - **Chi2 Statistic**: \( 5.59 \)
   - **p-value**: \( 0.018 \) (significant at \( p < 0.05 \), not significant at \( p < 0.01 \)).
   - **Degrees of Freedom**: 1
   - **Expected Frequencies**:
     - Balanced counts: \( C3 = [60, 60] \), \( C6 = [70, 70] \).
   - The chi-square test suggests significant association between gender and CEAP classes \( C3 \) and \( C6 \) at \( p < 0.05 \).

---

### Key Insights:
- Adjusted residuals reveal significant deviations for \( C3 \) and \( C6 \) at \( p < 0.05 \), which is consistent for both genders.
- The chi-square test supports the hypothesis of a gender association with \( C3 \) and \( C6 \) at \( p < 0.05 \).
- At \( p < 0.01 \), no residuals or associations are significant.

---

### Next Steps:
1. **Interpret Clinical Implications**:
   - Explore why \( C3 \) and \( C6 \) show significant deviations for both genders. Are these CEAP classes clinically distinct in prevalence or severity for males and females?

2. **Further Statistical Tests**:
   - Perform post-hoc tests or investigate the adjusted residuals in greater detail to understand which gender contributes more to the significant deviations.
   - Use logistic regression or other modeling techniques to predict \( C3 \) or \( C6 \) based on gender and other patient factors.

3. **Visualization**:
   - Enhance plots to highlight the significant residuals and chi-square test results for better interpretation.

Would you like to proceed with additional analyses or visualizations?
'''

'''
The distinction between **standardized residuals** and **adjusted residuals** lies in how they account for the structure of the contingency table, particularly in the presence of unequal marginal totals (row/column totals).

---

### **1. Standardized Residuals**
**Formula**:  
\[
R_{\text{std}} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}
\]

- **Interpretation**: Standardized residuals measure the difference between the observed (\(O_{ij}\)) and expected (\(E_{ij}\)) frequencies in each cell, normalized by the standard deviation under the assumption of independence (\(\sqrt{E_{ij}}\)).  
- **Key Points**:
  - They are useful for identifying cells with unusually large deviations compared to the expected values.
  - They assume independence but **do not account for the marginal totals** (row/column proportions).
  - Values are interpreted similarly to \(z\)-scores:
    - \(R_{\text{std}} > 1.96\): Significant positive deviation.
    - \(R_{\text{std}} < -1.96\): Significant negative deviation.
  - However, these residuals may overstate the significance of deviations if the marginal totals are very unequal.

---

### **2. Adjusted Residuals**
**Formula**:  
\[
R_{\text{adj}} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij} \cdot (1 - R_i) \cdot (1 - C_j)}}
\]
Where:  
- \(R_i = \frac{\text{Row Total}_i}{\text{Grand Total}}\) (row proportion).  
- \(C_j = \frac{\text{Column Total}_j}{\text{Grand Total}}\) (column proportion).  

- **Interpretation**: Adjusted residuals are modified to account for the marginal totals of the contingency table, making them more reliable in cases where row or column totals differ significantly.  
- **Key Points**:
  - They incorporate the proportions of the total in each row and column (\(R_i, C_j\)).
  - By adjusting for these proportions, they control for potential biases introduced by unequal row/column totals.
  - Adjusted residuals provide a more **precise measure of deviation significance** when the data is imbalanced.
  - They too are interpreted as \(z\)-scores:
    - \(R_{\text{adj}} > 1.96\): Significant positive deviation after accounting for marginal effects.
    - \(R_{\text{adj}} < -1.96\): Significant negative deviation.

---

### **Why Adjusted Residuals Are Important**
- In your data, the row (sex: M/F) and column (age bins) totals are imbalanced. For instance, males contribute more to certain age bins (e.g., 50-59, 40-49), while females dominate in others. Standardized residuals might exaggerate the significance of deviations due to this imbalance.  
- Adjusted residuals correct for this by factoring in row and column proportions, thus reflecting a more nuanced view of the deviation.

---

### **Comparison in Practice**
Using your data:
- **Example (Standardized Residual for M, 30-39)**:
  \[
  R_{\text{std}} = \frac{10 - 14.22}{\sqrt{14.22}} = \frac{-4.22}{3.77} \approx -1.12
  \]

- **Example (Adjusted Residual for M, 30-39)**:
  \[
  R_{\text{adj}} = \frac{10 - 14.22}{\sqrt{14.22 \cdot (1 - 0.420) \cdot (1 - 0.103)}} \approx \frac{-4.22}{2.70} \approx -1.56
  \]
  Here, row proportion (\(R_i\)) and column proportion (\(C_j\)) reduce the denominator, making the residual larger in magnitude.

---

### Summary
- **Standardized Residuals**: Quick and useful for initial checks but might be misleading with imbalanced margins.
- **Adjusted Residuals**: Corrected for row and column proportions, offering a more accurate significance measure in the presence of marginal imbalances.

'''
'''
Standard vs adjusted residuals
------------------------------
Given the formula for residuals:

$$ \text{residuals} = \frac{\text{df} - \text{expected\_df}}{\sqrt{\text{expected\_df}}} $$

This formula calculates the standardized residuals, which are similar to adjusted residuals but not exactly the same. Let's compare these concepts:

1. Standardized Residuals:
The formula you provided calculates standardized residuals. These measure the difference between observed and expected frequencies in units of standard deviations[3]. They are useful for identifying cells that contribute significantly to the chi-square statistic.

2. Adjusted Residuals:
Adjusted residuals are a more refined version of standardized residuals. They account for the variation due to sample size and are calculated as:

$$ \text{Adjusted Residual} = \frac{\text{Observed} - \text{Expected}}{\sqrt{\text{Expected} \times (1 - \text{row proportion}) \times (1 - \text{column proportion})}} $$

The key differences between standardized and adjusted residuals are:

- Adjusted residuals account for row and column proportions, making them more accurate for larger tables[1][2].
- Under the null hypothesis of independence, adjusted residuals follow a standard normal distribution more closely[2][4].

Interpretation:
- For both types, values outside the range of -2 to 2 are generally considered significant[2][3].
- Adjusted residuals > 1.96 or < -1.96 indicate significantly more or fewer cases than expected at a 0.05 significance level[2].
- Larger absolute values indicate stronger deviations from expected frequencies[5].

In practice, adjusted residuals are often preferred for their more accurate representation of cell-wise deviations, especially in larger contingency tables[1][4].

Citations:
[1] https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/tables/how-to/cross-tabulation-and-chi-square/interpret-the-results/all-statistics-and-graphs/tabulated-statistics/
[2] https://www.ibm.com/support/pages/interpreting-adjusted-residuals-crosstabs-cell-statistics
[3] https://www.statisticshowto.com/what-is-a-standardized-residuals/
[4] https://stats.stackexchange.com/questions/585735/how-to-properly-interpret-adjusted-residuals-in-crosstabs-with-chi-squares-large
[5] https://www.1ka.si/d/en/help/manuals/residuals-crosstabs
[6] https://en.wikipedia.org/wiki/Statistical_error
'''
'''
The standardized residuals you've computed provide a measure of how much 
the observed frequencies deviate from the expe_arra frequencies under the assumption of independence.
'''
