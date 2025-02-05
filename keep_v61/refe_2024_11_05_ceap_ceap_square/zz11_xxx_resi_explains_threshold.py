from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
'''
Input : Observed:
ceap  NA  C0  C1  C2  C3  C4  C5  C6
ceap
NA     0   0   0   2  18   5   0  22
C0     0   0   0   1   8   2   2  14
C1     0   0   1   0   3   2   1   0
C2     3   1   1  10  14   3   3  18
C3    20   9   0  15  80  16   3  14
C4     9   2   1   3  17  23   7   5
C5     7   5   1   5   4   2   6   8
C6    39  29   3  33   9   5   7  21
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object', name='ceap')
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object', name='ceap')
'''
'''
Output Residuals : Synthesis threshold_05:1.960, threshold_01:2.576
      NA  C0  C1  C2  C3  C4  C5  C6
ceap
NA    --   -   .   .   .   .   .  ++
C0     -   .   .   .   .   .   .  ++
C1     .   .  ++   .   .   .   .   .
C2     .   .   .   .   .   .   .   +
C3     .   .   .   .  ++   .   .  --
C4     .   .   .   .   .  ++   .   -
C5     .   .   .   .   -   .  ++   .
C6    ++  ++   .  ++  --  --   .   .

'''
def resi_chck_deta(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
 
    # Exec
    print(f"Input : Observed:\n{df}\n:{df.index}\n:{df.columns}")
    total_sum = df.to_numpy().sum()
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)  
    # Expected values : native
    expected = np.outer(row_sums, col_sums) / total_sum
    expected_df = pd.DataFrame(expected, index=indx_cate_list, columns=colu_cate_list)
    # Expected values
    chi2, pval, dof, expected_chi2 = chi2_contingency(df)
    expected_chi2_df = pd.DataFrame(expected_chi2, index=indx_cate_list, columns=colu_cate_list)
    # Residuals
    residuals = (df - expected_df) / np.sqrt(expected_df)
    residuals_df = pd.DataFrame(residuals, index=indx_cate_list, columns=colu_cate_list)
    # Resu
    def oup1(df, expected_df, residuals_df):
        def format_cell(cell):
            return round(cell, 2)
        expected_df_form = expected_df.map(format_cell)
        residuals_df_form = residuals_df.map(format_cell)
        print(f"Observed:\n{df}\n:{df.index}\n:{df.columns}")
        print(f"Expected:\n{expected_df_form}\n:{df.index}\n:{df.columns}")
        print(f"Residuals:\n{residuals_df_form}\n:{df.index}\n:{df.columns}")
    
    # Resu
    alpha_01 = 0.01
    alpha_05 = 0.05
    threshold_01 = norm.ppf(1 - alpha_01 / 2)  # Critical value for alpha = 0.01 : thres = 2.58
    threshold_05 = norm.ppf(1 - alpha_05 / 2)  # Critical value for alpha = 0.05 ; thres = 1.96
    
    def oup2(resi_arra, threshold_05, threshold_01):
        def residual_symbol(residual, threshold_05, threshold_01):
            if residual > threshold_01:
                return '++'
            elif residual > threshold_05:
                return '+'
            elif residual < -threshold_01:
                return '--'
            elif residual < -threshold_05:
                return '-'
            else:
                return '.'
        #
        symbol_df = resi_arra.apply(lambda row: row.apply(residual_symbol, args=(threshold_05, threshold_01)), axis=1)
        symbol_df = symbol_df.rename_axis(indx_name, axis='index')
        threshold_05_form = f"{threshold_05:.3e}" if threshold_05 < 0.001 else f"{threshold_05:.3f}"
        threshold_01_form = f"{threshold_01:.3e}" if threshold_01 < 0.001 else f"{threshold_01:.3f}"
        print(f"Residuals : Synthesis threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_df}")
        return symbol_df
    
    # Resu
    print(f"\nData : {what}\nResiduals")
    oup1(df, expected_df, residuals_df)
    dfs = oup2(residuals_df, threshold_05, threshold_01)
    pass

'''
Observed:
ceap  G1   G2   G3
ceap
G1     0    2   45
G2     3   14   70
G3    75  106  227
:Index(['G1', 'G2', 'G3'], dtype='object', name='ceap')
:Index(['G1', 'G2', 'G3'], dtype='object', name='ceap')
'''
'''
       G1  G2  G3
group
G1     --  --  ++
G2     --   .   +
G3      +   .   .
'''
def resi_chck_grop(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    # Print the input observed data
    print(f"Input : Observed:\n{df}\n:{df.index}\n:{df.columns}")

    # Calculate the total sum of the observed data
    total_sum = df.to_numpy().sum()

    # Calculate row and column sums
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)

    # Calculate the expected values under the null hypothesis of independence
    expected = np.outer(row_sums, col_sums) / total_sum
    expected_df = pd.DataFrame(expected, index=indx_cate_list, columns=colu_cate_list)

    # Perform chi-square test of independence
    chi2, pval, dof, expected_chi2 = chi2_contingency(df)
    expected_chi2_df = pd.DataFrame(expected_chi2, index=indx_cate_list, columns=colu_cate_list)

    # Calculate the residuals
    residuals = (df - expected_df) / np.sqrt(expected_df)
    residuals_df = pd.DataFrame(residuals, index=indx_cate_list, columns=colu_cate_list)

    # Function to format and print the observed, expected, and residual values
    def oup1(df, expected_df, residuals_df):
        def format_cell(cell):
            return round(cell, 2)
        expected_df_form = expected_df.applymap(format_cell)
        residuals_df_form = residuals_df.applymap(format_cell)
        print(f"Observed:\n{df}\n:{df.index}\n:{df.columns}")
        print(f"Expected:\n{expected_df_form}\n:{df.index}\n:{df.columns}")
        print(f"Residuals:\n{residuals_df_form}\n:{df.index}\n:{df.columns}")

    # Define significance levels and corresponding thresholds
    alpha_01 = 0.01
    alpha_05 = 0.05
    threshold_01 = norm.ppf(1 - alpha_01 / 2)  # Critical value for alpha = 0.01
    threshold_05 = norm.ppf(1 - alpha_05 / 2)  # Critical value for alpha = 0.05

    # Function to categorize residuals based on thresholds
    def oup2(resi_arra, threshold_05, threshold_01):
        def residual_symbol(residual, threshold_05, threshold_01):
            if residual > threshold_01:
                return '++'
            elif residual > threshold_05:
                return '+'
            elif residual < -threshold_01:
                return '--'
            elif residual < -threshold_05:
                return '-'
            else:
                return '.'
        symbol_df = resi_arra.applymap(lambda x: residual_symbol(x, threshold_05, threshold_01))
        symbol_df = symbol_df.rename_axis(indx_name, axis='index')
        threshold_05_form = f"{threshold_05:.3e}" if threshold_05 < 0.001 else f"{threshold_05:.3f}"
        threshold_01_form = f"{threshold_01:.3e}" if threshold_01 < 0.001 else f"{threshold_01:.3f}"
        print(f"Residuals : Synthesis threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_df}")
        return symbol_df

    # Print the results
    print(f"\nData : {what}\nResiduals")
    oup1(df, expected_df, residuals_df)
    dfs = oup2(residuals_df, threshold_05, threshold_01)

def regroup_data(df):
    # Define the group mappings
    group_mapping = {
        'NA': 'G1',
        'C0': 'G2',
        'C1': 'G2',
        'C2': 'G2',
        'C3': 'G3',
        'C4': 'G3',
        'C5': 'G3',
        'C6': 'G3'
    }

    # Regroup the data
    df_regrouped = df.groupby(group_mapping, axis=0).sum().groupby(group_mapping, axis=1).sum()
    return df_regrouped

if __name__ == "__main__":
    
    if True:
        data = [
            [0, 0, 0, 2, 18, 5, 0, 22],
            [0, 0, 0, 1, 8, 2, 2, 14],
            [0, 0, 1, 0, 3, 2, 1, 0],
            [3, 1, 1, 10, 14, 3, 3, 18],
            [20, 9, 0, 15, 80, 16, 3, 14],
            [9, 2, 1, 3, 17, 23, 7, 5],
            [7, 5, 1, 5, 4, 2, 6, 8],
            [39, 29, 3, 33, 9, 5, 7, 21]
        ]
        index = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap')
        columns = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap')
        df = pd.DataFrame(data, index=index, columns=columns)

        # Regroup the data
        df_regrouped = regroup_data(df)

        # Define the new categories and names
        clin_sign = ['G1', 'G2', 'G3']
        indx_cate_list = colu_cate_list = clin_sign
        indx_name = 'group'
        colu_name = 'group'

        # Perform the analysis on the regrouped data
        resi_chck_grop("Regrouped Data", df_regrouped, indx_cate_list, colu_cate_list, indx_name, colu_name)



    if False:

        # Create the data
        data = [
            [0, 0, 0, 2, 18, 5, 0, 22],
            [0, 0, 0, 1, 8, 2, 2, 14],
            [0, 0, 1, 0, 3, 2, 1, 0],
            [3, 1, 1, 10, 14, 3, 3, 18],
            [20, 9, 0, 15, 80, 16, 3, 14],
            [9, 2, 1, 3, 17, 23, 7, 5],
            [7, 5, 1, 5, 4, 2, 6, 8],
            [39, 29, 3, 33, 9, 5, 7, 21]
        ]
        index = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap')
        columns = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap')
        df = pd.DataFrame(data, index=index, columns=columns)
        clin_sign = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        indx_cate_list = colu_cate_list = clin_sign
        indx_name = 'ceap'
        colu_name = 'ceap'
        resi_chck_deta("what", df, indx_cate_list, colu_cate_list, indx_name, colu_name)
        pass
'''
The thresholds for the residuals in a chi-square test of independence are based on the standard normal distribution and are not directly dependent on the size of the contingency table. Let's break down why this is the case:

### Understanding Residuals and Thresholds

1. **Residuals Calculation**:
   - Residuals are calculated as the difference between the observed and expected values, normalized by the square root of the expected value. The formula is:
     \[
     \text{Residual} = \frac{\text{Observed} - \text{Expected}}{\sqrt{\text{Expected}}}
     \]
   - This normalization ensures that the residuals are standardized, meaning they are on a comparable scale regardless of the magnitude of the expected values.

2. **Standard Normal Distribution**:
   - The thresholds (`threshold_01` and `threshold_05`) are derived from the standard normal distribution (Z-distribution). These thresholds represent the critical values for a given significance level (alpha).
   - For example, `threshold_05` corresponds to the critical value for a two-tailed test at the 5% significance level. This value is approximately 1.96, meaning that about 5% of the values in a standard normal distribution are beyond ±1.96.

3. **Independence from Table Size**:
   - The thresholds are based on the properties of the standard normal distribution, which is independent of the size of the contingency table. The standard normal distribution is a theoretical distribution with a mean of 0 and a standard deviation of 1.
   - The residuals, being standardized, follow this distribution regardless of the table size. Therefore, the critical values (thresholds) for determining significance do not change with the table size.

### Why Thresholds Are Independent of Table Size

- **Standardization**: The residuals are standardized by dividing by the square root of the expected value. This standardization makes the residuals comparable across different table sizes.
- **Distribution Properties**: The standard normal distribution's properties (mean, standard deviation, and critical values) are fixed and do not depend on the sample size or table size.
- **Significance Levels**: The significance levels (alpha) and their corresponding critical values are based on the cumulative distribution function (CDF) of the standard normal distribution, which is independent of the table size.

### Example

Consider the following:

- For a 2x2 table, the residuals are standardized and follow the standard normal distribution.
- For a 3x3 table, the residuals are also standardized and follow the same standard normal distribution.
- The critical value for a 5% significance level (alpha = 0.05) is approximately 1.96 for both tables because it is derived from the standard normal distribution.

### Conclusion

The thresholds for the residuals are based on the standard normal distribution and are independent of the size of the contingency table. This is because the residuals are standardized, making them comparable across different table sizes, and the properties of the standard normal distribution are fixed.
'''
'''
Certainly! Creating an acronym for the table format can help in easily referencing it throughout your scientific publication. Here are a few suggestions for an acronym that captures the essence of the table format:

1. **CRST**: Categorized Residuals Synthesis Table
2. **CART**: Chi-Square Analysis Residuals Table
3. **RCT**: Residuals Categorization Table
4. **SRT**: Standardized Residuals Table
5. **CRAT**: Chi-Square Residuals Analysis Table

Each of these acronyms highlights different aspects of the table, such as the categorization of residuals, the chi-square analysis, or the standardization process. You can choose the one that best fits the context of your publication.

For example, if you choose **CRST** (Categorized Residuals Synthesis Table), you can introduce it in your publication as follows:

---

### Results

The chi-square test of independence was performed on the contingency table of clinical signs for left and right limbs in patients suffering from chronic venous insufficiency (CVI). The residuals were calculated to identify significant deviations from the expected values under the null hypothesis of independence. The residuals were categorized based on predefined thresholds corresponding to significance levels of 0.05 and 0.01.

#### Residuals Synthesis

The thresholds for the residuals were determined as follows:
- **Threshold for α = 0.05**: 1.960
- **Threshold for α = 0.01**: 2.576

The categorized residuals are presented in the Categorized Residuals Synthesis Table (CRST) below. The symbols indicate the significance and direction of the deviations:
- `++`: Significant positive deviation (p < 0.01)
- `+`: Positive deviation (p < 0.05)
- `--`: Significant negative deviation (p < 0.01)
- `-`: Negative deviation (p < 0.05)
- `.`: No significant deviation

**Table 1: Categorized Residuals Synthesis Table (CRST)**

|       | NA  | C0  | C1  | C2  | C3  | C4  | C5  | C6  |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| **NA**| --  | -   | .   | .   | .   | .   | .   | ++  |
| **C0**| -   | .   | .   | .   | .   | .   | .   | ++  |
| **C1**| .   | .   | ++  | .   | .   | .   | .   | .   |
| **C2**| .   | .   | .   | .   | .   | .   | .   | +   |
| **C3**| .   | .   | .   | .   | ++  | .   | .   | --  |
| **C4**| .   | .   | .   | .   | .   | ++  | .   | -   |
| **C5**| .   | .   | .   | .   | -   | .   | ++  | .   |
| **C6**| ++  | ++  | .   | ++  | --  | --  | .   | .   |

**Notes:**
- Rows represent the clinical signs for the left limb.
- Columns represent the clinical signs for the right limb.
- `NA`: Limb 'not affected' (by disease)
- `C0` to `C6`: Classical Clinical-Etiology-Anatomy-Pathophysiology (CEAP) classification

### Discussion

The residuals analysis highlights significant deviations from the expected values, indicating potential associations between specific clinical signs in the left and right limbs. For example, the significant positive deviation (`++`) in the cells (C6, NA) and (NA, C6) suggests a strong association between severe clinical signs in one limb and the absence of clinical signs in the other limb. Conversely, the significant negative deviation (`--`) in the cell (C3, C6) indicates a lower than expected co-occurrence of these clinical signs.

These findings provide insights into the patterns of clinical signs in patients with CVI and may have implications for clinical management and further research.

---

Using an acronym like **CRST** makes it easier to reference the table throughout your publication and ensures clarity for the readers.
'''
'''
CRST: Categorized Residuals Synthesis Table
CART: Chi-Square Analysis Residuals Table
RCT: Residuals Categorization Table
SRT: Standardized Residuals Table
CRAT: Chi-Square Residuals Analysis Table

CART : Categorized Residuals Table
'''