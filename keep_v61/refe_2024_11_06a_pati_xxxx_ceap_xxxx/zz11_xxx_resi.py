import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency, norm

# -------------------------------
# Residuals
# -------------------------------
def resi(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
 
    trac = True
    if trac: print (df)
 
    # Exec 1
    # ------
    # Total sum of the entire table
    totl_sums = df.to_numpy().sum()
    # Row sums and column sums
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)  
    # Expected values : native
    expected = np.outer(row_sums, col_sums) / totl_sums
    expected_df = pd.DataFrame(expected, index=indx_cate_list, columns=colu_cate_list)
    if trac: print (expected_df)
    
    # Exec 2 : Expected values : chi2 [same result]
    # ------
    chi2, pval, dof, expected_chi2 = chi2_contingency(df)
    expected_chi2_df = pd.DataFrame(expected_chi2, index=indx_cate_list, columns=colu_cate_list)
    if trac: print (expected_chi2_df)
    if not expected_df.compare(expected_chi2_df).empty:
        raise Exception()
    
    # Exec 3 : Residuals standardized
    # ------
    residuals = (df - expected_df) / np.sqrt(expected_df)
    residuals_standard_df = pd.DataFrame(residuals, index=indx_cate_list, columns=colu_cate_list)
    if trac: print (residuals_standard_df)
    
    # Exec 4 : Residuals adjusted
    # ------
    def calculate_adjusted_residuals(observed, expected):
        
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total = observed.sum().sum()
        
        row_props = row_totals / total
        col_props = col_totals / total
        
        adj_residuals = observed.copy()
        adj_residuals = adj_residuals.astype(float)  # Convert the entire DataFrame to float

        for i in observed.index:
            for j in observed.columns:
                numer = observed.loc[i, j] - expected.loc[i, j]
                denom = np.sqrt(expected.loc[i, j] * (1 - row_props[i]) * (1 - col_props[j]))
                adj_residuals.loc[i, j] = numer / denom
        return adj_residuals

    residuals_adjusted_df = calculate_adjusted_residuals(df, expected_df) 
    if trac: print (residuals_adjusted_df)

    # Exec 5 : Freeman-Tukey Deviates
    # ------
    freeman_df = np.sign(df - expected) * np.sqrt(4 * df + 1) + np.sqrt(4 * expected + 1)
    if trac: print (freeman_df)
    
    # Util
    # ----
    def oup1(df, expected_df, residuals_standard_df, residuals_adjusted_df, freeman_df):
        def format_cell(cell):
            return round(cell, 2) # f"{cell:.2e}" if cell < 0.01 else f"{cell:.2f}"
        expected_df_form = expected_df.map(format_cell)
        residuals_df_form = residuals_standard_df.map(format_cell)
        adjusted_df_form = residuals_adjusted_df.map(format_cell)
        freeman_df_form = freeman_df.map(format_cell)
        print(f"Residuals : Observed:\n{df}\n:{df.index}\n:{df.columns}")
        write(f"Residuals : Observed:\n{df}\n:{df.index}\n:{df.columns}")
        print(f"Residuals : Expected:\n{expected_df_form}\n:{df.index}\n:{df.columns}")
        write(f"Residuals : Expected:\n{expected_df_form}\n:{df.index}\n:{df.columns}")
        print(f"Residuals : Residuals standardized:\n{residuals_df_form}\n:{df.index}\n:{df.columns}")
        write(f"Residuals : Residuals standardized:\n{residuals_df_form}\n:{df.index}\n:{df.columns}")
        print(f"Residuals : Residuals adjusted:\n{adjusted_df_form}\n:{df.index}\n:{df.columns}")
        write(f"Residuals : Residuals adjusted:\n{adjusted_df_form}\n:{df.index}\n:{df.columns}")
        print(f"Residuals : Freeman-Tukey Deviates:\n{freeman_df_form}\n:{df.index}\n:{df.columns}")
        write(f"Residuals : Freeman-Tukey Deviates:\n{freeman_df_form}\n:{df.index}\n:{df.columns}")
        observed_df_form = df.copy()
        observed_df_form.index = observed_df_form.index.map(lambda x: x + 'O')
        expected_df_form.index = expected_df_form.index.map(lambda x: x + 'E')
        residuals_df_form.index = residuals_df_form.index.map(lambda x: x + 'R')
        adjusted_df_form.index = adjusted_df_form.index.map(lambda x: x + 'A')
        freeman_df_form.index = freeman_df_form.index.map(lambda x: x + 'D')
        df_full_form = pd.concat([observed_df_form, expected_df_form, residuals_df_form, adjusted_df_form, freeman_df_form])
        #df_full_form = df_full_form.map(format_cell)
        
        # Exit
        return df_full_form
    
    # Symbolic table
    # --------------
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
        # (old) symbol_df = resi_arra.applymap(lambda x: residual_symbol(x, threshold_05, threshold_01))
        symbol_df = symbol_df.rename_axis(indx_name, axis='index')
                
        # Exit
        return symbol_df
    
    # Symbolic lists
    # --------------
    def oup3(dfo,dfs):
     
        counts = {'++': 0, '--': 0, '+': 0, '-': 0, '.': 0}
        diagonal_counts = {'++': 0, '--': 0, '+': 0, '-': 0, '.': 0}
        off_diagonal_counts = {'++': 0, '--': 0, '+': 0, '-': 0, '.': 0}
        counts_C6_row = {'++': 0, '--': 0, '+': 0, '-': 0, '.': 0}
        counts_C6_col = {'++': 0, '--': 0, '+': 0, '-': 0, '.': 0}
        for i in dfs.index:
            for j in dfs.columns:
                category = dfs.loc[i, j]
                count_value = dfo.loc[i, j]
                counts[category] += count_value
                # Check if the cell is diagonal or off-diagonal
                if i == j:  # Diagonal cell
                    diagonal_counts[category] += count_value
                else:  # Off-diagonal cell
                    off_diagonal_counts[category] += count_value   
                # Check for counts related to C6 row and C6 column
                if i == 'C6' and j != 'C6':  # Row index is C6, column is not C6
                    counts_C6_row[category] += count_value
                if j == 'C6' and i != 'C6':  # Column index is C6, row is not C6
                    counts_C6_col[category] += count_value 
        #
        df_counts = pd.DataFrame(list(counts.items()), columns=['Classes', 'Obs(abs)'])
        df_counts['Diagonal Obs(abs)'] = [diagonal_counts[c] for c in df_counts['Classes']]
        df_counts['Off-Diagonal Obs(abs)'] = [off_diagonal_counts[c] for c in df_counts['Classes']]
        df_counts['C6 Row Obs(abs)'] = [counts_C6_row[c] for c in df_counts['Classes']]
        df_counts['C6 Column Obs(abs)'] = [counts_C6_col[c] for c in df_counts['Classes']]
        
        total_observations = df_counts['Obs(abs)'].sum()
        df_counts['Obs(%)'] = (df_counts['Obs(abs)'] / total_observations * 100).round().astype(int)
        sum_abs = df_counts['Obs(abs)'].sum()
        sum_percentage = df_counts['Obs(%)'].sum()
        sum_diagonal = df_counts['Diagonal Obs(abs)'].sum()
        sum_off_diagonal = df_counts['Off-Diagonal Obs(abs)'].sum()
        sum_C6_row = df_counts['C6 Row Obs(abs)'].sum()
        sum_C6_col = df_counts['C6 Column Obs(abs)'].sum()
        sum_row = pd.DataFrame({
            'Classes': ['Total'], 
            'Obs(abs)': [sum_abs], 
            'Obs(%)': [sum_percentage], 
            'Diagonal Obs(abs)': [sum_diagonal], 
            'Off-Diagonal Obs(abs)': [sum_off_diagonal],
            'C6 Row Obs(abs)': [sum_C6_row],
            'C6 Column Obs(abs)': [sum_C6_col]
            })
        df_counts = pd.concat([df_counts, sum_row], ignore_index=True)
        
        # Exit
        return df_counts
    
    # Resu
    print(f"\nData : {what}\nResiduals")
    write(f"\nData : {what}\nResiduals") 
    # All resu
    df_full_form = oup1(df, expected_df, residuals_standard_df, residuals_adjusted_df, freeman_df)
    print(f"Residuals : Observed,Expected,Residuals std,Residuals adj,Deviates:\n{df_full_form}")
    write(f"Residuals : Observed,Expected,Residuals std,Residuals adj,Deviates:\n{df_full_form}")
    # Thresholds
    alpha_01 = 0.01
    alpha_05 = 0.05
    threshold_01 = norm.ppf(1 - alpha_01 / 2)  # Critical value for alpha = 0.01 : thres = 2.58
    threshold_05 = norm.ppf(1 - alpha_05 / 2)  # Critical value for alpha = 0.05 ; thres = 1.96
    threshold_05_form = f"{threshold_05:.3e}" if threshold_05 < 0.001 else f"{threshold_05:.3f}"
    threshold_01_form = f"{threshold_01:.3e}" if threshold_01 < 0.001 else f"{threshold_01:.3f}"
    # Residuals std : Symbol table
    symbol_std_df = oup2(residuals_standard_df, threshold_05, threshold_01)
    print(f"Residuals standard: Symbol table threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_std_df}")
    write(f"Residuals standard: Symbol table threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_std_df}")
    # Residuals std : Symbol list
    symbol_std_list = oup3(df, symbol_std_df)
    print(f"Residuals standard: Symbol list\n{symbol_std_list.to_string(index=False)}")
    write(f"Residuals standard: Symbol list\n{symbol_std_list.to_string(index=False)}")
    # Residuals adj : Symbol table
    symbol_adj_df = oup2(residuals_adjusted_df, threshold_05, threshold_01)
    print(f"Residuals adjusted: Symbol table threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_adj_df}")
    write(f"Residuals adjusted: Symbol table threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_adj_df}")
    # Residuals adj : Symbol list
    symbol_adj_list = oup3(df, symbol_adj_df)
    print(f"Residuals adjusted: Symbol list\n{symbol_adj_list.to_string(index=False)}")
    write(f"Residuals adjusted: Symbol list\n{symbol_adj_list.to_string(index=False)}")
    pass
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

---

### **Why Adjusted Residuals Are Important**
- In your data, the row (sex: M/F) and column (age bins) totals are imbalanced. For instance, males contribute more to certain age bins (e.g., 50-59, 40-49), while females dominate in others. Standardized residuals might exaggerate the significance of deviations due to this imbalance.  
- Adjusted residuals correct for this by factoring in row and column proportions, thus reflecting a more nuanced view of the deviation.

---

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
if __name__ == "__main__":
   
    def perplexity(): 
        # ke15_agbi_sexe_abso.c3c6_full.abso
        data = np.array([[3, 2],
                        [6, 7],
                        [10, 23],
                        [26, 32],
                        [41, 46],
                        [35, 46],
                        [29, 38],
                        [6, 11],
                        [0, 1]])
        age_bins = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
        df = pd.DataFrame(data, index=age_bins, columns=['M', 'F'])

        # Calculate row totals and column totals
        row_totals = df.sum(axis=1)
        col_totals = df.sum(axis=0)
        grand_total = df.values.sum()
        # Calculate expected frequencies
        expected = np.outer(row_totals, col_totals) / grand_total
        # Convert expected frequencies to DataFrame
        expected_df = pd.DataFrame(expected, index=age_bins, columns=['M (Expected)', 'F (Expected)'])
        # Calculate standardized residuals
        standardized_residuals = (df.values - expected) / np.sqrt(expected)
        # Convert standardized residuals to DataFrame
        std_residuals_df = pd.DataFrame(standardized_residuals, index=age_bins, columns=['M (Std Residual)', 'F (Std Residual)'])

        # Display results
        print("\nPERPLEXITY : Observed Frequencies:")
        print(df)
        print("\nPERPLEXITY : Expected Frequencies:")
        print(expected_df)
        print("\nPERPLEXITY : Standardized Residuals:")
        print(std_residuals_df)
        
    def openai():

        # Original data
        data = np.array([[3, 2],
                        [6, 7],
                        [10, 23],
                        [26, 32],
                        [41, 46],
                        [35, 46],
                        [29, 38],
                        [6, 11],
                        [0, 1]])

        age_bins = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
        df = pd.DataFrame(data, index=age_bins, columns=['M', 'F'])

        # Total sum of the entire table
        total_sum = df.to_numpy().sum()
        # Row sums and column sums
        row_sums = df.sum(axis=1)
        col_sums = df.sum(axis=0)
        # Expected values
        expected = np.outer(row_sums, col_sums) / total_sum
        expected_df = pd.DataFrame(expected, index=age_bins, columns=['M', 'F'])
        # Residuals
        residuals = (df - expected_df) / np.sqrt(expected_df)
        residuals_standard_df = pd.DataFrame(residuals, index=age_bins, columns=['M', 'F'])

        # Output the dataframes
        print("\nOPENAI : Observed Data:\n", df)
        print("\nOPENAI : Expected Data:\n", expected_df)
        print("\nOPENAI : Residuals:\n", residuals_standard_df)
        
    def tvc(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

        # Exec
        # Total sum of the entire table
        total_sum = df.to_numpy().sum()
        # Row sums and column sums
        row_sums = df.sum(axis=1)
        col_sums = df.sum(axis=0)  
        # Expected values
        expected = np.outer(row_sums, col_sums) / total_sum
        expected_df = pd.DataFrame(expected, index=age_bins, columns=colu_cate_list)
        # Residuals
        residuals = (df - expected_df) / np.sqrt(expected_df)
        residuals_standard_df = pd.DataFrame(residuals, index=age_bins, columns=colu_cate_list)
        
        # Resu
        def oup1(df, expected_df, residuals_standard_df):
            def format_cell(cell):
                return round(cell, 2) # f"{cell:.2e}" if cell < 0.01 else f"{cell:.2f}"
            expected_df_form = expected_df.applymap(format_cell)
            residuals_df_form = residuals_standard_df.applymap(format_cell)
            print(f"Residuals : Observed:\n{df}")
            print(f"Residuals : Expected:\n{expected_df_form}")
            print(f"Residuals : Residuals:\n{residuals_df_form}")
        
            colu_expe_list = [col + "_expe" for col in colu_cate_list]
            df_expe = pd.DataFrame(expected, index=df.index)
            df_expe.columns = colu_expe_list
            colu_resi_list = [col + "_resi" for col in colu_cate_list]
            df_resi = pd.DataFrame(residuals, index=df.index)
            df_resi.columns = colu_resi_list
            df_full = pd.concat([df, df_expe, df_resi], axis=1)
            df_full_form = df_full.applymap(format_cell)
            print(f"Residuals : Observed,Expected,Residuals:\n{df_full_form}")
            #write(f"Residuals : Observed,Expected,Residuals:\n{df_full_form}")
        
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
            symbol_df = resi_arra.applymap(lambda x: residual_symbol(x, threshold_05, threshold_01))
            threshold_05_form = f"{threshold_05:.3e}" if threshold_05 < 0.001 else f"{threshold_05:.3f}"
            threshold_01_form = f"{threshold_01:.3e}" if threshold_01 < 0.001 else f"{threshold_01:.3f}"
            print(f"Residuals : Synthesis threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_df}")
            #write(f"Residuals : Synthesis threshold_05:{threshold_05_form}, threshold_01:{threshold_01_form} \n{symbol_df}")
        
        # Resu
        oup1(df, expected_df, residuals_standard_df)
        oup2(residuals_standard_df, threshold_05, threshold_01)
    #perplexity()
    #openai()
    data = np.array([[3, 2],
                    [6, 7],
                    [10, 23],
                    [26, 32],
                    [41, 46],
                    [35, 46],
                    [29, 38],
                    [6, 11],
                    [0, 1]])
    age_bins = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    df = pd.DataFrame(data, index=age_bins, columns=['M', 'F'])
    what = "test"
    indx_cate_list = None
    colu_cate_list = ['M', 'F']
    indx_name = None
    colu_name = None
    tvc(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    
    '''
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

    For the off-diagonal, the correct list is :

    ++
    NA C6
    C0 C6
    C6 NA
    C6 C0
    C6 C2 
    +
    C2 C6
    --
    C3 C6
    C6 C3
    C6 C4
    -
    NA C0
    C0 NA
    C4 C6
    C5 C3

    For the diagonal cells :

    ++
    C1 C1
    C3 C3
    C4 C4
    C5 C5
    --
    NA NA
    '''
    print ("**************")
    print ("CEAP RESIDUALS")
    print ("**************")
    
    import pandas as pd
    import numpy as np
    from scipy import stats

    # Recreate the contingency table
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

    index = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap_L')
    columns = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap_R')

    df = pd.DataFrame(data, index=index, columns=columns)

    # Calculate expected frequencies
    chi2, p, dof, expected = stats.chi2_contingency(df)

    # Calculate standardized residuals
    observed = df.values
    standardized_residuals = (observed - expected) / np.sqrt(expected)

    # Define thresholds
    threshold_05 = stats.norm.ppf(1 - 0.05 / 2)  # Critical value for alpha = 0.05
    threshold_01 = stats.norm.ppf(1 - 0.01 / 2)  # Critical value for alpha = 0.01

    # Function to categorize residuals
    def categorize_residual(residual):
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

    # Apply categorization
    categorized_residuals = np.vectorize(categorize_residual)(standardized_residuals)

    # Count residuals
    def count_residuals(residuals, diagonal=True):
        if diagonal:
            residuals = np.diag(residuals)
        else:
            residuals = residuals[~np.eye(residuals.shape[0], dtype=bool)]
        
        counts = {
            '++': np.sum(residuals == '++'),
            '+': np.sum(residuals == '+'),
            '--': np.sum(residuals == '--'),
            '-': np.sum(residuals == '-'),
            '.': np.sum(residuals == '.')
        }
        return counts

    # Calculate counts and percentages
    diagonal_counts = count_residuals(categorized_residuals, diagonal=True)
    off_diagonal_counts = count_residuals(categorized_residuals, diagonal=False)

    diagonal_total = sum(diagonal_counts.values())
    off_diagonal_total = sum(off_diagonal_counts.values())

    diagonal_percentages = {k: v / diagonal_total * 100 for k, v in diagonal_counts.items()}
    off_diagonal_percentages = {k: v / off_diagonal_total * 100 for k, v in off_diagonal_counts.items()}

    # Calculate total counts and percentages
    total_counts = {k: diagonal_counts[k] + off_diagonal_counts[k] for k in diagonal_counts.keys()}
    total_total = sum(total_counts.values())
    total_percentages = {k: v / total_total * 100 for k, v in total_counts.items()}

    # Print results
    print("Diagonal Counts:")
    for k, v in diagonal_counts.items():
        print(f"{k}: {v} ({diagonal_percentages[k]:.2f}%)")

    print("\nOff-Diagonal Counts:")
    for k, v in off_diagonal_counts.items():
        print(f"{k}: {v} ({off_diagonal_percentages[k]:.2f}%)")

    print("\nTotal Counts (Diagonal + Off-Diagonal):")
    for k, v in total_counts.items():
        print(f"{k}: {v} ({total_percentages[k]:.2f}%)")
