from datetime import datetime
import os
import sys
import pandas as pd
from util_file_mngr import set_file_objc, write
from util_file_inpu_mbre import inp1
from ke30_ceap_xxxx import ke30_main
from scipy.stats import chi2_contingency
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


def main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
        print (jrnl_file_path) 
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
        
        # ----
        # Create multi index df
        # ----
        if False:
            data = {
                'sexe': ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                'ceap': ['C0', 'C0', 'C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
                'mbre': ['D', 'G', 'D', 'G', 'D', 'G', 'D', 'G'],
                'coun': [23, 13, 3, 3, 34, 20, 78, 78]
            }
            df_inp = pd.DataFrame(data)
        df11, df12, df13 = inp1(file_path, filt_name, filt_valu)
        print(f"df11:{type(df11)} len:{len(df11)}\n{df11}\n:{df11.index}\n:{df11.columns}")
        write(f"df11:{type(df11)} len:{len(df11)}\n{df11}\n:{df11.index}\n:{df11.columns}")
        df_ini1 = df11.groupby(['sexe', 'ceap', 'mbre'])['#'].count().reset_index(name='coun')
        print(f"df_ini1:{type(df_ini1)}\n{df_ini1}\n:{df_ini1.index}\n:{df_ini1.columns}")
        write(f"df_ini1:{type(df_ini1)}\n{df_ini1}\n:{df_ini1.index}\n:{df_ini1.columns}")
        
        # Lists for CEAP, sexe, and mbre
        ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        sexe_list = ['M', 'F']
        mbre_list = ['G', 'D']
        # Create the multi-index for the columns
        mbre_sexe_columns = pd.MultiIndex.from_product([mbre_list, sexe_list], names=['mbre', 'sexe'])
        # Initialize the output DataFrame with the desired index and columns
        df_out = pd.DataFrame(index=ceap_list, columns=mbre_sexe_columns)
        # Fill the DataFrame with the counts from df_inp
        for index, row in df_ini1.iterrows():
            ceap = row['ceap']
            sexe = row['sexe']
            mbre = row['mbre']
            coun = row['coun']
            df_out.at[ceap, (mbre, sexe)] = coun
        df_out.fillna(0, inplace=True)
        df_out = df_out.astype(int)
        print(f"df_out:{type(df_out)}\n{df_out}\n:{df_out.index}\n:{df_out.columns}\nSum:{df_out.sum().sum()}")
        write(f"df_out:{type(df_out)}\n{df_out}\n:{df_out.index}\n:{df_out.columns}\nSum:{df_out.sum().sum()}")

        # ----
        # Perform chi2 test
        # ----
        # Reset the index to convert 'ceap' to a column
        df_out = df_out.reset_index()
        # Flatten the MultiIndex columns
        df_out.columns = ['ceap'] + ['_'.join(col).strip() for col in df_out.columns[1:].values]
        # Perform the Chi-Square test
        chi2, p, dof, ex = chi2_contingency(df_out.iloc[:, 1:])
        print(f"Chi-Square Statistic: {chi2} P-value: {p} Degrees of Freedom: {dof} Expected Frequencies: {ex}")
        write(f"Chi-Square Statistic: {chi2} P-value: {p} Degrees of Freedom: {dof} Expected Frequencies: {ex}")
      
        # ----
        # Identification of pairs
        # ----
        # Flatten the DataFrame for pairwise comparisons
        df_flat = df_out.melt(id_vars='ceap', var_name='mbre_sexe', value_name='count')
        print(f"df_flat:{type(df_flat)}\n{df_flat}\n:{df_flat.index}\n:{df_flat.columns}")
        write(f"df_flat:{type(df_flat)}\n{df_flat}\n:{df_flat.index}\n:{df_flat.columns}")
        
        # Lists for mbre and sexe
        mbre_list = ['G', 'D']
        sexe_list = ['M', 'F']

        # Create a categorical data type with the specified order
        mbre_sexe_order = [f"{mbre}_{sexe}" for mbre in mbre_list for sexe in sexe_list]
        df_flat['mbre_sexe'] = pd.Categorical(df_flat['mbre_sexe'], categories=mbre_sexe_order, ordered=True)

        # Sort the DataFrame based on the categorical mbre_sexe column
        df_sorted = df_flat.sort_values(by='mbre_sexe')

        # Display the sorted DataFrame
        print(df_sorted)
        df_flat = df_sorted
        
        # Generate combinations in the sorted order
        grouped = df_flat.groupby('mbre_sexe')
        combinations_list = list(combinations(grouped, 2))

        # Ensure the combinations are evaluated in the same order
        for (mbre_sexe1, df1), (mbre_sexe2, df2) in combinations_list:
            print(f"Comparison: {mbre_sexe1} vs {mbre_sexe2}")
        
        
        # Perform pairwise Chi-Square tests
        results = []
        for (mbre_sexe1, df1), (mbre_sexe2, df2) in combinations_list: # combinations(df_flat.groupby('mbre_sexe'), 2):
            # Create a contingency table for the pairwise comparison
            contingency_table = pd.DataFrame(index=ceap_list, columns=[mbre_sexe1, mbre_sexe2])
            for ceap in ceap_list:
                contingency_table.at[ceap, mbre_sexe1] = df1[df1['ceap'] == ceap]['count'].values[0]
                contingency_table.at[ceap, mbre_sexe2] = df2[df2['ceap'] == ceap]['count'].values[0]

            # Fill NaN values with 0
            contingency_table.fillna(0, inplace=True)
            # Convert the DataFrame to integer type
            contingency_table = contingency_table.astype(int)
            # print (contingency_table)
            # Perform the Chi-Square test
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            results.append((mbre_sexe1, mbre_sexe2, chi2, p, dof))

        # Print the results
        for result in results:
            print(f"Comparison: {result[0]} vs {result[1]} Chi-Square Statistic: {result[2]} P-value: {result[3]} Degrees of Freedom: {result[4]}")
            write(f"Comparison: {result[0]} vs {result[1]} Chi-Square Statistic: {result[2]} P-value: {result[3]} Degrees of Freedom: {result[4]}")
            pass

        # Assuming 'results' is the list of tuples containing the comparison results
        if False:
            results = [
                ('D_F', 'D_M', 6.063521218388672, 0.5323522210849896, 7),
                ('D_F', 'G_F', 10.826717211095708, 0.14635816731147444, 7),
                ('D_F', 'G_M', 12.695960302024911, 0.07987199683240653, 7),
                ('D_M', 'G_F', 18.73556628755698, 0.009057327949839707, 7),
                ('D_M', 'G_M', 12.804680014219015, 0.07701276447638543, 7),
                ('G_F', 'G_M', 8.484395703592256, 0.2918217174674129, 7)
            ]

        # Create a DataFrame from the results
        df_results = pd.DataFrame(results, columns=['Comparison1', 'Comparison2', 'Chi-Square Statistic', 'P-value', 'Degrees of Freedom'])
        df_results_sorted = df_results.sort_values(by='P-value')
        
        # Function to swap the labels
        def swap_labels(label):
            parts = label.split('_')
            return f"{parts[1]}_{parts[0]}"

        # Create a new DataFrame with swapped labels
        df_swapped = df_results_sorted.copy()
        df_swapped['Comparison1'] = df_swapped['Comparison1'].apply(swap_labels)
        df_swapped['Comparison2'] = df_swapped['Comparison2'].apply(swap_labels)
        df_swapped['Chi-Square Statistic'] = df_swapped['Chi-Square Statistic'].round(3)
        df_swapped['P-value'] = df_swapped['P-value'].round(3)
        print(f"df_swapped:{type(df_swapped)}\n{df_swapped}\n:{df_swapped.index}\n:{df_swapped.columns}")
        write(f"df_swapped:{type(df_swapped)}\n{df_swapped}\n:{df_swapped.index}\n:{df_swapped.columns}")
        xlsx = True
        if xlsx: 
            file_name = 'ke51_ceap_sexe_mbre_c3c6_full_abso.xlsx'
            df_swapped.to_excel(file_name, index=False)
        # Create a mosaic plot
        plot = False
        if plot:
            sns.catplot(data=df_flat, kind="count", x="ceap", hue="mbre_sexe", col="mbre_sexe", col_wrap=2, height=4, aspect=1)
            plt.title('Mosaic Plot of CEAP Classifications by Leg and Gender')
            plt.show()
            pass
        
def ke51_ceap_sexe_mbre_c3c6_full_abso():

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
    #
    ceap_mono = False
    indx_name = 'age_bin'  
    indx_cate_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    filt_name = 'sexe'
    filt_valu = None # 'G' 'D'
    #    
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}_trac.txt' if filt_valu is not None else f'{script_name}_trac.txt')
    main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    ke51_ceap_sexe_mbre_c3c6_full_abso()
    
'''
Thank you for clarifying the meaning of the symbols. Let's rephrase the interpretation of the results with the correct terminology:

### Summary of Results

1. **Comparison: Right Leg Female (D_F) vs Right Leg Male (D_M)**
   - Chi-Square Statistic: 6.064
   - P-value: 0.532
   - Degrees of Freedom: 7
   - **Interpretation**: There is no significant association between the CEAP classifications for females with right legs and males with right legs. The p-value is much higher than 0.05, so we fail to reject the null hypothesis.

2. **Comparison: Right Leg Female (D_F) vs Left Leg Female (G_F)**
   - Chi-Square Statistic: 10.827
   - P-value: 0.146
   - Degrees of Freedom: 7
   - **Interpretation**: There is no significant association between the CEAP classifications for females with right legs and females with left leg affected. The p-value is higher than 0.05, so we fail to reject the null hypothesis.

3. **Comparison: Right Leg Female (D_F) vs Left Leg Male (G_M)**
   - Chi-Square Statistic: 12.696
   - P-value: 0.0799
   - Degrees of Freedom: 7
   - **Interpretation**: There is no significant association between the CEAP classifications for females with right legs and males with left leg affected. The p-value is slightly higher than 0.05, so we fail to reject the null hypothesis, but it's close to the significance threshold.

4. **Comparison: Right Leg Male (D_M) vs Left Leg Female (G_F)**
   - Chi-Square Statistic: 18.736
   - P-value: 0.0091
   - Degrees of Freedom: 7
   - **Interpretation**: There is a significant association between the CEAP classifications for males with right legs and females with left leg affected. The p-value is less than 0.05, so we reject the null hypothesis.

5. **Comparison: Right Leg Male (D_M) vs Left Leg Male (G_M)**
   - Chi-Square Statistic: 12.805
   - P-value: 0.077
   - Degrees of Freedom: 7
   - **Interpretation**: There is no significant association between the CEAP classifications for males with right legs and males with left leg affected. The p-value is slightly higher than 0.05, so we fail to reject the null hypothesis, but it's close to the significance threshold.

6. **Comparison: Left Leg Female (G_F) vs Left Leg Male (G_M)**
   - Chi-Square Statistic: 8.484
   - P-value: 0.292
   - Degrees of Freedom: 7
   - **Interpretation**: There is no significant association between the CEAP classifications for females with left leg affected and males with left leg affected. The p-value is much higher than 0.05, so we fail to reject the null hypothesis.

### Conclusion

- **Significant Association**: The only significant association found is between the CEAP classifications for males with right legs (D_M) and females with left leg affected (G_F), with a p-value of 0.0091.
- **No Significant Association**: For the other comparisons, the p-values are higher than 0.05, indicating no significant association between the CEAP classifications for those combinations of leg and gender.

### Next Steps

1. **Further Investigation**: If the significant association between males with right legs (D_M) and females with left leg affected (G_F) is of interest, you might want to investigate further to understand the nature of this association. This could involve additional statistical tests or exploratory data analysis.
2. **Visualization**: Create visualizations (e.g., mosaic plots, bar charts) to better understand the distribution of CEAP classifications across different combinations of leg and gender.
3. **Additional Factors**: Consider other factors that might influence the CEAP classifications and include them in your analysis.

### Example Visualization

Here's an example of how you can create a mosaic plot to visualize the associations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a mosaic plot
sns.catplot(data=df_flat, kind="count", x="ceap", hue="mbre_sexe", col="mbre_sexe", col_wrap=2, height=4, aspect=1)
plt.title('Mosaic Plot of CEAP Classifications by Leg and Gender')
plt.show()
```

This visualization can help you better understand the distribution and associations between the CEAP classifications and the combinations of leg and gender.
'''

'''
The statement "There is a significant association between the CEAP classifications for males with right legs (D_M) and females with left leg affected (G_F)" means that the distribution of CEAP classifications (NA, C0, C1, C2, C3, C4, C5, C6) is significantly different between males with right legs affected and females with left leg affected. This significant difference is indicated by the p-value being less than 0.05, which leads us to reject the null hypothesis of independence.

### Clarification

The significant association does not mean that the association is independent of the CEAP classifications (NA, C0, C1, C2, C3, C4, C5, C6). Instead, it means that the distribution of these CEAP classifications is significantly different between the two groups (males with right legs affected and females with left leg affected).

### Detailed Explanation

1. **CEAP Classifications**: The CEAP classifications (NA, C0, C1, C2, C3, C4, C5, C6) represent different stages or categories of a condition.
2. **Groups Compared**: The groups being compared are males with right legs affected (D_M) and females with left leg affected (G_F).
3. **Significant Association**: The significant association means that the distribution of the CEAP classifications is not the same between these two groups. For example, the proportion of individuals in each CEAP classification might be different between males with right legs affected and females with left leg affected.

### Example

To illustrate, let's consider the contingency table for the comparison between D_M and G_F:

```plaintext
         D_M  G_F
NA       34   32
C0       20   23
C1        3    3
C2       23   34
C3       45   78
C4       16   29
C5       10   14
C6       39   44
```

The Chi-Square test compares the observed frequencies in this table to the expected frequencies under the null hypothesis of no association. The significant p-value (less than 0.05) indicates that the observed frequencies are significantly different from the expected frequencies, suggesting that the distribution of CEAP classifications is different between males with right legs affected and females with left leg affected.

### Conclusion

- **Significant Association**: There is a significant association between the CEAP classifications for males with right legs affected (D_M) and females with left leg affected (G_F). 
 !!!!!! This means that the distribution of CEAP classifications is significantly different between these two groups. !!!!!!!
- **Not Independent of CEAP Classifications**: The significant association is not independent of the CEAP classifications. 
Instead, it indicates that the distribution of the CEAP classifications (NA, C0, C1, C2, C3, C4, C5, C6) 
is significantly different between the two groups.

### Next Steps

1. **Further Investigation**: To understand the nature of this significant association, you might want to perform additional analyses, such as examining the proportions of each CEAP classification within each group or conducting post-hoc tests to identify which specific CEAP classifications contribute most to the significant association.
2. **Visualization**: Create visualizations (e.g., mosaic plots, bar charts) to better understand the distribution of CEAP classifications across different combinations of leg and gender.
3. **Additional Factors**: Consider other factors that might influence the CEAP classifications and include them in your analysis.

### Example Visualization

Here's an example of how you can create a mosaic plot to visualize the associations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a mosaic plot
sns.catplot(data=df_flat, kind="count", x="ceap", hue="mbre_sexe", col="mbre_sexe", col_wrap=2, height=4, aspect=1)
plt.title('Mosaic Plot of CEAP Classifications by Leg and Gender')
plt.show()
```

This visualization can help you better understand the distribution and associations between the CEAP classifications and the combinations of leg and gender.
'''
'''
IMPORTANT

The significant association found between males with right legs affected (D_M) and females with both legs affected (G_F) suggests that there is a notable difference in the distribution of CEAP classifications between these two groups. Although the mosaic plot shows normalized counts, the Chi-Square test 
indicates that the observed frequencies differ significantly from the expected frequencies under the null hypothesis of no association.

'''