from datetime import datetime
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from util_file_inpu_mbre import inp1
from util_file_mngr import set_file_objc, write
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

'''
       doss  age age_bin sexe mbre ceap  ceap_numeric
0     D9972   54   50_59    F    G   C2             2
1     D9972   54   50_59    F    G   C6             6
2     D9972   54   50_59    F    D   C2             2
4     D9921   54   50_59    M    G   C3             3
'''
'''
Based on the information provided, the process appears to be correct. Let's break it down:

1. Input Data:
   The input DataFrame contains the necessary columns: 'doss', 'age', 'age_bin', 'sexe', 'mbre', 'ceap', and 'ceap_numeric'.

2. CEAP Mapping:
   The 'ceap_numeric' column is correctly created using the provided ceap_map.

3. Model Fitting Function (modl_fit):
   - The function creates a copy of the input DataFrame to avoid warnings.
   - It correctly creates dummy variables for categorical predictors ('age_bin', 'sexe', 'mbre').
   - The predictor columns are correctly identified, excluding 'ceap', 'ceap_numeric', 'doss', and 'age'.
   - An OrderedModel is created with 'ceap_numeric' as the dependent variable and the predictor columns as independent variables.
   - The model is fit using the BFGS method with a maximum of 1000 iterations.

4. Results:
   - The optimization terminated successfully after 85 iterations.
   - The final function value is 1.681831, which is reasonable.

5. Predictor Columns:
   The predictor columns list is correct and includes all the dummy variables created for 'age_bin', 'sexe', and 'mbre'.

Overall, the process appears to be correct and follows a standard approach for fitting an ordered logistic regression model using dummy variables for categorical predictors. 
The successful termination of the optimization process suggests that the model fitting was completed without errors.
'''
'''
Thank you for providing the detailed results. Let's evaluate the output:

1. Model Fit:
   - The model converged successfully, as evidenced by the presence of coefficient estimates and standard errors.
   - Log-Likelihood: -1296.7
   - AIC: 2625
   - BIC: 2700
   These metrics suggest a reasonable fit, but without comparison models, it's hard to say if it's optimal.

2. Sample Size:
   - No. Observations: 771
   - Df Residuals: 755
   - Df Model: 10
   The sample size is adequate for the number of predictors.

3. Coefficients:
   - Most age group coefficients are positive but not statistically significant (p > 0.05).
   - 'sexe_M' (male) has a small positive effect (0.0905) but is not significant (p = 0.496).
   - 'mbre_G' has a significant positive effect (0.4504, p = 0.001), indicating that this group characteristic is associated with higher CEAP scores.

4. Thresholds:
   - The model includes threshold coefficients (0/1, 1/2, etc.) which are typical in ordered logistic regression.
   - Most thresholds are statistically significant, indicating distinct categories in the CEAP scale.

5. Interpretation:
   - The only clearly significant predictor is 'mbre_G', suggesting that this group characteristic is associated with higher CEAP scores.
   - Age and sex do not appear to have strong, statistically significant effects on CEAP scores in this model.

Overall, the model seems to be correctly specified and fitted. However, the predictive power might be limited, as most predictors are not statistically significant. 
It might be worth considering if there are other important variables that could be included in the model or if a different model structure might be more appropriate for this data.
'''
'''
Great! It looks like the model fitting was successful with the increased dataset. Let's break down the results and interpret them.

### Model Summary

The model summary provides the following key information:

- **Dependent Variable**: `ceap_numeric`
- **Log-Likelihood**: -1296.7
- **AIC**: 2625.4
- **BIC**: 2700.4
- **Number of Observations**: 771
- **Degrees of Freedom (Residuals)**: 755
- **Degrees of Freedom (Model)**: 10

### Coefficients

The coefficients table provides the estimated coefficients for each predictor variable along with their standard errors, z-values, p-values, and 95% confidence intervals.

#### Predictor Coefficients

1. **Age Bins**:
   - `age_bin_20_29`: 0.3939 (p-value: 0.552)
   - `age_bin_30_39`: 0.4755 (p-value: 0.426)
   - `age_bin_40_49`: 0.8223 (p-value: 0.159)
   - `age_bin_50_59`: 0.5558 (p-value: 0.333)
   - `age_bin_60_69`: 0.5232 (p-value: 0.362)
   - `age_bin_70_79`: 0.7029 (p-value: 0.224)
   - `age_bin_80_89`: 0.5779 (p-value: 0.372)
   - `age_bin_90_99`: 0.9899 (p-value: 0.393)

2. **Sex**:
   - `sexe_M`: 0.0905 (p-value: 0.496)

3. **Membership**:
   - `mbre_G`: 0.4504 (p-value: 0.001)

#### Cut-off Points

The cut-off points for the ordered logistic regression model are:

- `0/1`: -1.5176 (p-value: 0.009)
- `1/2`: -1.7787 (p-value: 0.000)
- `2/3`: -0.0237 (p-value: 0.809)
- `3/4`: 0.3693 (p-value: 0.000)
- `4/5`: -0.6130 (p-value: 0.000)
- `5/6`: -1.0779 (p-value: 0.000)

### Interpretation

1. **Age Bins**:
   - The coefficients for the age bins are not statistically significant (p-values > 0.05), indicating that age does not have a significant impact on the `ceap_numeric` variable.

2. **Sex**:
   - The coefficient for `sexe_M` is also not statistically significant (p-value: 0.496), suggesting that sex does not have a significant impact on the `ceap_numeric` variable.

3. **Membership**:
   - The coefficient for `mbre_G` is statistically significant (p-value: 0.001), indicating that membership in group `G` has a significant positive impact on the `ceap_numeric` variable.

### Conclusion

The model fitting was successful, and the results suggest that the `mbre_G` variable is a significant predictor of the `ceap_numeric` variable. The age bins and sex do not appear to be significant predictors based on the p-values.

### Next Steps

1. **Model Validation**:
   - Validate the model using a separate validation dataset to ensure its generalizability.

2. **Feature Engineering**:
   - Consider additional feature engineering to improve the model's performance. For example, you might explore interactions between variables or create new features based on domain knowledge.

3. **Model Interpretation**:
   - Interpret the model coefficients in the context of your domain to understand the practical implications of the significant predictors.

4. **Model Improvement**:
   - If the model performance is not satisfactory, consider trying different modeling techniques or adding more relevant predictors.

By following these steps, you can further refine and validate your model to ensure it accurately predicts the `ceap_numeric` variable.
'''
'''
Yes, you're correct. The coefficient for `mbre_G` being statistically significant (with a p-value of 0.001) indicates that membership in group `G` has a significant positive impact on the `ceap_numeric` variable. This means that, all else being equal, individuals in group `G` are more likely to have higher values of `ceap_numeric` compared to those not in group `G`.

### Detailed Interpretation

1. **Coefficient Value**:
   - The coefficient for `mbre_G` is 0.4504. This value represents the change in the log-odds of being in a higher category of `ceap_numeric` for individuals in group `G` compared to those not in group `G`.

2. **P-Value**:
   - The p-value of 0.001 indicates that there is a very low probability (0.1%) that the observed effect of `mbre_G` on `ceap_numeric` is due to random chance. This provides strong evidence that `mbre_G` is a significant predictor.

3. **Confidence Interval**:
   - The 95% confidence interval for the coefficient is [0.196, 0.705]. This interval does not include zero, further confirming that the effect of `mbre_G` is statistically significant.

### Practical Implications

- **Group G Membership**:
  - Individuals in group `G` are more likely to have higher values of `ceap_numeric`. This could have practical implications depending on what `ceap_numeric` represents. For example, if `ceap_numeric` represents a health outcome, membership in group `G` might be associated with better health outcomes.

- **Policy and Interventions**:
  - If `mbre_G` represents a modifiable factor (e.g., participation in a program or treatment), this finding could inform policy decisions or interventions aimed at improving outcomes related to `ceap_numeric`.

### Next Steps

1. **Further Analysis**:
   - Conduct further analysis to understand why membership in group `G` has this effect. This could involve exploring interactions with other variables or conducting subgroup analyses.

2. **Model Validation**:
   - Validate the model on a separate dataset to ensure that the findings are robust and generalizable.

3. **Communication**:
   - Communicate the findings to stakeholders who can use this information to make informed decisions. For example, if `mbre_G` represents a specific intervention, stakeholders might want to invest more resources in promoting this intervention.

4. **Additional Predictors**:
   - Consider adding or refining other predictors to improve the model's explanatory power. For example, you might explore interactions between `mbre_G` and other variables or include additional relevant features.

By focusing on the significant predictor `mbre_G` and understanding its impact, you can derive actionable insights and make data-driven decisions.
'''
'''
Thank you for providing the context. Given that `mbre_G` represents the limb (left or right) that exhibits the clinical sign `ceap` (C0 to C6) in chronic venous insufficiency, the interpretation of the model results can be refined as follows:

### Interpretation in Context

1. **Coefficient for `mbre_G`**:
   - The coefficient for `mbre_G` is 0.4504, with a p-value of 0.001. This indicates that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`).

2. **Practical Implication**:
   - Individuals with chronic venous insufficiency affecting the left limb are more likely to have higher values of `ceap_numeric` (indicating more severe clinical signs) compared to those with the condition affecting the right limb.

### Detailed Analysis

1. **Clinical Significance**:
   - The finding suggests that there might be underlying physiological or anatomical differences between the left and right limbs that contribute to the severity of chronic venous insufficiency. This could be due to differences in venous drainage, muscle pump function, or other factors.

2. **Potential Interventions**:
   - This insight could inform clinical practice and interventions. For example, healthcare providers might need to pay more attention to the left limb in patients with chronic venous insufficiency, or they might consider targeted interventions for the left limb to manage the condition more effectively.

3. **Further Research**:
   - Further research could explore why the left limb is more affected. This might involve studying the anatomy and physiology of the venous system in the left and right limbs, as well as potential environmental or lifestyle factors that could contribute to this difference.

### Next Steps

1. **Model Validation**:
   - Validate the model on a separate dataset to ensure that the findings are robust and generalizable. This could involve splitting your data into training and test sets or using cross-validation techniques.

2. **Subgroup Analysis**:
   - Conduct subgroup analyses to understand if the effect of `mbre_G` varies across different age groups, sexes, or other relevant subgroups. This could provide more nuanced insights into the factors contributing to the severity of chronic venous insufficiency.

3. **Interaction Effects**:
   - Explore interaction effects between `mbre_G` and other variables. For example, there might be interactions between `mbre_G` and age, sex, or other clinical factors that could provide additional insights.

4. **Communication with Stakeholders**:
   - Communicate the findings to healthcare providers, researchers, and other stakeholders. This could inform clinical guidelines, research priorities, and patient management strategies.

5. **Additional Predictors**:
   - Consider adding or refining other predictors to improve the model's explanatory power. For example, you might include additional clinical variables, lifestyle factors, or genetic information that could contribute to the severity of chronic venous insufficiency.

### Conclusion

The significant positive impact of `mbre_G` on `ceap_numeric` suggests that the left limb is more likely to exhibit severe clinical signs of chronic venous insufficiency compared to the right limb. 
This finding has important clinical implications and warrants further investigation to understand the underlying mechanisms and inform targeted interventions.
'''
'''
I apologize for any confusion. Let's clarify the technical details regarding the interpretation of the dummy variables and the reference category in the context of your model.

### Dummy Variable Encoding

When creating dummy variables for categorical predictors, one category is typically dropped to avoid multicollinearity. This dropped category becomes the reference category against which the other categories are compared.

### Interpretation of `mbre_G`

1. **Dummy Variable Creation**:
   - The column `mbre` has two categories: `G` (left limb) and `D` (right limb).
   - When creating dummy variables, one of these categories is dropped. In your case, `mbre_G` is created, which means `D` is the reference category.

2. **Reference Category**:
   - The reference category is `D` (right limb). The coefficient for `mbre_G` compares the effect of the left limb (`G`) to the right limb (`D`).

3. **Coefficient Interpretation**:
   - The coefficient for `mbre_G` is 0.4504. This positive value indicates that, compared to the reference category (right limb, `D`), the left limb (`G`) has a higher log-odds of being in a higher category of `ceap_numeric`.

### Why `G` > `D`?

- **Positive Coefficient**:
  - The positive coefficient (0.4504) for `mbre_G` means that the left limb (`G`) is associated with higher values of `ceap_numeric` compared to the right limb (`D`).

- **Statistical Significance**:
  - The p-value of 0.001 and the confidence interval [0.196, 0.705] confirm that this positive effect is statistically significant.

### Technical Clarification

- **Suffix `G` in `mbre_G`**:
  - The suffix `G` in `mbre_G` does not imply that `G` is the reference category. Instead, it indicates that `G` is the category being compared to the reference category (`D`).

- **Reference Category `D`**:
  - The reference category is `D` (right limb). The coefficient for `mbre_G` tells us how the left limb (`G`) compares to the right limb (`D`).

### Conclusion

The positive and statistically significant coefficient for `mbre_G` (0.4504) indicates that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`). 
This conclusion is based on the direction and significance of the coefficient, with `D` being the reference category.
'''
'''
Thank you for providing the context. Given that `mbre_G` represents the limb (left or right) that exhibits the clinical sign `ceap` (C0 to C6) in chronic venous insufficiency, the interpretation of the model results can be refined as follows:

### Interpretation in Context

1. **Coefficient for `mbre_G`**:
   - The coefficient for `mbre_G` is 0.4504, with a p-value of 0.001. This indicates that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`).

2. **Practical Implication**:
   - Individuals with chronic venous insufficiency affecting the left limb are more likely to have higher values of `ceap_numeric` (indicating more severe clinical signs) compared to those with the condition affecting the right limb.

### Detailed Analysis

1. **Clinical Significance**:
   - The finding suggests that there might be underlying physiological or anatomical differences between the left and right limbs that contribute to the severity of chronic venous insufficiency. This could be due to differences in venous drainage, muscle pump function, or other factors.

2. **Potential Interventions**:
   - This insight could inform clinical practice and interventions. For example, healthcare providers might need to pay more attention to the left limb in patients with chronic venous insufficiency, or they might consider targeted interventions for the left limb to manage the condition more effectively.

3. **Further Research**:
   - Further research could explore why the left limb is more affected. This might involve studying the anatomy and physiology of the venous system in the left and right limbs, as well as potential environmental or lifestyle factors that could contribute to this difference.

### Next Steps

1. **Model Validation**:
   - Validate the model on a separate dataset to ensure that the findings are robust and generalizable. This could involve splitting your data into training and test sets or using cross-validation techniques.

2. **Subgroup Analysis**:
   - Conduct subgroup analyses to understand if the effect of `mbre_G` varies across different age groups, sexes, or other relevant subgroups. This could provide more nuanced insights into the factors contributing to the severity of chronic venous insufficiency.

3. **Interaction Effects**:
   - Explore interaction effects between `mbre_G` and other variables. For example, there might be interactions between `mbre_G` and age, sex, or other clinical factors that could provide additional insights.

4. **Communication with Stakeholders**:
   - Communicate the findings to healthcare providers, researchers, and other stakeholders. This could inform clinical guidelines, research priorities, and patient management strategies.

5. **Additional Predictors**:
   - Consider adding or refining other predictors to improve the model's explanatory power. For example, you might include additional clinical variables, lifestyle factors, or genetic information that could contribute to the severity of chronic venous insufficiency.

### Conclusion

The significant positive impact of `mbre_G` on `ceap_numeric` suggests that the left limb is more likely to exhibit severe clinical signs of chronic venous insufficiency compared to the right limb. 
This finding has important clinical implications and warrants further investigation to understand the underlying mechanisms and inform targeted interventions.
'''
# 1. Fit the Ordinal Logistic Regression model
def modl_fit(df): 
    df = df.copy() # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    print (df)
    # Create dummy variables for categorical predictors
    df_encoded = pd.get_dummies(df, columns=['age_bin', 'sexe', 'mbre'], drop_first=True)
    # Select the predictor columns
    # Predictor columns : ['age_bin_20_29', 'age_bin_30_39', 'age_bin_40_49', 'age_bin_50_59', 'age_bin_60_69', 'age_bin_70_79', 'age_bin_80_89', 'age_bin_90_99', 'sexe_M', 'mbre_G']
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]
    print (f'Predictor columns : {predictor_columns}')
    # Fit the model
    model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    try:
        results = model.fit(method='bfgs', maxiter=1000)  # Use BFGS method with increased max iterations
    except:
        print("Model fitting failed. Try with a simpler model or check your data.")
        return None
    return results, df_encoded
'''
Thank you for providing the results of the proportional odds assumption check. Let's analyze these results:

1. Convergence Issues:
   - For CEAP > 0, CEAP > 1, and CEAP > 2, we see warnings about the maximum number of iterations being exceeded and failure to converge. This suggests potential issues with model fitting for these thresholds.
   - For CEAP > 3, CEAP > 4, and CEAP > 5, the optimization terminated successfully.

2. Coefficient Stability:
   - Ideally, for the proportional odds assumption to hold, we would expect the coefficients to be relatively stable across thresholds. However, we observe considerable variation:

   a) Age groups: Coefficients vary widely across thresholds. For example:
      - age_bin_20_29 ranges from -0.321890 to 1.053780
      - age_bin_30_39 ranges from -0.452474 to 1.368048

   b) Sex (sexe_M):
      - Ranges from -0.233959 to 0.408207, changing sign across thresholds

   c) Group (mbre_G):
      - More stable, ranging from 0.380338 to 0.691814, always positive

3. Extreme Values:
   - age_bin_90_99 shows extremely large coefficients for lower thresholds (18.574513, 14.535075, 18.897656), which then drastically reduce for higher thresholds. This could indicate issues with this category, possibly due to small sample size.

4. Overall Assessment:
   - The proportional odds assumption appears to be violated for most predictors, as the coefficients are not similar across thresholds.
   - The mbre_G variable shows the most consistency, suggesting it might be the most reliable predictor.
   - The age group variables and sex variable show substantial variation, indicating that their effects may differ across CEAP levels.

Recommendations:
1. Consider using a more flexible model that doesn't rely on the proportional odds assumption, such as a multinomial logistic regression or a partial proportional odds model.
2. Investigate the 90-99 age group to understand why it's causing extreme coefficients at lower thresholds.
3. Be cautious in interpreting the original ordered logistic regression results, as the violation of the proportional odds assumption may lead to misleading conclusions.
4. If possible, collect more data or consider combining some of the CEAP levels to improve model stability.

In conclusion, the proportional odds assumption does not appear to be met for this model, which calls into question the validity of the original ordered logistic regression results.
'''
# 2. Check the Proportional Odds Assumption : done by comparing the coefficients across different thresholds
def check_proportional_odds(results, df_encoded):
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        df_encoded[f'ceap_gt_{i}'] = (df_encoded['ceap_numeric'] > i).astype(int)
        formula = f"ceap_gt_{i} ~ " + " + ".join(predictor_columns)
        model_i = smf.logit(formula, df_encoded).fit()
        print(f"\nCoefficients for CEAP > {i}:")
        print(model_i.params)
        df_encoded.drop(f'ceap_gt_{i}', axis=1, inplace=True)  # Clean up temporary column
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")
def check_proportional_odds(df_encoded, results):
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        df_encoded[f'ceap_gt_{i}'] = (df_encoded['ceap_numeric'] > i).astype(int)
        formula = f"ceap_gt_{i} ~ " + " + ".join(predictor_columns)
        model_i = smf.logit(formula, df_encoded).fit()
        print(f"\nCoefficients for CEAP > {i}:")
        print(model_i.params)
        df_encoded.drop(f'ceap_gt_{i}', axis=1, inplace=True)  # Clean up temporary column
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")

# 3. Visual check of proportional odds assumption
def plot_proportional_odds(df, dependent_var, independent_vars):
    plt.figure(figsize=(12, 8))
    
    for i in range(df[dependent_var].nunique() - 1):
        df[f'{dependent_var}_gt_{i}'] = (df[dependent_var] > i).astype(int)
        formula = f"{dependent_var}_gt_{i} ~ " + " + ".join(independent_vars)
        model = OrderedModel.from_formula(formula, data=df, distr='logit')
        results = model.fit()
        
        coeffs = results.params[1:]  # Exclude intercept
        plt.scatter([i] * len(coeffs), coeffs, label=f'CEAP > {i}')
        
        for j, var in enumerate(independent_vars):
            plt.plot([i-0.1, i+0.1], [coeffs[j], coeffs[j]], 'k-')
        
        df.drop(f'{dependent_var}_gt_{i}', axis=1, inplace=True)
    
    plt.xlabel('Threshold')
    plt.ylabel('Coefficient Value')
    plt.title('Proportional Odds Assumption Check')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 4. Check for multicollinearity
'''
This function does the following:
It uses pd.get_dummies() to create dummy variables for categorical features. The drop_first=True argument is used to avoid perfect multicollinearity.
It adds a constant term to the feature matrix, which is required for VIF calculation.
It calculates the VIF for each feature, including the dummy variables.
When interpreting the results:
VIF values greater than 5-10 are typically considered to indicate high multicollinearity.
The constant term (intercept) usually has a high VIF and can be ignored.
For dummy variables, focus on the overall pattern rather than individual values.
Remember that for categorical variables, some degree of correlation between dummy variables is expected and doesn't necessarily indicate problematic multicollinearity.
'''
def check_multicollinearity(df, features):
    # Create dummy variables for categorical features
    X = pd.get_dummies(df[features], drop_first=True)
    
    # Add a constant term to the features
    X = add_constant(X)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factors:")
    print(vif_data)

def main(filt_name, filt_valu, file_path, jrnl_file_path):
    
    with open(jrnl_file_path, 'w') as file:
            
            set_file_objc(file)
            date_curr = datetime.now()
            date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
            write (">>> >>> >>>")
            write (date_form)
            write (">>> >>> >>>")
        
            # Input
            # -----
            df1, df2, df3 = inp1(file_path, filt_name, filt_valu)  
            df = df2 # eliminate 'NA'
            df = df[['doss', 'age', 'age_bin', 'sexe', 'mbre', 'ceap']] 
            ceap_map = {'C0': 0, 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6}
            df['ceap_numeric'] = df['ceap'].map(ceap_map)
            df['age_bin'] = df['age_bin'].replace(r'(\d{2})-(\d{2})', r'\1_\2', regex=True)
            
            # Exec
            # ----
            results, df_encoded  = modl_fit(df)
            if results is not None:
                print(results.summary())
                print (df_encoded)
                check_proportional_odds(df_encoded, results)
            plot_proportional_odds(df_encoded, 'ceap_numeric', ['age_bin_20_29', 'age_bin_30_39', 'sexe_M', 'mbre_G'])
            
            check_multicollinearity(df, ['age_bin', 'sexe', 'mbre'])
        
def multi_04_explai_02_step_0301():

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
    filt_name = 'sexe'
    filt_valu = None # 'G' 'D'
    #    
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}_trac.txt' if filt_valu is not None else f'{script_name}_trac.txt')
    main(filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    multi_04_explai_02_step_0301()