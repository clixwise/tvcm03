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
import pandas as pd
import statsmodels.api as sm


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
# 1. Fit the Ordinal Logistic Regression model
def modl_fit_variant_1(df): 
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

'''
'''
The conclusion that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`) 
is based on the interpretation of the coefficient for `mbre_G` in the ordered logistic regression model. Let's break down how this conclusion is derived from the data and results:

### Understanding the Coefficient

1. **Coefficient Value**:
   - The coefficient for `mbre_G` is 0.4504. This positive value indicates that, all else being equal, the log-odds of being in a higher category of `ceap_numeric` increase by 0.4504 
     for individuals with the left limb (`G`) affected compared to those with the right limb (`D`) affected.

2. **P-Value**:
   - The p-value for `mbre_G` is 0.001, which is statistically significant. This means there is strong evidence that the effect of `mbre_G` on `ceap_numeric` is not due to random chance.

3. **Confidence Interval**:
   - The 95% confidence interval for the coefficient is [0.196, 0.705]. Since this interval does not include zero, it further confirms that the effect is statistically significant and positive.

### Interpreting the Direction of the Effect

- **Positive Coefficient**:
  - A positive coefficient (0.4504) for `mbre_G` means that the left limb (`G`) is associated with higher values of `ceap_numeric` compared to the right limb (`D`). In other words, the presence of `G` (left limb) increases the likelihood of being in a higher category of `ceap_numeric`.

- **Reference Category**:
  - In the dummy variable encoding, `mbre_D` (right limb) is the reference category. The coefficient for `mbre_G` compares the effect of the left limb to this reference category. A positive coefficient indicates that the left limb has a higher impact on `ceap_numeric` than the right limb.

### Why Not the Other Way Around (D > G)?

- **Sign of the Coefficient**:
  - If the coefficient for `mbre_G` were negative, it would indicate that the left limb (`G`) is associated with lower values of `ceap_numeric` compared to the right limb (`D`). However, the coefficient is positive (0.4504), so this is not the case.

- **Statistical Significance**:
  - The p-value and confidence interval confirm that the positive effect of `mbre_G` is statistically significant. If the effect were not significant, we could not conclude a directional relationship.

### Conclusion

The positive and statistically significant coefficient for `mbre_G` (0.4504) indicates that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`). 
This conclusion is derived from the direction and significance of the coefficient in the ordered logistic regression model.
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
Thank you for providing the results of the model with interaction terms. Let's interpret these results to understand how the effect of `mbre_G` (left limb) varies across different age bins.

### Interpretation of Results

1. **Main Effects**:
   - **Age Bins**: The coefficients for the age bins (e.g., `age_bin_20_29`, `age_bin_30_39`, etc.) are generally positive, but most are not statistically significant (p-values > 0.05). This suggests that age alone does not have a significant impact on `ceap_numeric`.
   - **Sex**: The coefficient for `sexe_M` is not statistically significant (p-value: 0.513), indicating that sex does not have a significant impact on `ceap_numeric`.
   - **mbre_G**: The main effect of `mbre_G` is positive (2.0403) but not statistically significant (p-value: 0.088). This suggests that, on average, the left limb (`G`) does not have a significant impact on `ceap_numeric` when considering all age bins together.

2. **Interaction Terms**:
   - The interaction terms (e.g., `mbre_G:age_bin_20_29`, `mbre_G:age_bin_30_39`, etc.) are generally negative, and some are approaching statistical significance. This indicates that the effect of `mbre_G` (left limb) on `ceap_numeric` varies across different age bins.
   - For example, the interaction term `mbre_G:age_bin_80_89` has a coefficient of -2.5661 with a p-value of 0.060, suggesting that the effect of the left limb is significantly different for individuals in the 80-89 age bin compared to the reference age bin.

### Key Findings

1. **Age-Specific Effects**:
   - The interaction terms suggest that the impact of the left limb (`G`) on `ceap_numeric` varies by age. For example, the negative coefficients for the interaction terms indicate that the positive effect of `mbre_G` is reduced in older age bins.
   - The interaction term `mbre_G:age_bin_80_89` is particularly notable, with a p-value of 0.060, suggesting a marginally significant difference in the effect of the left limb for individuals in the 80-89 age bin.

2. **Generalizability**:
   - The results indicate that the effect of `mbre_G` is not consistent across all age bins. Therefore, it is not appropriate to generalize the effect of `mbre_G` across all age groups. Instead, the impact of the left limb on `ceap_numeric` should be considered in the context of specific age bins.

### Next Steps

1. **Stratified Analysis**:
   - Conduct separate analyses for different age bins to understand the specific effects of `mbre_G` within each age group. This can help tailor interventions and clinical guidelines for different age groups.
   - **Stratified Models**:
     ```python
     age_bins = ['20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89', '90_99']
     for age_bin in age_bins:
         df_age_bin = df_encoded[df_encoded[f'age_bin_{age_bin}'] == 1]
         model_age_bin = OrderedModel(df_age_bin['ceap_numeric'], df_age_bin[predictor_columns], distr='logit')
         results_age_bin = model_age_bin.fit(method='bfgs', maxiter=1000)
         print(f"Results for age bin {age_bin}:")
         print(results_age_bin.summary())
     ```

2. **Further Investigation**:
   - Explore other potential interactions and confounding factors that might influence the relationship between `mbre_G` and `ceap_numeric`. This could include lifestyle factors, comorbidities, or other clinical variables.
   - **Additional Predictors**:
     ```python
     additional_predictors = ['bmi', 'comorbidity_count', 'physical_activity']  # Example additional predictors
     df_encoded_extended = pd.get_dummies(df.drop(columns=['ceap']), columns=additional_predictors, drop_first=True)
     model_extended = OrderedModel(df_encoded_extended['ceap_numeric'], df_encoded_extended[predictor_columns + additional_predictors], distr='logit')
     results_extended = model_extended.fit(method='bfgs', maxiter=1000)
     print(results_extended.summary())
     ```

3. **Clinical Implications**:
   - Based on the findings, develop age-specific clinical guidelines and interventions for managing chronic venous insufficiency. 
   For example, healthcare providers might need to pay more attention to the left limb in older patients.
   - **Targeted Interventions**:
     ```python
     # Example targeted intervention for older patients
     older_patients = df_encoded[df_encoded['age'] >= 80]
     intervention_effect = older_patients.groupby('mbre_G')['ceap_numeric'].mean()
     print("Intervention effect for older patients:", intervention_effect)
     ```

By conducting these additional analyses, you can gain a more nuanced understanding of the factors contributing to the 
severity of chronic venous insufficiency and develop evidence-based strategies to improve patient outcomes.
'''
def mbre_G_subgroup_age_bins(df_encoded):
    # Assuming df_encoded is your DataFrame with dummy variables
    df_encoded['mbre_G:age_bin_20_29'] = df_encoded['mbre_G'] * df_encoded['age_bin_20_29']
    df_encoded['mbre_G:age_bin_30_39'] = df_encoded['mbre_G'] * df_encoded['age_bin_30_39']
    df_encoded['mbre_G:age_bin_40_49'] = df_encoded['mbre_G'] * df_encoded['age_bin_40_49']
    df_encoded['mbre_G:age_bin_50_59'] = df_encoded['mbre_G'] * df_encoded['age_bin_50_59']
    df_encoded['mbre_G:age_bin_60_69'] = df_encoded['mbre_G'] * df_encoded['age_bin_60_69']
    df_encoded['mbre_G:age_bin_70_79'] = df_encoded['mbre_G'] * df_encoded['age_bin_70_79']
    df_encoded['mbre_G:age_bin_80_89'] = df_encoded['mbre_G'] * df_encoded['age_bin_80_89']
    df_encoded['mbre_G:age_bin_90_99'] = df_encoded['mbre_G'] * df_encoded['age_bin_90_99']

    # Select predictor columns including interaction terms
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]

    # Fit the model with interaction terms
    model_with_interactions = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    results_with_interactions = model_with_interactions.fit(method='bfgs', maxiter=1000)

    # Print the results
    print(results_with_interactions.summary())
    pass
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
    
# ****************************************
# PLOTS of the LOGIT : Coefficient Plot
# ****************************************
def plo1():
    # Coefficients and confidence intervals
    coefficients = {
        'mbre_G': (0.4504, 0.196, 0.705),
        'age_bin_20_29': (1.3917, -0.739, 3.522),
        'age_bin_30_39': (0.9998, -1.038, 3.037),
        'age_bin_40_49': (1.6435, -0.364, 3.651),
        'age_bin_50_59': (1.7161, -0.272, 3.704),
        'age_bin_60_69': (1.7329, -0.251, 3.717),
        'age_bin_70_79': (2.0693, 0.071, 4.067),
        'age_bin_80_89': (2.1853, 0.025, 4.345),
        'age_bin_90_99': (2.8538, -0.409, 6.116),
        'sexe_M': (0.0877, -0.175, 0.350)
    }

    # Prepare data for plotting
    predictors = list(coefficients.keys())
    coef_values = [coefficients[pred][0] for pred in predictors]
    lower_bounds = [coefficients[pred][1] for pred in predictors]
    upper_bounds = [coefficients[pred][2] for pred in predictors]

    # Calculate error bars
    yerr_lower = np.array(coef_values) - np.array(lower_bounds)
    yerr_upper = np.array(upper_bounds) - np.array(coef_values)
    yerr = [yerr_lower, yerr_upper]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(predictors, coef_values, yerr=yerr, fmt='o', color='blue', capsize=5)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Predictors')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Plot with 95% Confidence Intervals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    pass
'''
In the context of an ordered logistic regression model, the coefficients labeled `0/1`, `1/2`, `2/3`, `3/4`, `4/5`, and `5/6` represent the cut-off points (or thresholds) between the different categories of the ordinal dependent variable (`ceap_numeric`). These thresholds are used to determine the probability of being in a particular category or higher.

### Explanation of Cut-off Points

1. **Cut-off Points**:
   - The cut-off points are the thresholds that divide the continuous latent variable (which is a linear combination of the predictors) into the observed ordinal categories.
   - For example, `0/1` represents the threshold between the first and second categories, `1/2` represents the threshold between the second and third categories, and so on.

2. **Interpretation**:
   - The coefficients for these cut-off points indicate the values at which the latent variable crosses from one category to the next.
   - These thresholds are estimated along with the coefficients for the predictors to fit the ordered logistic regression model.

### Example

Suppose you have an ordinal dependent variable `ceap_numeric` with categories 0, 1, 2, 3, 4, 5, and 6. The cut-off points would be:

- `0/1`: The threshold between category 0 and category 1.
- `1/2`: The threshold between category 1 and category 2.
- `2/3`: The threshold between category 2 and category 3.
- `3/4`: The threshold between category 3 and category 4.
- `4/5`: The threshold between category 4 and category 5.
- `5/6`: The threshold between category 5 and category 6.

### Extracting Cut-off Points

You can extract the cut-off points from the model results using the following Python code:

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Assuming df_encoded is your DataFrame with dummy variables
# Fit the model (you can use your existing model or fit a new one)
model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
results = model.fit(method='bfgs', maxiter=1000)

# Extract cut-off points
cut_off_points = results.params[-6:]  # Assuming the last 6 parameters are the cut-off points

# Print the cut-off points
print("Cut-off Points:")
for i, cut_off in enumerate(cut_off_points):
    print(f"{i}/{i+1}: {cut_off}")
```

### Example Output

The output might look something like this:

```
Cut-off Points:
0/1: -1.5176
1/2: -1.7787
2/3: -0.0237
3/4: 0.3693
4/5: -0.6130
5/6: -1.0779
```

### Conclusion
The cut-off points (`0/1`, `1/2`, `2/3`, `3/4`, `4/5`, `5/6`) represent the thresholds between the different categories of the ordinal dependent variable (`ceap_numeric`). These thresholds are estimated along with the coefficients for the predictors to fit the ordered logistic regression model. 
They help determine the probability of being in a particular category or higher based on the values of the predictors.
'''
# *****************************
# Plot 2 : Predicted Probabilities Plot
# *****************************
def plo2(df_encoded):
    # Assuming df_encoded is your DataFrame with dummy variables
    # Fit the model (you can use your existing model or fit a new one)
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]
    model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    results = model.fit(method='bfgs', maxiter=1000)
    # Predict probabilities
    predicted_probs = results.predict(df_encoded[predictor_columns])

    # Create a plot for the predicted probabilities of the highest category
    plt.figure(figsize=(10, 6))
    plt.scatter(df_encoded['age'], predicted_probs.iloc[:, -1], c=df_encoded['mbre_G'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='mbre_G (Left Limb)')
    plt.xlabel('Age')
    plt.ylabel('Predicted Probability of Highest ceap_numeric Category')
    plt.title('Predicted Probabilities by Age and Limb')
    plt.show()
    pass
 
# *****************************
# Plot 3 : Predicted Probabilities Plot
# *****************************
def plo3(df_encoded):
    # Assuming df_encoded is your DataFrame with dummy variables
    # Fit the model (you can use your existing model or fit a new one)
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss', 'age']]
    model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    results = model.fit(method='bfgs', maxiter=1000)
    # Predict probabilities
    predicted_probs = results.predict(df_encoded[predictor_columns])
    # Create subplots for each category
    fig, axes = plt.subplots(nrows=predicted_probs.shape[1], ncols=1, figsize=(10, 15), sharex=True)

    for i, ax in enumerate(axes):
        ax.scatter(df_encoded['age'], predicted_probs.iloc[:, i], c=df_encoded['mbre_G'], cmap='coolwarm', alpha=0.6)
        ax.set_ylabel(f'Predicted Probability of Category {i}')
        ax.set_title(f'Category {i}')

    axes[-1].set_xlabel('Age')
    fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', label='mbre_G (Left Limb)')
    plt.tight_layout()
    plt.show()
    pass

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
            results, df_encoded  = modl_fit_variant_1(df)
            if results is not None:
                print(results.summary())
                print (df_encoded)
                mbre_G_subgroup_age_bins(df_encoded)
                #
                plot = False
                if plot:
                  plo1()
                  plo2(df_encoded)
                  plo3(df_encoded)
                #
                check_proportional_odds(df_encoded, results)
            plot_proportional_odds(df_encoded, 'ceap_numeric', ['age_bin_20_29', 'age_bin_30_39', 'sexe_M', 'mbre_G'])
            
            check_multicollinearity(df, ['age_bin', 'sexe', 'mbre'])
        
def multi_04_explai_02_step_0403_fit_variant_1():

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
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}jrnl.txt' if filt_valu is not None else f'{script_name}jrnl.txt')
    main(filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    multi_04_explai_02_step_0403_fit_variant_1()
    
'''
CONCLUSIONS
***********
### Synthetic Rephrasing for Scientific Publication

The conclusion that the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable compared to the right limb (`D`) is derived from the ordered logistic regression model. 
The coefficient for `mbre_G` is 0.4504 (95% CI: [0.196, 0.705], p-value: 0.001), indicating that individuals with the left limb affected have a 0.4504 increase in the log-odds of 
being in a higher category of `ceap_numeric` compared to those with the right limb affected. This statistically significant positive coefficient, 
along with the confidence interval not including zero, confirms that the effect is not due to random chance and that the left limb is associated with higher values of `ceap_numeric`. 
The reference category in the dummy variable encoding is the right limb (`D`), against which the effect of the left limb (`G`) is compared.
'''
'''
Certainly! Here is a synthetic summary of the absence of predictability for the other variables, based on the ordered logistic regression model:

### Synthetic Summary for Scientific Publication

The ordered logistic regression model revealed that, apart from the significant positive impact of the left limb (`G`) on the `ceap_numeric` variable, other variables such as age bins and sex did not demonstrate significant predictability.

1. **Age Bins**:
   - The coefficients for age bins (e.g., `age_bin_20_29`, `age_bin_30_39`, etc.) were not statistically significant (p-values > 0.05). This indicates that age alone does not have a significant impact on the `ceap_numeric` variable. The lack of statistical significance suggests that age is not a reliable predictor of the severity of chronic venous insufficiency as measured by `ceap_numeric`.

2. **Sex**:
   - The coefficient for `sexe_M` (male sex) was also not statistically significant (p-value: 0.513). This implies that sex does not have a significant impact on the `ceap_numeric` variable. The absence of a significant effect indicates that sex is not a reliable predictor of the severity of chronic venous insufficiency.

### Statistical Process

The conclusions are based on the coefficients, p-values, and confidence intervals obtained from the ordered logistic regression model. The model assessed the impact of various predictors on the `ceap_numeric` variable, which represents the severity of chronic venous insufficiency. The lack of statistical significance for age bins and sex suggests that these variables do not contribute meaningfully to the prediction of `ceap_numeric`.

### Conclusion

While the left limb (`G`) has a significant positive impact on the `ceap_numeric` variable, age bins and sex do not demonstrate significant predictability. 
These findings highlight the importance of the left limb as a predictor of chronic venous insufficiency severity, while age and sex do not appear to be reliable predictors in this context. 
Further research may be warranted to explore additional factors that could influence the severity of chronic venous insufficiency.
'''
'''
IMPROVED
********
Your statement is mostly accurate, but there are a few points that could be clarified for better precision and understanding. Here is a refined version:

### Refined Statement

The coefficient for the limb variable (`mbre_G`) is 0.4504 (95% CI: [0.196, 0.705], p-value: 0.001), indicating that the left limb (`G`) is associated with a 0.4504 increase in the log-odds of being in a higher category of the `ceap_numeric` variable compared to the right limb (`D`). The confidence interval, which does not include zero, confirms that this effect is statistically significant and positive. This suggests that individuals with the left limb affected are more likely to have higher values of the clinical sign (`ceap_numeric`). Other variables, such as age and sex, did not show significant predictability for the `ceap_numeric` variable (p-values > 0.05).

### Explanation

1. **Coefficient Interpretation**:
   - The coefficient of 0.4504 represents the increase in the log-odds of being in a higher category of `ceap_numeric` for individuals with the left limb affected compared to those with the right limb affected.
   - The confidence interval [0.196, 0.705] not including zero confirms that this effect is statistically significant.

2. **Other Variables**:
   - Age and sex did not show significant predictability for the `ceap_numeric` variable, as indicated by their p-values being greater than 0.05.

### Statistical Process

The conclusions are based on the ordered logistic regression model, which assesses the impact of various predictors on the `ceap_numeric` variable. The model provides coefficients, p-values, and confidence intervals for each predictor, allowing for the evaluation of their significance and direction of effect.

### Conclusion

The left limb (`G`) has a significant positive impact on the `ceap_numeric` variable, while age and sex do not demonstrate significant predictability. These findings highlight the importance of the left limb as a predictor of chronic venous insufficiency severity, while age and sex do not appear to be reliable predictors in this context. Further research may be warranted to explore additional factors that could influence the severity of chronic venous insufficiency.
'''