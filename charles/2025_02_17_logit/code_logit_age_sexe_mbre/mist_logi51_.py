import pandas as pd
import statsmodels.api as sm

# -------------------------------
# Logit
# -------------------------------
def mist_logi01(what, df_table, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    if False:
        data = {
            'dossier': ['P1', 'P1', 'P1', 'P2'],
            'sexe': ['M', 'M', 'M', 'F'],
            'age': [54, 54, 54, 45],
            'age_bin': ['50_59', '50_59', '50_59', '40_49'],
            'mbre': ['G', 'G', 'D', 'G'],
            'ceap': ['C4', 'C5', 'C3', 'C2']
        }
        df = pd.DataFrame(data)

    # Prec
    # ----
    df = df_line[['age','sexe', 'ceap']]
    
    # Data
    # ----
    df['sexe_nume'] = df['sexe'].map({'M': 0, 'F': 1}) # 'sexe' must be 'numeric'
    df['C3'] = (df['ceap'] == 'C3').astype(int) # Create binary outcome for C3
    df = df[['sexe_nume', 'age','C3', 'sexe', 'ceap']]
    print (df)
    
    # Exec
    # ----
    # Prepare the data for logistic regression
    X = pd.get_dummies(df[['sexe_nume', 'age']], drop_first=True)
    y = df['C3'] 
    log_model = sm.Logit(y, sm.add_constant(X)) # Fit the logistic regression model
    result = log_model.fit()
    print(result.summary())
    pass
'''
Certainly! I'll explain the purpose of the code and interpret the results for you.

## 1. Purpose of the Code

The code is performing a logistic regression analysis to examine the relationship between the presence of C3 (a specific CEAP classification) and two predictor variables: sex and age. Here's a breakdown of what the code does:

1. Prepares the data:
   - Creates a binary outcome variable 'C3' (1 if CEAP = C3, 0 otherwise)
   - Encodes 'sexe' (sex) as a numeric variable (0 for Male, 1 for Female)

2. Sets up the logistic regression model:
   - Uses 'sexe_nume' (sex) and 'age' as predictor variables
   - Uses 'C3' as the dependent variable

3. Fits the logistic regression model and prints a summary of the results

The purpose is to determine if sex and age are significant predictors of having the C3 classification in the CEAP system.

## 2. Interpretation of Results

The output provides several important pieces of information:

1. Model Fit:
   - Pseudo R-squared: 0.004617 (very low, indicating the model explains only about 0.46% of the variance)
   - Log-Likelihood: -520.49
   - LLR p-value: 0.08944 (marginally significant at α = 0.1, but not at α = 0.05)

2. Coefficients:
   - const (Intercept): -1.1540 (p < 0.001)
   - sexe_nume: 0.3350 (p = 0.029)
   - age: 0.0006 (p = 0.902)

3. Interpretation of coefficients:
   - sexe_nume: The positive coefficient (0.3350) suggests that being female (coded as 1) is associated with higher odds of C3 classification. This effect is statistically significant (p = 0.029).
   - age: The coefficient is very close to zero (0.0006) and not statistically significant (p = 0.902), suggesting that age does not have a meaningful impact on C3 classification in this model.

In summary, the results suggest that sex may have a significant relationship with C3 classification, with females having higher odds of C3. 
However, age does not appear to be a significant predictor. 
The overall model fit is poor, indicating that these variables alone do not explain much of the variation in C3 classification.
'''
