from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from util_file_inpu_mbre import inp1
from util_file_mngr import set_file_objc, write

# Assuming your data is in a DataFrame called 'df'

# 1. Fit the Ordinal Logistic Regression model
def modl_fit1(df):
    print (df)
    model = OrderedModel(df['ceap'], df[['age_bin', 'sexe', 'mbre']], distr='logit')
    results = model.fit()

    print(results.summary())
    return results
def modl_fit2(df):
    print (df)
    # Convert 'ceap' to numeric
    ceap_map = {'C0': 0, 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6}
    df['ceap_numeric'] = df['ceap'].map(ceap_map)
    # Create dummy variables for categorical predictors
    df_encoded = pd.get_dummies(df, columns=['age_bin', 'sexe', 'mbre'], drop_first=True)
    # Select the predictor columns (all columns except 'ceap', 'ceap_numeric', and 'doss')
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss']]
    # Fit the model
    model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    results = model.fit()
    return results

def modl_fit(df):
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    print (df)
    # Convert 'ceap' to numeric
    ceap_map = {'C0': 0, 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6}
    df['ceap_numeric'] = df['ceap'].map(ceap_map)
    # Create dummy variables for categorical predictors
    df_encoded = pd.get_dummies(df, columns=['age_bin', 'sexe', 'mbre'], drop_first=True)
    # Select the predictor columns
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss']]
    # Fit the model
    model = OrderedModel(df_encoded['ceap_numeric'], df_encoded[predictor_columns], distr='logit')
    try:
        results = model.fit(method='bfgs', maxiter=1000)  # Use BFGS method with increased max iterations
    except:
        print("Model fitting failed. Try with a simpler model or check your data.")
        return None
    return results, df_encoded

# 2. Check the Proportional Odds Assumption : done by comparing the coefficients across different thresholds
def check_proportional_odds1(df, results):
    thresholds = results.model.transform_threshold_params(results.params[-6:])
    coeffs = results.params[:-6]
    
    for i in range(len(thresholds)-1):
        model_i = smf.logit(f"ceap > {i}", df[['age_bin', 'sexe', 'mbre']]).fit()
        print(f"\nCoefficients for ceap > {i}:")
        print(model_i.params)
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")
def check_proportional_odds1(results, df_encoded):
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss']]
    
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        formula = f"ceap_numeric > {i} ~ " + " + ".join(predictor_columns)
        model_i = smf.logit(formula, df_encoded).fit()
        print(f"\nCoefficients for CEAP > {i}:")
        print(model_i.params)
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")
def check_proportional_odds2(results, df_encoded):
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss']]
    
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        formula = f"ceap_numeric > {i} ~ " + " + ".join(predictor_columns)
        model_i = smf.logit(formula, df_encoded).fit()
        print(f"\nCoefficients for CEAP > {i}:")
        print(model_i.params)
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")
def check_proportional_odds(results, df_encoded):
    predictor_columns = [col for col in df_encoded.columns if col not in ['ceap', 'ceap_numeric', 'doss']]
    
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        df_encoded[f'ceap_gt_{i}'] = (df_encoded['ceap_numeric'] > i).astype(int)
        formula = f"ceap_gt_{i} ~ " + " + ".join(predictor_columns)
        model_i = smf.logit(formula, df_encoded).fit()
        print(f"\nCoefficients for CEAP > {i}:")
        print(model_i.params)
        df_encoded.drop(f'ceap_gt_{i}', axis=1, inplace=True)  # Clean up temporary column
    
    print("\nIf coefficients are similar across thresholds, the assumption is likely met.")
# 3. Visual check of proportional odds assumption
def plot_proportional_odds(df, variable):
    plt.figure(figsize=(10, 6))
    for i in range(6):  # CEAP has 7 levels (0-6), so 6 cumulative probabilities
        y = (df['ceap'] > i).astype(int)
        sns.regplot(x=variable, y=y, lowess=True, logistic=True, ci=None, scatter=False)
    plt.title(f'Proportional Odds Check for {variable}')
    plt.xlabel(variable)
    plt.ylabel('Cumulative Probability')
    plt.show()
# 4. Check for multicollinearity
def check_multicollinearity(df, features):
    X = df[features]
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
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
        
            # Selector
            # --------
            df1, df2, df3 = inp1(file_path, filt_name, filt_valu)  

            df = df2 # eliminate 'NA'
            df = df[['doss', 'age_bin', 'sexe', 'mbre', 'ceap']] 
            df['age_bin'] = df['age_bin'].replace(r'(\d{2})-(\d{2})', r'\1_\2', regex=True)
            results, df_encoded  = modl_fit(df)
            if results is not None:
                print(results.summary())
                check_proportional_odds(results, df_encoded)
            check_proportional_odds(df, results)
            plot_proportional_odds(df, 'age') # Assuming 'age' is available as a continuous variable
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