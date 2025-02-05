
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")

    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def stat(df):
   df = df[['sexe', 'mbre', 'age', 'ceap']].copy()
   # Convert 'sexe' and 'mbre' to numeric using one-hot encoding
   df = pd.get_dummies(df, columns=['sexe', 'mbre'], drop_first=True)
   # Convert 'age' to numeric
   df['age'] = pd.to_numeric(df['age'], errors='coerce')
   # Convert CEAP to numeric
   df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
   print(df.dtypes)
   # Check for missing values
   print(df.isnull().sum())
   print (df)
   # Prepare features and target
   df['sexe_M'] = df['sexe_M'].astype(int)
   df['mbre_G'] = df['mbre_G'].astype(int)
   print (df)
   X = df[['age', 'sexe_M', 'mbre_G']]  # Only include numeric columns
   y = df['ceap_numeric']
   # Add a constant to the model (intercept)
   # X = sm.add_constant(X)
   # Fit the ordinal logistic regression model
   model = OrderedModel(y, X, distr='logit')
   result = model.fit()

   # Print the summary of the model
   print(result.summary())
    
if __name__ == "__main__":

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
    
    # Step 2
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f"{script_name}jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        
        # set_file_objc(file)
        
        # Step 21
        filt_valu = None
        filt_name = 'sexe'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        #print (df12)
        stat(df12)
        

        pass
'''  
Yes, the approach you've outlined for predicting CEAP (Clinical-Etiology-Anatomy-Pathophysiology) classification using logistic regression has a direct equivalent in the statistical world. In statistical terms, what you are doing can be framed as an ordinal logistic regression or proportional odds model, given that CEAP classes (C0 to C6) are ordinal in nature.

### Statistical Equivalent: Ordinal Logistic Regression

Ordinal logistic regression is specifically designed to handle ordinal dependent variables. It models the log odds of the response variable being in or below a particular category.

Here's how you can perform ordinal logistic regression using statistical software like R or Python's `statsmodels` library:

#### Using Python's `statsmodels` Library

1. **Install the necessary library**:
   ```bash
   pip install statsmodels
   ```

2. **Perform Ordinal Logistic Regression**:
  

### Interpreting the Results

- **Coefficients**: The coefficients from the ordinal logistic regression model will tell you the log odds of being in a higher category of CEAP for a one-unit increase in the predictor variable, holding other variables constant.
- **P-values**: These will help you determine the statistical significance of each predictor.
- **Odds Ratios**: You can exponentiate the coefficients to get the odds ratios, which are easier to interpret.

### Feature Importance

In the context of ordinal logistic regression, feature importance can be inferred from the magnitude and significance of the coefficients. Larger absolute values of coefficients indicate stronger effects on the ordinal outcome.

### Comparison with Your Approach

- **Logistic Regression (Multinomial)**: Your approach uses multinomial logistic regression, which treats the CEAP classes as nominal categories rather than ordinal. This might not fully capture the ordered nature of the CEAP classes.
- **Ordinal Logistic Regression**: This approach explicitly models the ordinal nature of the CEAP classes, which can provide more meaningful insights and better model performance for ordinal outcomes.

### Conclusion

While your approach using multinomial logistic regression is valid, using ordinal logistic regression is more statistically appropriate for ordinal outcomes like CEAP classes. This method will likely provide better insights and more accurate predictions.
'''