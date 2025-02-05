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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import pandas as pd
import numpy as np

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"mbre : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'mbre', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")

    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def sta11(df):

    # Ordinal encoding for CEAP
    ordinal_encoder = OrdinalEncoder()
    df['ceap_encoded'] = ordinal_encoder.fit_transform(df[['ceap']])
    
    # Prepare features
    X = df[['age', 'mbre']]
    X = pd.get_dummies(X, columns=['mbre'], drop_first=True)
    
    # Scale continuous variable
    scaler = StandardScaler()
    X['age_scaled'] = scaler.fit_transform(X[['age']])
    
    # Ordinal logistic regression
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X[['age_scaled', 'mbre_G']], df['ceap_encoded'])
    
    # Coefficients interpretation
    print("Age coefficient:", model.coef_[0][0])
    print("Leg coefficient:", model.coef_[0][1])

def sta12(df, feature_column):
    # Ordinal encoding for CEAP
    ordinal_encoder = OrdinalEncoder()
    df['ceap_encoded'] = ordinal_encoder.fit_transform(df[['ceap']])
    
    # Prepare features
    X = df[['age', feature_column]]
    print (X)
    X = pd.get_dummies(X, columns=[feature_column], drop_first=True)
    
    
    # Scale continuous variable
    scaler = StandardScaler()
    X['age_scaled'] = scaler.fit_transform(X[['age']])
    
    # Identify the dummy column name dynamically
    dummy_col = [col for col in X.columns if col.startswith(feature_column + '_')][0]
    print (dummy_col)
    
    # Prepare target
    y = df['ceap_encoded'].to_numpy().ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ordinal logistic regression
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train[['age_scaled', dummy_col]], y_train)
    
    # Predictions
    y_pred = model.predict(X_test[['age_scaled', dummy_col]])
    
    # Detailed Report
    print("Logistic Regression Detailed Report:")
    print("\n1. Model Coefficients:")
    print(f"Age coefficient: {model.coef_[0][0]}")
    print(f"{feature_column} coefficient: {model.coef_[0][1]}")
    
    print("\n2. Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("\n3. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n4. Cross-Validation:")
    cv_scores = cross_val_score(model, X[['age_scaled', dummy_col]], y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())
    
    # Statistical test
    contingency = pd.crosstab(df[feature_column], df['ceap'])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    print(f"\n5. {feature_column}-CEAP Association:")
    print(f"Chi-square statistic: {chi2}")
    print(f"p-value: {p_value}")
    
    return model, y_pred



def sta2(df):
    
    # Chi-square test for sex vs CEAP
    contingency = pd.crosstab(df['mbre'], df['ceap'])
    chi2, p_sex = stats.chi2_contingency(contingency)[:2]
    print("Leg-CEAP Association: p-value =", p_sex)
    
    # Correlation between age and CEAP
    tau, p_age = stats.kendalltau(df['age'], df['ceap'])
    print("Age-CEAP Correlation: Kendall's tau =", tau, "p-value =", p_age)
    
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
        filt_name = 'mbre'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        if False:
            sta11(df12)
        model, predictions = sta12(df12, 'mbre')
        sta2(df12)
        

        pass
    
'''
Interpretation of new results:

1. Logistic Regression Coefficients:
- Age: Slight negative coefficient (-0.070)
- Leg: Negative coefficient (-0.619) suggests potential impact on CEAP severity

2. Statistical Tests:
- Leg-CEAP Association: p-value (0.035) < 0.05, indicating a statistically significant association
- Age-CEAP Correlation: Remains unchanged from previous analysis (no significant correlation)

Key Findings:
- Leg involvement shows a significant relationship with CEAP severity
- Age continues to show no significant correlation
- The leg variable appears more informative for predicting CEAP severity compared to sex

Recommendation: Further investigate the leg-CEAP relationship, potentially exploring interaction effects or more detailed analysis of leg involvement.
'''
