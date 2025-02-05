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

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
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

    # Ordinal encoding for CEAP
    ordinal_encoder = OrdinalEncoder()
    df['ceap_encoded'] = ordinal_encoder.fit_transform(df[['ceap']])
    
    # Prepare features
    X = df[['age', 'sexe']]
    X = pd.get_dummies(X, columns=['sexe'], drop_first=True)
    
    # Scale continuous variable
    scaler = StandardScaler()
    X['age_scaled'] = scaler.fit_transform(X[['age']])
    
    # Ordinal logistic regression
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X[['age_scaled', 'sexe_M']], df['ceap_encoded'])
    
    # Coefficients interpretation
    print("Age coefficient:", model.coef_[0][0])
    print("Sex coefficient:", model.coef_[0][1])

def sta2(df):
    
    # Chi-square test for sex vs CEAP
    contingency = pd.crosstab(df['sexe'], df['ceap'])
    chi2, p_sex = stats.chi2_contingency(contingency)[:2]
    print("Sex-CEAP Association: p-value =", p_sex)
    
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
        filt_name = 'sexe'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        stat(df12)
        sta2(df12)
        

        pass
    
'''
Interpretation of results:

1. Logistic Regression Coefficients:
- Age: Negative coefficient (-0.074) suggests a slight decrease in CEAP severity with increasing age
- Sex: Positive coefficient (0.171) indicates males might have slightly higher CEAP severity

2. Statistical Tests:
- Sex-CEAP Association: p-value (0.157) > 0.05, suggesting no statistically significant association
- Age-CEAP Correlation: 
  - Kendall's tau (0.023) is very close to 0
  - p-value (0.385) > 0.05, indicating no statistically significant correlation

Conclusions:
- Neither age nor sex shows a strong, statistically significant relationship with CEAP severity
- The logistic regression model suggests weak potential influences
- Further investigation or additional variables might be needed to explain CEAP severity

Recommendations:
- Consider other potential predictors
- Explore more advanced modeling techniques
- Collect more data if possible
'''
