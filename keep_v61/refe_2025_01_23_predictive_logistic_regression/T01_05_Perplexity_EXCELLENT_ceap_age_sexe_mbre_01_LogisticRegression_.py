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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    #print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        #print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        pass
    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def sta1(what, df): # linear

      print (f"=================")
      print (f"sta1 - Linear fit : {what}")
      print (f"=================")

      # Convert CEAP to numeric
      df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
      print(df['ceap_numeric'].unique())

      # Prepare features and target
      X = pd.get_dummies(df[['age', 'sexe', 'mbre']], drop_first=True)
      y = df['ceap_numeric']

      # Split data
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Train logistic regression model
      model = LogisticRegression(multi_class='multinomial', max_iter=1000)
      model.fit(X_train, y_train)

      # Evaluate model
      y_pred = model.predict(X_test)
      report = classification_report(y_test, y_pred)
      print("\nAccuracy:", accuracy_score(y_test, y_pred))
      print("\nClassification Report:\n", report)

      # Feature importance
      feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.mean(np.abs(model.coef_), axis=0)})
      # Calculer les valeurs d'importance relatives
      total_importance = feature_importance['importance'].sum()
      feature_importance['relative_importance'] = feature_importance['importance'] / total_importance
      print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))
      # Variables indicatrices
      print (X.columns)

      # ----
      # Plot
      # ----
      plot = True
      if plot:      
         # Generate classification report dictionary
         report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

         # Convert report to DataFrame
         report_df = pd.DataFrame(report).T  # Transpose to include all metrics
         report_df = report_df.iloc[:-3, :3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg' rows
         print (report_df)

         # Transpose the dataframe
         df_transposed = report_df.T

         # Rename the columns
         df_transposed.columns = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

         # Create the heatmap
         plt.figure(figsize=(10, 6))
         ax = sns.heatmap(df_transposed, annot=True, cmap='YlOrRd', fmt='.2', cbar=False)
         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
         plt.title('Classification Report Heatmap')
         plt.ylabel('Metrics')
         plt.xlabel('Classes')
         plt.show()

         # Update class labels for heatmap x-axis
         class_labels = [str(int(float(label))) if label.isdigit() or label.replace('.', '', 1).isdigit() else label 
                        for label in report_df.index]

         # Heatmap
         plt.figure(figsize=(10, 6))
         sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, xticklabels=["Precision", "Recall", "F1-Score"], yticklabels=class_labels)
         plt.title("Classification Report Heatmap")
         plt.ylabel("CEAP Classes")
         plt.xlabel("Metrics")
         plt.show()
         
         # Sort features by importance
         feature_importance = feature_importance.sort_values('importance', ascending=False)

         # Bar plot
         plt.figure(figsize=(8, 6))
         sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
         plt.title("Feature Importance")
         plt.xlabel("Importance")
         plt.ylabel("Features")
         plt.show()

         # Confusion Matrix
         cm = confusion_matrix(y_test, y_pred)

         # Heatmap
         plt.figure(figsize=(8, 6))
         sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.sort(y.unique()), yticklabels=np.sort(y.unique()))
         plt.title("Confusion Matrix")
         plt.xlabel("Predicted Labels")
         plt.ylabel("True Labels")
         plt.show()

         # Calculate per-class accuracy
         cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         class_accuracy = cm_normalized.diagonal()

         # Bar plot for class accuracy
         plt.figure(figsize=(8, 6))
         sns.barplot(x=np.sort(y.unique()), y=class_accuracy, palette='coolwarm')
         plt.title("Per-Class Accuracy")
         plt.xlabel("CEAP Class")
         plt.ylabel("Accuracy")
         plt.ylim(0, 1)
         plt.show()

         # Prediction probabilities
         y_prob = model.predict_proba(X_test)

         # Convert to DataFrame
         prob_df = pd.DataFrame(y_prob, columns=[f"Class {i}" for i in np.sort(y.unique())])

         # Boxplot for probabilities
         plt.figure(figsize=(12, 6))
         sns.boxplot(data=prob_df, palette="Set3")
         plt.title("Prediction Probabilities by Class")
         plt.xlabel("CEAP Class")
         plt.ylabel("Probability")
         plt.show()

      # Step 2
      # ------
      print (f"------------------")
      print (f"sta1 - Linear fit : {what} : Prevalence for 'sexe'")
      print (f"------------------")
      # Create a dummy variable for 'sexe'
      df_copy = pd.get_dummies(df, columns=['sexe'], drop_first=True)
      # Group by 'ceap_numeric' and calculate the mean of 'sexe_M'
      grouped = df_copy.groupby('ceap_numeric')['sexe_M'].mean()
      grouped_df = grouped.to_frame()
      grouped_df['prevalent_gender'] = grouped_df['sexe_M'].apply(lambda x: 'M' if x > 0.5 else 'F')
      print(f"Sexe: grouped\n{grouped_df}") 

      # Create a contingency table
      contingency_table = pd.crosstab(df_copy['ceap_numeric'], df_copy['sexe_M'])
      # Add a column to indicate the prevalent gender in the contingency table
      contingency_table['prevalent_gender'] = contingency_table.apply(lambda row: 'M' if row[1] > row[0] else 'F', axis=1)
      print(f"Sexe: contingency_table\n{contingency_table}")
      pass
   
      if plot:
         # Bar plot for grouped data (mean of sexe_M)
         plt.figure(figsize=(10, 6))
         sns.barplot(x=grouped_df.index, y=grouped_df['sexe_M'], palette="viridis")
         plt.axhline(0.5, color='red', linestyle='--', label="50% Threshold")
         plt.title("Proportion of Male Patients (`sexe_M`) by CEAP Class")
         plt.xlabel("CEAP Class")
         plt.ylabel("Proportion of Males")
         plt.legend()
         plt.show()

         # Stacked bar chart for contingency table
         contingency_table_stacked = contingency_table.drop(columns=['prevalent_gender'])  # Exclude prevalent_gender column
         contingency_table_stacked.columns = ['Female', 'Male']  # Rename columns for clarity

         contingency_table_stacked.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#6baed6', '#2171b5'])
         plt.title("Gender Distribution by CEAP Class")
         plt.xlabel("CEAP Class")
         plt.ylabel("Count")
         plt.legend(title="Gender", labels=["Female", "Male"])
         plt.show()
      
      # Step 3
      # ------
      print (f"------------------")
      print (f"sta1 - Linear fit : {what} : Prevalence for 'mbre'")
      print (f"------------------")
      # Create a dummy variable for 'sexe'
      df_copy = pd.get_dummies(df, columns=['mbre'], drop_first=True)
      # Group by 'ceap_numeric' and calculate the mean of 'sexe_M'
      grouped = df_copy.groupby('ceap_numeric')['mbre_G'].mean()
      grouped_df = grouped.to_frame()
      grouped_df['prevalent_leg'] = grouped_df['mbre_G'].apply(lambda x: 'G' if x > 0.5 else 'D')
      print(f"Mbre: grouped\n{grouped_df}") 
      # Create a contingency table
      contingency_table = pd.crosstab(df_copy['ceap_numeric'], df_copy['mbre_G'])
      contingency_table['prevalent_leg'] = contingency_table.apply(lambda row: 'G' if row[1] > row[0] else 'D', axis=1)
      print(f"Mbre: contingency_table\n{contingency_table}")

      pass

def sta2(what, df): # polynomial
  
    print (f"=================")
    print (f"sta2 - Linear and Polynomial (2) fit : {what}")
    print (f"=================")

    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Assuming df is your DataFrame and 'ceap_numeric' is your target variable
    X = pd.get_dummies(df[['age', 'sexe', 'mbre']], drop_first=True)
    y = df['ceap_numeric']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with polynomial features and logistic regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),  # You can adjust the degree as needed
        ('logreg', LogisticRegression(multi_class='multinomial', max_iter=1000))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def sta3(what, df, ceap):   
  
    print (f"=================")
    print (f"sta3 - Binary fit : {what}")
    print (f"=================")
    
    # Create binary target for C3
    df[f'is_{ceap}'] = (df['ceap'] == ceap).astype(int)

    # Prepare features and target
    X = pd.get_dummies(df[['age', 'sexe', 'mbre']], drop_first=True)
    y = df[f'is_{ceap}']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model_c3 = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_c3.fit(X_train, y_train)

    # Evaluate model
    y_pred_c3 = model_c3.predict(X_test)
    print(f"{ceap} Classification Report:\n", classification_report(y_test, y_pred_c3))

    # Feature importance
    feature_importance_c3 = pd.DataFrame({'feature': X.columns, 'importance': np.abs(model_c3.coef_[0])})
    print(f"\n{{ceap}} Feature Importance:\n", feature_importance_c3.sort_values('importance', ascending=False))
    pass

def sta4(what, df, ceap):   
  
    print (f"=================")
    print (f"sta4 - Binary fit : {what}")
    print (f"=================")
    
    # Create binary target for C3
    df[f'is_{ceap}'] = (df['ceap'] == ceap).astype(int)
    df['sexe_mbre'] = df['sexe'].astype(str) + '_' + df['mbre'].astype(str)

    # Prepare features and target
    X = pd.get_dummies(df[['age', 'sexe', 'mbre', 'sexe_mbre']], drop_first=True)
    y = df[f'is_{ceap}']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"{ceap} Classification Report:\n", classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(model.coef_[0])})
    print(f"\n{{ceap}} Feature Importance:\n", feature_importance.sort_values('importance', ascending=False))

    pass

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
        
        # Step
        # ----
        what = "ceap:C0..C6"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12.copy()
        sta1(what, df) # Linear
        sta2(what, df) # Polynomial
        
        # Step
        # ----    
        what = "ceap:C3,C6"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12[df12['ceap'].isin(['C3', 'C6'])]
        sta1(what, df) # Linear
        sta2(what, df) # Polynomial
        pass
      
        # Step
        # ----
        ceap = 'C3'
        what = f"ceap:{ceap}"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12.copy()
        sta3(what, df, ceap) # Linear
        
        ceap = 'C6'
        what = f"ceap:{ceap}"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12.copy()
        sta3(what, df, ceap) # Linear
        
        # Step
        # ----
        ceap = 'C3'
        what = f"ceap:{ceap}"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12.copy()
        sta4(what, df, ceap) # Linear
        
        ceap = 'C6'
        what = f"ceap:{ceap}"
        print (f">>>>>>>>>>>>>>>>>")
        print (f"Fit : {what}")
        print (f">>>>>>>>>>>>>>>>>")
        df = df12.copy()
        sta4(what, df, ceap) # Linear
   
'''
STA4 C6 RESULTS
---------------
The updated results for **C6** after including the `sexe_mbre` interaction term offer interesting insights into the model’s behavior:

---

### Observations:
1. **Model Performance**:
   - **Precision (0.29)** for `C6 (1)` remains low, indicating many false positives when predicting `C6`.
   - **Recall (0.60)** for `C6 (1)` is moderate, suggesting the model captures 60% of true `C6` cases.
   - **Accuracy (0.46)** and **F1-score (0.39)** are not ideal but consistent with the imbalanced nature of the dataset.
   - The **macro average** (`0.50`) shows the model equally weights both classes, but performance is weak overall.

2. **Feature Importance**:
   - **`sexe_M` (Male)** is the most significant feature (`0.291423`), slightly ahead of **`mbre_G` (Left Limb)** (`0.281939`).
   - Among the interaction terms:
     - **`sexe_mbre_M_G` (Male, Left Limb)** ranks highest (`0.168751`), indicating strong predictive power.
     - **`sexe_mbre_M_D` (Male, Right Limb)** and **`sexe_mbre_F_G` (Female, Left Limb)** follow with moderate importance.
   - **`age`** remains negligible (`0.002085`).

---

### Key Insights:
1. **Gender and Limb Side**:
   - The predictive power of `sexe_M` and `mbre_G`, as well as their interaction terms, suggests that **male patients and the left limb** are crucial for identifying C6.
   - Interestingly, **`sexe_mbre_F_G` (Female, Left Limb)** contributes as well, indicating a gender-specific pattern in left-limb cases.

2. **Low Precision for `C6 (1)`**:
   - Many false positives indicate that the model struggles to differentiate true `C6` cases from non-C6 cases.
   - This could be due to the overlapping characteristics between `C6` and other classes or insufficient features to capture subtle distinctions.

---

### Recommendations:
1. **Refine Feature Set**:
   - Investigate additional features that could help separate `C6` from other classes:
     - **Clinical variables**: Symptoms, duration, or history of venous disease.
     - **Environmental factors**: Lifestyle or activity levels.

2. **Threshold Tuning**:
   - Adjust the decision threshold to balance precision and recall, depending on the desired trade-off:
     ```python
     y_pred_proba = model.predict_proba(X_test)[:, 1]
     y_pred_adjusted = (y_pred_proba >= 0.4).astype(int)
     print("\nAdjusted Classification Report:\n", classification_report(y_test, y_pred_adjusted))
     ```

3. **Explore Non-Linear Models**:
   - Models like **Random Forest** or **Gradient Boosting** can handle complex feature interactions better than logistic regression:
     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     model = GradientBoostingClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     print("\nClassification Report:\n", classification_report(y_test, y_pred))
     ```

4. **Class Imbalance Handling**:
   - Use techniques like **SMOTE** or **class weights** in non-linear models to improve performance for the minority class (`C6`).

---

### Next Steps:
Would you like to proceed with:
1. Tuning the decision threshold for logistic regression?
2. Trying a non-linear model like Gradient Boosting or Random Forest?
3. Investigating the false positives for further insights?
'''     
'''
STA4 C3 RESULTS
---------------
The updated results for **C3** with the addition of the `sexe_mbre` interaction term are revealing:

---

### Observations:
1. **Model Performance**:
   - **Precision (0.38)** for `C3 (1)` remains low, indicating a high rate of false positives.
   - **Recall (0.62)** for `C3 (1)` is moderate, showing the model captures 62% of true `C3` cases.
   - The **accuracy (0.52)** and **F1-score (0.47)** for `C3 (1)` remain consistent with earlier results.
   - The **macro average** and **weighted average** metrics show the model struggles to balance between classes.

2. **Feature Importance**:
   - **`sexe_mbre_M_D` (Male, Right)** is now the most significant feature, with an importance score of `0.213868`, surpassing the standalone **`sexe_M`** (`0.200156`).
   - **`sexe_mbre_F_G` (Female, Left)** and **`mbre_G` (Left Limb)** have smaller but notable importance scores (`0.033159` and `0.019447`).
   - **`age`** remains insignificant (`0.000570`), confirming no direct relationship with C3 in this setup.

---

### Key Takeaways:
1. Adding the `sexe_mbre` interaction term provided **new insights**:
   - The combination of gender and limb side (e.g., `Male-Right`) is more predictive than gender (`sexe_M`) or limb side (`mbre_G`) alone.
   - This suggests that C3 prevalence may vary significantly based on these combined factors.

2. The low **precision for `C3 (1)`** indicates a need to further refine the model to reduce false positives. This may require:
   - Better feature engineering.
   - Exploring non-linear models that can capture complex relationships.

---

### Next Steps:
1. **Threshold Adjustment**:
   - Tune the prediction threshold to balance precision and recall, especially if you prioritize identifying true C3 cases.

2. **Alternative Models**:
   - Consider non-linear models like Random Forest or Gradient Boosting, which may better handle interactions between features.

3. **Investigate False Positives**:
   - Examine cases where the model predicts `C3` but the true label is not `C3`. This could reveal patterns in the misclassified data.

Would you like to try threshold tuning or switch to a non-linear model next?
'''
'''
STA3 C6 RESULTS
---------------
The **C6 classification results** reveal some key insights and challenges:

---

### Observations:
1. **Model Performance**:
   - **Precision (0.30)** for `C6 (1)` is low, indicating many false positives when predicting C6.
   - **Recall (0.56)** for `C6 (1)` is moderate, suggesting the model identifies over half of the true C6 cases.
   - Overall **accuracy (0.49)** and **F1-score (0.39)** for `C6 (1)` are suboptimal.
   - The imbalance in performance between `0` (non-C6) and `1` (C6) shows that the model struggles with C6 detection.

2. **Feature Importance**:
   - **`sexe_M`** (male) has the highest importance score (`0.375712`), suggesting males are more predictive of C6.
   - **`mbre_G`** follows closely behind (`0.363583`), indicating it also contributes significantly.
   - **`age`** remains almost irrelevant (`0.002057`), showing no relationship to C6 in the current model.

---

### Comparison Between C3 and C6:
- For both C3 and C6, **`sexe_M`** is the top predictor, but its importance for C6 (`0.375712`) is higher than for C3 (`0.275818`).
- **`mbre_G`** is also important for both but contributes more to predicting C6 than C3.
- **`age`** is not significant in either case, possibly requiring transformations or rethinking its utility.

---

### Recommendations for C6:
1. **Focus on Balancing Precision and Recall**:
   - Since precision is low, investigate false-positive cases:
     - Analyze errors (e.g., which non-C6 cases are misclassified as C6).
   - Consider lowering the classification threshold to improve recall (if identifying all C6 cases is more critical).

2. **Feature Engineering**:
   - **Explore interactions**:
     - Interaction between `sexe` and `mbre` might explain more variation.
     - Add an interaction term like `sexe * mbre` to the features:
       ```python
       df['sexe_mbre'] = df['sexe'].astype(str) + '_' + df['mbre'].astype(str)
       ```
   - **Group or bin `age`**:
     - Since `age` isn’t directly contributing, group it into meaningful bins (e.g., `<30`, `30-50`, `50-70`, etc.) to detect non-linear patterns.

3. **Model Upgrade**:
   - Logistic regression might not be sufficient for C6, as it relies on linear separability. Try:
     - **Random Forest**: Handles non-linear interactions and provides feature importance.
     - **Gradient Boosting**: Often outperforms other classifiers in structured/tabular data.
     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     model_c6 = GradientBoostingClassifier()
     model_c6.fit(X_train, y_train)
     ```

4. **Threshold Tuning**:
   - Adjust the prediction threshold for better recall or precision:
     ```python
     y_pred_proba = model_c6.predict_proba(X_test)[:, 1]
     y_pred_adjusted = (y_pred_proba >= 0.4).astype(int)  # Adjust threshold
     print("Adjusted Classification Report:\n", classification_report(y_test, y_pred_adjusted))
     ```

5. **Advanced Imbalance Handling**:
   - Use **SMOTE** or other resampling methods for the training dataset to address the imbalance between C6 and non-C6.

---

### Next Steps:
- Should we focus on engineering new features or transition to non-linear models like Random Forest or Gradient Boosting for C6?
'''     
'''
STA3 C3 RESULTS
------------
Model Performance:

Precision (0.38) for C3 (1) is low, meaning the model struggles to avoid false positives when predicting C3.
Recall (0.62) for C3 (1) is moderate, showing the model identifies 62% of true C3 cases.
The overall accuracy (0.52) and F1-score (0.47) for C3 (1) indicate the model is only partially effective.
The imbalance in performance between class 0 (non-C3) and 1 (C3) reflects some difficulty in distinguishing C3 from non-C3 cases.
Feature Importance:

sexe_M (male) is the most significant predictor for C3 with an importance score of 0.275818.
mbre_G (some grouping feature) has a minor importance (0.050979).
age has almost no importance (0.000578), meaning it’s not contributing to predicting C3 in this model.
'''
    
'''
Thank you for providing the results. Let's analyze the outcome of including the 'leg' variable in the logistic regression model:

1. Model Performance:
   - Accuracy: 0.39 (39%)
   - This is a slight improvement from the previous model (33%), but still indicates poor overall predictive power.

2. Classification Report:
   - The model still struggles with most CEAP grades (0, 1, 2, 4, 5), showing zero precision, recall, and f1-score for these classes.
   - Performance for C3 (grade 3) has improved:
     * Precision: 0.38 (up from 0.35)
     * Recall: 0.87 (up from 0.70)
     * F1-score: 0.53 (up from 0.47)
   - Performance for C6 (grade 6) has also improved:
     * Precision: 0.43 (up from 0.28)
     * Recall: 0.33 (slightly up from 0.31)
     * F1-score: 0.38 (up from 0.29)

3. Feature Importance:
   1. leg_L: 0.245365
   2. sexe_M: 0.184230
   3. age: 0.007093

Key Insights:
1. Improved but Still Limited Performance: The addition of the 'leg' variable has slightly improved the model's performance, particularly for C3 and C6 grades. However, the overall accuracy remains low.

2. 'Leg' as the Most Important Feature: Interestingly, the 'leg' variable (specifically, left leg) has emerged as the most important feature, even surpassing sex and age. This suggests that the affected leg (left or right) may be a stronger predictor of CEAP grade than previously considered factors.

3. Persistent Class Imbalance: The model still fails to predict several CEAP grades, indicating a persistent class imbalance issue or the need for more complex modeling approaches.

4. Relative Importance of Factors: The order of importance (leg > sex > age) provides new insights into the factors influencing CEAP grades in your dataset.

Recommendations:
1. Further Investigation of 'Leg' Factor: The high importance of the 'leg' variable warrants deeper analysis. Consider examining if there are physiological or lifestyle factors that might explain why the left leg is a stronger predictor.

2. Advanced Modeling Techniques: Given the persistent class imbalance, consider techniques like oversampling (SMOTE), undersampling, or using class weights to improve prediction for underrepresented classes.

3. Feature Engineering: Explore interactions between variables (e.g., age and leg, sex and leg) to capture more complex relationships.

4. Clinical Interpretation: Discuss these findings with medical experts to understand the potential clinical significance of the leg factor in CEAP classification.

5. Data Collection Review: If possible, review the data collection process to ensure there's no bias in recording leg-specific information that might be influencing these results.

While the model's overall performance is still limited, the inclusion of the 'leg' variable has provided valuable new insights, particularly highlighting its importance in predicting CEAP grades. This finding could potentially inform clinical assessments and guide further research into the factors influencing chronic venous disease progression.
'''
'''
RSEULT: COMMENT
The new results show the impact of introducing the variable 'mbre G,D' (leg Left, Right) into the model. Let's analyze the updated classification report and feature importance:

### Classification Report

The classification report provides metrics for each CEAP class (0 through 6):

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the all observations in actual class.
- **F1-Score**: The weighted average of Precision and Recall.
- **Support**: The number of actual occurrences of the class in the dataset.

#### Detailed Breakdown:

- **Class 0**:
  - Precision: 0.00
  - Recall: 0.00
  - F1-Score: 0.00
  - Support: 12

- **Class 1**:
  - Precision: 0.00
  - Recall: 0.00
  - F1-Score: 0.00
  - Support: 1

- **Class 2**:
  - Precision: 0.00
  - Recall: 0.00
  - F1-Score: 0.00
  - Support: 17

- **Class 3**:
  - Precision: 0.38
  - Recall: 0.87
  - F1-Score: 0.53
  - Support: 53

- **Class 4**:
  - Precision: 0.00
  - Recall: 0.00
  - F1-Score: 0.00
  - Support: 18

- **Class 5**:
  - Precision: 0.00
  - Recall: 0.00
  - F1-Score: 0.00
  - Support: 9

- **Class 6**:
  - Precision: 0.43
  - Recall: 0.33
  - F1-Score: 0.38
  - Support: 45

#### Overall Metrics:

- **Accuracy**: 0.39
  - This means that 39% of the predictions were correct, which is an improvement from the previous accuracy of 33%.

- **Macro Average**:
  - Precision: 0.12
  - Recall: 0.17
  - F1-Score: 0.13

- **Weighted Average**:
  - Precision: 0.26
  - Recall: 0.39
  - F1-Score: 0.29

### Feature Importance

The feature importance indicates the significance of each feature in the model's predictions:

- **mbre_G**: 0.245756
- **sexe_M**: 0.184782
- **age**: 0.006988

This suggests that the 'mbre_G' (leg Left) feature is the most important, followed by 'sexe_M' (gender being Male), and 'age' is the least important.

### Interpretation

1. **Classification Performance**:
   - The model's overall accuracy improved from 33% to 39% with the introduction of the 'mbre G,D' variable.
   - Classes 0, 1, 2, 4, and 5 still have zero precision, recall, and F1-score, indicating that the model continues to struggle with these classes.
   - Classes 3 and 6 show improved performance, particularly Class 3, which has a higher recall and F1-score.

2. **Feature Importance**:
   - The 'mbre_G' feature is the most important, followed by 'sexe_M'. This indicates that the leg (Left/Right) and gender play significant roles in predicting the CEAP class.
   - The 'age' feature remains relatively unimportant.

### Recommendations

1. **Model Improvement**:
   - Continue exploring additional features or feature engineering to further improve the model's performance.
   - Consider using more complex models or ensemble methods to capture more intricate patterns in the data.

2. **Data Balancing**:
   - Address the class imbalance issue, as some classes have very few samples (e.g., Class 1 with only 1 sample). Techniques like oversampling, undersampling, or using class weights can help.

3. **Feature Selection**:
   - Given the importance of 'mbre_G' and 'sexe_M', consider adding more relevant features or interactions between features that might capture additional information.

By addressing these points, you may be able to further improve the model's performance in predicting CEAP classes.

'''
