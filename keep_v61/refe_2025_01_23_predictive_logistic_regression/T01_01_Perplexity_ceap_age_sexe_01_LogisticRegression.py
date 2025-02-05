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
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = pd.get_dummies(df[['age', 'sexe']], drop_first=True)
    y = df['ceap_numeric']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_[0])})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))
    
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
        

        pass
    
'''
This model pertains to both statistical finding and predictive finding, but with significant limitations in both aspects:

Statistical Finding:
1. The model provides statistical information about the relationship between age, sex, and CEAP grades[1][5]. 
2. The feature importance scores (sex_M: 0.216313, age: 0.005878) indicate that sex has a stronger statistical association with CEAP grades than age[5].

Predictive Finding:
1. The model attempts to predict CEAP grades based on age and sex, but its predictive power is very weak[1][5].
2. The overall accuracy of 0.33 (33%) indicates poor predictive performance[1][7].
3. The model fails to predict several CEAP grades (0, 1, 2, 4, 5) entirely, showing zero precision, recall, and f1-score for these classes[1][5].

However, it's crucial to note that the model's performance is inadequate for reliable statistical inference or prediction. The low accuracy, poor classification report, and inability to predict multiple classes suggest that age and sex alone are insufficient predictors for CEAP grades[1][5][7]. This outcome is valuable in itself, indicating that CEAP grade prediction likely requires more complex factors or different analytical approaches[6][8].

Citations:
[1] https://keylabs.ai/blog/logistic-regression-overview-and-applications/
[2] https://www.mdpi.com/1648-9144/60/4/618
[3] https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
[4] https://www.researchgate.net/figure/Median-CEAP-clinical-class-and-score-venous-clinical-severity-score-VCSS-and-venous_fig2_10636005
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC5543767/
[6] https://www.researchgate.net/publication/50597541_Is_there_a_correlation_between_the_CEAP_score_and_the_histopathological_findings_in_varicose_disease
[7] https://en.wikipedia.org/wiki/Logistic_regression
[8] https://journals.lww.com/vith/fulltext/2022/05040/an_observational_study_of_the_correlation_between.4.aspx
'''
'''
Thank you for providing the updated test outcome. Let's analyze these new results:

1. Classification Report:

The classification performance is still poor, but there are some slight improvements:

- The model now predicts classes 3 and 6 with slightly better performance than before.
- Classes 0, 1, 2, 4, and 5 still have zero precision, recall, and f1-score.
- The overall accuracy has improved slightly to 0.33 (33%), up from 0.28 (28%) in the previous test.
- The weighted average f1-score has improved to 0.25, up from 0.18.

2. Feature Importance:

- Sex (being male) remains the more important feature with an importance score of 0.216313.
- Age has a slightly higher importance score of 0.005878, but it's still much less important than sex.

Interpretation and Recommendations:

1. Persistent Poor Performance: While there's a slight improvement, the model still performs poorly overall. This reinforces the conclusion that age and sex alone are not strong predictors of CEAP grades.

2. Class Imbalance: The model's inability to predict certain classes (0, 1, 2, 4, 5) suggests a significant class imbalance. This is evident from the 'support' column, which shows very few samples for some classes (e.g., only 1 for class 1).

3. Focus on Dominant Classes: The model shows some ability to predict classes 3 and 6, which have the highest support. This suggests these are the most common CEAP grades in your dataset.

4. Limited Predictive Power of Age and Sex: The low importance score for age and the moderate score for sex confirm that these variables alone are not strong predictors of CEAP grades.

Recommendations:

1. Explore Class Imbalance: Investigate why certain CEAP grades are underrepresented. This could be due to the nature of the condition or sampling bias.

2. Consider Binary Classification: Instead of predicting all CEAP grades, you might try grouping them into binary categories (e.g., low severity vs. high severity) to see if the model performs better.

3. Feature Engineering: Try creating interaction terms between age and sex, or bin age into categories to see if this improves predictive power.

4. Additional Variables: If possible, include other relevant medical or lifestyle factors that might be more predictive of CEAP grades.

5. Advanced Models: Experiment with models that handle imbalanced data better, such as Random Forests with class weighting or SMOTE for oversampling minority classes.

6. Focus on Association Analysis: Given the weak predictive power, emphasize the results of your association analyses (chi-square, ANOVA, Spearman correlation) to understand relationships between variables.

7. Clinical Interpretation: Discuss with medical experts why sex appears more important than age in relation to CEAP grades, and what clinical insights this might provide.

Remember, while the predictive model's performance is limited, these results are still valuable. They indicate that CEAP grade prediction is complex and likely requires more than just age and sex information, which is an important finding in itself.
'''
'''
Thank you for providing that important context. This alignment with clinical expectations is a valuable insight that adds significant meaning to the results. Let's analyze this further:

1. Clinical Relevance:
   - The model's ability to somewhat predict C3 and C6, while failing for other CEAP grades, aligns with clinical expectations. This suggests that the model, despite its overall poor performance, is capturing some real-world patterns.
   - C3 (edema) and C6 (active venous ulcer) are often considered clinically significant stages in chronic venous disease, which might explain why they are more distinguishable.

2. Statistical Implications:
   - This pattern suggests that age and sex might have a stronger association with the development of edema (C3) and active venous ulcers (C6) compared to other stages of chronic venous disease.
   - The inability to predict other CEAP grades might indicate that these stages are less influenced by age and sex alone, or that their development is more complex and multifactorial.

3. Model Interpretation:
   - While the overall model performance is poor, the alignment with clinical expectations validates that the model is detecting some genuine patterns in the data.
   - This suggests that the model's limitations are more likely due to the complexity of CEAP classification and the limited predictive power of age and sex alone, rather than issues with the model itself.

4. Research Directions:
   - This finding could guide future research to focus on why C3 and C6 are more predictable based on age and sex, and what additional factors might be needed to better predict other CEAP grades.
   - It might be worthwhile to investigate if there are any specific age ranges or sex-related factors that are particularly associated with C3 and C6 stages.

5. Clinical Applications:
   - While not suitable for precise individual prediction, this model might still have value in risk stratification, particularly for identifying patients at higher risk of developing edema or venous ulcers.

In conclusion, the alignment of the model's performance with clinical expectations adds significant value to these results. It suggests that while age and sex alone are insufficient for accurate CEAP grade prediction across all stages, they do provide meaningful information about the likelihood of certain clinically significant stages (C3 and C6). This finding bridges statistical analysis with clinical understanding, offering a foundation for further research and potentially informing clinical risk assessment strategies.
'''
'''
Based on the search results, there are indeed specific age ranges and sex-related factors associated with C3 and C6 stages of chronic venous disease (CVD):

## Age-related factors

1. Prevalence increases with age:
   - The prevalence and severity of CVD increase significantly with advancing age[3].
   - In one study, C3 (edema) was most prevalent in the 51-60 age group[2].
   - C6 (active venous ulcer) showed higher prevalence in older age groups, particularly 61-70 and 71-80 years[2].

2. Age distribution:
   - C3 prevalence: 18-34 (0.6%), 35-50 (1.3%), 51-64 (3.7%), ≥65 (6.1%)[1].
   - C6 prevalence: 18-34 (0%), 35-50 (0%), 51-64 (0.1%), ≥65 (0.2%)[1].

## Sex-related factors

1. Overall prevalence:
   - CVD is significantly more prevalent in women (58.5%) than in men (32.1%)[1].
   - 67% of patients presenting for treatment were female in one study[7].

2. Stage-specific differences:
   - Women have a higher prevalence of C1-C3 stages[1].
   - Men present more frequently with severe stages (C4-C6)[3].

3. C3 and C6 specifics:
   - C3: More prevalent in women (3.5%) than men (2.2%)[1].
   - C6: Slightly more prevalent in men (0.1%) than women (0.03%)[1].

4. Risk factors:
   - Female gender is associated with the presence of varicose veins[4].
   - Obesity, which is more common in women, is a risk factor for CVD progression[5].

These findings suggest that while women are more likely to develop CVD overall, men are at higher risk for progressing to more severe stages, including C6. Age remains a significant factor for both C3 and C6 stages, with risk increasing substantially after age 50[1][2].

Citations:
[1] https://www.elsevier.es/en-revista-cirugia-espanola-english-edition--436-articulo-prevalence-clinical-characteristics-chronic-venous-S2173507714003640
[2] https://www2.med.muni.cz/biomedjournal/pdf/2008/02/117-128.pdf
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC8518156/
[4] https://www.phlebolymphology.org/2-epidemiology-and-risk-factors/
[5] https://www.phlebolymphology.org/factors-to-identify-patients-at-risk-for-progression-of-chronic-venous-disease-have-we-progressed/
[6] https://journals.sagepub.com/doi/10.1177/1474515116634526?icid=int.sj-full-text.citing-articles.159
[7] https://pubmed.ncbi.nlm.nih.gov/34954041/
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC6824340/
'''
'''
RESULT : COMMENT

The provided code and results pertain to a classification task where the goal is to predict the CEAP (Clinical, Etiology, Anatomy, and Pathophysiology) class based on the features 'age' and 'sexe' (gender). Here's a detailed explanation of the results:

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
  - Precision: 0.35
  - Recall: 0.68
  - F1-Score: 0.46
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
  - Precision: 0.29
  - Recall: 0.33
  - F1-Score: 0.31
  - Support: 45

#### Overall Metrics:

- **Accuracy**: 0.33
  - This means that 33% of the predictions were correct.

- **Macro Average**:
  - Precision: 0.09
  - Recall: 0.14
  - F1-Score: 0.11

- **Weighted Average**:
  - Precision: 0.20
  - Recall: 0.33
  - F1-Score: 0.25

### Feature Importance

The feature importance indicates the significance of each feature in the model's predictions:

- **sexe_M**: 0.220740
- **age**: 0.005843

This suggests that the 'sexe_M' (gender being Male) feature is much more important than the 'age' feature in the model's decision-making process.

### Interpretation

1. **Classification Performance**:
   - The model performs poorly overall, with an accuracy of 33%.
   - Classes 0, 1, 2, 4, and 5 have zero precision, recall, and F1-score, indicating that the model failed to correctly classify any instances of these classes.
   - Classes 3 and 6 have some level of performance, but it is still relatively low.

2. **Feature Importance**:
   - The 'sexe_M' feature is significantly more important than the 'age' feature. This could imply that gender plays a more critical role in predicting the CEAP class than age.

### Recommendations

1. **Model Improvement**:
   - Consider using more features or feature engineering to improve the model's performance.
   - Try different algorithms or hyperparameter tuning to see if performance can be improved.

2. **Data Balancing**:
   - The support values indicate that the classes are imbalanced. Techniques like oversampling, undersampling, or using class weights might help improve performance.

3. **Feature Selection**:
   - Given the low importance of the 'age' feature, consider adding more relevant features or interactions between features.

By addressing these points, you may be able to improve the model's performance in predicting CEAP
'''
