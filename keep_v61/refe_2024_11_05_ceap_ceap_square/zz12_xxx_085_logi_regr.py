import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy import stats
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# Cohen kappa Test of Independence
# -------------------------------

def logi_regr(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True


    def ceap_to_numeric(ceap_list):
        return max([i for i, x in enumerate(ceap_list) if x == 1])
    # Assuming df1 is your DataFrame with 'ceaL' and 'ceaR' columns
    left_severity = df1['ceaL'].apply(ceap_to_numeric)
    right_severity = df1['ceaR'].apply(ceap_to_numeric)
    # Assuming left_severity and right_severity are already defined
    # We need to convert the severity scores to binary (0 or 1)
    # Let's consider severity > 0 as 1, and 0 as 0
    binary_left = (left_severity > 0).astype(int)
    binary_right = (right_severity > 0).astype(int)

 


    # Function to convert CEAP classification to binary (presence/absence of sign)
    def ceap_to_binary(ceap_list):
        return [1 if any(ceap_list[1:]) else 0]  # 1 if any sign present, 0 if only C0 or NA

    # Prepare the data
    X = np.array([ceap_to_binary(ceap) for ceap in df1['ceaL']])
    y = np.array([ceap_to_binary(ceap) for ceap in df1['ceaR']])

    # Initialize and fit the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X, y.ravel())

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y.ravel(), cv=5)

    # Make predictions
    y_pred = model.predict(X)

    # Print results
    print("\nLogistic Regression Results:")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f}")
    print(f"Model coefficient: {model.coef_[0][0]:.4f}")
    print(f"Model intercept: {model.intercept_[0]:.4f}")

    # Calculate and print odds ratio
    odds_ratio = np.exp(model.coef_[0][0])
    print(f"Odds ratio: {odds_ratio:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nInterpretation:")
    print(f"The mean cross-validation score of {np.mean(cv_scores):.4f} suggests that the model's performance is {'good' if np.mean(cv_scores) > 0.7 else 'moderate' if np.mean(cv_scores) > 0.5 else 'poor'}.")
    print(f"An odds ratio of {odds_ratio:.4f} indicates that the presence of CEAP signs in the left leg {'increases' if odds_ratio > 1 else 'decreases'}")
    print(f"the odds of CEAP signs in the right leg by a factor of {odds_ratio:.4f}.")
    if odds_ratio > 1:
        print(f"This suggests a positive association between CEAP signs in the left and right legs.")
    elif odds_ratio < 1:
        print(f"This suggests a negative association between CEAP signs in the left and right legs.")
    else:
        print(f"This suggests no association between CEAP signs in the left and right legs.")

    print("\nNote: This logistic regression model predicts the presence of CEAP signs in the right leg")
    print("based on the presence of CEAP signs in the left leg. The model's performance and")
    print("the odds ratio provide insights into the relationship between CEAP signs in both legs.")
   
    
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not   
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
    # Exec
    stat, pval = spearmanr(indx_list_stra, colu_list_ordi)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    asso_form = "positive LE 1" if stat > 0 else "negative GE -1" if stat < 0 else "none"
    print(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")
    write(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")  
   
    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho EQ 0."
    Ha = f"Ha : There is a monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho NE 0."
    alpha = 0.05
    if pval < alpha:
        print(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
        write(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
        write(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
    pass
