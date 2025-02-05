import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau

# -------------------------------
# Goodman and Kruskal's Lambda Test of Independence
# -------------------------------

def goodman_kruskal_lambda(x, y):
    tau, _ = kendalltau(x, y)
    return tau / (1 + tau)

def lambda_permutation_test(x, y, n_permutations=10000):
    observed_lambda = goodman_kruskal_lambda(x, y)
    permuted_lambdas = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_lambda = goodman_kruskal_lambda(x, y_permuted)
        permuted_lambdas.append(permuted_lambda)
    
    p_value = np.mean(np.abs(permuted_lambdas) >= np.abs(observed_lambda))
    return observed_lambda, p_value

def bootstrap_lambda(x, y, perm=10000):
    #original_lambda = goodman_kruskal_lambda(x, y)
    bootstrap_lambdas = []
    
    for _ in range(perm):
        # Resample with replacement
        indices = np.random.randint(0, len(x), len(x))
        x_resampled = x[indices]
        y_resampled = y[indices]
        
        # Calculate lambda for this bootstrap sample
        bootstrap_lambda = goodman_kruskal_lambda(x_resampled, y_resampled)
        bootstrap_lambdas.append(bootstrap_lambda)
    
    # Calculate confidence interval
    alpha=0.05
    ci_lower = np.percentile(bootstrap_lambdas, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_lambdas, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper
    
def goodkruslamb(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True
    
    # 'df_tabl' only 
    if df1 is None:
        print(f"\nData : {what}\n(df_table) : Goodman and Kruskal's Lambda : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Goodman and Kruskal's Lambda : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        return
    
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not  
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # ----
    # Exec 1
    # ----
    lambda_gender_age = goodman_kruskal_lambda(indx_list_stra, colu_list_ordi) # df['Gender_num'], df['Age_Ordinal'] OR df['Gender_num'].values, df['Age_Ordinal'].values
    lambda_age_gender = goodman_kruskal_lambda(colu_list_ordi, indx_list_stra) # df['Age_Ordinal'], df['Gender_num'] OR # df['Age_Ordinal'].values, df['Gender_num'].values
    #print(f"Lambda (Gender predicting Age): {lambda_gender_age}")
    #print(f"Lambda (Age predicting Gender): {lambda_age_gender}")
    lambda_gender_age_form = round(lambda_gender_age * 100)
    lambda_age_gender_form = round(lambda_age_gender * 100)
    
    # ----
    # Exec 2
    # ----
    perm = 5000
    lambda_gender_age, p_value_gender_age = lambda_permutation_test(indx_list_stra, colu_list_ordi, perm)
    lambda_age_gender, p_value_age_gender = lambda_permutation_test(colu_list_ordi, indx_list_stra, perm)
    #print(f"Lambda (Gender predicting Age): {lambda_gender_age:.4f}, p-value: {p_value_gender_age:.4f}")
    #print(f"Lambda (Age predicting Gender): {lambda_age_gender:.4f}, p-value: {p_value_age_gender:.4f}")
    lambda_gender_age_form = round(lambda_gender_age * 100)
    lambda_age_gender_form = round(lambda_age_gender * 100)
    
    p_value_gender_age_form = f"{p_value_gender_age:.3e}" if p_value_gender_age < 0.001 else f"{p_value_gender_age:.3f}"
    p_value_age_gender_form = f"{p_value_age_gender:.3e}" if p_value_age_gender < 0.001 else f"{p_value_age_gender:.3f}"
    
    # Conf Intv
    perm=5000
    ci_lower_ga, ci_upper_ga = bootstrap_lambda(indx_list_stra, colu_list_ordi, perm) # df['Gender_num'].values, df['Age_Ordinal'].values
    ci_lower_ag, ci_upper_ag = bootstrap_lambda(colu_list_ordi, indx_list_stra, perm) # df['Age_Ordinal'].values, df['Gender_num'].values
    ci_lower_g2, ci_upper_g2 = bootstrap_lambda(indx_list_stra.values, colu_list_ordi.values, perm) # df['Gender_num'].values, df['Age_Ordinal'].values
    ci_lower_a2, ci_upper_a2 = bootstrap_lambda(colu_list_ordi.values, indx_list_stra.values, perm) # df['Age_Ordinal'].values, df['Gender_num'].values
    
    ci_lower_ga_form = f"{ci_lower_ga:.3e}" if ci_lower_ga < 0.001 else f"{ci_lower_ga:.3f}"
    ci_upper_ga_form = f"{ci_upper_ga:.3e}" if ci_upper_ga < 0.001 else f"{ci_upper_ga:.3f}"
    ci_lower_ag_form = f"{ci_lower_ag:.3e}" if ci_lower_ag < 0.001 else f"{ci_lower_ag:.3f}"
    ci_upper_ag_form = f"{ci_upper_ag:.3e}" if ci_upper_ag < 0.001 else f"{ci_upper_ag:.3f}"

    #print(f"Lambda (Gender predicting Age): {lambda_gender_age:.4f} 95% CI: ({ci_lower_ga:.4f}, {ci_upper_ga:.4f})")
    #print(f"\nLambda (Age predicting Gender): {lambda_age_gender:.4f} 95% CI: ({ci_lower_ag:.4f}, {ci_upper_ag:.4f})")
    
    # Resu
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    # Claude
    # H0 : The '{colu_name}' variable does not provide any information about the'{indx_name}' variable.
    # Ha : The '{colu_name}' variable does provide information about the'{indx_name}' variable.
    if np.isnan(lambda_gender_age) or np.isnan(lambda_age_gender):
        raise Exception("Stat or Pval are NaN")
    p_value_gender_age
    print(f"\nData : {what}\nGoodman and Kruskal's Lambda : {indx_name} predicts {colu_name}")
    print(f"Stat: {lambda_gender_age_form} Pval : {p_value_gender_age_form} 95% CI: ({ci_lower_ga_form}, {ci_upper_ga_form})")
    print(f"Given '{indx_name}' value and asked to predict '{colu_name}' value, Lamda reduces by {lambda_gender_age_form} % the error probability")
    write(f"\nData : {what}\nGoodman and Kruskal's Lambda : {indx_name} predicts {colu_name}")
    write(f"Stat: {lambda_gender_age_form} Pval : {p_value_gender_age_form} 95% CI: ({ci_lower_ga_form}, {ci_upper_ga_form})")
    write(f"Given '{indx_name}' value and asked to predict '{colu_name}' value, Lamda reduces by {lambda_gender_age_form} % the error probability")
 
    # Intp
    alpha = 0.05
    H0_gender_age = f"H0 : The '{indx_name}' variable does not provide any information about the'{colu_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha_gender_age = f"Ha : The '{indx_name}' variable does provide information about the'{colu_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    if p_value_gender_age < alpha:
        print(f"Goodman and Kruskal's Lambda : Reject the null hypothesis:\n{Ha_gender_age}")
        write(f"Goodman and Kruskal's Lambda : Reject the null hypothesis:\n{Ha_gender_age}")
    else:
        print(f"Goodman and Kruskal's Lambda : Fail to reject the null hypothesis:\n{H0_gender_age}")
        write(f"Goodman and Kruskal's Lambda : Fail to reject the null hypothesis:\n{H0_gender_age}")
        
    # Resu
    print(f"\nData : {what}\nGoodman and Kruskal's Lambda : {colu_name} predicts {indx_name}")
    print(f"Stat: {lambda_age_gender_form} Pval : {p_value_age_gender_form} 95% CI: ({ci_lower_ag_form}, {ci_upper_ag_form})")
    print(f"Given '{colu_name}' value and asked to predict '{indx_name}' value, Lamda reduces by {lambda_age_gender_form} % the error probability")
    write(f"\nData : {what}\nGoodman and Kruskal's Lambda : {colu_name} predicts {indx_name}")
    write(f"Stat: {lambda_age_gender_form} Pval : {p_value_age_gender_form} 95% CI: ({ci_lower_ag_form}, {ci_upper_ag_form})")
    write(f"Given '{colu_name}' value and asked to predict '{indx_name}' value, Lamda reduces by {lambda_age_gender_form} % the error probability")
 
    # Intp
    alpha = 0.05
    H0_age_gender = f"H0 : The '{colu_name}' variable does not provide any information about the'{indx_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha_age_gender = f"Ha : The '{colu_name}' variable does provide information about the'{indx_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    if p_value_gender_age < alpha:
        print(f"Goodman and Kruskal's Lambda : Reject the null hypothesis:\n{Ha_age_gender}")
        write(f"Goodman and Kruskal's Lambda : Reject the null hypothesis:\n{Ha_age_gender}")
    else:
        print(f"Goodman and Kruskal's Lambda : Fail to reject the null hypothesis:\n{H0_age_gender}")
        write(f"Goodman and Kruskal's Lambda : Fail to reject the null hypothesis:\n{H0_age_gender}")
    pass
'''
2025_01_12 Mistral
The approach you've outlined for applying Goodman and Kruskal's Lambda to your data has a few issues. Goodman and Kruskal's Lambda is a measure of association for nominal variables, not ordinal variables. It measures the proportional reduction in error when predicting one nominal variable from another. The formula you've provided using Kendall's Tau is not correct for Goodman and Kruskal's Lambda.

Goodman and Kruskal's Lambda is calculated as follows:

1. **Lambda for predicting Y from X**:
   \[
   \lambda_Y = \frac{E_1 - E_2}{E_1}
   \]
   where \( E_1 \) is the number of errors made when predicting Y without knowledge of X, and \( E_2 \) is the number of errors made when predicting Y with knowledge of X.

2. **Lambda for predicting X from Y**:
   \[
   \lambda_X = \frac{E_1 - E_2}{E_1}
   \]
   where \( E_1 \) is the number of errors made when predicting X without knowledge of Y, and \( E_2 \) is the number of errors made when predicting X with knowledge of Y.

3. **Symmetric Lambda**:
   \[
   \lambda = \frac{\lambda_Y + \lambda_X}{2}
   \]

Given your data, you need to calculate the contingency table first and then apply the Goodman and Kruskal's Lambda formula. Here's how you can do it:

```python
import pandas as pd
import numpy as np

def goodman_kruskal_lambda(x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)

    # Calculate row and column sums
    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)
    total = contingency_table.sum().sum()

    # Calculate E1 (errors without knowledge of the other variable)
    E1_x = total - row_sums.max()
    E1_y = total - col_sums.max()

    # Calculate E2 (errors with knowledge of the other variable)
    E2_x = total - contingency_table.max(axis=1).sum()
    E2_y = total - contingency_table.max(axis=0).sum()

    # Calculate Lambda for X and Y
    lambda_x = (E1_x - E2_x) / E1_x
    lambda_y = (E1_y - E2_y) / E1_y

    # Calculate symmetric Lambda
    lambda_sym = (lambda_x + lambda_y) / 2

    return lambda_sym

# Example usage
data = {
    'sexe_stra': [1, 1, 1, 1, 0, ...],  # Replace with your actual data
    'ceap_ordi': [3, 7, 3, 7, 4, ...]   # Replace with your actual data
}

df = pd.DataFrame(data)

indx_list_stra = df['sexe_stra']
colu_list_ordi = df['ceap_ordi']

lambda_gender_age = goodman_kruskal_lambda(indx_list_stra, colu_list_ordi)
print(f"Goodman and Kruskal's Lambda: {lambda_gender_age}")
```

### Explanation:
1. **Contingency Table**: The `pd.crosstab` function creates a contingency table from the two variables.
2. **Row and Column Sums**: These sums are used to calculate the errors \( E_1 \) and \( E_2 \).
3. **Errors Calculation**: \( E_1 \) is the number of errors made without knowledge of the other variable, and \( E_2 \) is the number of errors made with knowledge of the other variable.
4. **Lambda Calculation**: The Lambda values for predicting X from Y and Y from X are calculated, and the symmetric Lambda is the average of these two values.

This approach correctly applies Goodman and Kruskal's Lambda to your data.

'''