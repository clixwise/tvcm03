import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import kendalltau

# -------------------------------
# Goodman and Kruskal's Lambda [Line] Test of Independence
# -------------------------------
def goodman_kruskal_lambda_open_2(indx_list_stra, colu_list_ordi):
    # Create a contingency table
    contingency_table = pd.crosstab(indx_list_stra, colu_list_ordi)

    # Calculate totals
    max_total = contingency_table.sum(axis=1).max()  # Most frequent category of Y
    total = contingency_table.values.sum()

    # Calculate maximum frequencies by row
    max_by_row = contingency_table.max(axis=1).sum()

    # Ensure no negative values for Lambda
    numerator = max_by_row - max_total
    denominator = total - max_total

    if denominator <= 0 or numerator < 0:
        return 0  # Lambda must be non-negative

    return numerator / denominator

def lambda_permutation_test_open_2(x, y, n_permutations=10000):
    np.random.seed(42)
    observed_lambda = goodman_kruskal_lambda_open_2(x, y)
    permuted_lambdas = []

    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_lambda = goodman_kruskal_lambda_open_2(x, y_permuted)
        permuted_lambdas.append(permuted_lambda)

    p_value = np.mean(np.abs(permuted_lambdas) >= np.abs(observed_lambda))
    return observed_lambda, p_value

def bootstrap_lambda_open_2(indx_list_stra, colu_list_ordi, perm=5000, alpha=0.05):
    np.random.seed(42)
    # Store observed Lambda
    observed_lambda = goodman_kruskal_lambda_open_2(indx_list_stra, colu_list_ordi)

    # Combine the input series for easier resampling
    data = pd.DataFrame({'indx': indx_list_stra, 'colu': colu_list_ordi})

    # Initialize an array to store bootstrapped Lambdas
    lambda_bootstrap = []

    # Perform bootstrap sampling
    for _ in range(perm):
        # Resample with replacement
        resampled_data = data.sample(frac=1, replace=True)
        resampled_lambda = goodman_kruskal_lambda_open_2(
            resampled_data['indx'], resampled_data['colu']
        )
        lambda_bootstrap.append(resampled_lambda)

    # Convert to numpy array for easier processing
    lambda_bootstrap = np.array(lambda_bootstrap)

    # Calculate confidence intervals
    ci_lower = np.percentile(lambda_bootstrap, 100 * (alpha / 2))
    ci_upper = np.percentile(lambda_bootstrap, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper
# ----
# NOTE : The result is strange : there is an association but when we perform the permutation, then 'PVAL = 1' !!! 
# Rather refer to DF_TABL version which is OK
# ----
def goodkruslamb_line(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True
    
    # 'df_tabl' only 
    if df1 is None:
        print(f"\nData : {what}\n(df_table) : Goodman and Kruskal's Lambda [Line] : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Goodman and Kruskal's Lambda [Line] : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
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
    if False:
        data = {
            'sexe_stra': indx_list_stra, # sexe [1, 1, 1, 1, 0, ...]
            'ceap_ordi': colu_list_ordi  # ceap [3, 7, 3, 7, 4, ...]
        }
        df = pd.DataFrame(data)
        indx_list_stra = df['sexe_stra']
        colu_list_ordi = df['ceap_ordi']
        #indx_list_stra = pd.Series([1, 1, 1, 1, 0, 1, 1, 0, 0, 1], name='sexe_stra')
        #colu_list_ordi = pd.Series([3, 7, 3, 7, 4, 0, 0, 0, 0, 0], name='ceap_ordi')
    lambda_open_1 = goodman_kruskal_lambda_open_2(indx_list_stra, colu_list_ordi)
    lambda_open_2 = goodman_kruskal_lambda_open_2(colu_list_ordi, indx_list_stra)
    main(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, indx_list_stra, colu_list_ordi)
    main(what, df, colu_cate_list, indx_cate_list, colu_name, indx_name, colu_name_ordi, indx_name_stra, colu_list_ordi, indx_list_stra)
    pass
def main(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, indx_list_stra, colu_list_ordi):
    
    # indx_list_stra: Independent variable (predictor)
    # colu_list_ordi: Dependent variable (predicted)
    predicted = colu_list_ordi
    predictor = indx_list_stra
    lambda_1 = goodman_kruskal_lambda_open_2(indx_list_stra, colu_list_ordi)
    lambda_1_form = round(lambda_1 * 100)
    print(f"Goodman and Kruskal's Lambda [Line] : ({colu_name_ordi} | {indx_name_stra}) ({colu_name_ordi}=f({indx_name_stra})) (predictor:{indx_name_stra} ; predicted:{colu_name_ordi}): {lambda_1_form}")

    # ----
    # Exec 2
    # ----
    perm = 5000
    print (indx_list_stra)
    print (colu_list_ordi)
    lambda_1_perm, p_value_1 = lambda_permutation_test_open_2(indx_list_stra, colu_list_ordi, perm)
    if lambda_1_perm != lambda_1:
        raise Exception()  
    p_value_1_form = f"{p_value_1:.3e}" if p_value_1 < 0.001 else f"{p_value_1:.3f}"
    
    # ----
    # Exec 3 : confidence interval
    # ----
    perm=5000
    ci_lower_1, ci_upper_1 = bootstrap_lambda_open_2(indx_list_stra, colu_list_ordi, perm) # df['Gender_num'].values, df['Age_Ordinal'].values   
    ci_lower_1_form = f"{ci_lower_1:.3e}" if ci_lower_1 < 0.001 else f"{ci_lower_1:.3f}"
    ci_upper_1_form = f"{ci_upper_1:.3e}" if ci_upper_1 < 0.001 else f"{ci_upper_1:.3f}"
    
    # Resu
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    # Claude
    # H0 : The '{colu_name}' variable does not provide any information about the'{indx_name}' variable.
    # Ha : The '{colu_name}' variable does provide information about the'{indx_name}' variable.
    if np.isnan(lambda_1_form):
        raise Exception("Stat or Pval are NaN")
    print(f"\nData : {what}\nGoodman and Kruskal's Lambda : {indx_name} predicts {colu_name}")
    print(f"Stat: {lambda_1_form} Pval : {p_value_1_form} 95% CI: ({ci_lower_1_form}, {ci_upper_1_form})")
    print(f"Given '{indx_name}' value and asked to predict '{colu_name}' value, Lamda reduces by {lambda_1_form} % the error probability")
 
    # Intp
    alpha = 0.05
    H0_1 = f"H0 : The '{indx_name}' variable does not provide any information about the'{colu_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha_1 = f"Ha : The '{indx_name}' variable does provide information about the'{colu_name}' variable.\n({colu_cate_list}) vs ({indx_cate_list})"
    if p_value_1 < alpha:
        print(f"Goodman and Kruskal's Lambda [Line] : Reject the null hypothesis:\n{Ha_1}")
        write(f"Goodman and Kruskal's Lambda [Line] : Reject the null hypothesis:\n{Ha_1}")
    else:
        print(f"Goodman and Kruskal's Lambda [Line] : Fail to reject the null hypothesis:\n{H0_1}")
        write(f"Goodman and Kruskal's Lambda [Line] : Fail to reject the null hypothesis:\n{H0_1}")

 
'''
2025_01_12 Mistral
The approach you've outlined for applying Goodman and Kruskal's Lambda [Line] to your data has a few issues. Goodman and Kruskal's Lambda [Line] is a measure of association for nominal variables, not ordinal variables. It measures the proportional reduction in error when predicting one nominal variable from another. The formula you've provided using Kendall's Tau is not correct for Goodman and Kruskal's Lambda [Line].

Goodman and Kruskal's Lambda [Line] is calculated as follows:

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

Given your data, you need to calculate the contingency table first and then apply the Goodman and Kruskal's Lambda [Line] formula. Here's how you can do it:

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
print(f"Goodman and Kruskal's Lambda [Line]: {lambda_gender_age}")
```

### Explanation:
1. **Contingency Table**: The `pd.crosstab` function creates a contingency table from the two variables.
2. **Row and Column Sums**: These sums are used to calculate the errors \( E_1 \) and \( E_2 \).
3. **Errors Calculation**: \( E_1 \) is the number of errors made without knowledge of the other variable, and \( E_2 \) is the number of errors made with knowledge of the other variable.
4. **Lambda Calculation**: The Lambda values for predicting X from Y and Y from X are calculated, and the symmetric Lambda is the average of these two values.

This approach correctly applies Goodman and Kruskal's Lambda [Line] to your data.

'''