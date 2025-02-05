import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix

import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix

# Create a sample DataFrame
data = {
    'doss': ['D00001', 'D00002', 'D00003', 'D00004', 'D00005', 'D00006'],
    'sexe': ['M', 'F', 'M', 'F', 'M', 'F'],
    'limb': ['L', 'R', 'L', 'R', 'L', 'R'],
    'age_bin': ['50_59', '60_69', '70_79', '40_49', '80_89', '50_59'],
    'ceap': ['C0', 'C2', 'C3', 'C1', 'C4', 'C2']
}

df = pd.DataFrame(data)

print("Sample DataFrame:")
print(df)
print("\n")

# Create design matrix without intercept and drop one level of each categorical variable
X = dmatrix('0 + C(sexe, Drop(1)) + C(limb, Drop(1)) + C(age_bin, Drop(1))', data=df, return_type='dataframe')

# Prepare y (outcome)
y = pd.Categorical(df['ceap']).codes

print("Predictors (X):")
print(X)
print("\n")

print("Outcome (y):")
print(pd.Series(y).value_counts())
print("\n")

# Fit the ordinal logistic regression model
try:
    model = OrderedModel(y, X, distr='logit')
    results = model.fit()

    # Print the summary of results
    print(results.summary())

    # Calculate and print odds ratios
    odds_ratios = np.exp(results.params)
    print("\nOdds Ratios:")
    print(odds_ratios)

except ValueError as e:
    print(f"An error occurred: {e}")