import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Example DataFrame
df = pd.DataFrame({
    'doss': ['D9972', 'D9972', 'D9921', 'D8930', 'D8956'],
    'age': [54, 54, 54, 45, 50],
    'sexe': ['F', 'F', 'M', 'F', 'F'],
    'unbi': ['B', 'B', 'B', 'U', 'U'],
    'mbas': ['G2', 'G2', 'G2', 'NA', 'NA'],
    'mbre': ['G', 'D', 'G', 'D', 'G'],
    'ceap': ['C2', 'C6', 'C3', 'NA', 'NA']
})

# 1. Binary Encoding for Sexe
df['sexe'] = df['sexe'].map({'F': 0, 'M': 1})

# 2. One-Hot Encoding for Nominal Categorical Variables
nominal_cols = ['unbi', 'mbas', 'mbre']
df_nominal = pd.get_dummies(df[nominal_cols], prefix=nominal_cols)

# 3. Ordinal Encoding for 'ceap'
ordinal_mapping = {'NA': 0, 'C0': 1, 'C1': 2, 'C2': 3, 'C3': 4, 'C4': 5, 'C5': 6, 'C6': 7}
df['ceap'] = df['ceap'].map(ordinal_mapping)

# 4. Combine Encoded Features
df_encoded = pd.concat([df.drop(columns=nominal_cols), df_nominal], axis=1)

# Display Result
print(df_encoded)