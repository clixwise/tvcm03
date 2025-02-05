import os
import sys
import pandas as pd
import scipy.stats as stats
import numpy as np

from statsmodels.stats.proportion import proportion_effectsize
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Step 1
exit_code = 0           
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.basename(__file__)
print (f"len(sys.argv): {len(sys.argv)}")
print (f"sys.argv: {sys.argv}")

file_path = script_dir
file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
path_inpu = os.path.join(file_path, file_inpu)
df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)
print (df1)





df =df1
cond_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
mbre_rnge = ['G', 'D']
inve_rnge = ['NA', 'VI']
inve_abso = df.copy()
inve_abso['inve'] = inve_abso['ceap'].apply(lambda x: 'VI' if x in cond_list else 'NA')
inve_abso = inve_abso.drop(columns=['ceap'])
inve_abso = inve_abso.groupby(['doss', 'mbre'], as_index=False).first()
print(f"inve_abso:{type(inve_abso)}\n{inve_abso}\n:{inve_abso.index}")
df1 = inve_abso
total_patients = len(df1)
male_patients = len(df1[df1['mbre'] == 'G'])
female_patients = len(df1[df1['mbre'] == 'D'])
ill_patients = len(df1[df1['inve'] == 'VI'])
healthy_patients = len(df1[df1['inve'] == 'NA'])

print ("\nDF1 STATS")
print(f"Total patients: {total_patients}")
print(f"Male patients: {male_patients} ({male_patients/total_patients:.2%})")
print(f"Female patients: {female_patients} ({female_patients/total_patients:.2%})")
print(f"Ill patients: {ill_patients} ({ill_patients/total_patients:.2%})")
print(f"Healthy patients: {healthy_patients} ({healthy_patients/total_patients:.2%})")




inve_abso = inve_abso.pivot_table(index='mbre', columns='inve', aggfunc='size', fill_value=0)
inve_abso = inve_abso.reindex(index=mbre_rnge)
inve_abso_tran = inve_abso.T
inve_abso_tran = inve_abso_tran.reindex(index=inve_rnge)
# Display
inve_abso_disp = inve_abso
#print(f"inve_abso_disp:{type(inve_abso_disp)}\n{inve_abso_disp}\n:{inve_abso_disp.index}")
inve_abso_tran_disp = inve_abso_tran
#rint(f"inve_abso_tran:{type(inve_abso_tran)}\n{inve_abso_tran}\n:{inve_abso_tran.index}")

# Rename the index
inve_abso.rename(index={'G': 'M', 'D': 'F'}, inplace=True)

# Rename the columns
inve_abso.rename(columns={'NA': 'N', 'VI': 'Y'}, inplace=True)
inve_abso.columns.name = 'Ill'
inve_abso.index.name = 'Gender'





if False:
    df1 = pd.DataFrame(data, index=index)
    # Assuming df1 is already loaded
    total_patients = len(df1)
    male_patients = len(df1[df1['Gender'] == 'M'])
    female_patients = len(df1[df1['Gender'] == 'F'])
    ill_patients = len(df1[df1['Ill'] == 'Y'])
    healthy_patients = len(df1[df1['Ill'] == 'N'])

    print(f"Total patients: {total_patients}")
    print(f"Male patients: {male_patients} ({male_patients/total_patients:.2%})")
    print(f"Female patients: {female_patients} ({female_patients/total_patients:.2%})")
    print(f"Ill patients: {ill_patients} ({ill_patients/total_patients:.2%})")
    print(f"Healthy patients: {healthy_patients} ({healthy_patients/total_patients:.2%})")
    # Assuming df1 is already loaded with your data
    # If not, you can load it like this:
    # df1 = pd.read_csv('your_data_file.csv')

    # Create the contingency table (df2)
    df2 = pd.crosstab(df1['Gender'], df1['Ill'])

    # Reorder columns to match the desired output (N, Y)
    df2 = df2[['N', 'Y']]

    print(df2)


if False:
    # Create the data dictionary
    data = {
        'N': [39, 66],
        'Y': [323, 296]
    }

    # Create the index
    index = pd.Index(['M', 'F'], name='gender')

    # Create the DataFrame
    df2 = pd.DataFrame(data, index=index)

    # Set the column name for the 'Ill' status
    df2.columns.name = 'ill'

    print(df2)
    
print ("\nDF2 STATS")
print(inve_abso)
df2 = inve_abso
chi2, p_value, dof, expected = stats.chi2_contingency(df2)
print(f"1/Chi-square statistic: {chi2:.4f} p-value: {p_value:.4f}")

odds_ratio, p_value = stats.fisher_exact(df2)
print(f"2/Odds ratio: {odds_ratio:.4f} p-value: {p_value:.4f}")

risk_male = df2.loc['M', 'Y'] / df2.loc['M'].sum()
risk_female = df2.loc['F', 'Y'] / df2.loc['F'].sum()
relative_risk = risk_male / risk_female
print(f"3/Relative risk (M/F): {relative_risk:.4f}")


male_ill = df2.loc['M', 'Y']
male_total = df2.loc['M'].sum()
male_proportion = male_ill / male_total
ci_male = proportion_effectsize(male_proportion, 0.5)  # 0.5 is a reference proportion, adjust as needed
female_ill = df2.loc['F', 'Y']
female_total = df2.loc['F'].sum()
female_proportion = female_ill / female_total
ci_female = proportion_effectsize(female_proportion, 0.5)  # 0.5 is a reference proportion, adjust as needed
print(f"4/Proportion effect size : ci_male ={ci_male} ci_female ={ci_female}")


print (df1)
# Logistic regression
if True:
    X = pd.get_dummies(df1['mbre'], drop_first=True) # X = pd.get_dummies(df1['Gender'], drop_first=True)
    #print (X)
    y = df1['inve'].map({'NA': 0, 'VI': 1}) # y = df1['Ill'].map({'N': 0, 'Y': 1})
    #print (y)
    X = X.astype(int)
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    print(model.summary())

# Confidence intervals for proportions
from statsmodels.stats.proportion import proportion_confint

alpha = 0.05  # 95% confidence interval
for gender in ['M', 'F']:
    n_ill = df2.loc[gender, 'Y']
    n_total = df2.loc[gender].sum()
    ci_lower, ci_upper = proportion_confint(n_ill, n_total, alpha=alpha, method='wilson')
    print(f"CI : {gender}: {ci_lower:.4f} - {ci_upper:.4f}")

# Risk difference
risk_m = df2.loc['M', 'Y'] / df2.loc['M'].sum()
risk_f = df2.loc['F', 'Y'] / df2.loc['F'].sum()
risk_diff = risk_m - risk_f
print(f"Risk Difference: {risk_diff:.4f}")

# Number to traet
nnt = 1 / abs(risk_diff)
print(f"Number Needed to Treat: {nnt:.2f}")

# PHI coef


chi2, _, _, _ = chi2_contingency(df2)
n = df2.sum().sum()
phi = (chi2 / n) ** 0.5
print(f"Phi Coefficient: {phi:.4f}")

#  Cramer V
min_dim = min(df2.shape) - 1
cramer_v = phi / (min_dim ** 0.5)
print(f"Cramer's V: {cramer_v:.4f}")

# Information Value (IV):
# This measures the predictive power of an independent variable in relation to the dependent variable.
def calculate_iv(df):
    iv = 0
    for col in df.columns:
        if col != 'Y':
            woe = np.log(df[col] / df['Y'])
            iv += (df[col] - df['Y']) * woe
    return iv.sum() / 100

iv = calculate_iv(df2)
print(f"Information Value: {iv:.4f}")