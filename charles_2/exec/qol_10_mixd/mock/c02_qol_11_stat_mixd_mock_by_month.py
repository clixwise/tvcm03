import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ****
#
# ****
def simulate_clinical_lmm(n_patients=50, timepoints=[0, 3, 6, 12]):
    data = []
    
    # --- STEP 1: Define Fixed Effects (The "Average" Patient) ---
    beta_0 = 60          # Intercept: Avg QoL at baseline
    beta_time = 0.5      # Improvement: +0.5 points per month
    beta_bmi = -0.7      # Impact: -0.7 points per unit of BMI
    beta_ceap = -4.0     # Impact: -4 points per CEAP class (C1=1, C2=2...)

    # --- STEP 2: Define Random Effects (Patient Heterogeneity) ---
    # Standard deviations for the random intercept and random slope
    sd_intercept = 5.0   
    sd_slope = 0.2       

    for i in range(n_patients):
        # Patient-specific baseline characteristics
        bmi = np.random.normal(28, 5) # BMI mean 28, sd 5
        ceap = np.random.randint(1, 7) # C1 to C6
        
        # Draw Random Effects for THIS patient
        u_0 = np.random.normal(0, sd_intercept) # Baseline shift
        u_1 = np.random.normal(0, sd_slope)     # Progress shift
        
        for t in timepoints:
            # --- STEP 3: The Missing Data Mechanism (Attrition) ---
            # Realism: Late timepoints are more likely to be missing
            if t > 0 and np.random.random() < (t * 0.02): # Up to 24% loss at month 12
                continue 

            # --- STEP 4: The Generative Equation ---
            # Fixed + Random + Individual Covariates + Residual Error
            error = np.random.normal(0, 2)
            
            qol = (beta_0 + u_0) + \
                  (beta_time + u_1) * t + \
                  (beta_ceap * ceap) + \
                  (beta_bmi * bmi) + \
                  error
            
            data.append({
                'PatientID': f'P{i:03d}',
                'Month': t,
                'CEAP': f'C{ceap}',
                'BMI': round(bmi, 1),
                'VEINES_QOL': round(qol, 2)
            })

    return pd.DataFrame(data)

df = simulate_clinical_lmm()
print(df)

# ****
#
# ****

# Define the model formula
# VEINES_QOL is the outcome
# Month, CEAP, and BMI are Fixed Effects
# (1 | PatientID) indicates a Random Intercept for each patient
model = smf.mixedlm("VEINES_QOL ~ Month + CEAP + BMI", 
                    df, 
                    groups=df["PatientID"])

# Fit the model
result = model.fit()

# View the results
print(result.summary())

# ****
#
# ****

# 1. Define your LMM results from Run 1
data = {
    'Variable': ['CEAP[T.C2]', 'CEAP[T.C3]', 'CEAP[T.C4]', 'CEAP[T.C5]', 'CEAP[T.C6]', 'Month', 'BMI'],
    'Coef': [-7.414, -11.752, -13.889, -14.531, -23.058, 0.461, -0.574],
    'StdErr': [2.831, 2.379, 2.644, 2.988, 2.488, 0.037, 0.172]
}

df = pd.DataFrame(data)

# 2. Calculate 95% CI (1.96 * StdErr)
df['low'] = df['Coef'] - (1.96 * df['StdErr'])
df['high'] = df['Coef'] + (1.96 * df['StdErr'])

# 3. Create the Plot
plt.figure(figsize=(10, 6))
plt.errorbar(df['Coef'], range(len(df)), 
             xerr=[df['Coef']-df['low'], df['high']-df['Coef']], 
             fmt='o', color='navy', capsize=5, elinewidth=2)

# 4. Add formatting and the "Zero line"
plt.axvline(x=0, color='crimson', linestyle='--')
plt.yticks(range(len(df)), df['Variable'])
plt.xlabel('Coefficient Value (Effect on VEINES-QOL)')
plt.title('Forest Plot of Predictors (LMM Run 1)')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('lmm_forest_plot.png')
plt.show()


'''
    PatientID  Month CEAP   BMI  VEINES_QOL
0        P000      0   C1  24.5       38.08
1        P000      3   C1  24.5       38.82
2        P000      6   C1  24.5       41.51
3        P001      0   C3  32.0       26.37
4        P001      3   C3  32.0       27.71
..        ...    ...  ...   ...         ...
178      P048      6   C3  26.7       20.86
179      P048     12   C3  26.7       21.72
180      P049      0   C3  38.2       19.03
181      P049      3   C3  38.2       15.92
182      P049     12   C3  38.2       15.69

[183 rows x 5 columns]
         Mixed Linear Model Regression Results
========================================================
Model:            MixedLM Dependent Variable: VEINES_QOL
No. Observations: 183     Method:             REML      
No. Groups:       50      Scale:              4.5016    
Min. group size:  2       Log-Likelihood:     -465.5918 
Max. group size:  4       Converged:          Yes       
Mean group size:  3.7
--------------------------------------------------------
            Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------
Intercept   54.241    5.551  9.771 0.000  43.361  65.120
CEAP[T.C2]  -7.414    2.831 -2.619 0.009 -12.962  -1.866
CEAP[T.C3] -11.752    2.379 -4.941 0.000 -16.414  -7.090
CEAP[T.C4] -13.889    2.644 -5.253 0.000 -19.072  -8.707
CEAP[T.C5] -14.531    2.988 -4.863 0.000 -20.388  -8.674
CEAP[T.C6] -23.058    2.488 -9.266 0.000 -27.935 -18.181
Month        0.461    0.037 12.494 0.000   0.389   0.534
BMI         -0.574    0.172 -3.329 0.001  -0.911  -0.236
Group Var   27.359    3.344
========================================================
'''
'''
         Mixed Linear Model Regression Results
========================================================
Model:            MixedLM Dependent Variable: VEINES_QOL
No. Observations: 176     Method:             REML      
No. Groups:       50      Scale:              4.4816    
Min. group size:  2       Log-Likelihood:     -454.2629 
Max. group size:  4       Converged:          Yes       
Mean group size:  3.5
--------------------------------------------------------
            Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------
Intercept   52.152    5.043 10.342 0.000  42.268  62.036
CEAP[T.C2]  -1.353    2.669 -0.507 0.612  -6.585   3.878
CEAP[T.C3]  -5.665    2.863 -1.979 0.048 -11.276  -0.054
CEAP[T.C4] -12.638    2.656 -4.759 0.000 -17.843  -7.432
CEAP[T.C5] -19.315    2.759 -7.002 0.000 -24.722 -13.908
CEAP[T.C6] -17.338    3.223 -5.379 0.000 -23.655 -11.021
Month        0.504    0.037 13.662 0.000   0.431   0.576
BMI         -0.580    0.177 -3.280 0.001  -0.926  -0.233
Group Var   34.960    4.296
========================================================
'''